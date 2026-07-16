import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { GpuSizerRequestSchema } from '../schemas'
import { callGpuSizer, generateRequestId } from '../gpu-sizer'
import type { GpuSizerResult, GpuSizerErrorResponse } from '../gpu-sizer'

// ─── Fixtures ────────────────────────────────────────────────────────────────

const VALID_REQUEST = {
  model_path: 'meta-llama/Llama-3.1-70B-Instruct',
  system: 'h200_sxm',
  isl: 2048,
  osl: 128,
  ttft: 1000,
}

const EXTERNAL_RESPONSE = {
  gpus_needed: 4,
  ttft_latency: 599.81,
  concurrency: 128.0,
  tpot_ms: 149.95,
  request_latency: 19643.78,
  tokens_per_second: 846.93,
  tokens_per_second_per_gpu: 211.73,
  tokens_per_second_per_user: 6.67,
  num_total_gpus: 4,
  memory: 58.61,
  tp_size: 4,
  pp_size: 1,
  dp_size: 1,
}

function mockFetchOk(data: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  })
}

// ─── Schema Tests ────────────────────────────────────────────────────────────

describe('GpuSizerRequestSchema', () => {
  it('accepts a valid minimal request', () => {
    const result = GpuSizerRequestSchema.safeParse(VALID_REQUEST)
    expect(result.success).toBe(true)
  })

  it('accepts a request with all optional fields', () => {
    const result = GpuSizerRequestSchema.safeParse({
      ...VALID_REQUEST,
      tps_per_user: 10,
      e2e: 5000,
      batch_size: 32,
    })
    expect(result.success).toBe(true)
  })

  it('rejects missing model_path', () => {
    const { model_path, ...rest } = VALID_REQUEST
    const result = GpuSizerRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects empty model_path', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, model_path: '' })
    expect(result.success).toBe(false)
  })

  it('rejects missing system', () => {
    const { system, ...rest } = VALID_REQUEST
    const result = GpuSizerRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects zero isl', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, isl: 0 })
    expect(result.success).toBe(false)
  })

  it('rejects negative osl', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, osl: -1 })
    expect(result.success).toBe(false)
  })

  it('rejects non-integer isl', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, isl: 2048.5 })
    expect(result.success).toBe(false)
  })

  it('rejects zero ttft', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, ttft: 0 })
    expect(result.success).toBe(false)
  })

  it('rejects negative ttft', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, ttft: -100 })
    expect(result.success).toBe(false)
  })

  it('rejects unknown fields (strict mode)', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, extra: 'nope' })
    expect(result.success).toBe(false)
  })

  it('rejects username field from client', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, username: 'hacker' })
    expect(result.success).toBe(false)
  })

  it('rejects password field from client', () => {
    const result = GpuSizerRequestSchema.safeParse({ ...VALID_REQUEST, password: 'secret' })
    expect(result.success).toBe(false)
  })
})

// ─── Service Tests ───────────────────────────────────────────────────────────

describe('callGpuSizer', () => {
  beforeEach(() => {
    vi.stubEnv('GPU_SIZER_USERNAME', 'test-user')
    vi.stubEnv('GPU_SIZER_PASSWORD', 'test-pass')
    vi.stubEnv('GPU_SIZER_URL', 'http://fake-gpu-sizer:7860/gpu_sizer')
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it('returns a normalized response on success', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('completed')
    const r = result as GpuSizerResult
    expect(r.requestId).toMatch(/^size_/)
    expect(r.recommendation.gpusNeeded).toBe(4)
    expect(r.recommendation.totalGpus).toBe(4)
    expect(r.recommendation.tensorParallelSize).toBe(4)
    expect(r.recommendation.pipelineParallelSize).toBe(1)
    expect(r.recommendation.dataParallelSize).toBe(1)
    expect(r.performance.ttftLatencyMs).toBe(599.81)
    expect(r.performance.tpotMs).toBe(149.95)
    expect(r.performance.concurrency).toBe(128)
    expect(r.throughput.tokensPerSecond).toBe(846.93)
    expect(r.throughput.tokensPerSecondPerGpu).toBe(211.73)
    expect(r.throughput.tokensPerSecondPerUser).toBe(6.67)
    expect(r.memory).toEqual({ value: 58.61, unit: 'GB', scope: 'unspecified' })
    expect(r.metadata.modelPath).toBe('meta-llama/Llama-3.1-70B-Instruct')
    expect(r.metadata.system).toBe('h200_sxm')
    expect(r.metadata.inputTokens).toBe(2048)
    expect(r.metadata.outputTokens).toBe(128)
    expect(r.metadata.targetTtftMs).toBe(1000)
    expect(r.metadata.durationMs).toBeGreaterThanOrEqual(0)
    expect(r.warnings).toEqual([])
  })

  it('includes optional fields in the upstream request when supplied', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callGpuSizer({ ...VALID_REQUEST, tps_per_user: 10, batch_size: 32 })

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.tps_per_user).toBe(10)
    expect(sentBody.batch_size).toBe(32)
  })

  it('omits optional fields from upstream when not supplied', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callGpuSizer(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody).not.toHaveProperty('tps_per_user')
    expect(sentBody).not.toHaveProperty('e2e')
    expect(sentBody).not.toHaveProperty('batch_size')
  })

  it('injects credentials into the upstream request', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callGpuSizer(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.username).toBe('test-user')
    expect(sentBody.password).toBe('test-pass')
  })

  it('never returns credentials in the response', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callGpuSizer(VALID_REQUEST)
    const json = JSON.stringify(result)

    expect(json).not.toContain('test-user')
    expect(json).not.toContain('test-pass')
    expect(json).not.toContain('username')
    expect(json).not.toContain('password')
  })

  it('returns GPU_SIZER_NOT_CONFIGURED when credentials are missing', async () => {
    vi.stubEnv('GPU_SIZER_USERNAME', '')
    vi.stubEnv('GPU_SIZER_PASSWORD', '')

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_NOT_CONFIGURED')
  })

  it('returns GPU_SIZER_NOT_CONFIGURED when password is missing', async () => {
    vi.stubEnv('GPU_SIZER_PASSWORD', '')

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_NOT_CONFIGURED')
  })

  it('returns GPU_SIZER_TIMEOUT on fetch timeout', async () => {
    const timeoutError = new Error('signal timed out')
    timeoutError.name = 'TimeoutError'
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(timeoutError))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_TIMEOUT')
  })

  it('returns GPU_SIZER_UNAVAILABLE on network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_UNAVAILABLE')
  })

  it('returns GPU_SIZER_AUTH_FAILED on 401', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: () => Promise.resolve({}),
    }))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_AUTH_FAILED')
  })

  it('returns GPU_SIZER_AUTH_FAILED on 403', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      json: () => Promise.resolve({}),
    }))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_AUTH_FAILED')
  })

  it('returns GPU_SIZER_UNAVAILABLE on 500', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({}),
    }))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_UNAVAILABLE')
    expect((result as GpuSizerErrorResponse).error.message).toContain('500')
  })

  it('returns GPU_SIZER_INVALID_RESPONSE on non-JSON response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error('invalid json')),
    }))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as GpuSizerErrorResponse).error.code).toBe('GPU_SIZER_INVALID_RESPONSE')
  })

  it('returns GPU_SIZER_INVALID_RESPONSE when response is missing required fields', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ gpus_needed: 4 }))

    const result = await callGpuSizer(VALID_REQUEST)

    expect(result.status).toBe('failed')
    const err = result as GpuSizerErrorResponse
    expect(err.error.code).toBe('GPU_SIZER_INVALID_RESPONSE')
    expect(err.error.message).toContain('missing required fields')
  })

  it('adds GPU_TOPOLOGY_MISMATCH warning when parallelism does not match GPU count', async () => {
    const mismatchResponse = {
      ...EXTERNAL_RESPONSE,
      tp_size: 2,
      pp_size: 1,
      dp_size: 1,
      num_total_gpus: 4,
    }
    vi.stubGlobal('fetch', mockFetchOk(mismatchResponse))

    const result = await callGpuSizer(VALID_REQUEST) as GpuSizerResult

    expect(result.status).toBe('completed')
    expect(result.warnings.some(w => w.code === 'GPU_TOPOLOGY_MISMATCH')).toBe(true)
  })

  it('adds GPU_COUNT_MISMATCH warning when gpus_needed differs from num_total_gpus', async () => {
    const mismatchResponse = {
      ...EXTERNAL_RESPONSE,
      gpus_needed: 8,
      num_total_gpus: 4,
    }
    vi.stubGlobal('fetch', mockFetchOk(mismatchResponse))

    const result = await callGpuSizer(VALID_REQUEST) as GpuSizerResult

    expect(result.status).toBe('completed')
    expect(result.warnings.some(w => w.code === 'GPU_COUNT_MISMATCH')).toBe(true)
  })

  it('handles missing optional response fields gracefully', async () => {
    const minimalResponse = {
      gpus_needed: 2,
      ttft_latency: 300,
      concurrency: 64,
      tpot_ms: 100,
      tokens_per_second: 500,
      memory: 30,
      tp_size: 2,
      pp_size: 1,
      dp_size: 1,
    }
    vi.stubGlobal('fetch', mockFetchOk(minimalResponse))

    const result = await callGpuSizer(VALID_REQUEST) as GpuSizerResult

    expect(result.status).toBe('completed')
    expect(result.recommendation.totalGpus).toBe(2)
    expect(result.performance.requestLatencyMs).toBe(0)
    expect(result.throughput.tokensPerSecondPerGpu).toBe(0)
    expect(result.throughput.tokensPerSecondPerUser).toBe(0)
  })

  it('includes durationMs in metadata', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callGpuSizer(VALID_REQUEST) as GpuSizerResult

    expect(result.metadata.durationMs).toBeTypeOf('number')
    expect(result.metadata.durationMs).toBeGreaterThanOrEqual(0)
  })
})

// ─── Request ID Tests ────────────────────────────────────────────────────────

describe('generateRequestId', () => {
  it('starts with size_ prefix', () => {
    expect(generateRequestId()).toMatch(/^size_/)
  })

  it('is 17 characters long', () => {
    expect(generateRequestId()).toHaveLength(17)
  })

  it('generates unique IDs', () => {
    const ids = new Set(Array.from({ length: 100 }, generateRequestId))
    expect(ids.size).toBe(100)
  })
})
