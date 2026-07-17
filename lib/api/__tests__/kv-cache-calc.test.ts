import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { KvCacheCalcRequestSchema } from '../schemas'
import { callKvCacheCalc, generateKvRequestId } from '../kv-cache-calc'
import type { KvCacheCalcResult, KvCacheCalcErrorResponse } from '../kv-cache-calc'

// ─── Fixtures ────────────────────────────────────────────────────────────────

const VALID_REQUEST = {
  model_path: 'meta-llama/Llama-3.1-70B-Instruct',
  system: 'h200_sxm',
  backend: 'vllm',
  max_num_tokens: 4096,
  max_batch_size: 128,
}

const EXTERNAL_RESPONSE = {
  total_gpu_capacity_bytes: 85899345920,
  total_kv_size_bytes: 65343062016,
  kv_size_per_token_bytes: 131072,
  total_kv_size_tokens: 498528,
  source: 'native',
  memory_breakdown: {
    weights_bytes: 16059990016,
    activations_bytes: 738197504,
    runtime_overhead_bytes: 3758096384,
    comm_overhead_bytes: 0,
  },
}

function mockFetchOk(data: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  })
}

// ─── Schema Tests ────────────────────────────────────────────────────────────

describe('KvCacheCalcRequestSchema', () => {
  it('accepts a valid request', () => {
    const result = KvCacheCalcRequestSchema.safeParse(VALID_REQUEST)
    expect(result.success).toBe(true)
  })

  it('rejects missing model_path', () => {
    const { model_path, ...rest } = VALID_REQUEST
    const result = KvCacheCalcRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects empty model_path', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, model_path: '' })
    expect(result.success).toBe(false)
  })

  it('rejects missing system', () => {
    const { system, ...rest } = VALID_REQUEST
    const result = KvCacheCalcRequestSchema.safeParse(rest)
    expect(result.success).toBe(false)
  })

  it('rejects empty system', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, system: '' })
    expect(result.success).toBe(false)
  })

  it('rejects unknown fields (strict mode)', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, extra: 'nope' })
    expect(result.success).toBe(false)
  })

  it('rejects username field from client', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, username: 'hacker' })
    expect(result.success).toBe(false)
  })

  it('rejects password field from client', () => {
    const result = KvCacheCalcRequestSchema.safeParse({ ...VALID_REQUEST, password: 'secret' })
    expect(result.success).toBe(false)
  })

  it('applies defaults when optional fields are omitted', () => {
    const result = KvCacheCalcRequestSchema.safeParse({
      model_path: 'meta-llama/Llama-3.1-8B',
      system: 'h100_sxm',
    })
    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data.backend).toBe('vllm')
      expect(result.data.max_num_tokens).toBe(4096)
      expect(result.data.max_batch_size).toBe(128)
    }
  })
})

// ─── Service Tests ───────────────────────────────────────────────────────────

describe('callKvCacheCalc', () => {
  beforeEach(() => {
    vi.stubEnv('AICONFIGURATOR_USERNAME', 'test-user')
    vi.stubEnv('AICONFIGURATOR_PASSWORD', 'test-pass')
    vi.stubEnv('AICONFIGURATOR_API_URL', 'http://fake-aiconfigurator:7860')
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it('returns a normalized response on success', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('completed')
    const r = result as KvCacheCalcResult
    expect(r.requestId).toMatch(/^kv_/)
    expect(r.kvCache.totalBytes).toBe(65343062016)
    expect(r.kvCache.perTokenBytes).toBe(131072)
    expect(r.kvCache.totalTokens).toBe(498528)
    expect(r.memoryBreakdown.weightsBytes).toBe(16059990016)
    expect(r.memoryBreakdown.activationsBytes).toBe(738197504)
    expect(r.memoryBreakdown.runtimeOverheadBytes).toBe(3758096384)
    expect(r.memoryBreakdown.commOverheadBytes).toBe(0)
    expect(r.gpuCapacity.totalBytes).toBe(85899345920)
    expect(r.metadata.modelPath).toBe('meta-llama/Llama-3.1-70B-Instruct')
    expect(r.metadata.backend).toBe('vllm')
    expect(r.metadata.system).toBe('h200_sxm')
    expect(r.metadata.maxNumTokens).toBe(4096)
    expect(r.metadata.maxBatchSize).toBe(128)
    expect(r.metadata.source).toBe('native')
    expect(r.metadata.durationMs).toBeGreaterThanOrEqual(0)
  })

  it('injects credentials into the upstream request', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.username).toBe('test-user')
    expect(sentBody.password).toBe('test-pass')
  })

  it('sends backend, max_num_tokens, max_batch_size in the upstream request', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.backend).toBe('vllm')
    expect(sentBody.max_num_tokens).toBe(4096)
    expect(sentBody.max_batch_size).toBe(128)
  })

  it('sends allow_hf_config_download: true in the upstream request', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
    expect(sentBody.allow_hf_config_download).toBe(true)
  })

  it('sends request to /kv_cache_calc endpoint', async () => {
    const mockFetch = mockFetchOk(EXTERNAL_RESPONSE)
    vi.stubGlobal('fetch', mockFetch)

    await callKvCacheCalc(VALID_REQUEST)

    expect(mockFetch.mock.calls[0][0]).toBe('http://fake-aiconfigurator:7860/kv_cache_calc')
  })

  it('never returns credentials in the response', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callKvCacheCalc(VALID_REQUEST)
    const json = JSON.stringify(result)

    expect(json).not.toContain('test-user')
    expect(json).not.toContain('test-pass')
    expect(json).not.toContain('"username"')
    expect(json).not.toContain('"password"')
  })

  it('returns KV_CACHE_NOT_CONFIGURED when credentials are missing', async () => {
    vi.stubEnv('AICONFIGURATOR_USERNAME', '')
    vi.stubEnv('AICONFIGURATOR_PASSWORD', '')

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_NOT_CONFIGURED')
  })

  it('returns KV_CACHE_NOT_CONFIGURED when password is missing', async () => {
    vi.stubEnv('AICONFIGURATOR_PASSWORD', '')

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_NOT_CONFIGURED')
  })

  it('returns KV_CACHE_TIMEOUT on fetch timeout', async () => {
    const timeoutError = new Error('signal timed out')
    timeoutError.name = 'TimeoutError'
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(timeoutError))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_TIMEOUT')
  })

  it('returns KV_CACHE_UNAVAILABLE on network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_UNAVAILABLE')
  })

  it('returns KV_CACHE_AUTH_FAILED on 401', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: () => Promise.resolve({}),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_AUTH_FAILED')
  })

  it('returns KV_CACHE_AUTH_FAILED on 403', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      json: () => Promise.resolve({}),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_AUTH_FAILED')
  })

  it('returns KV_CACHE_UNAVAILABLE on 500', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({}),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_UNAVAILABLE')
    expect((result as KvCacheCalcErrorResponse).error.message).toContain('500')
  })

  it('returns KV_CACHE_INVALID_RESPONSE on non-JSON response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error('invalid json')),
    }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.code).toBe('KV_CACHE_INVALID_RESPONSE')
  })

  it('returns KV_CACHE_INVALID_RESPONSE when total_kv_size_bytes is missing', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ memory_breakdown: {} }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    const err = result as KvCacheCalcErrorResponse
    expect(err.error.code).toBe('KV_CACHE_INVALID_RESPONSE')
    expect(err.error.message).toContain('total_kv_size_bytes')
  })

  it('surfaces API error messages from the error field', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ error: 'unsupported model' }))

    const result = await callKvCacheCalc(VALID_REQUEST)

    expect(result.status).toBe('failed')
    expect((result as KvCacheCalcErrorResponse).error.message).toBe('unsupported model')
  })

  it('handles missing optional response fields gracefully', async () => {
    vi.stubGlobal('fetch', mockFetchOk({ total_kv_size_bytes: 1024 }))

    const result = await callKvCacheCalc(VALID_REQUEST) as KvCacheCalcResult

    expect(result.status).toBe('completed')
    expect(result.kvCache.totalBytes).toBe(1024)
    expect(result.kvCache.perTokenBytes).toBe(0)
    expect(result.kvCache.totalTokens).toBe(0)
    expect(result.memoryBreakdown.weightsBytes).toBe(0)
    expect(result.gpuCapacity.totalBytes).toBe(0)
    expect(result.metadata.source).toBe('unknown')
  })

  it('includes durationMs in metadata', async () => {
    vi.stubGlobal('fetch', mockFetchOk(EXTERNAL_RESPONSE))

    const result = await callKvCacheCalc(VALID_REQUEST) as KvCacheCalcResult

    expect(result.metadata.durationMs).toBeTypeOf('number')
    expect(result.metadata.durationMs).toBeGreaterThanOrEqual(0)
  })
})

// ─── Request ID Tests ────────────────────────────────────────────────────────

describe('generateKvRequestId', () => {
  it('starts with kv_ prefix', () => {
    expect(generateKvRequestId()).toMatch(/^kv_/)
  })

  it('is 15 characters long', () => {
    expect(generateKvRequestId()).toHaveLength(15)
  })

  it('generates unique IDs', () => {
    const ids = new Set(Array.from({ length: 100 }, generateKvRequestId))
    expect(ids.size).toBe(100)
  })
})
