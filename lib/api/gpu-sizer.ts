import type { GpuSizerRequest } from './schemas'

// ─── Types ───────────────────────────────────────────────────────────────────

interface GpuSizerWarning {
  code: string
  message: string
}

export interface GpuSizerResult {
  requestId: string
  status: 'completed'
  recommendation: {
    gpusNeeded: number
    totalGpus: number
    tensorParallelSize: number
    pipelineParallelSize: number
    dataParallelSize: number
  }
  performance: {
    ttftLatencyMs: number
    tpotMs: number
    requestLatencyMs: number
    concurrency: number
  }
  throughput: {
    tokensPerSecond: number
    tokensPerSecondPerGpu: number
    tokensPerSecondPerUser: number
  }
  memory: {
    value: number
    unit: 'GB'
    scope: 'unspecified'
  }
  metadata: {
    modelPath: string
    system: string
    inputTokens: number
    outputTokens: number
    targetTtftMs: number
    durationMs: number
  }
  warnings: GpuSizerWarning[]
}

export interface GpuSizerErrorResponse {
  requestId: string
  status: 'failed'
  error: {
    code: string
    message: string
  }
}

export type GpuSizerResponse = GpuSizerResult | GpuSizerErrorResponse

// ─── Request ID ──────────────────────────────────────────────────────────────

export function generateRequestId(): string {
  return 'size_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
}

// ─── Service ─────────────────────────────────────────────────────────────────

const DEFAULT_URL = 'http://163.74.81.138:7860/gpu_sizer'
const DEFAULT_TIMEOUT_SECONDS = 90

export async function callGpuSizer(
  request: GpuSizerRequest
): Promise<GpuSizerResponse> {
  const requestId = generateRequestId()
  const startTime = performance.now()

  const url = process.env.GPU_SIZER_URL || DEFAULT_URL
  const username = process.env.GPU_SIZER_USERNAME
  const password = process.env.GPU_SIZER_PASSWORD
  const timeoutSeconds = parseInt(process.env.GPU_SIZER_TIMEOUT_SECONDS || '', 10) || DEFAULT_TIMEOUT_SECONDS

  if (!username || !password) {
    return makeError(requestId, 'GPU_SIZER_NOT_CONFIGURED', 'GPU sizing service is not configured')
  }

  const externalPayload: Record<string, unknown> = {
    model_path: request.model_path,
    system: request.system,
    isl: request.isl,
    osl: request.osl,
    ttft: request.ttft,
    username,
    password,
  }

  if (request.tps_per_user !== undefined) externalPayload.tps_per_user = request.tps_per_user
  if (request.e2e !== undefined) externalPayload.e2e = request.e2e
  if (request.batch_size !== undefined) externalPayload.batch_size = request.batch_size

  let response: Response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(externalPayload),
      signal: AbortSignal.timeout(timeoutSeconds * 1000),
    })
  } catch (err: unknown) {
    const durationMs = Math.round(performance.now() - startTime)
    if (err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')) {
      return makeError(requestId, 'GPU_SIZER_TIMEOUT', `The GPU sizing engine did not respond within ${timeoutSeconds} seconds (waited ${durationMs}ms)`)
    }
    return makeError(requestId, 'GPU_SIZER_UNAVAILABLE', 'GPU sizing service is unreachable')
  }

  if (response.status === 401 || response.status === 403) {
    return makeError(requestId, 'GPU_SIZER_AUTH_FAILED', 'Authentication with GPU sizing service failed')
  }

  if (!response.ok) {
    return makeError(requestId, 'GPU_SIZER_UNAVAILABLE', `GPU sizing service returned HTTP ${response.status}`)
  }

  let rawData: Record<string, unknown>
  try {
    const parsed = await response.json()
    if (parsed == null || typeof parsed !== 'object') {
      return makeError(requestId, 'GPU_SIZER_NO_CONFIGURATION', 'No valid GPU configuration found for this model and hardware combination. Try a different GPU system or adjust your parameters.')
    }
    rawData = parsed as Record<string, unknown>
  } catch {
    return makeError(requestId, 'GPU_SIZER_INVALID_RESPONSE', 'GPU sizing service returned non-JSON response')
  }

  const requiredFields = [
    'gpus_needed', 'ttft_latency', 'concurrency', 'tpot_ms',
    'tokens_per_second', 'memory', 'tp_size', 'pp_size', 'dp_size',
  ] as const

  const missingFields = requiredFields.filter(f => typeof rawData[f] !== 'number')
  if (missingFields.length > 0) {
    return makeError(
      requestId,
      'GPU_SIZER_INVALID_RESPONSE',
      `GPU sizing response missing required fields: ${missingFields.join(', ')}`
    )
  }

  const gpusNeeded = rawData.gpus_needed as number
  const totalGpus = (typeof rawData.num_total_gpus === 'number' ? rawData.num_total_gpus : gpusNeeded) as number
  const tp = rawData.tp_size as number
  const pp = rawData.pp_size as number
  const dp = rawData.dp_size as number

  const warnings: GpuSizerWarning[] = []

  const parallelismProduct = tp * pp * dp
  if (parallelismProduct !== totalGpus) {
    warnings.push({
      code: 'GPU_TOPOLOGY_MISMATCH',
      message: `Parallelism dimensions (TP=${tp} x PP=${pp} x DP=${dp} = ${parallelismProduct}) do not equal total GPU count (${totalGpus})`,
    })
  }

  if (gpusNeeded !== totalGpus) {
    warnings.push({
      code: 'GPU_COUNT_MISMATCH',
      message: `gpus_needed (${gpusNeeded}) differs from num_total_gpus (${totalGpus})`,
    })
  }

  const durationMs = Math.round(performance.now() - startTime)

  return {
    requestId,
    status: 'completed',
    recommendation: {
      gpusNeeded,
      totalGpus,
      tensorParallelSize: tp,
      pipelineParallelSize: pp,
      dataParallelSize: dp,
    },
    performance: {
      ttftLatencyMs: rawData.ttft_latency as number,
      tpotMs: rawData.tpot_ms as number,
      requestLatencyMs: typeof rawData.request_latency === 'number' ? rawData.request_latency as number : 0,
      concurrency: rawData.concurrency as number,
    },
    throughput: {
      tokensPerSecond: rawData.tokens_per_second as number,
      tokensPerSecondPerGpu: typeof rawData.tokens_per_second_per_gpu === 'number' ? rawData.tokens_per_second_per_gpu as number : 0,
      tokensPerSecondPerUser: typeof rawData.tokens_per_second_per_user === 'number' ? rawData.tokens_per_second_per_user as number : 0,
    },
    memory: {
      value: rawData.memory as number,
      unit: 'GB',
      scope: 'unspecified',
    },
    metadata: {
      modelPath: request.model_path,
      system: request.system,
      inputTokens: request.isl,
      outputTokens: request.osl,
      targetTtftMs: request.ttft,
      durationMs,
    },
    warnings,
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): GpuSizerErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}
