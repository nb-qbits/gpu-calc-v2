import type { KvCacheCalcRequest } from './schemas'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface KvCacheCalcResult {
  requestId: string
  status: 'completed'
  kvCache: {
    totalBytes: number
    perTokenBytes: number
    totalTokens: number
  }
  memoryBreakdown: {
    weightsBytes: number
    activationsBytes: number
    runtimeOverheadBytes: number
    commOverheadBytes: number
  }
  gpuCapacity: {
    totalBytes: number
  }
  metadata: {
    modelPath: string
    backend: string
    system: string
    maxNumTokens: number
    maxBatchSize: number
    source: string
    durationMs: number
  }
}

export interface KvCacheCalcErrorResponse {
  requestId: string
  status: 'failed'
  error: {
    code: string
    message: string
  }
}

export type KvCacheCalcResponse = KvCacheCalcResult | KvCacheCalcErrorResponse

// ─── Request ID ──────────────────────────────────────────────────────────────

export function generateKvRequestId(): string {
  return 'kv_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
}

// ─── Service ─────────────────────────────────────────────────────────────────

const DEFAULT_URL = 'http://163.74.81.138:7860'
const DEFAULT_TIMEOUT_SECONDS = 90

export async function callKvCacheCalc(
  request: KvCacheCalcRequest
): Promise<KvCacheCalcResponse> {
  const requestId = generateKvRequestId()
  const startTime = performance.now()

  const baseUrl = process.env.AICONFIGURATOR_API_URL || DEFAULT_URL
  const username = process.env.AICONFIGURATOR_USERNAME
  const password = process.env.AICONFIGURATOR_PASSWORD
  const timeoutSeconds = parseInt(process.env.AICONFIGURATOR_TIMEOUT_SECONDS || '', 10) || DEFAULT_TIMEOUT_SECONDS

  if (!username || !password) {
    return makeError(requestId, 'KV_CACHE_NOT_CONFIGURED', 'KV cache calculator service is not configured')
  }

  const externalPayload: Record<string, unknown> = {
    model_path: request.model_path,
    backend: request.backend,
    system: request.system,
    max_num_tokens: request.max_num_tokens,
    max_batch_size: request.max_batch_size,
    allow_hf_config_download: true,
    username,
    password,
  }

  let response: Response
  try {
    response = await fetch(`${baseUrl}/kv_cache_calc`, {
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
      return makeError(requestId, 'KV_CACHE_TIMEOUT', `The KV cache calculator did not respond within ${timeoutSeconds} seconds (waited ${durationMs}ms)`)
    }
    return makeError(requestId, 'KV_CACHE_UNAVAILABLE', 'KV cache calculator service is unreachable')
  }

  if (response.status === 401 || response.status === 403) {
    return makeError(requestId, 'KV_CACHE_AUTH_FAILED', 'Authentication with KV cache calculator service failed')
  }

  if (!response.ok) {
    return makeError(requestId, 'KV_CACHE_UNAVAILABLE', `KV cache calculator service returned HTTP ${response.status}`)
  }

  let rawData: Record<string, unknown>
  try {
    const parsed = await response.json()
    if (parsed == null || typeof parsed !== 'object') {
      return makeError(requestId, 'KV_CACHE_INVALID_RESPONSE', 'No valid KV cache data found for this model and hardware combination.')
    }
    rawData = parsed as Record<string, unknown>
  } catch {
    return makeError(requestId, 'KV_CACHE_INVALID_RESPONSE', 'KV cache calculator service returned non-JSON response')
  }

  if (typeof rawData.error === 'string') {
    return makeError(requestId, 'KV_CACHE_INVALID_RESPONSE', rawData.error as string)
  }

  if (typeof rawData.total_kv_size_bytes !== 'number') {
    return makeError(
      requestId,
      'KV_CACHE_INVALID_RESPONSE',
      'KV cache response missing required field: total_kv_size_bytes'
    )
  }

  const breakdown = (rawData.memory_breakdown ?? {}) as Record<string, unknown>
  const durationMs = Math.round(performance.now() - startTime)

  return {
    requestId,
    status: 'completed',
    kvCache: {
      totalBytes: rawData.total_kv_size_bytes as number,
      perTokenBytes: asNumber(rawData.kv_size_per_token_bytes, 0),
      totalTokens: asNumber(rawData.total_kv_size_tokens, 0),
    },
    memoryBreakdown: {
      weightsBytes: asNumber(breakdown.weights_bytes, 0),
      activationsBytes: asNumber(breakdown.activations_bytes, 0),
      runtimeOverheadBytes: asNumber(breakdown.runtime_overhead_bytes, 0),
      commOverheadBytes: asNumber(breakdown.comm_overhead_bytes, 0),
    },
    gpuCapacity: {
      totalBytes: asNumber(rawData.total_gpu_capacity_bytes, 0),
    },
    metadata: {
      modelPath: request.model_path,
      backend: request.backend,
      system: request.system,
      maxNumTokens: request.max_num_tokens,
      maxBatchSize: request.max_batch_size,
      source: typeof rawData.source === 'string' ? rawData.source : 'unknown',
      durationMs,
    },
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeError(requestId: string, code: string, message: string): KvCacheCalcErrorResponse {
  return { requestId, status: 'failed', error: { code, message } }
}

function asNumber(val: unknown, fallback: number): number {
  return typeof val === 'number' ? val : fallback
}
