// vLLM Configuration Defaults Module
// Smart parameter selection based on workload characteristics

import type { InferenceRequest, VLLMConfig } from './types'

/**
 * Compute max_model_len - always exact ISL + OSL, never rounded.
 * PDF spec: "never round to 4096/8192 — use exact workload value"
 *
 * @param isl - Input sequence length
 * @param osl - Output sequence length
 * @returns Exact max_model_len
 */
export function computeMaxModelLen(isl: number, osl: number): number {
  return isl + osl
}

/**
 * Compute max_num_seqs with 20% headroom, capped by KV memory budget.
 *
 * PDF spec:
 *   max_num_seqs = min(
 *     ceil(concurrent_users / replicas × 1.2),
 *     max_sequences_from_memory
 *   )
 *
 * @param concurrent_users - Peak concurrent users
 * @param replicas - Number of replicas
 * @param max_sequences_from_memory - Memory-based sequence limit
 * @returns max_num_seqs value
 */
export function computeMaxNumSeqs(
  concurrent_users: number,
  replicas: number,
  max_sequences_from_memory: number
): number {
  const per_replica = Math.ceil((concurrent_users / replicas) * 1.2)
  return Math.min(per_replica, max_sequences_from_memory)
}

/**
 * Determine max_num_batched_tokens based on ISL and SLA priority.
 *
 * PDF spec:
 *   Default: 512 for chunked prefill
 *   = 1024 if ISL > 4000 AND sla_priority == throughput
 *   = 256 if sla_priority == tpot AND OSL > 1000
 *
 * @param isl - Input sequence length
 * @param osl - Output sequence length
 * @param sla_priority - SLA priority (ttft, tpot, throughput)
 * @returns max_num_batched_tokens value
 */
export function computeMaxNumBatchedTokens(
  isl: number,
  osl: number,
  sla_priority: string
): number {
  if (isl > 4000 && sla_priority === 'throughput') return 1024
  if (sla_priority === 'tpot' && osl > 1000) return 256
  return 512  // default
}

/**
 * Determine enable_chunked_prefill.
 *
 * PDF spec:
 *   Enable if:
 *   - ISL > 1000, OR
 *   - workload_type in [rag, web_search, coding]
 *   Disable if:
 *   - ISL < 500 AND workload_type == chat
 *
 * @param isl - Input sequence length
 * @param workload_type - Type of workload
 * @returns Whether to enable chunked prefill
 */
export function shouldEnableChunkedPrefill(
  isl: number,
  workload_type: string
): boolean {
  if (isl > 1000) return true
  if (['rag', 'web_search', 'coding'].includes(workload_type)) return true
  if (isl < 500 && workload_type === 'chat') return false
  return true  // default to enabled
}

/**
 * Determine enable_prefix_caching.
 *
 * PDF spec:
 *   Enable if shared_prefix_tokens > 0 AND hit_rate_estimate > 0.5
 *   Disable if retrieved_context > shared_prefix
 *   (RAG: shared_prefix / total_isl < 0.3)
 *
 * vLLM docs: Best for long documents, multi-round chat, long system prompts
 * https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
 *
 * Note: For now, we don't have shared_prefix detection, so use heuristics.
 * This will be enhanced when we add prompt analysis.
 *
 * @param workload_type - Type of workload
 * @param isl - Input sequence length
 * @returns Whether to enable prefix caching
 */
export function shouldEnablePrefixCaching(
  workload_type: string,
  isl: number
): boolean {
  // RAG with long contexts likely has reusable system prompts
  // Updated based on vLLM APC docs - good for long document queries
  if (workload_type === 'rag' && isl > 2000) return true

  // Chat and coding with long contexts - multi-round conversations
  if (['chat', 'coding'].includes(workload_type) && isl > 2000) return true

  // Web search - likely has repeated query patterns
  if (workload_type === 'web_search' && isl > 1000) return true

  // Batch workloads don't benefit from prefix caching
  if (workload_type === 'batch') return false

  return false
}

/**
 * Build complete vLLM config from all smart defaults.
 *
 * IMPORTANT: Chunked prefill and prefix caching cannot be enabled simultaneously.
 * vLLM V1 docs: https://docs.vllm.ai/en/v0.8.2/performance/optimization.html
 * "This feature cannot be used simultaneously with automatic prefix caching (APC)"
 *
 * Resolution priority:
 * - If both would be enabled, prioritize chunked_prefill (better for most workloads)
 * - Prefix caching is only enabled if chunked_prefill is disabled
 *
 * @param req - Inference request
 * @param memory - Memory analysis results
 * @returns Complete vLLM configuration
 */
export function computeVLLMConfig(
  req: InferenceRequest,
  memory: {
    tp_size: number
    replicas: number
    max_sequences_from_memory: number
    gpu_memory_utilization: number
  }
): VLLMConfig {
  // Compute both flags independently first
  const would_enable_chunked_prefill = shouldEnableChunkedPrefill(
    req.isl,
    req.workload_type
  )
  const would_enable_prefix_caching = shouldEnablePrefixCaching(
    req.workload_type,
    req.isl
  )

  // Resolve conflict: Chunked prefill takes priority
  let enable_chunked_prefill = would_enable_chunked_prefill
  let enable_prefix_caching = would_enable_prefix_caching

  if (would_enable_chunked_prefill && would_enable_prefix_caching) {
    // Conflict! Prioritize chunked_prefill (better for most workloads)
    enable_prefix_caching = false
  }

  return {
    tensor_parallel_size: memory.tp_size,
    max_model_len: computeMaxModelLen(req.isl, req.osl),
    max_num_seqs: computeMaxNumSeqs(
      req.concurrent_users,
      memory.replicas,
      memory.max_sequences_from_memory
    ),
    gpu_memory_utilization: memory.gpu_memory_utilization,
    max_num_batched_tokens: computeMaxNumBatchedTokens(
      req.isl,
      req.osl,
      req.sla_priority
    ),
    enable_chunked_prefill,
    enable_prefix_caching,
    quantization: req.precision.toLowerCase()
  }
}
