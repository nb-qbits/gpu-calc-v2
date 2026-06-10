// llm-d Disaggregated Configuration Module
// Computes llm-d prefill/decode split configuration (optional feature)

import type { LLMDConfig, InferenceRequest } from './types'

/**
 * Compute llm-d disaggregated prefill/decode configuration.
 * Only used when user toggles llm-d view (enable_llmd = true).
 *
 * PDF spec:
 * - Prefill is fast, needs fewer instances (30% of concurrent users)
 * - Decode holds state longer, needs more instances (120% headroom)
 * - KV transfer size determines network requirements
 *
 * @param req - Inference request
 * @param tp_size - Tensor parallel size
 * @param model_layers - Number of model layers
 * @param kv_heads - Number of KV heads
 * @param head_dim - Head dimension
 * @returns llm-d configuration with prefill/decode instances and KV transfer specs
 */
export function computeLLMDConfig(
  req: InferenceRequest,
  tp_size: number,
  model_layers: number,
  kv_heads: number,
  head_dim: number
): LLMDConfig {
  // Prefill is compute-bound and fast - fewer instances needed
  const prefill_count = Math.ceil(req.concurrent_users * 0.3)

  // Decode is memory-bound and holds state - more instances with headroom
  const decode_count = Math.ceil(req.concurrent_users * 1.2)

  // KV transfer size calculation
  // Formula: num_layers × kv_heads × head_dim × 2 (K and V) × ISL × bytes_per_dtype
  const bytes_per_dtype = req.precision === 'FP16' ? 2 : 1
  const transfer_size_mb =
    (model_layers * kv_heads * head_dim * 2 * req.isl * bytes_per_dtype) / 1e6

  // Network requirement based on transfer size
  // PDF spec: InfiniBand HDR minimum if transfer_size_mb > 100
  let network_requirement: string
  if (transfer_size_mb > 100) {
    network_requirement = 'InfiniBand HDR minimum (KV transfer > 100 MB)'
  } else {
    network_requirement = 'Standard networking OK (KV transfer < 100 MB)'
  }

  return {
    prefill_instances: {
      count: prefill_count,
      tp_size: tp_size,
      max_num_seqs: prefill_count,
      enable_chunked_prefill: false,  // Prefill node does full prefill
      gpu_memory_utilization: 0.95    // Maximize compute, less KV needed
    },
    decode_instances: {
      count: decode_count,
      tp_size: tp_size,
      max_num_seqs: decode_count,
      gpu_memory_utilization: 0.90,
      max_model_len: req.isl + req.osl,
      quantization: 'fp8'  // Decode is memory-bandwidth bound, FP8 helps most
    },
    kv_transfer: {
      block_size: 16,  // vLLM default block size
      transfer_size_mb: Math.round(transfer_size_mb * 100) / 100,  // Round to 2 decimals
      network_requirement
    }
  }
}
