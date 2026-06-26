// Weight Memory Calculation
// Derives actual bytes-per-param from quantization_config, respecting storage dtype

import type { ExtractedConfig, QuantizationConfig } from '../kv-types'

/**
 * Get storage bytes per parameter from quantization config.
 *
 * For quantized models (FP8, INT8, INT4, GPTQ, AWQ, BnB), returns the
 * quantized bytes-per-param. For unquantized models, returns bytes from
 * the base dtype (torch_dtype).
 *
 * Critical: This is STORAGE dtype (on-disk / in-GPU-memory weight size),
 * NOT compute dtype. FP8 models store weights at 1 byte/param but compute
 * in BF16 (2 bytes/elem).
 *
 * @param quantConfig - Quantization config from config.json
 * @param baseDtype - Base dtype (torch_dtype) for unquantized case
 * @returns Bytes per parameter for weight storage
 */
export function getStorageBytesPerParam(
  quantConfig: QuantizationConfig,
  baseDtype: string
): number {
  // Check quantization type
  switch (quantConfig.type) {
    case 'fp8':
      return 1  // FP8: 1 byte per param

    case 'int8':
      return 1  // INT8: 1 byte per param

    case 'int4':
      return 0.5  // INT4: 0.5 bytes per param

    case 'gptq':
    case 'awq':
      // GPTQ/AWQ: use bits field if present, otherwise default to 4-bit
      const bits = quantConfig.bits ?? 4
      return bits / 8  // 4-bit → 0.5, 8-bit → 1.0, etc.

    case 'bnb':
      // BitsAndBytes: default to 4-bit (most common)
      return 0.5

    case 'none':
    case 'unknown':
      // No quantization or unknown → use base dtype
      return getDtypeBytes(baseDtype)

    default:
      // Unknown quantization type → fallback to base dtype
      console.warn(`⚠️ Unknown quantization type "${quantConfig.type}" - using base dtype`)
      return getDtypeBytes(baseDtype)
  }
}

/**
 * Map dtype string to bytes per element.
 */
function getDtypeBytes(dtype: string): number {
  const DTYPE_BYTES: Record<string, number> = {
    float32: 4,
    bfloat16: 2,
    float16: 2,
    float8: 1,
    float8_e4m3fn: 1,
    float8_e4m3: 1,
    float8_e5m2: 1,
    int8: 1,
    int4: 0.5,
  }
  return DTYPE_BYTES[dtype] ?? 2  // Default to 2 bytes (BF16/FP16)
}

/**
 * Estimate weight memory in bytes, respecting quantization.
 *
 * For quantized models, computes:
 *   quantized_params × quant_bytes + unquantized_params × base_bytes
 *
 * Where unquantized_params are from modules_to_not_convert (embedding,
 * lm_head, norms, gates, etc.).
 *
 * @param cfg - Extracted config from config.json
 * @returns Weight memory in bytes
 */
export function estimateWeightMemoryBytes(cfg: ExtractedConfig): number {
  const baseBytes = getDtypeBytes(cfg.dtype)
  const quantConfig = cfg.quantization_config
  const storageBytes = getStorageBytesPerParam(quantConfig, cfg.dtype)

  // If no quantization or unknown, use simple formula
  if (quantConfig.type === 'none' || quantConfig.type === 'unknown') {
    return estimateTotalParams(cfg) * baseBytes
  }

  // Quantized model: split params into quantized vs unquantized buckets
  const totalParams = estimateTotalParams(cfg)

  // Estimate unquantized params (embedding + lm_head)
  // Most quantization schemes skip embedding table and lm_head
  const embeddingParams = cfg.vocab_size * cfg.hidden_size
  const lmHeadParams = cfg.vocab_size * cfg.hidden_size  // Usually untied
  const unquantizedParams = embeddingParams + lmHeadParams

  // Remaining params are quantized
  const quantizedParams = Math.max(0, totalParams - unquantizedParams)

  const quantizedBytes = quantizedParams * storageBytes
  const unquantizedBytes = unquantizedParams * baseBytes

  console.log(`📦 Weight memory breakdown:`)
  console.log(`   Total params: ${(totalParams / 1e9).toFixed(2)}B`)
  console.log(`   Quantized (${storageBytes}B/param): ${(quantizedParams / 1e9).toFixed(2)}B → ${(quantizedBytes / 1e9).toFixed(1)} GB`)
  console.log(`   Unquantized (${baseBytes}B/param): ${(unquantizedParams / 1e9).toFixed(2)}B → ${(unquantizedBytes / 1e9).toFixed(1)} GB`)
  console.log(`   Total: ${((quantizedBytes + unquantizedBytes) / 1e9).toFixed(1)} GB`)

  return quantizedBytes + unquantizedBytes
}

/**
 * Estimate total parameter count from architecture.
 * Same formula as kv-config.ts but extracted for reuse.
 */
function estimateTotalParams(cfg: ExtractedConfig): number {
  const attnPerLayer =
    cfg.H_q * cfg.d * cfg.hidden_size +  // Q
    cfg.H_kv * cfg.d * cfg.hidden_size +  // K
    cfg.H_kv * cfg.d * cfg.hidden_size +  // V
    cfg.H_q * cfg.d * cfg.hidden_size    // O

  let ffnPerLayer: number
  if (cfg.is_moe && cfg.total_routed_experts != null) {
    const expertSize = cfg.moe_intermediate_size ?? cfg.intermediate_size
    ffnPerLayer =
      cfg.total_routed_experts * cfg.hidden_size * expertSize * 3 +
      cfg.shared_experts * cfg.hidden_size * expertSize * 3
  } else {
    ffnPerLayer = cfg.hidden_size * cfg.intermediate_size * 3
  }

  const embedding = cfg.vocab_size * cfg.hidden_size
  return (attnPerLayer + ffnPerLayer) * cfg.L + embedding
}
