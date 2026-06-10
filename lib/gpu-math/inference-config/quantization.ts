// Quantization Recommendations Module
// Recommends quantization based on memory pressure and workload type with quality guardrails

/**
 * Recommend quantization based on memory pressure and workload type.
 *
 * PDF spec rules:
 * - Don't quantize if weight_gb_per_gpu < 40GB (fits comfortably)
 * - FP8 if TPOT-bound (2x bandwidth, <0.5% quality loss)
 * - INT4 if memory-constrained BUT never for coding/math/RAG
 *
 * @param weight_gb_per_gpu - Weight memory per GPU in GB
 * @param workload_type - Type of workload
 * @param sla_priority - SLA priority (ttft, tpot, throughput)
 * @returns Quantization recommendation with reason and optional warning
 */
export function recommendQuantization(
  weight_gb_per_gpu: number,
  workload_type: string,
  sla_priority: string
): { quantization: string; reason: string; warning?: string } {
  // Fits comfortably - no quantization needed
  if (weight_gb_per_gpu < 40) {
    return {
      quantization: 'none',
      reason: 'Model fits comfortably in GPU memory'
    }
  }

  // Quality-sensitive workloads - never INT4
  const quality_sensitive = ['coding', 'rag'].includes(workload_type)

  // TPOT-bound workloads benefit from FP8 (decode is memory-bandwidth bound)
  if (sla_priority === 'tpot' || weight_gb_per_gpu > 60) {
    return {
      quantization: 'fp8',
      reason: '2x bandwidth reduction for decode, <0.5% quality loss',
      warning: 'Check perplexity before/after. Threshold: <1% degradation'
    }
  }

  // Memory-constrained but quality-tolerant - INT4 possible
  if (weight_gb_per_gpu > 70 && !quality_sensitive) {
    return {
      quantization: 'int4',
      reason: '4x memory reduction enables model to fit',
      warning: '3-5% quality loss expected. Not recommended for coding/math/RAG.'
    }
  }

  // Default: no quantization
  return {
    quantization: 'none',
    reason: 'No quantization needed for current memory profile'
  }
}
