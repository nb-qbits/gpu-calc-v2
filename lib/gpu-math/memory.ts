/**
 * GPU memory estimation formulas.
 * Ported from the original gpu-calc static site.
 */

export interface MemoryEstimateInput {
  /** Number of model parameters in billions (e.g. 7 for a 7B model) */
  parametersBillions: number;
  /** Bytes per parameter — 2 for fp16/bf16, 4 for fp32, 1 for int8, 0.5 for int4 */
  bytesPerParam: number;
  /** KV cache overhead multiplier (default 1.2) */
  kvCacheMultiplier?: number;
}

export interface MemoryEstimateResult {
  /** Raw model weights memory in GB */
  weightsGb: number;
  /** Total estimated GPU memory required in GB */
  totalGb: number;
}

/**
 * Estimate GPU memory required to serve a model.
 */
export function estimateMemory(input: MemoryEstimateInput): MemoryEstimateResult {
  const { parametersBillions, bytesPerParam, kvCacheMultiplier = 1.2 } = input;

  const weightsGb = (parametersBillions * 1e9 * bytesPerParam) / 1e9;
  const totalGb = weightsGb * kvCacheMultiplier;

  return { weightsGb, totalGb };
}

/**
 * How many GPUs are needed for a given model memory requirement.
 */
export function gpusRequired(totalMemoryGb: number, gpuMemoryGb: number): number {
  return Math.ceil(totalMemoryGb / gpuMemoryGb);
}
