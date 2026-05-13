/**
 * Throughput estimation formulas.
 */

export interface ThroughputEstimateInput {
  /** Model parameters in billions */
  parametersBillions: number;
  /** GPU memory bandwidth in GB/s */
  memoryBandwidthGbps: number;
  /** Bytes per parameter */
  bytesPerParam: number;
}

/**
 * Estimate tokens per second based on memory bandwidth.
 * Uses the roofline model: throughput is memory-bandwidth bound for autoregressive decoding.
 */
export function estimateTokensPerSecond(input: ThroughputEstimateInput): number {
  const { parametersBillions, memoryBandwidthGbps, bytesPerParam } = input;
  const modelSizeGb = (parametersBillions * 1e9 * bytesPerParam) / 1e9;
  return memoryBandwidthGbps / modelSizeGb;
}
