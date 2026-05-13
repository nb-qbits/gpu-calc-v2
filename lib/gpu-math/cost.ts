/**
 * Cost modeling formulas.
 */

export interface CloudCostInput {
  /** Tokens per second per GPU */
  tokensPerSecondPerGpu: number;
  /** Number of GPUs */
  gpuCount: number;
  /** Cost per GPU per hour in USD */
  costPerGpuHour: number;
}

/**
 * Cost per million tokens from cloud GPU pricing.
 */
export function costPerMillionTokens(input: CloudCostInput): number {
  const { tokensPerSecondPerGpu, gpuCount, costPerGpuHour } = input;
  const totalTps = tokensPerSecondPerGpu * gpuCount;
  const tokensPerHour = totalTps * 3600;
  const costPerToken = costPerGpuHour * gpuCount / tokensPerHour;
  return costPerToken * 1_000_000;
}
