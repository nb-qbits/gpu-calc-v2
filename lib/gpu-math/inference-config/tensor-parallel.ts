// Tensor Parallel Sizing Module
// Pure functions for TP sizing, replica calculation, and usable HBM computation

/**
 * Returns next power of 2: 1 → 1, 3 → 4, 5 → 8, etc.
 * TP must be power of 2 for NCCL AllReduce efficiency.
 *
 * @param n - Input number
 * @returns Next power of 2 >= n
 */
export function nextPowerOf2(n: number): number {
  if (n <= 1) return 1
  return Math.pow(2, Math.ceil(Math.log2(n)))
}

/**
 * Compute minimum TP size needed to fit model weights in GPU memory.
 * Returns power-of-2 rounded value.
 *
 * Formula from PDF spec:
 *   min_gpus_for_weights = ceil(weight_gb / usable_hbm)
 *   tp_size = next_power_of_2(min_gpus_for_weights)
 *
 * @param weight_gb - Total model weight size in GB
 * @param usable_hbm_per_gpu - Usable HBM per GPU in GB
 * @returns TP size (1, 2, 4, 8, or 16)
 */
export function computeTensorParallelSize(
  weight_gb: number,
  usable_hbm_per_gpu: number
): number {
  const min_gpus = Math.ceil(weight_gb / usable_hbm_per_gpu)
  return nextPowerOf2(min_gpus)
}

/**
 * Compute number of replicas given total GPU count and TP size.
 * Replicas scale throughput; TP size fits model in memory.
 *
 * Formula: replicas = floor(gpu_count / tp_size)
 *
 * @param gpu_count - Total GPUs available
 * @param tp_size - Tensor parallel size
 * @returns Number of replicas
 */
export function computeReplicas(
  gpu_count: number,
  tp_size: number
): number {
  return Math.floor(gpu_count / tp_size)
}

/**
 * Compute usable HBM per GPU with gpu_memory_utilization factor.
 * Default 0.90, reduced to 0.85 if model is close to filling HBM.
 *
 * PDF spec: gpu_memory_utilization = 0.90 default
 *           = 0.85 if weight_gb_per_gpu > 60GB
 *
 * @param gpu_hbm_gb - Total GPU HBM in GB
 * @param weight_gb_per_gpu - Weight memory per GPU in GB
 * @returns Object with usable GB and utilization factor
 */
export function computeUsableHBM(
  gpu_hbm_gb: number,
  weight_gb_per_gpu: number
): { usable_gb: number; utilization: number } {
  const utilization = weight_gb_per_gpu > 60 ? 0.85 : 0.90
  return {
    usable_gb: gpu_hbm_gb * utilization,
    utilization
  }
}
