import { GPU_CATALOG } from '@/lib/gpu-math/gpus'

// Hardcoded median cloud rates ($/hr) from major providers.
// Used when live Cloudflare pricing is unavailable.
const CLOUD_RATE_FALLBACK: Record<string, number> = {
  'a100-40gb':    2.40,
  'a100-80gb':    2.20,
  'h100-80gb':    2.85,
  'h200-141gb':   3.85,
  'l40s-48gb':    1.10,
  'mi300x-192gb': 2.50,
}

// GPU catalog name → catalog id mapping for live pricing lookup
const NAME_TO_ID: Record<string, string> = {}
for (const g of GPU_CATALOG) {
  NAME_TO_ID[g.name] = g.id
}

export function getCloudRate(
  gpuId: string,
  livePricing?: Record<string, number>,
): number {
  if (livePricing) {
    const gpu = GPU_CATALOG.find(g => g.id === gpuId)
    if (gpu && livePricing[gpu.name] !== undefined) {
      return livePricing[gpu.name]
    }
  }

  if (CLOUD_RATE_FALLBACK[gpuId] !== undefined) {
    return CLOUD_RATE_FALLBACK[gpuId]
  }

  const gpu = GPU_CATALOG.find(g => g.id === gpuId)
  if (gpu) return gpu.hardware_cost_usd / (36 * 730)
  return 3.00
}

export function getOwnedRate(gpuId: string, amortMonths = 36): number {
  const gpu = GPU_CATALOG.find(g => g.id === gpuId)
  if (!gpu) return 1.00
  return gpu.hardware_cost_usd / (amortMonths * 730)
}
