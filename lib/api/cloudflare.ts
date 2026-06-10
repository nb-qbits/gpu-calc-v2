// Cloudflare Worker integration for live GPU pricing
// Fetches from deployed gpu-pricing-worker

const CLOUDFLARE_WORKER_URL = 'https://gpu-pricing-worker.vikasgrover2004.workers.dev'
const CACHE_TTL_SECONDS = 60 * 60 * 6 // 6 hours (match Cloudflare KV)

export interface CloudflareGPUPrice {
  id: string
  provider: string
  category: 'gpu_cloud' | 'api_token'
  gpu: string | null
  vram_gb: number | null
  model: string | null
  pricing_type: 'on_demand' | 'spot' | 'reserved_1yr' | 'api' | 'spot_median'
  price_usd: number
  unit: string
  gpu_count: number
  region: string | null
  source_url: string
  confidence: 'high' | 'medium' | 'low'
  status: 'approved' | 'pending_review' | 'rejected'
  fetched_at: string
  updated_at: string
}

export interface CloudflarePricesResponse {
  prices: CloudflareGPUPrice[]
  count: number
  source: 'cache' | 'db'
  timestamp: string
  filters_applied?: string[]
}

/**
 * Fetch GPU pricing from Cloudflare Worker
 * @param filters - Query parameters for filtering (gpu, provider, pricing_type)
 */
export async function fetchGPUPricing(
  filters?: {
    gpu?: string
    provider?: string
    pricing_type?: string
    category?: string
  }
): Promise<CloudflarePricesResponse> {
  const params = new URLSearchParams({
    category: 'gpu_cloud',
    ...filters
  })

  const url = `${CLOUDFLARE_WORKER_URL}/prices?${params.toString()}`

  const response = await fetch(url, {
    next: { revalidate: CACHE_TTL_SECONDS }
  })

  if (!response.ok) {
    throw new Error(`Cloudflare Worker returned ${response.status}`)
  }

  return response.json()
}

/**
 * Get pricing statistics for a specific GPU type
 * Returns min, median, max pricing across providers
 */
export function aggregateGPUPricing(
  prices: CloudflareGPUPrice[],
  pricingType: 'on_demand' | 'spot' = 'on_demand'
): {
  min: number | null
  median: number | null
  max: number | null
  count: number
  providers: Array<{ provider: string; price_per_gpu: number; region: string }>
} {
  const filtered = prices
    .filter(p => p.pricing_type === pricingType && p.price_usd > 0)
    .map(p => ({
      provider: p.provider,
      price_per_gpu: p.price_usd / (p.gpu_count || 1),
      region: p.region || 'global'
    }))
    .sort((a, b) => a.price_per_gpu - b.price_per_gpu)

  if (filtered.length === 0) {
    return { min: null, median: null, max: null, count: 0, providers: [] }
  }

  return {
    min: filtered[0].price_per_gpu,
    median: filtered[Math.floor(filtered.length / 2)].price_per_gpu,
    max: filtered[filtered.length - 1].price_per_gpu,
    count: filtered.length,
    providers: filtered
  }
}

/**
 * Fetch API token pricing (for model inference cost estimation)
 */
export async function fetchAPITokenPricing(
  filters?: {
    provider?: string
    model?: string
  }
): Promise<CloudflarePricesResponse> {
  const params = new URLSearchParams({
    category: 'api_token',
    ...filters
  })

  const url = `${CLOUDFLARE_WORKER_URL}/prices?${params.toString()}`

  const response = await fetch(url, {
    next: { revalidate: CACHE_TTL_SECONDS }
  })

  if (!response.ok) {
    throw new Error(`Cloudflare Worker returned ${response.status}`)
  }

  return response.json()
}
