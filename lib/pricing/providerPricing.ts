// Provider pricing from Cloudflare Worker
// Fetches live GPU pricing from multiple cloud providers

export interface ProviderGpu {
  model: string
  price: number | null // $/hr, null if unavailable
}

export interface Provider {
  id: string
  label: string
  gpus: ProviderGpu[]
}

interface CacheEntry {
  data: Provider[]
  timestamp: number
}

const CACHE_TTL_MS = 30 * 60 * 1000 // 30 minutes
let cache: CacheEntry | null = null

const WORKER_URL = process.env.NEXT_PUBLIC_PRICING_WORKER_URL ||
  'https://gpu-pricing-worker.vikasgrover2004.workers.dev/'

/**
 * Fetch all providers from worker. Tries multiple endpoints until success.
 * Returns [] on error and logs the issue.
 */
export async function fetchAllProviders(): Promise<Provider[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_TTL_MS) {
    return cache.data
  }

  const endpoints = ['/', '/api/prices', '/prices']

  for (const endpoint of endpoints) {
    try {
      const url = WORKER_URL + endpoint.replace(/^\//, '')
      console.log(`[Pricing] Trying ${url}`)

      const response = await fetch(url, {
        headers: { 'Accept': 'application/json' },
        signal: AbortSignal.timeout(10000), // 10s timeout
      })

      if (!response.ok) {
        console.warn(`[Pricing] ${url} returned ${response.status}`)
        continue
      }

      const data = await response.json()
      console.log(`[Pricing] Response shape:`, Object.keys(data))

      // Adapt to whatever shape the worker returns
      const providers = parseWorkerResponse(data)

      if (providers.length > 0) {
        console.log(`[Pricing] ✓ Loaded ${providers.length} providers from ${url}`)
        cache = { data: providers, timestamp: Date.now() }
        return providers
      }
    } catch (error) {
      console.warn(`[Pricing] Failed to fetch from ${endpoint}:`, error)
    }
  }

  console.error('[Pricing] All endpoints failed, returning empty array')
  return []
}

/**
 * Parse worker response - adapts to actual shape returned
 */
function parseWorkerResponse(data: any): Provider[] {
  // Try common response shapes
  if (Array.isArray(data)) {
    return data as Provider[]
  }

  if (data.providers && Array.isArray(data.providers)) {
    return data.providers as Provider[]
  }

  if (data.data && Array.isArray(data.data)) {
    return data.data as Provider[]
  }

  if (data.data && data.data.providers && Array.isArray(data.data.providers)) {
    return data.data.providers as Provider[]
  }

  console.warn('[Pricing] Unknown response shape, trying to extract providers')

  // If it's an object with provider-like keys, try to parse it
  if (typeof data === 'object' && data !== null) {
    const providers: Provider[] = []
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === 'object' && value !== null && 'gpus' in value) {
        providers.push({
          id: key,
          label: (value as any).label || key,
          gpus: (value as any).gpus || [],
        })
      }
    }
    if (providers.length > 0) return providers
  }

  return []
}

/**
 * Get effective rate for a provider/GPU combination.
 * Checks overrides first, then worker data.
 */
export function getEffectiveRate(
  providerId: string,
  gpuModel: string,
  providers: Provider[],
  overrides: Record<string, number | undefined>
): number | null {
  const key = `${providerId}_${gpuModel}`

  // Check override first
  if (overrides[key] !== undefined) {
    return overrides[key]!
  }

  // Check worker data
  const provider = providers.find(p => p.id === providerId)
  const gpu = provider?.gpus.find(g => g.model === gpuModel)
  return gpu?.price ?? null
}

/**
 * Load user overrides from localStorage
 */
export function loadUserOverrides(): Record<string, number | undefined> {
  if (typeof window === 'undefined') return {}

  const overrides: Record<string, number | undefined> = {}

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key?.startsWith('price_')) {
      const priceKey = key.substring(6) // Remove 'price_' prefix
      const value = localStorage.getItem(key)
      if (value) {
        const parsed = parseFloat(value)
        if (!isNaN(parsed)) {
          overrides[priceKey] = parsed
        }
      }
    }
  }

  return overrides
}

/**
 * Save user override to localStorage
 */
export function setUserOverride(providerId: string, gpuModel: string, price: number | undefined): void {
  if (typeof window === 'undefined') return

  const key = `price_${providerId}_${gpuModel}`

  if (price === undefined) {
    localStorage.removeItem(key)
  } else {
    localStorage.setItem(key, String(price))
  }
}

/**
 * Clear user override from localStorage
 */
export function clearUserOverride(providerId: string, gpuModel: string): void {
  setUserOverride(providerId, gpuModel, undefined)
}

/**
 * Load selected GPU per provider from localStorage
 */
export function loadSelectedGpus(providers: Provider[]): Record<string, string> {
  if (typeof window === 'undefined') return {}

  const selected: Record<string, string> = {}

  providers.forEach(p => {
    const stored = localStorage.getItem(`selectedGpu_${p.id}`)
    if (stored && p.gpus.some(g => g.model === stored)) {
      selected[p.id] = stored
    } else {
      // Default to first GPU
      selected[p.id] = p.gpus[0]?.model || ''
    }
  })

  return selected
}

/**
 * Save selected GPU for a provider to localStorage
 */
export function saveSelectedGpu(providerId: string, gpuModel: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(`selectedGpu_${providerId}`, gpuModel)
}
