// GPU Catalog - Hardware specifications
// Source: gpu-catalog.json (converted from docs/gpu-data)

import gpuCatalogJson from './gpu-catalog.json'

export interface GpuSpec {
  // New JSON schema fields (from gpu-catalog.json)
  id: string
  name: string
  vendor: 'nvidia' | 'amd'
  vram_gb: number
  hardware_cost_usd: number
  memory_bandwidth_tbps: number
  tokens_per_dollar: number
  tflops_bf16: number
  tflops_fp8: number | null
  mfu_prefill: number  // Model FLOPs Utilization for prefill (compute-bound)
  mfu_decode: number   // Memory bandwidth utilization for decode
  tdp_watts: number
  architecture: 'ampere' | 'hopper' | 'blackwell' | 'ada' | 'cdna2' | 'cdna3' | 'cdna4'
  nvlink_bandwidth_gbps?: number
  color: string  // Hex color for visualization

  // Legacy field names (backward compatibility - computed at runtime)
  vramGb: number
  memoryBandwidthGbps: number
  bandwidthTbps: number
  tflops: number
  pricePerHour: number
  hardwareCostPerGpu: number
  powerWatts: number
  cloudAvailabilityPct: number
  tpuAvailabilityPct: number

  // Live pricing from Cloudflare Worker (optional - populated at runtime)
  livePricing?: {
    onDemand?: {
      min: number | null
      median: number | null
      max: number | null
      count: number
      providers: Array<{ provider: string; price_per_gpu: number; region: string }>
    }
    spot?: {
      min: number | null
      median: number | null
      max: number | null
      count: number
      providers: Array<{ provider: string; price_per_gpu: number; region: string }>
    }
    lastUpdated: string
  }
}

// Load GPU catalog from JSON and add computed fields
const rawGpuCatalog: GpuSpec[] = gpuCatalogJson as GpuSpec[]

// Export with both new and legacy field names for backward compatibility
export const GPU_CATALOG = rawGpuCatalog.map(gpu => ({
  // New JSON schema fields
  ...gpu,

  // Legacy field names (for backward compatibility)
  vramGb: gpu.vram_gb,
  memoryBandwidthGbps: gpu.memory_bandwidth_tbps * 1000,
  bandwidthTbps: gpu.memory_bandwidth_tbps,
  tflops: gpu.tflops_bf16,
  pricePerHour: 0, // Will be populated from Cloudflare Worker live pricing
  hardwareCostPerGpu: gpu.hardware_cost_usd,
  powerWatts: gpu.tdp_watts,
  cloudAvailabilityPct: 0, // Deprecated - use Cloudflare Worker
  tpuAvailabilityPct: 0    // Deprecated
}))

// Helper: Get GPU by ID
export function getGpuById(id: string): GpuSpec | undefined {
  return GPU_CATALOG.find(gpu => gpu.id === id)
}

// Helper: Get GPUs by vendor
export function getGpusByVendor(vendor: 'nvidia' | 'amd'): GpuSpec[] {
  return GPU_CATALOG.filter(gpu => gpu.vendor === vendor)
}

// Helper: Get GPUs by architecture
export function getGpusByArchitecture(arch: string): GpuSpec[] {
  return GPU_CATALOG.filter(gpu => gpu.architecture === arch)
}

// Helper: Get GPUs with minimum VRAM
export function getGpusByMinVram(minVram: number): GpuSpec[] {
  return GPU_CATALOG.filter(gpu => gpu.vram_gb >= minVram)
}

// Default GPU for initial state
export const DEFAULT_GPU = GPU_CATALOG.find(gpu => gpu.id === 'h100-80gb') || GPU_CATALOG[0]

// Backward compatibility - map new JSON schema to old API response format
export interface GpuSpecLegacy {
  id: string
  name: string
  vramGb: number
  memoryBandwidthGbps: number
  bandwidthTbps: number
  tflops: number
  pricePerHour: number
  hardwareCostPerGpu: number
  powerWatts: number
  cloudAvailabilityPct: number
  tpuAvailabilityPct: number
}

// Convert new GPU spec to legacy format for backward compatibility
export function toLegacyFormat(gpu: GpuSpec): GpuSpecLegacy {
  return {
    id: gpu.id,
    name: gpu.name,
    vramGb: gpu.vram_gb,
    memoryBandwidthGbps: gpu.memory_bandwidth_tbps * 1000,
    bandwidthTbps: gpu.memory_bandwidth_tbps,
    tflops: gpu.tflops_bf16,
    pricePerHour: 0, // Will be populated from Cloudflare Worker live pricing
    hardwareCostPerGpu: gpu.hardware_cost_usd,
    powerWatts: gpu.tdp_watts,
    cloudAvailabilityPct: 0, // Deprecated - use Cloudflare Worker for availability
    tpuAvailabilityPct: 0
  }
}
