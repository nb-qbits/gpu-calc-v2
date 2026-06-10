// API Response Formatting
// Standardized success responses

import type { InferenceConfigResult } from '../gpu-math/inference-config'
import type { GpuSpec } from '../gpu-math/gpus'
import type { ModelSpec } from '../gpu-math/models'

/**
 * Format inference config result for API response.
 * Adds metadata and ensures consistent structure.
 */
export function formatInferenceConfigResponse(result: InferenceConfigResult) {
  return {
    success: true,
    data: result,
    metadata: {
      generated_at: new Date().toISOString(),
      version: 'v1'
    }
  }
}

/**
 * Format GPU catalog for API response.
 */
export function formatGpuCatalogResponse(gpus: GpuSpec[]) {
  return {
    success: true,
    data: {
      gpus: gpus.map(gpu => ({
        id: gpu.id,
        name: gpu.name,
        memory_gb: gpu.vramGb,
        price_per_hour: gpu.pricePerHour,
        hardware_cost: gpu.hardwareCostPerGpu,
        memory_bandwidth_gbps: gpu.memoryBandwidthGbps,
        tflops: gpu.tflops,
        power_watts: gpu.powerWatts,
        cloud_availability_pct: gpu.cloudAvailabilityPct,
        // Include live pricing if available
        ...(gpu.livePricing && { live_pricing: gpu.livePricing })
      })),
      count: gpus.length
    }
  }
}

/**
 * Format model catalog for API response.
 */
export function formatModelCatalogResponse(models: ModelSpec[]) {
  return {
    success: true,
    data: {
      models: models.map(model => ({
        id: model.id,
        hf_id: model.hfId,
        name: model.name,
        vendor: model.vendor,
        param_label: model.paramLabel,
        tags: model.tags || [],
        is_new: model.isNew || false
      })),
      count: models.length
    }
  }
}
