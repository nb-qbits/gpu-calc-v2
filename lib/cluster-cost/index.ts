// Cluster Cost Module
// Calculate real monthly cost of GPU clusters beyond headline GPU-hour pricing

export { computeClusterCost } from './engine'
export { getProviderProfile, listProviderProfiles, PROVIDER_PROFILES } from './profiles'
export type {
  ClusterCostInput,
  ClusterCostResult,
  ProviderProfile,
  ProviderType,
  PricingModel,
} from './types'

// Helper: Merge provider profile defaults with user overrides
import type { ClusterCostInput, ProviderProfile } from './types'

export function applyProviderProfile(
  profile: ProviderProfile,
  overrides: Partial<ClusterCostInput>
): ClusterCostInput {
  const merged = JSON.parse(JSON.stringify(profile.defaults)) as ClusterCostInput

  // Deep merge overrides
  if (overrides.cluster) {
    Object.assign(merged.cluster, overrides.cluster)
  }
  if (overrides.compute) {
    Object.assign(merged.compute, overrides.compute)
  }
  if (overrides.storage) {
    Object.assign(merged.storage, overrides.storage)
  }
  if (overrides.network) {
    Object.assign(merged.network, overrides.network)
  }
  if (overrides.controlPlane) {
    Object.assign(merged.controlPlane, overrides.controlPlane)
  }
  if (overrides.support) {
    Object.assign(merged.support, overrides.support)
  }
  if (overrides.operations) {
    Object.assign(merged.operations, overrides.operations)
  }
  if (overrides.goodput) {
    Object.assign(merged.goodput, overrides.goodput)
  }

  return merged
}

// Helper: Format cost breakdown as percentages
import type { ClusterCostResult } from './types'

export function formatBreakdownPercentages(result: ClusterCostResult): Record<string, string> {
  const total = result.monthlyCost

  return {
    gpu: `${((result.breakdown.gpu / total) * 100).toFixed(1)}%`,
    storage: `${((result.breakdown.storage / total) * 100).toFixed(1)}%`,
    network: `${((result.breakdown.network / total) * 100).toFixed(1)}%`,
    controlPlane: `${((result.breakdown.controlPlane / total) * 100).toFixed(1)}%`,
    support: `${((result.breakdown.support / total) * 100).toFixed(1)}%`,
    setup: `${((result.breakdown.setup / total) * 100).toFixed(1)}%`,
    debugging: `${((result.breakdown.debugging / total) * 100).toFixed(1)}%`,
    goodput: `${((result.breakdown.goodput / total) * 100).toFixed(1)}%`,
  }
}
