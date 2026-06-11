// Cluster Cost Module Types
// Based on SemiAnalysis TCO methodology

export type ProviderType = "aws" | "azure" | "gcp" | "neocloud" | "on_prem" | "custom"
export type PricingModel = "rental" | "capex"

export interface ClusterCostInput {
  cluster: {
    name: string
    providerType: ProviderType
    gpuType: string
    gpuCount: number
    gpusPerNode: number
    hoursPerMonth: number // default 720
    utilizationTarget: number // default 0.7
    durationMonths: number // default 36
  }

  compute: {
    pricingModel: PricingModel
    gpuHourPrice?: number
    hardwareCapex?: number
    hardwareLifetimeMonths?: number
    discountPercent?: number
  }

  storage: {
    hotTb: number
    warmTb: number
    coldTb: number
    hotPricePerTbMonth: number
    warmPricePerTbMonth: number
    coldPricePerTbMonth: number
  }

  network: {
    egressTbMonth: number
    egressPricePerTb: number
    loadBalancerMonthly: number
    natFirewallMonthly: number
    dataTransferMonthly: number
  }

  controlPlane: {
    monthlyCost: number
    cpuHelperNodeMonthly: number
    cpuHelperNodeCount: number
  }

  support: {
    included: boolean
    percentOfBill: number // e.g. AWS support uplift
    fixedMonthly: number
  }

  operations: {
    engineerHourlyRate: number
    setupHours: number
    setupAmortizationMonths: number
    debuggingHoursPerMonth: number
  }

  goodput: {
    enabled: boolean
    lossPercent: number // v1 simple model
  }
}

export interface ClusterCostResult {
  monthlyCost: number
  annualCost: number
  totalCostForDuration: number
  rawGpuHourCost: number
  effectiveUsableGpuHourCost: number

  breakdown: {
    gpu: number
    storage: number
    network: number
    controlPlane: number
    support: number
    setup: number
    debugging: number
    goodput: number
  }

  assumptions: string[]
  warnings: string[]
}

export interface ProviderProfile {
  id: string
  name: string
  defaults: Partial<ClusterCostInput>
}
