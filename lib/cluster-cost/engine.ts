// Cluster Cost Calculation Engine
// Pure functional module - provider-neutral

import type { ClusterCostInput, ClusterCostResult } from './types'

export function computeClusterCost(input: ClusterCostInput): ClusterCostResult {
  const assumptions: string[] = []
  const warnings: string[] = []

  // ═══ GPU COST ═══
  let gpuCost: number

  if (input.compute.pricingModel === 'rental') {
    if (!input.compute.gpuHourPrice) {
      throw new Error('gpuHourPrice is required for rental pricing model')
    }

    const discountMultiplier = 1 - (input.compute.discountPercent || 0)
    gpuCost =
      input.cluster.gpuCount *
      input.cluster.hoursPerMonth *
      input.compute.gpuHourPrice *
      discountMultiplier

    assumptions.push(
      `GPU rental: ${input.cluster.gpuCount} GPUs × ${input.cluster.hoursPerMonth} hours × $${input.compute.gpuHourPrice}/hr`
    )

    if (input.compute.discountPercent) {
      assumptions.push(`Applied ${(input.compute.discountPercent * 100).toFixed(1)}% discount`)
    }
  } else {
    // CAPEX model
    if (!input.compute.hardwareCapex || !input.compute.hardwareLifetimeMonths) {
      throw new Error('hardwareCapex and hardwareLifetimeMonths are required for capex pricing model')
    }

    gpuCost = input.compute.hardwareCapex / input.compute.hardwareLifetimeMonths

    assumptions.push(
      `GPU capex: $${input.compute.hardwareCapex.toLocaleString()} amortized over ${input.compute.hardwareLifetimeMonths} months`
    )

    if (input.compute.hardwareLifetimeMonths < 36) {
      warnings.push(
        `Hardware lifetime is ${input.compute.hardwareLifetimeMonths} months. Industry standard is 36-48 months.`
      )
    }
  }

  // ═══ STORAGE COST ═══
  const storageCost =
    (input.storage.hotTb * input.storage.hotPricePerTbMonth) +
    (input.storage.warmTb * input.storage.warmPricePerTbMonth) +
    (input.storage.coldTb * input.storage.coldPricePerTbMonth)

  if (storageCost > 0) {
    const totalTb = input.storage.hotTb + input.storage.warmTb + input.storage.coldTb
    assumptions.push(
      `Storage: ${totalTb.toFixed(1)} TB (hot: ${input.storage.hotTb}, warm: ${input.storage.warmTb}, cold: ${input.storage.coldTb})`
    )
  }

  // ═══ NETWORK COST ═══
  const networkCost =
    (input.network.egressTbMonth * input.network.egressPricePerTb) +
    input.network.loadBalancerMonthly +
    input.network.natFirewallMonthly +
    input.network.dataTransferMonthly

  if (networkCost > 0) {
    assumptions.push(
      `Network: ${input.network.egressTbMonth} TB egress/month @ $${input.network.egressPricePerTb}/TB`
    )
  }

  // ═══ CONTROL PLANE COST ═══
  const controlPlaneCost =
    input.controlPlane.monthlyCost +
    (input.controlPlane.cpuHelperNodeMonthly * input.controlPlane.cpuHelperNodeCount)

  if (input.controlPlane.cpuHelperNodeCount > 0) {
    assumptions.push(
      `Control plane: ${input.controlPlane.cpuHelperNodeCount} CPU helper nodes @ $${input.controlPlane.cpuHelperNodeMonthly}/month`
    )
  }

  // ═══ BASE BEFORE SUPPORT ═══
  const baseBeforeSupport = gpuCost + storageCost + networkCost + controlPlaneCost

  // ═══ SUPPORT COST ═══
  let supportCost: number

  if (input.support.included) {
    supportCost = 0
    assumptions.push('Support included in base price')
  } else {
    supportCost =
      input.support.fixedMonthly +
      (baseBeforeSupport * input.support.percentOfBill)

    if (input.support.percentOfBill > 0) {
      assumptions.push(
        `Support: ${(input.support.percentOfBill * 100).toFixed(1)}% of infrastructure cost`
      )
    }
  }

  // ═══ SETUP COST (AMORTIZED) ═══
  const setupCost =
    (input.operations.setupHours * input.operations.engineerHourlyRate) /
    input.operations.setupAmortizationMonths

  if (setupCost > 0) {
    assumptions.push(
      `Setup: ${input.operations.setupHours} hours @ $${input.operations.engineerHourlyRate}/hr amortized over ${input.operations.setupAmortizationMonths} months`
    )
  }

  // ═══ DEBUGGING COST ═══
  const debuggingCost =
    input.operations.debuggingHoursPerMonth * input.operations.engineerHourlyRate

  if (debuggingCost > 0) {
    assumptions.push(
      `Debugging: ${input.operations.debuggingHoursPerMonth} hours/month @ $${input.operations.engineerHourlyRate}/hr`
    )
  }

  // ═══ PRE-GOODPUT TOTAL ═══
  const preGoodputTotal =
    baseBeforeSupport + supportCost + setupCost + debuggingCost

  // ═══ GOODPUT COST (RELIABILITY LOSS) ═══
  let goodputCost: number

  if (input.goodput.enabled) {
    goodputCost = preGoodputTotal * input.goodput.lossPercent

    assumptions.push(
      `Goodput loss: ${(input.goodput.lossPercent * 100).toFixed(2)}% downtime/reliability uplift`
    )

    if (input.goodput.lossPercent > 0.05) {
      warnings.push(
        `Goodput loss of ${(input.goodput.lossPercent * 100).toFixed(1)}% is high. Consider improving reliability.`
      )
    }
  } else {
    goodputCost = 0
  }

  // ═══ MONTHLY COST ═══
  const monthlyCost = preGoodputTotal + goodputCost

  // ═══ DERIVED METRICS ═══
  const annualCost = monthlyCost * 12
  const totalCostForDuration = monthlyCost * input.cluster.durationMonths

  // Raw GPU-hour cost (headline price only)
  const rawGpuHourCost = input.compute.pricingModel === 'rental'
    ? (input.compute.gpuHourPrice || 0)
    : (input.compute.hardwareCapex || 0) / (input.compute.hardwareLifetimeMonths || 1) / input.cluster.hoursPerMonth / input.cluster.gpuCount

  // Effective usable GPU-hour cost (accounts for utilization and all TCO components)
  const totalGpuHours = input.cluster.gpuCount * input.cluster.hoursPerMonth * input.cluster.utilizationTarget
  const effectiveUsableGpuHourCost = totalGpuHours > 0 ? monthlyCost / totalGpuHours : 0

  // Warnings
  if (input.cluster.utilizationTarget < 0.5) {
    warnings.push(
      `Utilization target is ${(input.cluster.utilizationTarget * 100).toFixed(0)}%. Low utilization increases effective cost per GPU-hour.`
    )
  }

  if (effectiveUsableGpuHourCost > rawGpuHourCost * 2) {
    const ratio = effectiveUsableGpuHourCost / rawGpuHourCost
    warnings.push(
      `Effective GPU-hour cost is ${ratio.toFixed(1)}x higher than headline price. Hidden costs: storage, network, support, ops, downtime.`
    )
  }

  const nodeCount = Math.ceil(input.cluster.gpuCount / input.cluster.gpusPerNode)
  assumptions.push(
    `Cluster: ${input.cluster.gpuCount} GPUs across ${nodeCount} nodes (${input.cluster.gpusPerNode} GPUs/node)`
  )

  assumptions.push(
    `Target utilization: ${(input.cluster.utilizationTarget * 100).toFixed(0)}%`
  )

  return {
    monthlyCost,
    annualCost,
    totalCostForDuration,
    rawGpuHourCost,
    effectiveUsableGpuHourCost,

    breakdown: {
      gpu: gpuCost,
      storage: storageCost,
      network: networkCost,
      controlPlane: controlPlaneCost,
      support: supportCost,
      setup: setupCost,
      debugging: debuggingCost,
      goodput: goodputCost,
    },

    assumptions,
    warnings,
  }
}
