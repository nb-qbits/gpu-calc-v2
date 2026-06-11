#!/usr/bin/env npx tsx

// Test script for Cluster Cost Module

import { computeClusterCost, getProviderProfile, applyProviderProfile, formatBreakdownPercentages } from './index'

console.log('═══ CLUSTER COST MODULE TEST ═══\n')

// Test 1: AWS H200 with defaults
console.log('Test 1: AWS H200 Inference Cluster (defaults)')
console.log('─'.repeat(60))

const awsProfile = getProviderProfile('aws.h200.inference')!
const awsInput = applyProviderProfile(awsProfile, {})
const awsResult = computeClusterCost(awsInput)

console.log(`Monthly Cost: $${awsResult.monthlyCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`Annual Cost: $${awsResult.annualCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`3-Year Total: $${awsResult.totalCostForDuration.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`\nRaw GPU-Hour: $${awsResult.rawGpuHourCost.toFixed(2)}`)
console.log(`Effective GPU-Hour: $${awsResult.effectiveUsableGpuHourCost.toFixed(2)} (${(awsResult.effectiveUsableGpuHourCost / awsResult.rawGpuHourCost).toFixed(2)}x)`)

console.log(`\nCost Breakdown:`)
const awsPercentages = formatBreakdownPercentages(awsResult)
console.log(`  GPU:          $${awsResult.breakdown.gpu.toLocaleString()} (${awsPercentages.gpu})`)
console.log(`  Storage:      $${awsResult.breakdown.storage.toLocaleString()} (${awsPercentages.storage})`)
console.log(`  Network:      $${awsResult.breakdown.network.toLocaleString()} (${awsPercentages.network})`)
console.log(`  Control:      $${awsResult.breakdown.controlPlane.toLocaleString()} (${awsPercentages.controlPlane})`)
console.log(`  Support:      $${awsResult.breakdown.support.toLocaleString()} (${awsPercentages.support})`)
console.log(`  Setup:        $${awsResult.breakdown.setup.toLocaleString()} (${awsPercentages.setup})`)
console.log(`  Debugging:    $${awsResult.breakdown.debugging.toLocaleString()} (${awsPercentages.debugging})`)
console.log(`  Goodput:      $${awsResult.breakdown.goodput.toLocaleString()} (${awsPercentages.goodput})`)

console.log(`\nWarnings: ${awsResult.warnings.length}`)
awsResult.warnings.forEach(w => console.log(`  ⚠️  ${w}`))

console.log('\n\n')

// Test 2: On-Prem H100 CAPEX
console.log('Test 2: On-Prem H100 Cluster (CAPEX model)')
console.log('─'.repeat(60))

const onPremProfile = getProviderProfile('on_prem.h100')!
const onPremInput = applyProviderProfile(onPremProfile, {})
const onPremResult = computeClusterCost(onPremInput)

console.log(`Monthly Cost: $${onPremResult.monthlyCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`Annual Cost: $${onPremResult.annualCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`4-Year Total: $${onPremResult.totalCostForDuration.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`\nRaw GPU-Hour: $${onPremResult.rawGpuHourCost.toFixed(2)}`)
console.log(`Effective GPU-Hour: $${onPremResult.effectiveUsableGpuHourCost.toFixed(2)} (${(onPremResult.effectiveUsableGpuHourCost / onPremResult.rawGpuHourCost).toFixed(2)}x)`)

console.log(`\nCost Breakdown:`)
const onPremPercentages = formatBreakdownPercentages(onPremResult)
console.log(`  GPU:          $${onPremResult.breakdown.gpu.toLocaleString()} (${onPremPercentages.gpu})`)
console.log(`  Storage:      $${onPremResult.breakdown.storage.toLocaleString()} (${onPremPercentages.storage})`)
console.log(`  Network:      $${onPremResult.breakdown.network.toLocaleString()} (${onPremPercentages.network})`)
console.log(`  Control:      $${onPremResult.breakdown.controlPlane.toLocaleString()} (${onPremPercentages.controlPlane})`)
console.log(`  Support:      $${onPremResult.breakdown.support.toLocaleString()} (${onPremPercentages.support})`)
console.log(`  Setup:        $${onPremResult.breakdown.setup.toLocaleString()} (${onPremPercentages.setup})`)
console.log(`  Debugging:    $${onPremResult.breakdown.debugging.toLocaleString()} (${onPremPercentages.debugging})`)
console.log(`  Goodput:      $${onPremResult.breakdown.goodput.toLocaleString()} (${onPremPercentages.goodput})`)

console.log('\n\n')

// Test 3: Custom override
console.log('Test 3: Custom Configuration (64 GPUs, high utilization)')
console.log('─'.repeat(60))

const customProfile = getProviderProfile('neocloud.h100')!
const customInput = applyProviderProfile(customProfile, {
  cluster: {
    ...customProfile.defaults.cluster!,
    name: 'Large Production Cluster',
    gpuCount: 64,
    utilizationTarget: 0.9,
  },
  compute: {
    ...customProfile.defaults.compute!,
    discountPercent: 0.15, // 15% volume discount
  }
})
const customResult = computeClusterCost(customInput)

console.log(`Monthly Cost: $${customResult.monthlyCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
console.log(`Effective GPU-Hour: $${customResult.effectiveUsableGpuHourCost.toFixed(2)}`)

console.log(`\nAssumptions: ${customResult.assumptions.length}`)
customResult.assumptions.slice(0, 5).forEach(a => console.log(`  • ${a}`))

console.log('\n\n✅ All tests completed successfully!')
