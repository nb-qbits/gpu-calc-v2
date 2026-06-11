# Cluster Cost Module

Calculate the real monthly cost of GPU clusters beyond headline GPU-hour pricing.

Based on SemiAnalysis TCO methodology, this module accounts for 8 cost categories:
1. GPU (rental or capex)
2. Storage (hot/warm/cold tiers)
3. Network (egress, load balancer, NAT/firewall)
4. Control Plane (K8s, helper nodes)
5. Support (vendor support contracts)
6. Setup (engineering time, amortized)
7. Debugging (ongoing operational costs)
8. Goodput (downtime/reliability loss)

## Design Principles

- **Pure calculation engine** - No UI, no cloud-specific logic
- **Provider-neutral** - All provider defaults come from profiles, not hardcoded
- **All defaults overridable** - User can customize any assumption
- **Transparent** - Returns assumptions and warnings with every result
- **Two pricing models** - Rental (cloud) and CAPEX (on-prem)

## Usage

### Basic Example

```typescript
import { computeClusterCost, getProviderProfile, applyProviderProfile } from '@/lib/cluster-cost'

// Use AWS H200 defaults
const profile = getProviderProfile('aws.h200.inference')!
const input = applyProviderProfile(profile, {
  cluster: {
    gpuCount: 16, // Override: use 16 GPUs instead of default 8
  }
})

const result = computeClusterCost(input)

console.log(`Monthly Cost: $${result.monthlyCost.toLocaleString()}`)
console.log(`Effective GPU-Hour: $${result.effectiveUsableGpuHourCost.toFixed(2)}`)
console.log(`Headline GPU-Hour: $${result.rawGpuHourCost.toFixed(2)}`)
```

### Manual Configuration

```typescript
import { computeClusterCost } from '@/lib/cluster-cost'
import type { ClusterCostInput } from '@/lib/cluster-cost'

const input: ClusterCostInput = {
  cluster: {
    name: 'Production Inference',
    providerType: 'aws',
    gpuType: 'h100-80gb',
    gpuCount: 32,
    gpusPerNode: 8,
    hoursPerMonth: 720,
    utilizationTarget: 0.8,
    durationMonths: 36,
  },
  compute: {
    pricingModel: 'rental',
    gpuHourPrice: 0.98,
    discountPercent: 0.1, // 10% volume discount
  },
  storage: {
    hotTb: 20,
    warmTb: 100,
    coldTb: 500,
    hotPricePerTbMonth: 200,
    warmPricePerTbMonth: 23,
    coldPricePerTbMonth: 4,
  },
  network: {
    egressTbMonth: 10,
    egressPricePerTb: 90,
    loadBalancerMonthly: 100,
    natFirewallMonthly: 200,
    dataTransferMonthly: 300,
  },
  controlPlane: {
    monthlyCost: 1000,
    cpuHelperNodeMonthly: 250,
    cpuHelperNodeCount: 4,
  },
  support: {
    included: false,
    percentOfBill: 0.03,
    fixedMonthly: 0,
  },
  operations: {
    engineerHourlyRate: 150,
    setupHours: 120,
    setupAmortizationMonths: 12,
    debuggingHoursPerMonth: 15,
  },
  goodput: {
    enabled: true,
    lossPercent: 0.0002, // 0.02% downtime
  },
}

const result = computeClusterCost(input)
```

### Compare Cloud vs On-Prem

```typescript
import { computeClusterCost, getProviderProfile, applyProviderProfile } from '@/lib/cluster-cost'

// AWS rental
const awsProfile = getProviderProfile('aws.h200.inference')!
const awsInput = applyProviderProfile(awsProfile, { cluster: { gpuCount: 64 } })
const awsResult = computeClusterCost(awsInput)

// On-prem CAPEX
const onPremProfile = getProviderProfile('on_prem.h100')!
const onPremInput = applyProviderProfile(onPremProfile, { cluster: { gpuCount: 64 } })
const onPremResult = computeClusterCost(onPremInput)

console.log('AWS 3-Year Total:', awsResult.totalCostForDuration)
console.log('On-Prem 4-Year Total:', onPremResult.totalCostForDuration)

// Breakeven analysis
const months = onPremResult.totalCostForDuration / awsResult.monthlyCost
console.log(`Breakeven: ${months.toFixed(1)} months`)
```

## Available Provider Profiles

- `aws.h200.inference` - AWS H200 141GB rental
- `azure.h100.inference` - Azure H100 80GB rental
- `on_prem.h100` - On-premise H100 CAPEX
- `neocloud.h100` - Neocloud H100 rental (competitive GPU cloud)

## Output Schema

```typescript
interface ClusterCostResult {
  monthlyCost: number              // Total monthly cost
  annualCost: number               // 12 × monthly
  totalCostForDuration: number     // monthlyCost × durationMonths
  rawGpuHourCost: number           // Headline GPU-hour price
  effectiveUsableGpuHourCost: number  // Real cost accounting for utilization and TCO

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

  assumptions: string[]  // What went into the calculation
  warnings: string[]     // Things to watch out for
}
```

## Key Metrics

### Raw GPU-Hour Cost
Headline price: `gpuHourPrice` for rental, or `hardwareCapex / (lifetime × hours × GPUs)` for CAPEX.

### Effective Usable GPU-Hour Cost
Real cost per usable GPU-hour, accounting for:
- Utilization (not all GPUs run 24/7)
- Storage, network, control plane overhead
- Support contracts
- Engineering time (setup, debugging)
- Downtime/reliability loss

Formula:
```
effectiveUsableGpuHourCost = monthlyCost / (gpuCount × hoursPerMonth × utilizationTarget)
```

This is typically **2-4x higher than headline price** for cloud deployments.

## Testing

```bash
npx tsx lib/cluster-cost/test-cluster-cost.ts
```

Runs 3 test scenarios:
1. AWS H200 with defaults
2. On-prem H100 CAPEX
3. Custom high-scale configuration

## Integration with GPU Calc

This module is designed to plug into the main gpu-calc workflow:

```
Workload Module (Quick Estimate)
  ↓
  "Needs 64 H100 GPUs"
  ↓
Cluster Cost Module ← YOU ARE HERE
  ↓
  "$X/month cluster cost"
  ↓
Decision Module
  ↓
  "Cloud / Hybrid / On-prem comparison"
```

Two modes:
- **Manual Mode**: User enters cluster details directly
- **Auto Mode**: Workload engine passes recommended cluster shape

## Future Enhancements (v2)

Current goodput model uses simple `lossPercent`. Future versions could model:
- MTBF (mean time between failures)
- MTTR (mean time to repair)
- Failure detection time
- Hot spares
- Blast radius
- Retry behavior

This would provide more accurate reliability cost modeling for different provider SLAs.

## References

Based on analysis methodology from:
- SemiAnalysis GPU cluster TCO reports
- Real-world deployment cost breakdowns
- Cloud provider pricing calculators (AWS, Azure, GCP)
- Specialized GPU cloud providers (Nebius, Lambda, CoreWeave)
