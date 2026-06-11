// Provider Profiles for Cluster Cost Module
// Defines defaults for different cloud providers and deployment types

import type { ProviderProfile } from './types'

export const PROVIDER_PROFILES: Record<string, ProviderProfile> = {
  'aws.h200.inference': {
    id: 'aws.h200.inference',
    name: 'AWS H200 Inference Cluster',
    defaults: {
      cluster: {
        name: 'AWS H200 Cluster',
        providerType: 'aws',
        gpuType: 'h200-141gb',
        gpuCount: 8,
        gpusPerNode: 8,
        hoursPerMonth: 720,
        utilizationTarget: 0.7,
        durationMonths: 36,
      },
      compute: {
        pricingModel: 'rental',
        gpuHourPrice: 1.14,
        discountPercent: 0,
      },
      storage: {
        hotTb: 10,
        warmTb: 50,
        coldTb: 100,
        hotPricePerTbMonth: 200,
        warmPricePerTbMonth: 23,
        coldPricePerTbMonth: 4,
      },
      network: {
        egressTbMonth: 5,
        egressPricePerTb: 90,
        loadBalancerMonthly: 50,
        natFirewallMonthly: 100,
        dataTransferMonthly: 200,
      },
      controlPlane: {
        monthlyCost: 500,
        cpuHelperNodeMonthly: 200,
        cpuHelperNodeCount: 2,
      },
      support: {
        included: false,
        percentOfBill: 0.03, // AWS support tier
        fixedMonthly: 0,
      },
      operations: {
        engineerHourlyRate: 150,
        setupHours: 80,
        setupAmortizationMonths: 12,
        debuggingHoursPerMonth: 10,
      },
      goodput: {
        enabled: true,
        lossPercent: 0.0002, // 0.02% downtime (modern cloud)
      },
    },
  },

  'azure.h100.inference': {
    id: 'azure.h100.inference',
    name: 'Azure H100 Inference Cluster',
    defaults: {
      cluster: {
        name: 'Azure H100 Cluster',
        providerType: 'azure',
        gpuType: 'h100-80gb',
        gpuCount: 8,
        gpusPerNode: 8,
        hoursPerMonth: 720,
        utilizationTarget: 0.7,
        durationMonths: 36,
      },
      compute: {
        pricingModel: 'rental',
        gpuHourPrice: 0.98,
        discountPercent: 0,
      },
      storage: {
        hotTb: 10,
        warmTb: 50,
        coldTb: 100,
        hotPricePerTbMonth: 180,
        warmPricePerTbMonth: 20,
        coldPricePerTbMonth: 3.5,
      },
      network: {
        egressTbMonth: 5,
        egressPricePerTb: 87,
        loadBalancerMonthly: 40,
        natFirewallMonthly: 90,
        dataTransferMonthly: 150,
      },
      controlPlane: {
        monthlyCost: 450,
        cpuHelperNodeMonthly: 180,
        cpuHelperNodeCount: 2,
      },
      support: {
        included: false,
        percentOfBill: 0.025,
        fixedMonthly: 0,
      },
      operations: {
        engineerHourlyRate: 150,
        setupHours: 80,
        setupAmortizationMonths: 12,
        debuggingHoursPerMonth: 10,
      },
      goodput: {
        enabled: true,
        lossPercent: 0.0002,
      },
    },
  },

  'on_prem.h100': {
    id: 'on_prem.h100',
    name: 'On-Prem H100 Cluster',
    defaults: {
      cluster: {
        name: 'On-Prem H100 Cluster',
        providerType: 'on_prem',
        gpuType: 'h100-80gb',
        gpuCount: 64,
        gpusPerNode: 8,
        hoursPerMonth: 720,
        utilizationTarget: 0.8,
        durationMonths: 48,
      },
      compute: {
        pricingModel: 'capex',
        hardwareCapex: 2000000, // $2M for 64 GPUs
        hardwareLifetimeMonths: 48,
      },
      storage: {
        hotTb: 50,
        warmTb: 200,
        coldTb: 500,
        hotPricePerTbMonth: 150,
        warmPricePerTbMonth: 15,
        coldPricePerTbMonth: 2,
      },
      network: {
        egressTbMonth: 0, // No egress fees on-prem
        egressPricePerTb: 0,
        loadBalancerMonthly: 0,
        natFirewallMonthly: 0,
        dataTransferMonthly: 500, // Internal network costs
      },
      controlPlane: {
        monthlyCost: 2000, // Bare metal control plane
        cpuHelperNodeMonthly: 150,
        cpuHelperNodeCount: 4,
      },
      support: {
        included: false,
        percentOfBill: 0,
        fixedMonthly: 5000, // Vendor support contract
      },
      operations: {
        engineerHourlyRate: 150,
        setupHours: 320, // On-prem setup is more intensive
        setupAmortizationMonths: 24,
        debuggingHoursPerMonth: 20, // More hands-on debugging
      },
      goodput: {
        enabled: true,
        lossPercent: 0.001, // 0.1% - slightly higher than cloud
      },
    },
  },

  'neocloud.h100': {
    id: 'neocloud.h100',
    name: 'Neocloud H100 Cluster',
    defaults: {
      cluster: {
        name: 'Neocloud H100 Cluster',
        providerType: 'neocloud',
        gpuType: 'h100-80gb',
        gpuCount: 8,
        gpusPerNode: 8,
        hoursPerMonth: 720,
        utilizationTarget: 0.75,
        durationMonths: 36,
      },
      compute: {
        pricingModel: 'rental',
        gpuHourPrice: 0.82, // Competitive pricing
        discountPercent: 0,
      },
      storage: {
        hotTb: 10,
        warmTb: 50,
        coldTb: 100,
        hotPricePerTbMonth: 160,
        warmPricePerTbMonth: 18,
        coldPricePerTbMonth: 3,
      },
      network: {
        egressTbMonth: 5,
        egressPricePerTb: 75, // Lower egress than hyperscalers
        loadBalancerMonthly: 30,
        natFirewallMonthly: 60,
        dataTransferMonthly: 100,
      },
      controlPlane: {
        monthlyCost: 300,
        cpuHelperNodeMonthly: 120,
        cpuHelperNodeCount: 2,
      },
      support: {
        included: true, // Often included for specialized providers
        percentOfBill: 0,
        fixedMonthly: 0,
      },
      operations: {
        engineerHourlyRate: 150,
        setupHours: 40, // Managed service - less setup
        setupAmortizationMonths: 12,
        debuggingHoursPerMonth: 5, // Less debugging with managed service
      },
      goodput: {
        enabled: true,
        lossPercent: 0.0003, // 0.03% - newer infra
      },
    },
  },
}

export function getProviderProfile(id: string): ProviderProfile | undefined {
  return PROVIDER_PROFILES[id]
}

export function listProviderProfiles(): ProviderProfile[] {
  return Object.values(PROVIDER_PROFILES)
}
