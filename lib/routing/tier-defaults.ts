export interface TierConfig {
  id: string
  label: string
  color: string
  defaultPct: number
  defaultTokensIn: number
  defaultTokensOut: number
  defaultFrontierModelId: string
  defaultOssModelId: string
  defaultGpuType: string
  defaultGpuPerReplica: number
  defaultCapacityPerReplica: number
}

export const DEFAULT_TIERS: TierConfig[] = [
  {
    id: 'simple',
    label: 'Simple queries',
    color: '#0066cc',
    defaultPct: 60,
    defaultTokensIn: 200,
    defaultTokensOut: 100,
    defaultFrontierModelId: 'claude-3.5-haiku',
    defaultOssModelId: 'llama-3.1-8b',
    defaultGpuType: 'l40s-48gb',
    defaultGpuPerReplica: 1,
    defaultCapacityPerReplica: 150,
  },
  {
    id: 'standard',
    label: 'Standard queries',
    color: '#f0ab00',
    defaultPct: 30,
    defaultTokensIn: 800,
    defaultTokensOut: 400,
    defaultFrontierModelId: 'claude-3.5-sonnet',
    defaultOssModelId: 'llama-3.1-70b',
    defaultGpuType: 'h100-80gb',
    defaultGpuPerReplica: 2,
    defaultCapacityPerReplica: 60,
  },
  {
    id: 'complex',
    label: 'Complex reasoning',
    color: '#5e40be',
    defaultPct: 10,
    defaultTokensIn: 2000,
    defaultTokensOut: 1000,
    defaultFrontierModelId: 'gpt-4o',
    defaultOssModelId: 'llama-3.1-405b',
    defaultGpuType: 'h100-80gb',
    defaultGpuPerReplica: 8,
    defaultCapacityPerReplica: 15,
  },
]
