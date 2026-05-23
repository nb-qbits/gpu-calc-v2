// gpu-math — core GPU sizing and inference economics engine.
// All GPU sizing logic lives here, isolated from React components.

// ── Static catalogs (display metadata, not used in computation) ───────────────
export * from './models'
export * from './gpus'

// ── KV cache engine ───────────────────────────────────────────────────────────
export * from './kv-types'
export * from './kv-config'
export * from './kv-detect'
export * from './kv-formulas'
export * from './kv-budget'

// ── Legacy exports kept for cost and throughput pages ─────────────────────────
export * from './cost'
export * from './throughput'

// ── Context length constants (used by Quick Estimate UI) ──────────────────────
export type ContextLength = 'short' | 'medium' | 'long' | 'verylong'
export type DeploymentType = 'cloud' | 'onprem' | 'hybrid'
export type TensorParallelism = 'auto' | 1 | 2 | 4 | 8

export const CONTEXT_TOKENS: Record<ContextLength, number> = {
  short:    8_192,
  medium:   65_536,
  long:     131_072,
  verylong: 1_048_576,
}

export const CONTEXT_LABELS: Record<ContextLength, string> = {
  short:    'Short',
  medium:   'Medium',
  long:     'Long',
  verylong: 'Very long',
}

export const CONTEXT_SUBLABELS: Record<ContextLength, string> = {
  short:    '8K tokens',
  medium:   '64K tokens',
  long:     '128K tokens',
  verylong: '1M tokens',
}

// Derives ISL and OSL from a total context window selection.
// ISL = 75% (the prompt), OSL = 25% (the generated reply).
export function contextToDeployParams(tokens: number): { ISL: number; OSL: number; max_model_len: number } {
  return {
    ISL:           Math.round(tokens * 0.75),
    OSL:           Math.round(tokens * 0.25),
    max_model_len: tokens,
  }
}
