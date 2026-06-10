// Inference Configuration Engine
// Public API exports

// Main entry point
export { computeInferenceConfig } from './core'

// Types
export type {
  InferenceRequest,
  InferenceConfigResult,
  MemoryAnalysis,
  VLLMConfig,
  BottleneckAnalysis,
  ParallelismStrategy,
  LLMDConfig
} from './types'

// Validation
export { validateInferenceRequest, validateOrThrow } from './validation'
export type { ValidationResult } from './validation'

// Individual modules (for advanced use cases)
export {
  nextPowerOf2,
  computeTensorParallelSize,
  computeReplicas,
  computeUsableHBM
} from './tensor-parallel'

export {
  computeMaxModelLen,
  computeMaxNumSeqs,
  computeMaxNumBatchedTokens,
  shouldEnableChunkedPrefill,
  shouldEnablePrefixCaching,
  computeVLLMConfig
} from './vllm-defaults'

export { classifyBottleneck } from './bottleneck'
export { determineParallelismStrategy } from './parallelism'
export { recommendQuantization } from './quantization'
export { computeLLMDConfig } from './llmd'
