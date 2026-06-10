// Validation Module
// Single validation point for all inference requests
// Catches invalid inputs, conflicts, and provides recommendations

import type { InferenceRequest } from './types'
import { nextPowerOf2 } from './tensor-parallel'

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
  recommendations?: {
    gpu_count?: number
    reason?: string
  }
}

const VALID_PRECISIONS = ['FP16', 'FP8', 'INT8', 'INT4'] as const
const VALID_WORKLOAD_TYPES = ['chat', 'web_search', 'rag', 'batch', 'coding'] as const
const VALID_SLA_PRIORITIES = ['ttft', 'tpot', 'throughput'] as const
const VALID_NETWORK_TOPOLOGIES = ['nvlink', 'infiniband', 'ethernet'] as const

/**
 * Validate inference request at entry point.
 * Catches all invalid inputs before any computation.
 *
 * @param req - Inference request to validate
 * @returns Validation result with errors, warnings, and recommendations
 */
export function validateInferenceRequest(req: InferenceRequest): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  let recommendations: ValidationResult['recommendations'] = undefined

  // ═══ CRITICAL VALIDATIONS (block execution) ═══

  // Validate ISL (Input Sequence Length)
  if (typeof req.isl !== 'number' || req.isl <= 0) {
    errors.push('ISL (input sequence length) must be a positive number')
  }

  // Validate OSL (Output Sequence Length)
  if (typeof req.osl !== 'number' || req.osl <= 0) {
    errors.push('OSL (output sequence length) must be a positive number')
  }

  // Validate concurrent users
  if (typeof req.concurrent_users !== 'number' || req.concurrent_users <= 0) {
    errors.push('concurrent_users must be a positive number')
  }

  // Validate precision enum
  if (!VALID_PRECISIONS.includes(req.precision as any)) {
    errors.push(`precision must be one of: ${VALID_PRECISIONS.join(', ')}`)
  }

  // Validate workload_type enum
  if (!VALID_WORKLOAD_TYPES.includes(req.workload_type as any)) {
    errors.push(`workload_type must be one of: ${VALID_WORKLOAD_TYPES.join(', ')}`)
  }

  // Validate sla_priority enum
  if (!VALID_SLA_PRIORITIES.includes(req.sla_priority as any)) {
    errors.push(`sla_priority must be one of: ${VALID_SLA_PRIORITIES.join(', ')}`)
  }

  // Validate network_topology if provided
  if (req.network_topology && !VALID_NETWORK_TOPOLOGIES.includes(req.network_topology as any)) {
    errors.push(`network_topology must be one of: ${VALID_NETWORK_TOPOLOGIES.join(', ')}`)
  }

  // Validate model_name (basic check - not empty)
  if (!req.model_name || req.model_name.trim().length === 0) {
    errors.push('model_name cannot be empty')
  }

  // Validate gpu_type (basic check - not empty)
  if (!req.gpu_type || req.gpu_type.trim().length === 0) {
    errors.push('gpu_type cannot be empty')
  }

  // ═══ GPU COUNT VALIDATION & RECOMMENDATIONS ═══

  if (req.gpu_count !== undefined) {
    if (typeof req.gpu_count !== 'number' || req.gpu_count <= 0) {
      errors.push('gpu_count must be a positive number if provided')
    } else {
      // Check if gpu_count will result in valid TP size
      // This is a preliminary check - actual TP sizing happens in core.ts
      // We're checking if the user's gpu_count makes sense

      // If gpu_count is not a power of 2 and less than 8, warn about potential waste
      const isPowerOf2 = (n: number) => n > 0 && (n & (n - 1)) === 0
      if (!isPowerOf2(req.gpu_count) && req.gpu_count < 8) {
        warnings.push(
          `GPU count ${req.gpu_count} is not a power of 2. ` +
          `If TP sizing requires power-of-2, some GPUs may be unused.`
        )
      }
    }
  }

  // ═══ VLLM FEATURE CONFLICT CHECKS ═══

  // Chunked prefill and prefix caching cannot be used together in vLLM
  // This is a soft check - we'll auto-resolve in vllm-defaults.ts
  // Just warn the user
  if (req.workload_type === 'chat' || req.workload_type === 'coding') {
    // These workloads might benefit from prefix caching
    // But if ISL > 1000, chunked prefill would also be enabled
    // We'll auto-prioritize in vllm-defaults.ts, but warn here
    if (req.isl > 1000) {
      warnings.push(
        'Long ISL detected. Chunked prefill will be prioritized over prefix caching ' +
        '(they cannot be enabled simultaneously in vLLM).'
      )
    }
  }

  // ═══ WORKLOAD WARNINGS ═══

  // Very high concurrent users
  if (req.concurrent_users > 1000) {
    warnings.push(
      `Very high concurrent users (${req.concurrent_users}). ` +
      `Consider using multiple replicas for load distribution.`
    )
  }

  // Very long context (ISL + OSL)
  const total_context = req.isl + req.osl
  if (total_context > 100000) {
    warnings.push(
      `Very long context (${total_context.toLocaleString()} tokens). ` +
      `KV cache memory will be significant.`
    )
  }

  // Batch workload with interactive SLA
  if (req.workload_type === 'batch' && req.sla_priority === 'ttft') {
    warnings.push(
      'Batch workload with TTFT priority is unusual. ' +
      'Batch workloads typically optimize for throughput.'
    )
  }

  // ═══ QUANTIZATION WARNINGS ═══

  // INT4 for quality-sensitive workloads
  if (req.precision === 'INT4' && ['coding', 'rag'].includes(req.workload_type)) {
    warnings.push(
      `INT4 quantization with ${req.workload_type} workload may degrade quality. ` +
      `Consider FP8 or FP16 for better accuracy.`
    )
  }

  // ═══ RETURN VALIDATION RESULT ═══

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    recommendations
  }
}

/**
 * Validate and throw if invalid.
 * Convenience function for use in core.ts.
 *
 * @param req - Inference request
 * @throws Error with validation messages if invalid
 * @returns Validation result (warnings only, since valid=true)
 */
export function validateOrThrow(req: InferenceRequest): ValidationResult {
  const result = validateInferenceRequest(req)

  if (!result.valid) {
    const errorMsg = [
      'Invalid inference request:',
      ...result.errors.map(e => `  - ${e}`)
    ].join('\n')
    throw new Error(errorMsg)
  }

  return result
}
