// Bottleneck Classification Module
// Classifies primary bottleneck based on ISL:OSL ratio and workload type

import type { InferenceRequest, BottleneckAnalysis } from './types'

/**
 * Classify primary bottleneck based on workload characteristics.
 *
 * PDF spec rules:
 * - ISL >> OSL (ratio > 5:1) → TTFT/prefill-bound
 * - OSL >> ISL (ratio > 3:1) → TPOT/decode-bound / memory-bandwidth
 * - ISL ≈ OSL + high concurrency → MIXED
 * - workload_type == batch → THROUGHPUT
 *
 * @param req - Inference request
 * @returns Bottleneck analysis with primary bottleneck and fix suggestions
 */
export function classifyBottleneck(
  req: InferenceRequest
): BottleneckAnalysis {
  const ratio = req.isl / req.osl

  // Batch workload always throughput-bound
  if (req.workload_type === 'batch') {
    return {
      primary: 'THROUGHPUT',
      risk: 'Batch processing is throughput-bound',
      fix_suggestions: [
        'Increase replicas for higher throughput',
        'Disable chunked_prefill for batch workloads',
        'Increase max_num_seqs to batch more requests'
      ]
    }
  }

  // Prefill-bound: long prompts, short outputs
  if (ratio > 5) {
    return {
      primary: 'TTFT',
      risk: 'Long prompts cause high time-to-first-token',
      fix_suggestions: [
        'Enable chunked_prefill to reduce P90 TTFT',
        'Enable prefix_caching if shared prefixes exist',
        'Consider FP8 quantization for faster prefill compute'
      ]
    }
  }

  // Decode-bound: short prompts, long outputs
  if (ratio < 1 / 3) {
    return {
      primary: 'TPOT',
      risk: 'Long generation is memory-bandwidth bound',
      fix_suggestions: [
        'Use FP8 quantization (2x bandwidth reduction)',
        'Tune max_num_seqs for better decode batching',
        'Consider reducing gpu_memory_utilization if memory-constrained'
      ]
    }
  }

  // Mixed workload with high concurrency
  if (req.concurrent_users > 50) {
    return {
      primary: 'MIXED',
      risk: 'Balanced ISL/OSL with high concurrency stresses both prefill and decode',
      fix_suggestions: [
        'Enable chunked_prefill to balance prefill and decode',
        'Right-size max_model_len (no overallocation)',
        'Tune max_num_seqs for workload concurrency'
      ]
    }
  }

  // Default: TTFT if unknown pattern
  return {
    primary: 'TTFT',
    risk: 'Moderate workload, watch for prefill latency',
    fix_suggestions: [
      'Monitor TTFT metrics',
      'Enable chunked_prefill if TTFT exceeds SLA'
    ]
  }
}
