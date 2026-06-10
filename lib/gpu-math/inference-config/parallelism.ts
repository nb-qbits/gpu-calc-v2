// Parallelism Strategy Module
// Determines optimal parallelism strategy based on topology and scale

import type { ParallelismStrategy } from './types'

/**
 * Determine parallelism strategy based on TP size and network topology.
 *
 * PDF spec rules:
 * - TP fits in single node → TP_ONLY (optimal with NVLink)
 * - TP across nodes + InfiniBand → TP_ACROSS_NODES (15-20% penalty)
 * - TP across nodes + Ethernet → PP_ACROSS_NODES + TP_WITHIN_NODE
 * - >2 nodes at production scale → DISAGGREGATED
 *
 * @param tp_size - Tensor parallel size
 * @param gpus_per_node - GPUs per node (default 8)
 * @param network_topology - Network topology (nvlink, infiniband, ethernet)
 * @returns Parallelism strategy with topology notes
 */
export function determineParallelismStrategy(
  tp_size: number,
  gpus_per_node: number = 8,
  network_topology: 'nvlink' | 'infiniband' | 'ethernet' = 'nvlink'
): ParallelismStrategy {
  // TP fits in single node - optimal case
  if (tp_size <= gpus_per_node) {
    return {
      strategy: 'TP_ONLY',
      pp_size: 1,
      topology_note: 'NVLink within node — optimal'
    }
  }

  // Multi-node required
  const nodes_needed = Math.ceil(tp_size / gpus_per_node)

  // More than 2 nodes - recommend disaggregated architecture
  if (nodes_needed > 2) {
    return {
      strategy: 'DISAGGREGATED',
      pp_size: 1,
      topology_note: 'Requires llm-d or Mooncake-style orchestration for >2 nodes'
    }
  }

  // Cross-node with InfiniBand - acceptable but with latency penalty
  if (network_topology === 'infiniband') {
    return {
      strategy: 'TP_ACROSS_NODES',
      pp_size: 1,
      topology_note: '15-20% latency penalty vs NVLink (InfiniBand cross-node TP)'
    }
  }

  // Cross-node with Ethernet - must use pipeline parallelism
  if (network_topology === 'ethernet') {
    return {
      strategy: 'PP_ACROSS_NODES',
      pp_size: nodes_needed,
      topology_note: 'Ethernet too slow for cross-node TP AllReduce — using PP instead'
    }
  }

  // Default fallback
  return {
    strategy: 'TP_ONLY',
    pp_size: 1,
    topology_note: 'Default strategy'
  }
}
