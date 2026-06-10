// Inference Configuration Engine Types
// All TypeScript interfaces for vLLM/llm-d configuration

import type { HFModelConfig } from '@/lib/huggingface/fetch-config'

export interface InferenceRequest {
  model_name: string
  precision: 'FP16' | 'FP8' | 'INT8' | 'INT4'  // Weight precision
  gpu_type: string
  gpu_count?: number  // Optional - engine recommends if not provided
  concurrent_users: number
  isl: number  // Input sequence length
  osl: number  // Output sequence length
  workload_type: 'chat' | 'web_search' | 'rag' | 'batch' | 'coding'
  sla_priority: 'ttft' | 'tpot' | 'throughput'

  // Optional overrides
  kv_cache_precision?: 'FP16' | 'FP8'  // KV cache dtype (defaults to match weight precision if not specified)
  network_topology?: 'nvlink' | 'infiniband' | 'ethernet'
  enable_llmd?: boolean

  // Optional: provide fetched HuggingFace config directly
  hf_config?: HFModelConfig
}

export interface MemoryAnalysis {
  weight_gb: number
  weight_gb_per_gpu: number
  usable_hbm_per_gpu: number
  tp_size: number
  replicas: number
  kv_cache_budget_gb: number  // Available memory for KV cache
  kv_cache_used_gb?: number   // Actual KV cache consumed by concurrent users
  max_sequences_from_memory: number
  kv_category?: string        // KV-1, KV-2, KV-3a, KV-3b, etc.
  kv_category_label?: string  // Human-readable description
}

export interface VLLMConfig {
  tensor_parallel_size: number
  max_model_len: number
  max_num_seqs: number
  gpu_memory_utilization: number
  max_num_batched_tokens: number
  enable_chunked_prefill: boolean
  enable_prefix_caching: boolean
  quantization: string
}

export interface BottleneckAnalysis {
  primary: 'TTFT' | 'TPOT' | 'THROUGHPUT' | 'MIXED'
  risk: string
  fix_suggestions: string[]
}

export interface ParallelismStrategy {
  strategy: 'TP_ONLY' | 'TP_ACROSS_NODES' | 'PP_ACROSS_NODES' | 'DISAGGREGATED'
  pp_size: number
  topology_note: string
}

export interface LLMDConfig {
  prefill_instances: {
    count: number
    tp_size: number
    max_num_seqs: number
    enable_chunked_prefill: boolean
    gpu_memory_utilization: number
  }
  decode_instances: {
    count: number
    tp_size: number
    max_num_seqs: number
    gpu_memory_utilization: number
    max_model_len: number
    quantization: string
  }
  kv_transfer: {
    block_size: number
    transfer_size_mb: number
    network_requirement: string
  }
}

export interface InferenceConfigResult {
  memory_analysis: MemoryAnalysis
  vllm_config: VLLMConfig
  parallelism_strategy: ParallelismStrategy
  bottleneck_analysis: BottleneckAnalysis
  llmd_config?: LLMDConfig
  diagnostics: {
    nvidia_smi_watch: string
    dcgm_metrics: string[]
    vllm_metrics: string[]
  }
  warnings: string[]  // Validation warnings and recommendations
}
