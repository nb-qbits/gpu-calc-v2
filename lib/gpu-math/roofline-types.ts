export type Dtype = 'fp16' | 'bf16' | 'fp8' | 'mxfp4' | 'int8' | 'int4' | 'fp32'
export type TrafficClass = 'realtime' | 'mixed' | 'batch'
export type GpuArch = 'hopper' | 'ampere' | 'ada' | 'blackwell'
export type MemoryType = 'hbm' | 'gddr'
export type BindingConstraint = 'prefill-bound' | 'decode-bound' | 'kv-memory-bound'
export type ConfidenceLevel = 'high' | 'medium' | 'default'
export type GeometrySource = 'known' | 'estimated'

export interface RooflineGpu {
  id: string
  display_name: string
  arch: GpuArch
  memory_type: MemoryType
  mem_gb: number
  hbm_bandwidth_gbps: number
  peak_flops: Partial<Record<Dtype, number>>  // TFLOP/s dense (no sparsity)
  default_mfu_prefill: number
  default_bw_efficiency_decode: number
}

export interface RooflineModel {
  id: string
  display_name: string
  is_moe: boolean
  total_params: number
  active_params: number
  num_layers: number
  d_model: number
  num_q_heads: number
  num_kv_heads: number
  head_dim: number
  kv_dtype_bytes: number
  kv_bytes_per_token: number  // 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
  geometry_source: GeometrySource
  resident_weights_gb?: number
  num_experts?: number
  experts_per_token?: number
}

export interface WorkloadInputs {
  model_id: string
  gpu_id: string
  dtype: Dtype
  tp: number
  requests_per_day: number
  peak_multiplier: number
  isl: number
  osl: number
  ttft_slo_ms: number
  traffic_class: TrafficClass
  gpu_mem_util: number
}

export interface Traffic {
  requests_per_day: number
  avg_rps: number
  peak_rps: number
  input_tps_avg: number
  output_tps_avg: number
  input_tps_peak: number
  output_tps_peak: number
  total_tokens_day: number
}

export interface KvBudget {
  kv_bytes_per_token: number
  weights_resident_bytes: number
  usable_mem_bytes: number
  kv_cache_budget_bytes: number
  max_kv_tokens: number
  max_concurrent_seqs: number
}

export interface TtftEstimate {
  ttft_compute_ms: number
  ttft_queue_ms: number
  ttft_ms: number
  utilization: number
  slo_ms: number
  slo_met: boolean
  slo_breach_reason: string | null
}

export interface CapacityEstimate {
  traffic: Traffic
  kv_budget: KvBudget
  prefill_tps_gpu: number
  decode_tps_gpu: number
  replicas: number
  replicas_low: number
  replicas_high: number
  binding_constraint: BindingConstraint
  ttft_estimate: TtftEstimate
  confidence: ConfidenceLevel
  assumptions: string[]
  warnings: string[]
  replicas_prefill: number
  replicas_decode: number
  replicas_concurrency: number
  mfu_used: number
  decode_bw_eff_used: number
  tp_used: number
  total_gpus: number
  tpot_ms: number
  eff_batch_used: number
  kv_ratio: number
  headroom_factor: number
}

export interface CostEstimate {
  on_demand_per_day: number
  on_demand_per_month: number
  reserved_per_day: number
  reserved_per_month: number
  cost_per_1m_tokens_on_demand: number
  cost_per_1m_tokens_reserved: number
}

export interface RooflineError {
  type: 'insufficient_vram' | 'no_kv_budget' | 'unknown_dtype'
  message: string
}

export type RooflineResult =
  | { ok: true; estimate: CapacityEstimate }
  | { ok: false; error: RooflineError }
