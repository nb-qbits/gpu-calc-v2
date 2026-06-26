// KV cache engine — core type definitions
// All computation in the engine derives from these types.
// No `any` types. No architecture values hardcoded outside of config extraction.

// ─── KV Category ─────────────────────────────────────────────────────────────

export type KVCategory =
  | 'KV-1'   // Standard Dense (GQA / MHA / MQA)
  | 'KV-2'   // MLA Low-rank KV (DeepSeek)
  | 'KV-3a'  // Full Sliding Window (Mistral)
  | 'KV-3b'  // Hybrid Sliding + Global (Gemma 3, Qwen3)
  | 'KV-4'   // Cross-Layer Attention Sharing (Hunyuan)
  | 'KV-5a'  // Pure SSM — zero KV cache (Mamba)
  | 'KV-5b'  // SSM + Attention Hybrid (Jamba, Nemotron)
  | 'KV-5c'  // Linear Recurrence + Local Attention (RecurrentGemma)

export const KV_CATEGORY_LABELS: Record<KVCategory, string> = {
  'KV-1':  'Standard Dense (GQA / MHA / MQA)',
  'KV-2':  'MLA low-rank KV',
  'KV-3a': 'Full sliding window',
  'KV-3b': 'Hybrid sliding + global',
  'KV-4':  'Cross-layer attention sharing',
  'KV-5a': 'Pure SSM — no KV cache',
  'KV-5b': 'SSM + attention hybrid',
  'KV-5c': 'Linear recurrence + local attention',
}

export const KV_CATEGORY_FORMULAS: Record<KVCategory, string> = {
  'KV-1':  '2 × kv_heads_per_gpu × head_dim × B × L',
  'KV-2':  '(kv_lora_rank + qk_rope_head_dim) × B × L',
  'KV-3a': '2 × kv_heads_per_gpu × head_dim × B × L  [bounded by sliding window]',
  'KV-3b': '[n_global × seq + n_sliding × min(seq, W)] × kv_bytes_per_layer',
  'KV-4':  '2 × kv_heads_per_gpu × head_dim × B × ceil(L / share_factor)',
  'KV-5a': '0  [no KV cache — pure recurrent state]',
  'KV-5b': '2 × kv_heads_per_gpu × head_dim × B × n_attn_layers',
  'KV-5c': '2 × kv_heads_per_gpu × head_dim × B × n_attn_layers  [bounded by window]',
}

// ─── Dtype ────────────────────────────────────────────────────────────────────

export const DTYPE_BYTES: Record<string, number> = {
  float32:          4,
  bfloat16:         2,
  float16:          2,
  float8:           1,
  float8_e4m3fn:    1,
  float8_e4m3:      1,
  float8_e5m2:      1,
  float8_e5m2fnuz:  1,
  float8_e4m3fnuz:  1,
  int8:             1,
  int4:             0.5,
}

export type KVDtypeSource =
  | 'user_provided'        // user set kv_cache_dtype explicitly in advanced panel
  | 'config_field'         // config.json contains kv_cache_dtype field
  | 'torch_dtype_fallback' // defaulted to compute dtype

export interface KVDtypeResolution {
  dtype:        string
  bytes:        number
  source:       KVDtypeSource
  weight_dtype: string
  warning?:     string
}

// ─── TP sharding ──────────────────────────────────────────────────────────────

export type KVTPMode =
  | 'sharded'        // H_kv >= TP: KV memory reduces with TP
  | 'replicated'     // TP > H_kv: no KV benefit, heads replicated
  | 'mla_replicated' // KV-2 MLA: latent vector, never sharded
  | 'zero'           // KV-5a: no KV cache at all

// ─── Extracted config (parsed from HuggingFace config.json) ───────────────────

export interface MoEInfo {
  total_routed_experts:   number
  shared_experts:         number
  total_experts:          number
  active_routed_per_tok:  number
  active_experts_per_tok: number
  active_ratio:           number
  moe_intermediate_size:  number | null
  expert_layer_period:    number | null
  expert_layer_offset:    number
}

export interface QuantizationConfig {
  type:                    'fp8' | 'int8' | 'int4' | 'gptq' | 'awq' | 'bnb' | 'none' | 'unknown'
  bits?:                   number
  group_size?:             number
  modules_to_not_convert?: string[]
  quant_type?:             string
}

export interface ExtractedConfig {
  model_type:  string
  // Attention geometry
  L:           number   // num_hidden_layers
  H_q:         number   // num_attention_heads
  H_kv:        number   // num_key_value_heads
  d:           number   // head_dim
  d_source:    'explicit' | 'computed' | 'model-families.json'
  hidden_size: number
  intermediate_size: number
  vocab_size:  number
  B:           number   // bytes per compute element (from torch_dtype)
  dtype:       string   // torch_dtype string

  // Sliding window
  sliding_window:             number | null
  sliding_window_pattern:     number | null
  use_sliding_window:         boolean | null
  global_attn_every_n_layers: number | null
  layer_types:                string[] | null
  max_window_layers:          number | null

  // MLA (DeepSeek)
  kv_lora_rank:     number | null
  qk_rope_head_dim: number | null

  // CLA (Hunyuan)
  use_cla:          boolean | null
  cla_share_factor: number | null

  // SSM / Mamba
  ssm_cfg:       Record<string, unknown> | null
  mamba_d_state: number | null
  mamba_d_conv:  number | null
  mamba_expand:  number | null

  // Hybrid attention layers
  attn_layer_period:    number | null
  attn_layer_offset:    number | null
  attention_layers_idx: number[] | null

  // Linear recurrence (RecurrentGemma)
  block_types:            string[] | null
  attention_window_size:  number | null
  lru_width:              number | null
  conv1d_width:           number | null
  residual_in_fp32:       boolean | null

  // MoE
  is_moe:               boolean
  total_routed_experts: number | null
  shared_experts:       number
  active_routed_per_tok: number | null
  total_experts:        number | null
  active_experts_per_tok: number | null
  active_ratio:         number | null
  moe_intermediate_size: number | null
  expert_layer_period:  number | null
  expert_layer_offset:  number

  // Multimodal
  is_multimodal:       boolean
  mm_tokens_per_image: number | null

  // Quantization
  quantization_config: QuantizationConfig

  // KV cache dtype from config (optional field)
  kv_cache_dtype?: string
}

// ─── Deployment parameters (from UI inputs) ───────────────────────────────────

export interface DeploymentParams {
  tp:                     number   // tensor parallel degree
  max_model_len:          number   // maps from context length selection
  max_num_seqs:           number   // concurrent users
  gpu_memory_utilization: number   // 0.0–1.0
  ISL:                    number   // input sequence length (75% of max_model_len)
  OSL:                    number   // output sequence length (25% of max_model_len)
  block_size:             16 | 32 | 64 | 128
  kv_cache_dtype?:        string   // user override: 'auto' | 'fp8' | undefined
  mamba_ssm_cache_dtype:  'float32' | 'float16' | 'bfloat16'
}

// ─── Weight memory provenance ─────────────────────────────────────────────────

export type WeightMemorySource =
  | 'safetensors_exact'       // from model.safetensors.index.json total_size
  | 'safetensors_overestimate'// multimodal: includes vision weights
  | 'hf_api_exact'            // from HF API parameter count × dtype bytes
  | 'estimated'               // computed from config fields
  | 'metadata_lookup_failure' // all HF lookups failed

export type WeightMemoryConfidence = 'exact' | 'high' | 'medium' | 'low'

// ─── KV cache result ──────────────────────────────────────────────────────────

export interface KVCacheResult {
  kv_category:         KVCategory
  kv_category_label:   string
  formula:             string
  kv_bytes_per_token:  number
  kv_bytes_per_token_fallback?: number  // KV-4 only: without CLA support
  kv_heads_per_gpu:    number
  kv_tp_mode:          KVTPMode
  tp_inflection:       number
  tp_gives_kv_benefit: boolean
  is_bounded:          boolean
  bound_tokens?:       number
  kv_dtype:            KVDtypeResolution
  is_moe:              boolean
  warnings:            string[]
}

// ─── KV memory — 3 scenarios ──────────────────────────────────────────────────

export interface KVMemory3Cases {
  optimistic:   number   // ISL only (bytes)
  expected:     number   // ISL + OSL (bytes)
  conservative: number   // max_model_len (bytes)
}

// ─── Recurrent state (KV-5a / 5b / 5c) ───────────────────────────────────────

export interface RecurrentState {
  n_recurrent_layers:        number
  state_bytes_per_seq:       number
  conv_bytes_per_seq:        number
  total_state_bytes_per_seq: number
  total_state_memory_bytes:  number
}

// ─── Full memory budget ───────────────────────────────────────────────────────

export interface MemoryBudget {
  weight_memory_total:  number   // total across all GPUs (reference)
  weight_per_gpu:       number   // weight_memory_total / TP
  activation_per_gpu:   number   // act_coeff × weight_per_gpu (MoE-aware)
  recurrent_per_gpu:    number   // recurrent state / TP
  safety_buffer:        number   // per GPU
  overhead:             number   // per GPU (CUDA context + logits buffer)
  kv_memory:            KVMemory3Cases  // already per-GPU
  total_per_gpu_optimistic:   number
  total_per_gpu_expected:     number
  total_per_gpu_conservative: number
}

// ─── GPU count result ─────────────────────────────────────────────────────────

export interface GPUCountResult {
  optimistic:   number
  expected:     number
  conservative: number
  headroom_gb: {
    optimistic:   number
    expected:     number
    conservative: number
  }
  tp_used:  number
  warnings: string[]
}

// ─── TP validation ────────────────────────────────────────────────────────────

export interface TPValidation {
  is_valid:           boolean
  hard_reject:        boolean
  reject_reason?:     string
  warn_kv_split:      boolean
  kv_heads_per_gpu:   number
  kv_replication:     boolean
  kv_tp_mode:         KVTPMode
  tp_inflection:      number
  tp_gives_kv_benefit: boolean
  warning?:           string
}

// ─── Detection result ─────────────────────────────────────────────────────────

export type SourceType =
  | 'config.json'
  | 'computed'
  | 'model-families.json'
  | 'inferred'
  | 'user_input'
  | 'default'

export type Confidence = 'high' | 'medium-high' | 'medium' | 'low'

export interface DetectionResult {
  category:   KVCategory
  source:     SourceType
  confidence: Confidence
  fields:     string[]
  warnings?:  string[]
}

// ─── Model families (from lib/data/model-families.json) ──────────────────────

export interface ModelFamilyEntry {
  kv_category?:               KVCategory
  is_multimodal?:             boolean
  text_config_field?:         string
  pattern_field?:             string
  pattern_default?:           number
  n_global_formula?:          string
  max_window_layers_semantics?: 'first_N_layers_are_full' | 'version_sensitive_or_ambiguous'
  state_defaults?: {
    d_state: { value: number; confidence: Confidence }
    d_conv:  { value: number; confidence: Confidence }
    expand:  { value: number; confidence: Confidence }
  }
  block_types_fields?: string[]
  fallback?:           string
  verified?:           boolean
  config_fallbacks_by_layers?: Record<string, {
    num_attention_heads: number
    num_key_value_heads: number
    head_dim: number
  }>
  notes?: string
}

export type ModelFamilies = Record<string, ModelFamilyEntry>

// ─── Full engine output (what the UI consumes) ────────────────────────────────

export interface EngineResult {
  // Weight
  weight_bytes:            number
  weight_source:           WeightMemorySource
  weight_confidence:       WeightMemoryConfidence
  weight_warnings:         string[]

  // KV
  kv:                      KVCacheResult
  kv_memory:               KVMemory3Cases

  // Recurrent (null for non-SSM models)
  recurrent:               RecurrentState | null

  // Budget
  budget:                  MemoryBudget

  // GPU count
  gpu_count:               GPUCountResult

  // TP
  resolved_tp:             number
  tp_validation:           TPValidation
  auto_tp_used:            boolean
}
