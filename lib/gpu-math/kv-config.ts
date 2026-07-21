import {
  DTYPE_BYTES,
  ExtractedConfig,
  KVDtypeResolution,
  KVDtypeSource,
  QuantizationConfig,
} from './kv-types'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getDtypeBytes(dtype: string | undefined): number {
  if (!dtype) return 2
  return DTYPE_BYTES[dtype] ?? 2
}

function num(v: unknown): number | null {
  return typeof v === 'number' ? v : null
}

function bool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null
}

function numArr(v: unknown): number[] | null {
  return Array.isArray(v) && v.every((x) => typeof x === 'number')
    ? (v as number[])
    : null
}

function strArr(v: unknown): string[] | null {
  return Array.isArray(v) && v.every((x) => typeof x === 'string')
    ? (v as string[])
    : null
}

// ─── Quantization extraction ──────────────────────────────────────────────────

function extractQuantConfig(raw: Record<string, unknown>): QuantizationConfig {
  const qcfg =
    (raw.quantization_config as Record<string, unknown> | undefined) ??
    (raw.quant_config as Record<string, unknown> | undefined) ??
    null

  if (!qcfg) {
    const dtype = (raw.torch_dtype as string | undefined) ?? ''
    if (dtype.includes('float8') || dtype.includes('fp8')) {
      return { type: 'fp8', quant_type: dtype }
    }
    return { type: 'none' }
  }

  const rawType =
    (qcfg.quant_type as string | undefined) ??
    (qcfg.quantization_algo as string | undefined) ??
    (qcfg.quant_method as string | undefined) ??
    'unknown'

  const type = (['fp8', 'int8', 'int4', 'gptq', 'awq', 'bnb', 'mxfp4', 'none'] as const).includes(
    rawType as 'fp8' | 'int8' | 'int4' | 'gptq' | 'awq' | 'bnb' | 'mxfp4' | 'none'
  )
    ? (rawType as QuantizationConfig['type'])
    : 'unknown'

  return {
    type,
    bits: num(qcfg.bits) ?? num(qcfg.num_bits) ?? undefined,
    group_size: num(qcfg.group_size) ?? num(qcfg.q_group_size) ?? undefined,
    modules_to_not_convert: strArr(qcfg.modules_to_not_convert) ?? undefined,
    quant_type: (qcfg.quant_type as string | undefined) ?? undefined,
  }
}

// ─── Main extraction ──────────────────────────────────────────────────────────

export function extractConfig(
  rawConfig: Record<string, unknown>,
  families?: import('./kv-types').ModelFamilies
): ExtractedConfig {
  // Multimodal models nest attention arch under text_config
  const cfg = (rawConfig.text_config as Record<string, unknown> | undefined) ?? rawConfig

  const isMultimodal = !!(
    rawConfig.text_config ||
    rawConfig.vision_config ||
    (Array.isArray(rawConfig.architectures) &&
      (rawConfig.architectures as string[]).some(
        (a) => a.includes('Conditional') || a.includes('VL') || a.includes('Vision')
      ))
  )

  // head_dim: always use explicit value if present; compute only as fallback.
  // Using hiddenSize / H_q would give the wrong answer for Gemma-2 (12.5% error).
  let H_q = num(cfg.num_attention_heads) ?? 1
  const headDimExplicit = num(cfg.head_dim)
  const hiddenSize = num(cfg.hidden_size) ?? num(cfg.d_model) ?? 1
  let d = headDimExplicit ?? hiddenSize / H_q
  let d_source: ExtractedConfig['d_source'] = headDimExplicit != null ? 'explicit' : 'computed'

  // MoE — resolve all known field name variants
  const total_routed_experts =
    num(cfg.num_experts) ??
    num(cfg.num_local_experts) ??
    num(cfg.n_routed_experts) ??
    null

  const shared_experts =
    num(cfg.num_shared_experts) ??
    num(cfg.n_shared_experts) ??
    0

  const active_routed_per_tok =
    num(cfg.num_experts_per_tok) ??
    num(cfg.moe_topk) ??
    null

  const total_experts =
    total_routed_experts != null ? total_routed_experts + shared_experts : null

  const active_experts_per_tok =
    active_routed_per_tok != null && shared_experts != null
      ? active_routed_per_tok + shared_experts
      : null

  const active_ratio =
    total_experts != null && active_experts_per_tok != null && total_experts > 0
      ? active_experts_per_tok / total_experts
      : null

  const is_moe = !!(
    (total_routed_experts != null && total_routed_experts > 1) ||
    cfg.moe_intermediate_size != null
  )

  // Block types — try all known field names
  const block_types =
    strArr(cfg._block_types) ??
    strArr(cfg.block_types) ??
    strArr(cfg.layers_block_type) ??
    null

  // torch_dtype: prefer top-level (may differ from text_config in multimodal)
  const torchDtype =
    (rawConfig.torch_dtype as string | undefined) ??
    (cfg.torch_dtype as string | undefined)

  const model_type =
    (cfg.model_type as string | undefined) ??
    (rawConfig.model_type as string | undefined) ??
    'unknown'

  const L = num(cfg.num_hidden_layers) ?? num(cfg.n_layer) ?? 1
  let H_kv = num(cfg.num_key_value_heads) ?? H_q

  // ── Fallback: Fill missing architecture fields from model-families.json ──────
  // For models with incomplete configs (e.g., Gemma 3 4B), use verified values
  // keyed by num_hidden_layers. Only fills null fields — never overwrites.
  if (families?.[ model_type ]?.config_fallbacks_by_layers) {
    const fallback = families[ model_type ].config_fallbacks_by_layers![ String(L) ]
    if (fallback) {
      if (num(cfg.num_attention_heads) == null) {
        H_q = fallback.num_attention_heads
      }
      if (num(cfg.num_key_value_heads) == null) {
        H_kv = fallback.num_key_value_heads
      }
      if (headDimExplicit == null) {
        d = fallback.head_dim
        d_source = 'model-families.json' as const
      }
    }
  }

  return {
    model_type,

    L,
    H_q,
    H_kv,
    d,
    d_source,
    hidden_size: hiddenSize,
    intermediate_size: num(cfg.intermediate_size) ?? 0,
    vocab_size:
      num(cfg.vocab_size) ??
      num(rawConfig.vocab_size) ??
      0,
    B:     getDtypeBytes(torchDtype),
    dtype: torchDtype ?? 'bfloat16',

    // Sliding window
    sliding_window:
      num(cfg.sliding_window) ??
      num(cfg.sliding_window_size) ??
      null,
    sliding_window_pattern: num(cfg.sliding_window_pattern) ?? null,
    use_sliding_window:     bool(cfg.use_sliding_window),
    global_attn_every_n_layers: num(cfg.global_attn_every_n_layers) ?? null,
    layer_types:            strArr(cfg.layer_types),
    max_window_layers:      num(cfg.max_window_layers) ?? null,

    // MLA
    kv_lora_rank:    num(cfg.kv_lora_rank) ?? null,
    qk_rope_head_dim: num(cfg.qk_rope_head_dim) ?? null,

    // CLA
    use_cla:          bool(cfg.use_cla),
    cla_share_factor: num(cfg.cla_share_factor) ?? null,

    // SSM / Mamba — resolve field aliases
    ssm_cfg: (cfg.ssm_cfg as Record<string, unknown> | undefined) ?? null,
    mamba_d_state:
      num(cfg.mamba_d_state) ??
      num(cfg.ssm_state_size) ??   // Nemotron alias
      null,
    mamba_d_conv:
      num(cfg.mamba_d_conv) ??
      num(cfg.conv_kernel) ??       // Nemotron alias
      null,
    mamba_expand: num(cfg.mamba_expand) ?? num(cfg.expand) ?? null,

    // Hybrid attention layers
    attn_layer_period:
      num(cfg.attn_layer_period) ??
      num(cfg.attention_layer_period) ??
      null,
    attn_layer_offset:
      num(cfg.attn_layer_offset) ??
      num(cfg.attention_layer_offset) ??
      null,
    attention_layers_idx: numArr(cfg.attention_layers_idx),

    // Linear recurrence
    block_types,
    attention_window_size: num(cfg.attention_window_size) ?? null,
    lru_width:             num(cfg.lru_width) ?? null,
    conv1d_width:          num(cfg.conv1d_width) ?? null,
    residual_in_fp32:      bool(cfg.residual_in_fp32),

    // MoE
    is_moe,
    total_routed_experts,
    shared_experts,
    active_routed_per_tok,
    total_experts,
    active_experts_per_tok,
    active_ratio,
    moe_intermediate_size: num(cfg.moe_intermediate_size) ?? null,
    expert_layer_period:   num(cfg.expert_layer_period) ?? null,
    expert_layer_offset:   num(cfg.expert_layer_offset) ?? 0,

    // Multimodal
    is_multimodal: isMultimodal,
    mm_tokens_per_image:
      num(rawConfig.mm_tokens_per_image) ??
      num(cfg.mm_tokens_per_image) ??
      null,

    // Quantization
    quantization_config: extractQuantConfig(rawConfig),

    // KV cache dtype field (optional — exists in some configs)
    kv_cache_dtype: (cfg.kv_cache_dtype as string | undefined),
  }
}

// ─── KV dtype resolution — 4-priority chain ───────────────────────────────────
//
// Priority 1: user explicitly set kv_cache_dtype in the advanced panel
// Priority 2: config.json contains a kv_cache_dtype field
// Priority 3: torch_dtype (compute dtype — NOT weight storage dtype)
//
// Critical: for FP8 weight models, torch_dtype is bfloat16 (compute dtype).
// KV cache runs in bfloat16 by default even when weights are fp8.
// Only explicit user opt-in via --kv-cache-dtype fp8 changes this.

export function resolveKVCacheDtype(
  userOverride: string | undefined,  // from UI advanced panel: 'auto' | 'fp8' | undefined
  cfg:          ExtractedConfig,
  weightDtype:  string               // storage dtype from safetensors / quantization_config
): KVDtypeResolution {
  let dtype:  string
  let source: KVDtypeSource

  if (userOverride != null && userOverride !== '') {
    dtype  = userOverride === 'auto' ? deriveComputeDtype(cfg.dtype) : userOverride
    source = 'user_provided'
  } else if (cfg.kv_cache_dtype != null) {
    dtype  = cfg.kv_cache_dtype
    source = 'config_field'
  } else {
    dtype  = deriveComputeDtype(cfg.dtype)
    source = 'torch_dtype_fallback'
  }

  const bytes = DTYPE_BYTES[dtype] ?? 2
  const wgtBytes = DTYPE_BYTES[weightDtype] ?? 2

  let warning: string | undefined
  if (dtype !== weightDtype) {
    if (bytes > wgtBytes) {
      warning =
        `KV cache dtype (${dtype}, ${bytes}B/elem) uses more memory than ` +
        `weight storage dtype (${weightDtype}, ${wgtBytes}B/elem). ` +
        `Set kv cache dtype to ${weightDtype} in the advanced panel to halve KV memory.`
    } else {
      warning =
        `KV cache dtype (${dtype}, ${bytes}B/elem) differs from ` +
        `weight storage dtype (${weightDtype}, ${wgtBytes}B/elem).`
    }
  }

  return { dtype, bytes, source, weight_dtype: weightDtype, warning }
}

// FP8 weights are dequantized to bfloat16/float16 for compute.
// torch_dtype in config.json = compute dtype, not storage dtype.
function deriveComputeDtype(torchDtype: string): string {
  if (torchDtype.includes('float16') || torchDtype === 'float16') return 'float16'
  return 'bfloat16'
}

// ─── Weight memory fallback estimate ─────────────────────────────────────────
// Re-exported from inference-config/weight-memory for backward compatibility.
// That module has the quantization-aware implementation.

export { estimateWeightMemoryBytes, getStorageBytesPerParam } from './inference-config/weight-memory'
