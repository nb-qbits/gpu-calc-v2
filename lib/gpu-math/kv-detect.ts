import { DetectionResult, ExtractedConfig, KVCategory, ModelFamilies } from './kv-types'

// ─── KV category detection ────────────────────────────────────────────────────
// Rules are checked in this exact order. First match wins.
//
// KV-2  → KV-4  → KV-5b → KV-5a → KV-5c → KV-3 → KV-1
//
// Within KV-3 (sliding window), sub-rules determine 3a vs 3b:
//   Rule 1: layer_types array (explicit, highest priority)
//   Rule 2: global_attn_every_n_layers field
//   Rule 3: model-families.json override (e.g. Gemma 3)
//   Rule 4: max_window_layers with known semantics
//   Rule 5: sliding_window fallback → KV-3a
//   Guard:  use_sliding_window=false → skip all SWA, fall to KV-1

export function detectKVCategory(
  cfg:      ExtractedConfig,
  families: ModelFamilies
): DetectionResult {

  // ── KV-2: MLA low-rank KV (DeepSeek) ──────────────────────────────────────
  if (
    cfg.kv_lora_rank != null && cfg.kv_lora_rank > 0 &&
    cfg.qk_rope_head_dim != null
  ) {
    return {
      category:   'KV-2',
      source:     'config.json',
      confidence: 'high',
      fields:     ['kv_lora_rank', 'qk_rope_head_dim'],
    }
  }

  // ── KV-4: Cross-layer attention sharing (Hunyuan) ─────────────────────────
  if (cfg.use_cla === true && cfg.cla_share_factor != null && cfg.cla_share_factor > 1) {
    return {
      category:   'KV-4',
      source:     'config.json',
      confidence: 'medium',
      fields:     ['use_cla', 'cla_share_factor'],
      warnings:   [
        'CLA KV saving requires backend CLA-aware support. ' +
        'Standard vLLM without the Hunyuan patch will not realise this saving. ' +
        'Both CLA and non-CLA estimates are shown.',
      ],
    }
  }

  // ── SSM signals ───────────────────────────────────────────────────────────
  const ssmSignal =
    cfg.mamba_d_state != null ||
    cfg.mamba_d_conv  != null ||
    cfg.mamba_expand  != null ||
    cfg.ssm_cfg       != null ||
    (cfg as any).ssm_state_size != null  // Nemotron-H uses this field

  const attnSignal =
    cfg.H_q != null && (
      cfg.attn_layer_period   != null ||
      cfg.attn_layer_offset   != null ||
      cfg.attention_layers_idx != null
    )

  // ── KV-5b: SSM + Attention hybrid (Jamba, Nemotron-H) ────────────────────

  // First priority: Config signals (SSM + attention fields both present)
  if (ssmSignal && attnSignal) {
    return {
      category:   'KV-5b',
      source:     'config.json',
      confidence: 'high',
      fields:     ['attn_layer_period', 'mamba_d_state'],
    }
  }

  // Second priority: model-families.json override (only if signals confirm OR model family is trusted)
  const family = families[cfg.model_type]
  if (family?.kv_category === 'KV-5b') {
    // If config has SSM signal but missing attn signal, still trust family override
    // (e.g., Jamba with attn_layer_period missing but model_type='jamba' is reliable)
    if (ssmSignal || attnSignal) {
      return {
        category:   'KV-5b',
        source:     'model-families.json',
        confidence: 'high',
        fields:     ssmSignal && attnSignal
          ? ['model_type', 'mamba_d_state', 'attn_layer_period']
          : ssmSignal
          ? ['model_type', 'mamba_d_state']
          : ['model_type'],
      }
    } else {
      // Model family says KV-5b but NO config signals — very suspicious!
      return {
        category:   'KV-5b',
        source:     'model-families.json',
        confidence: 'low',
        fields:     ['model_type'],
        warnings:   [
          'KV-5b inferred from model family name only — no SSM config fields found. ' +
          'Verify this is a hybrid SSM model.',
        ],
      }
    }
  }

  // ── KV-5a: Pure SSM — zero KV cache (Mamba, Falcon-Mamba, RWKV) ──────────
  const pureSsmTypes = new Set(['mamba', 'mamba2', 'rwkv', 'falcon_mamba'])
  if (ssmSignal && !attnSignal) {
    return {
      category:   'KV-5a',
      source:     'config.json',
      confidence: 'high',
      fields:     ['ssm_cfg'],
    }
  }
  if (pureSsmTypes.has(cfg.model_type) && !attnSignal) {
    return {
      category:   'KV-5a',
      source:     'model-families.json',
      confidence: 'high',
      fields:     ['model_type'],
    }
  }

  // ── KV-5c: Linear recurrence + local attention (RecurrentGemma, Griffin) ──
  const linearRecTypes = new Set(['recurrent_gemma', 'hawk', 'griffin'])
  if (linearRecTypes.has(cfg.model_type) || cfg.block_types != null) {
    return {
      category:   'KV-5c',
      source:     cfg.block_types != null ? 'config.json' : 'model-families.json',
      confidence: cfg.block_types != null ? 'high' : 'medium-high',
      fields:     cfg.block_types != null
        ? ['_block_types', 'attention_window_size']
        : ['model_type'],
    }
  }

  // ── KV-3: Sliding window present ──────────────────────────────────────────
  const hasSliding =
    cfg.sliding_window != null &&
    cfg.sliding_window > 0 &&
    cfg.use_sliding_window !== false  // Guard: explicit false disables all SWA

  if (hasSliding) {
    // Rule 1: layer_types array — explicit, highest priority
    if (cfg.layer_types != null && cfg.layer_types.length > 0) {
      const hasSliding3b = cfg.layer_types.includes('sliding_attention')
      const hasFull      = cfg.layer_types.includes('full_attention')
      if (hasSliding3b && hasFull) {
        return {
          category:   'KV-3b',
          source:     'config.json',
          confidence: 'high',
          fields:     ['layer_types'],
        }
      }
      if (hasSliding3b && !hasFull) {
        return {
          category:   'KV-3a',
          source:     'config.json',
          confidence: 'high',
          fields:     ['layer_types'],
        }
      }
    }

    // Rule 2: global_attn_every_n_layers
    if (cfg.global_attn_every_n_layers != null) {
      return {
        category:   'KV-3b',
        source:     'config.json',
        confidence: 'high',
        fields:     ['global_attn_every_n_layers'],
      }
    }

    // Rule 3: model-families.json override (e.g. Gemma 3 runtime pattern)
    const family = families[cfg.model_type]
    if (family?.kv_category === 'KV-3b') {
      return {
        category:   'KV-3b',
        source:     'model-families.json',
        confidence: 'medium-high',
        fields:     ['model_type'],
      }
    }

    // Rule 4: max_window_layers with known semantics
    if (cfg.max_window_layers != null) {
      const semantics = family?.max_window_layers_semantics
      if (semantics === 'first_N_layers_are_full') {
        return {
          category:   'KV-3b',
          source:     'model-families.json',
          confidence: 'medium-high',
          fields:     ['max_window_layers'],
        }
      }
      if (semantics === 'version_sensitive_or_ambiguous') {
        return {
          category:   'KV-3a',
          source:     'inferred',
          confidence: 'low',
          fields:     ['max_window_layers'],
          warnings:   [
            'max_window_layers semantics are ambiguous for this model type. ' +
            'KV-3a assumed conservatively. Provide layer_types in config for a precise estimate.',
          ],
        }
      }
    }

    // Rule 5: sliding_window fallback — KV-3a, medium confidence
    return {
      category:   'KV-3a',
      source:     'inferred',
      confidence: 'medium',
      fields:     ['sliding_window'],
    }
  }

  // ── KV-1: Standard Dense fallback (GQA / MHA / MQA) ──────────────────────
  return {
    category:   'KV-1',
    source:     'config.json',
    confidence: 'high',
    fields:     ['num_key_value_heads'],
  }
}

// ─── Hybrid layer counts ──────────────────────────────────────────────────────
// For KV-3b: how many global vs sliding layers are there?

export interface HybridLayerCounts {
  n_global:  number
  n_sliding: number
}

export function getHybridLayerCounts(
  cfg:      ExtractedConfig,
  families: ModelFamilies
): HybridLayerCounts {
  const L = cfg.L

  // From explicit layer_types array
  if (cfg.layer_types != null) {
    const n_global  = cfg.layer_types.filter((t) => t === 'full_attention').length
    const n_sliding = cfg.layer_types.filter((t) => t === 'sliding_attention').length
    return { n_global, n_sliding }
  }

  // From global_attn_every_n_layers
  if (cfg.global_attn_every_n_layers != null) {
    const n_global  = Math.floor(L / cfg.global_attn_every_n_layers)
    return { n_global, n_sliding: L - n_global }
  }

  // From model-families.json pattern (e.g. Gemma 3: every 6th layer is global)
  const family = families[cfg.model_type]
  if (family?.pattern_default != null) {
    const pattern   = cfg.sliding_window_pattern ?? family.pattern_default
    const n_global  = Math.floor(L / pattern)
    return { n_global, n_sliding: L - n_global }
  }

  // From max_window_layers with known semantics: first N layers are full attention
  if (
    cfg.max_window_layers != null &&
    family?.max_window_layers_semantics === 'first_N_layers_are_full'
  ) {
    return {
      n_global:  cfg.max_window_layers,
      n_sliding: L - cfg.max_window_layers,
    }
  }

  // Fallback: treat as all-global (conservative)
  return { n_global: L, n_sliding: 0 }
}

// ─── Attention layer counts for hybrid SSM (KV-5b) ────────────────────────────

export interface HybridSSMLayerCounts {
  n_attn: number
  n_ssm:  number
}

export function getHybridSSMLayerCounts(cfg: ExtractedConfig): HybridSSMLayerCounts {
  const L = cfg.L

  if (cfg.attention_layers_idx != null) {
    return { n_attn: cfg.attention_layers_idx.length, n_ssm: L - cfg.attention_layers_idx.length }
  }

  if (cfg.attn_layer_period != null) {
    const offset = cfg.attn_layer_offset ?? 0
    let n_attn = 0
    for (let i = 0; i < L; i++) {
      if (i >= offset && (i - offset) % cfg.attn_layer_period === 0) n_attn++
    }
    return { n_attn, n_ssm: L - n_attn }
  }

  // Fallback: assume all layers have attention (overestimates KV — conservative)
  return { n_attn: L, n_ssm: 0 }
}

// ─── Block type layer counts for KV-5c ────────────────────────────────────────

export interface LinearRecLayerCounts {
  n_attn:      number
  n_recurrent: number
}

export function getLinearRecLayerCounts(cfg: ExtractedConfig): LinearRecLayerCounts {
  const L = cfg.L

  if (cfg.block_types == null) {
    return { n_attn: Math.floor(L / 3), n_recurrent: L - Math.floor(L / 3) }
  }

  const pattern = cfg.block_types
  const fullReps = Math.floor(L / pattern.length)
  const remainder = pattern.slice(0, L % pattern.length)

  const attnInPattern    = pattern.filter((t) => t === 'attention').length
  const attnInRemainder  = remainder.filter((t) => t === 'attention').length
  const n_attn = fullReps * attnInPattern + attnInRemainder

  return { n_attn, n_recurrent: L - n_attn }
}
