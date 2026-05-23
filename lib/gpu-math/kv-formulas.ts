import {
  DeploymentParams,
  DetectionResult,
  ExtractedConfig,
  KVCacheResult,
  KVMemory3Cases,
  KVTPMode,
  ModelFamilies,
  RecurrentState,
  TPValidation,
} from './kv-types'
import { resolveKVCacheDtype } from './kv-config'
import {
  getHybridLayerCounts,
  getHybridSSMLayerCounts,
  getLinearRecLayerCounts,
} from './kv-detect'

// ─── TP validation ────────────────────────────────────────────────────────────

export function validateTP(H_q: number, H_kv: number, TP: number): TPValidation {
  if (TP < 1) {
    return {
      is_valid: false, hard_reject: true, warn_kv_split: false,
      reject_reason: `TP must be >= 1. Got TP=${TP}.`,
      kv_heads_per_gpu: H_kv, kv_replication: false,
      kv_tp_mode: 'sharded', tp_inflection: H_kv, tp_gives_kv_benefit: false,
    }
  }

  if (H_q % TP !== 0) {
    return {
      is_valid: false, hard_reject: true, warn_kv_split: false,
      reject_reason: `TP=${TP} invalid: H_q=${H_q} is not divisible by TP.`,
      kv_heads_per_gpu: H_kv, kv_replication: false,
      kv_tp_mode: 'sharded', tp_inflection: H_kv, tp_gives_kv_benefit: false,
    }
  }

  const kvClean       = (H_kv % TP === 0) || (TP % H_kv === 0)
  const kv_replication = TP > H_kv
  const kv_heads_per_gpu = Math.max(1, Math.floor(H_kv / TP))
  const tp_inflection  = H_kv
  const kv_tp_mode: KVTPMode = kv_replication ? 'replicated' : 'sharded'

  let warning: string | undefined
  if (kv_replication) {
    warning =
      `TP=${TP} exceeds H_kv=${H_kv}. KV heads replicated — no KV memory benefit ` +
      `beyond TP=${H_kv}. Additional TP only reduces weight memory.`
  } else if (!kvClean) {
    warning = `KV head split unclean for TP=${TP}, H_kv=${H_kv}. Verify framework supports this.`
  }

  return {
    is_valid: kvClean, hard_reject: false, warn_kv_split: !kvClean,
    kv_heads_per_gpu, kv_replication,
    kv_tp_mode, tp_inflection,
    tp_gives_kv_benefit: !kv_replication,
    warning,
  }
}

// ─── Block rounding helpers ───────────────────────────────────────────────────
// vLLM allocates KV in pages of block_size tokens. Sequences always consume
// ceil(seq_len / block_size) complete blocks — fractional blocks are not used.

export function blockAlignedSeqLen(seqLen: number, blockSize: number): number {
  return Math.ceil(seqLen / blockSize) * blockSize
}

// ─── KV bytes per token — per category ───────────────────────────────────────

export function computeKVCacheResult(
  cfg:        ExtractedConfig,
  detection:  DetectionResult,
  deploy:     DeploymentParams,
  families:   ModelFamilies,
  weightDtype: string
): KVCacheResult {
  const kvDtype = resolveKVCacheDtype(deploy.kv_cache_dtype, cfg, weightDtype)
  const B       = kvDtype.bytes
  const TP      = deploy.tp
  const L       = cfg.L
  const warnings: string[] = [...(detection.warnings ?? [])]
  if (kvDtype.warning) warnings.push(kvDtype.warning)

  const tpResult = validateTP(cfg.H_q, cfg.H_kv, TP)
  if (tpResult.warning) warnings.push(tpResult.warning)
  const kv_heads_per_gpu = tpResult.kv_heads_per_gpu

  switch (detection.category) {

    // ── KV-2: MLA ─────────────────────────────────────────────────────────
    case 'KV-2': {
      const r    = cfg.kv_lora_rank!
      const rope = cfg.qk_rope_head_dim!
      const kv_bytes_per_token = (r + rope) * B * L
      return {
        kv_category:         'KV-2',
        kv_category_label:   'MLA low-rank KV',
        formula:             `(${r} + ${rope}) × ${B}B × ${L} layers`,
        kv_bytes_per_token,
        kv_heads_per_gpu:    1,
        kv_tp_mode:          'mla_replicated',
        tp_inflection:       Infinity,
        tp_gives_kv_benefit: false,
        is_bounded:          false,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-4: CLA ─────────────────────────────────────────────────────────
    case 'KV-4': {
      const shareFactor     = cfg.cla_share_factor!
      const effectiveLayers = Math.ceil(L / shareFactor)
      const kv_bytes_per_token = 2 * kv_heads_per_gpu * cfg.d * B * effectiveLayers
      const kv_bytes_per_token_fallback = 2 * kv_heads_per_gpu * cfg.d * B * L
      return {
        kv_category:              'KV-4',
        kv_category_label:        'Cross-layer attention sharing',
        formula:                  `2 × ${kv_heads_per_gpu} heads × ${cfg.d} head_dim × ${B}B × ceil(${L}/${shareFactor})`,
        kv_bytes_per_token,
        kv_bytes_per_token_fallback,
        kv_heads_per_gpu,
        kv_tp_mode:               tpResult.kv_tp_mode,
        tp_inflection:            tpResult.tp_inflection,
        tp_gives_kv_benefit:      tpResult.tp_gives_kv_benefit,
        is_bounded:               false,
        kv_dtype:                 kvDtype,
        is_moe:                   cfg.is_moe,
        warnings,
      }
    }

    // ── KV-5a: Pure SSM ───────────────────────────────────────────────────
    case 'KV-5a': {
      return {
        kv_category:         'KV-5a',
        kv_category_label:   'Pure SSM — no KV cache',
        formula:             '0 bytes/token (pure recurrent — no KV cache)',
        kv_bytes_per_token:  0,
        kv_heads_per_gpu:    0,
        kv_tp_mode:          'zero',
        tp_inflection:       0,
        tp_gives_kv_benefit: false,
        is_bounded:          false,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-5b: SSM + Attention hybrid ─────────────────────────────────────
    case 'KV-5b': {
      const { n_attn } = getHybridSSMLayerCounts(cfg)
      const kv_bytes_per_token = 2 * kv_heads_per_gpu * cfg.d * B * n_attn
      return {
        kv_category:         'KV-5b',
        kv_category_label:   'SSM + attention hybrid',
        formula:             `2 × ${kv_heads_per_gpu} heads × ${cfg.d} head_dim × ${B}B × ${n_attn} attn layers`,
        kv_bytes_per_token,
        kv_heads_per_gpu,
        kv_tp_mode:          tpResult.kv_tp_mode,
        tp_inflection:       tpResult.tp_inflection,
        tp_gives_kv_benefit: tpResult.tp_gives_kv_benefit,
        is_bounded:          false,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-5c: Linear recurrence + local attention ────────────────────────
    case 'KV-5c': {
      const { n_attn }  = getLinearRecLayerCounts(cfg)
      const W           = cfg.attention_window_size
      const kv_bytes_per_token = 2 * kv_heads_per_gpu * cfg.d * B * n_attn
      return {
        kv_category:         'KV-5c',
        kv_category_label:   'Linear recurrence + local attention',
        formula:             `2 × ${kv_heads_per_gpu} heads × ${cfg.d} head_dim × ${B}B × ${n_attn} local-attn layers`,
        kv_bytes_per_token,
        kv_heads_per_gpu,
        kv_tp_mode:          tpResult.kv_tp_mode,
        tp_inflection:       tpResult.tp_inflection,
        tp_gives_kv_benefit: tpResult.tp_gives_kv_benefit,
        is_bounded:          W != null,
        bound_tokens:        W ?? undefined,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-3b: Hybrid sliding + global ────────────────────────────────────
    case 'KV-3b': {
      const kv_bytes_per_layer = 2 * kv_heads_per_gpu * cfg.d * B
      return {
        kv_category:         'KV-3b',
        kv_category_label:   'Hybrid sliding + global',
        formula:             `[n_global × seq + n_sliding × min(seq, W)] × ${kv_bytes_per_layer}B/layer`,
        kv_bytes_per_token:  kv_bytes_per_layer,  // per-layer bytes; memory calc uses n_global/n_sliding
        kv_heads_per_gpu,
        kv_tp_mode:          tpResult.kv_tp_mode,
        tp_inflection:       tpResult.tp_inflection,
        tp_gives_kv_benefit: tpResult.tp_gives_kv_benefit,
        is_bounded:          true,
        bound_tokens:        cfg.sliding_window ?? undefined,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-3a: Full sliding window ────────────────────────────────────────
    case 'KV-3a': {
      const kv_bytes_per_token = 2 * kv_heads_per_gpu * cfg.d * B * L
      return {
        kv_category:         'KV-3a',
        kv_category_label:   'Full sliding window',
        formula:             `2 × ${kv_heads_per_gpu} heads × ${cfg.d} head_dim × ${B}B × ${L} layers [bounded at W=${cfg.sliding_window}]`,
        kv_bytes_per_token,
        kv_heads_per_gpu,
        kv_tp_mode:          tpResult.kv_tp_mode,
        tp_inflection:       tpResult.tp_inflection,
        tp_gives_kv_benefit: tpResult.tp_gives_kv_benefit,
        is_bounded:          true,
        bound_tokens:        cfg.sliding_window ?? undefined,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }

    // ── KV-1: Standard dense (GQA / MHA / MQA) — default ─────────────────
    default: {
      const kv_bytes_per_token = 2 * kv_heads_per_gpu * cfg.d * B * L
      return {
        kv_category:         'KV-1',
        kv_category_label:   'Standard dense (GQA / MHA / MQA)',
        formula:             `2 × ${kv_heads_per_gpu} heads × ${cfg.d} head_dim × ${B}B × ${L} layers`,
        kv_bytes_per_token,
        kv_heads_per_gpu,
        kv_tp_mode:          tpResult.kv_tp_mode,
        tp_inflection:       tpResult.tp_inflection,
        tp_gives_kv_benefit: tpResult.tp_gives_kv_benefit,
        is_bounded:          false,
        kv_dtype:            kvDtype,
        is_moe:              cfg.is_moe,
        warnings,
      }
    }
  }
}

// ─── Effective sequence length (multimodal adds image tokens) ─────────────────

function effectiveSeqLen(seqLen: number, cfg: ExtractedConfig, deploy: DeploymentParams): number {
  if (!cfg.is_multimodal) return seqLen
  const imgTokens = 0 * (cfg.mm_tokens_per_image ?? 0) // num_images not in deploy for now
  return seqLen + imgTokens
}

// ─── KV memory — 3 scenarios ──────────────────────────────────────────────────

export function computeKVMemory(
  kv:       KVCacheResult,
  cfg:      ExtractedConfig,
  deploy:   DeploymentParams,
  families: ModelFamilies
): KVMemory3Cases {
  const { ISL, OSL, max_model_len, max_num_seqs, block_size } = deploy

  if (kv.kv_category === 'KV-5a') {
    return { optimistic: 0, expected: 0, conservative: 0 }
  }

  const eff  = (s: number) => effectiveSeqLen(s, cfg, deploy)
  const ba   = (s: number) => blockAlignedSeqLen(eff(s), block_size)

  // KV-3b: per-layer bytes × mixed sequence formula
  if (kv.kv_category === 'KV-3b') {
    const { n_global, n_sliding } = getHybridLayerCounts(cfg, families)
    const W   = cfg.sliding_window!
    const bpl = kv.kv_bytes_per_token   // bytes per layer (not per token × L)
    const compute = (seq: number): number => {
      const seqAligned = ba(seq)
      const wAligned   = blockAlignedSeqLen(Math.min(eff(seq), W), block_size)
      return (n_global * seqAligned + n_sliding * wAligned) * bpl * max_num_seqs
    }
    return {
      optimistic:   compute(ISL),
      expected:     compute(ISL + OSL),
      conservative: compute(max_model_len),
    }
  }

  // KV-5c: bounded by attention_window_size
  if (kv.kv_category === 'KV-5c' && kv.is_bounded && kv.bound_tokens) {
    const W = kv.bound_tokens
    const compute = (seq: number): number =>
      kv.kv_bytes_per_token *
      blockAlignedSeqLen(Math.min(eff(seq), W), block_size) *
      max_num_seqs
    return {
      optimistic:   compute(ISL),
      expected:     compute(ISL + OSL),
      conservative: compute(max_model_len),
    }
  }

  // KV-3a: bounded by sliding window
  if (kv.is_bounded && kv.bound_tokens) {
    const W = kv.bound_tokens
    const compute = (seq: number): number =>
      kv.kv_bytes_per_token *
      blockAlignedSeqLen(Math.min(eff(seq), W), block_size) *
      max_num_seqs
    return {
      optimistic:   compute(ISL),
      expected:     compute(ISL + OSL),
      conservative: compute(max_model_len),
    }
  }

  // KV-1, KV-2, KV-4, KV-5b: unbounded
  const compute = (seq: number): number =>
    kv.kv_bytes_per_token * ba(seq) * max_num_seqs
  return {
    optimistic:   compute(ISL),
    expected:     compute(ISL + OSL),
    conservative: compute(max_model_len),
  }
}

// ─── Recurrent state (KV-5a / 5b / 5c) ───────────────────────────────────────

export function computeRecurrentState(
  cfg:     ExtractedConfig,
  deploy:  DeploymentParams,
  families: ModelFamilies
): RecurrentState | null {
  const category = cfg.ssm_cfg != null || cfg.mamba_d_state != null || cfg.mamba_d_conv != null
    ? 'ssm' : cfg.block_types != null ? 'linear_rec' : null

  if (!category) return null

  const B_state = cfg.residual_in_fp32 === true ? 4 : cfg.B
  const maxSeqs = deploy.max_num_seqs

  if (category === 'ssm') {
    const family = families[cfg.model_type]
    const d_state  = cfg.mamba_d_state  ?? family?.state_defaults?.d_state.value ?? 16
    const d_conv   = cfg.mamba_d_conv   ?? family?.state_defaults?.d_conv.value  ?? 4
    const expand   = cfg.mamba_expand   ?? family?.state_defaults?.expand.value  ?? 2
    const d_inner  = cfg.hidden_size * expand
    const conv_len = d_conv - 1
    const { n_attn, n_ssm } = getHybridSSMLayerCounts(cfg)
    const n_recurrent = cfg.attn_layer_period != null ? n_ssm : cfg.L

    const state_bytes_per_seq = n_recurrent * d_inner * d_state * B_state
    const conv_bytes_per_seq  = n_recurrent * d_inner * conv_len * B_state
    const total_bytes_per_seq = state_bytes_per_seq + conv_bytes_per_seq

    return {
      n_recurrent_layers:        n_recurrent,
      state_bytes_per_seq,
      conv_bytes_per_seq,
      total_state_bytes_per_seq: total_bytes_per_seq,
      total_state_memory_bytes:  total_bytes_per_seq * maxSeqs,
    }
  }

  // Linear recurrence (KV-5c)
  if (category === 'linear_rec') {
    const { n_recurrent } = getLinearRecLayerCounts(cfg)
    const lru_width = cfg.lru_width ?? cfg.hidden_size
    const conv_len  = (cfg.conv1d_width ?? 4) - 1

    const state_bytes_per_seq = n_recurrent * lru_width * B_state
    const conv_bytes_per_seq  = n_recurrent * lru_width * conv_len * B_state
    const total_bytes_per_seq = state_bytes_per_seq + conv_bytes_per_seq

    return {
      n_recurrent_layers:        n_recurrent,
      state_bytes_per_seq,
      conv_bytes_per_seq,
      total_state_bytes_per_seq: total_bytes_per_seq,
      total_state_memory_bytes:  total_bytes_per_seq * maxSeqs,
    }
  }

  return null
}
