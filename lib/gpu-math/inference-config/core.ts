// Core Orchestrator
// Main entry point that ties all modules together

import type { InferenceRequest, InferenceConfigResult, MemoryAnalysis } from './types'
import { validateOrThrow } from './validation'
import { computeTensorParallelSize, computeReplicas, computeUsableHBM } from './tensor-parallel'
import { computeVLLMConfig } from './vllm-defaults'
import { classifyBottleneck } from './bottleneck'
import { determineParallelismStrategy } from './parallelism'
import { computeLLMDConfig } from './llmd'

// Import from existing KV cache engine
import { GPU_CATALOG } from '../gpus'
import { MODEL_CATALOG } from '../models'
import { detectKVCategory } from '../kv-detect'
import { KV_CATEGORY_LABELS } from '../kv-types'
import { extractConfig, resolveKVCacheDtype } from '../kv-config'
import { computeKVCacheResult } from '../kv-formulas'
import type { ModelFamilies, ExtractedConfig } from '../kv-types'

/**
 * Main entry point for inference configuration engine.
 *
 * Takes user inputs, validates them, and returns complete vLLM/llm-d configuration.
 *
 * Architecture:
 * 1. Validate inputs (single validation point)
 * 2. Look up model and GPU specs
 * 3. Compute tensor parallel sizing
 * 4. Determine replicas
 * 5. Compute memory budgets
 * 6. Generate vLLM configuration
 * 7. Classify bottleneck
 * 8. Determine parallelism strategy
 * 9. Optional: llm-d configuration
 * 10. Add diagnostics
 *
 * @param req - Inference request
 * @returns Complete inference configuration with vLLM/llm-d settings
 * @throws Error if validation fails or model/GPU not found
 */
export function computeInferenceConfig(
  req: InferenceRequest
): InferenceConfigResult {
  // ═══ STEP 1: VALIDATION ═══
  const validation = validateOrThrow(req)

  // ═══ STEP 2: LOOKUP MODEL & GPU SPECS ═══

  // Find GPU in catalog
  const gpu = GPU_CATALOG.find(g =>
    g.id === req.gpu_type ||
    g.name === req.gpu_type
  )

  if (!gpu) {
    throw new Error(
      `GPU "${req.gpu_type}" not found in catalog. ` +
      `Available GPUs: ${GPU_CATALOG.map(g => g.id).join(', ')}`
    )
  }

  // Find model in catalog (if it exists)
  const model = MODEL_CATALOG.find(m =>
    m.id === req.model_name ||
    m.hfId === req.model_name ||
    m.name === req.model_name
  )

  // If model not in catalog, we need HF config to proceed
  if (!model && !req.hf_config) {
    throw new Error(
      `Model "${req.model_name}" not found in catalog. ` +
      `Please fetch the model config from HuggingFace first, or select a model from the dropdown.`
    )
  }

  // Use HF config if provided (takes precedence over catalog)
  const useHFConfig = req.hf_config != null

  // Detect KV cache category
  let kv_category = 'KV-1'  // Default to standard dense
  let kv_category_label = KV_CATEGORY_LABELS['KV-1']

  try {
    console.log('🔍 KV category detection:', {
      useHFConfig,
      hasHfConfig: !!req.hf_config,
      hfConfigKeys: req.hf_config ? Object.keys(req.hf_config) : [],
      sliding_window: req.hf_config?.sliding_window,
      kv_lora_rank: req.hf_config?.kv_lora_rank,
      use_cla: req.hf_config?.use_cla,
      state_size: req.hf_config?.state_size,
      model_type: req.hf_config?.model_type,
      hasModel: !!model,
      modelTags: model?.tags
    })

    // Check HF config first for detailed detection
    let detectedFromHF = false
    if (useHFConfig && req.hf_config) {
      // Check for MLA (DeepSeek)
      if (req.hf_config.kv_lora_rank != null && req.hf_config.kv_lora_rank > 0) {
        kv_category = 'KV-2'
        kv_category_label = KV_CATEGORY_LABELS['KV-2']
        detectedFromHF = true
        console.log('✅ Detected KV-2 (MLA) from kv_lora_rank:', req.hf_config.kv_lora_rank)
      }
      // Check for CLA (Hunyuan)
      else if (req.hf_config.use_cla === true) {
        kv_category = 'KV-4'
        kv_category_label = KV_CATEGORY_LABELS['KV-4']
        detectedFromHF = true
        console.log('✅ Detected KV-4 (CLA) from use_cla')
      }
      // Check for SSM (Mamba, Jamba)
      else if (req.hf_config.state_size != null || req.hf_config.model_type?.toLowerCase().includes('mamba')) {
        kv_category = 'KV-5b'
        kv_category_label = KV_CATEGORY_LABELS['KV-5b']
        detectedFromHF = true
        console.log('✅ Detected KV-5b (SSM) from state_size or model_type')
      }
      // Check for sliding window (Mistral)
      else if (req.hf_config.sliding_window != null && req.hf_config.sliding_window > 0) {
        kv_category = 'KV-3a'
        kv_category_label = KV_CATEGORY_LABELS['KV-3a']
        detectedFromHF = true
        console.log('✅ Detected KV-3a (sliding window) from sliding_window:', req.hf_config.sliding_window)
      }
    }

    // Fallback to tag-based detection if HF config didn't match OR if model is in catalog
    if (!detectedFromHF && model) {
      if (model.tags?.includes('SSM')) {
        kv_category = 'KV-5b'
        kv_category_label = KV_CATEGORY_LABELS['KV-5b']
        console.log('✅ Detected KV-5b from model tags')
      } else if (model.tags?.includes('MLA')) {
        kv_category = 'KV-2'
        kv_category_label = KV_CATEGORY_LABELS['KV-2']
        console.log('✅ Detected KV-2 from model tags')
      } else if (model.tags?.includes('SlidingWindow')) {
        kv_category = 'KV-3a'
        kv_category_label = KV_CATEGORY_LABELS['KV-3a']
        console.log('✅ Detected KV-3a from model tags (SlidingWindow)')
      }
    }

    // Last resort: detect by model name pattern for common architectures
    if (!detectedFromHF && !model && req.model_name) {
      const modelName = req.model_name.toLowerCase()
      // Mistral models use sliding window (even if config.json has sliding_window: null)
      if (modelName.includes('mistral') && !modelName.includes('mixtral')) {
        kv_category = 'KV-3a'
        kv_category_label = KV_CATEGORY_LABELS['KV-3a']
        console.log('✅ Detected KV-3a from model name pattern (Mistral)')
      }
      // DeepSeek models use MLA
      else if (modelName.includes('deepseek')) {
        kv_category = 'KV-2'
        kv_category_label = KV_CATEGORY_LABELS['KV-2']
        console.log('✅ Detected KV-2 from model name pattern (DeepSeek)')
      }
    }

    console.log('📊 Final KV category:', kv_category, '-', kv_category_label)
  } catch (error) {
    console.warn('KV category detection failed, using default KV-1:', error)
  }

  // ═══ STEP 3: COMPUTE WEIGHT MEMORY ═══

  let params_billions: number

  if (useHFConfig && req.hf_config) {
    // Use existing config extraction (handles MoE, MLA, sliding window, all field variants)
    const cfg = extractConfig(req.hf_config as Record<string, unknown>)

    console.log(`📊 Extracted config: ${cfg.model_type}, layers=${cfg.L}, hidden=${cfg.hidden_size}, is_moe=${cfg.is_moe}`)
    if (cfg.is_moe) {
      console.log(`   MoE: ${cfg.total_routed_experts} routed + ${cfg.shared_experts} shared, ${cfg.active_routed_per_tok} active/tok`)
    }

    // Embedding params
    const embedding_params = cfg.vocab_size * cfg.hidden_size

    // Attention params
    let attention_params: number
    if (cfg.kv_lora_rank) {
      // MLA (Multi-head Latent Attention) - DeepSeek, GLM
      // Note: This is a simplified estimate. Actual MLA has more parameters.
      // For accurate sizing, we should fetch model.safetensors.index.json
      attention_params = 4 * cfg.hidden_size * cfg.hidden_size
      console.log(`   MLA detected (kv_lora=${cfg.kv_lora_rank}) - using standard attention estimate`)
    } else {
      // Standard attention
      attention_params = 4 * cfg.hidden_size * cfg.hidden_size
    }

    // FFN params
    let ffn_params: number
    if (cfg.is_moe && cfg.total_routed_experts) {
      // MoE: ALL expert weights (vLLM loads all into GPU)
      const expert_size = cfg.moe_intermediate_size || cfg.intermediate_size
      const shared_ffn = cfg.shared_experts * 2 * cfg.hidden_size * expert_size
      const routed_ffn = cfg.total_routed_experts * 2 * cfg.hidden_size * expert_size
      ffn_params = shared_ffn + routed_ffn
      console.log(`   Shared FFN: ${(shared_ffn / 1e9).toFixed(2)}B, Routed: ${(routed_ffn / 1e9).toFixed(2)}B (per layer)`)
    } else {
      // Dense FFN
      ffn_params = 2 * cfg.hidden_size * cfg.intermediate_size
    }

    const layer_params = attention_params + ffn_params
    const total_params = embedding_params + (cfg.L * layer_params) + embedding_params
    params_billions = total_params / 1e9

    console.log(`📊 ${params_billions.toFixed(1)}B params (${cfg.is_moe ? 'MoE' : 'dense'})`)
  } else if (model) {
    // Extract parameter count from paramLabel (e.g., "70B" → 70)
    const paramMatch = model.paramLabel.match(/(\d+)B/)
    if (!paramMatch) {
      throw new Error(`Cannot parse parameter count from "${model.paramLabel}"`)
    }
    params_billions = parseInt(paramMatch[1], 10)
  } else {
    throw new Error('Cannot determine parameter count - no model or config available')
  }

  // Compute weight memory based on precision
  const bytes_per_param: Record<string, number> = {
    FP16: 2, FP8: 1, INT8: 1, INT4: 0.5
  }
  const weight_gb = (params_billions * 1e9 * bytes_per_param[req.precision]) / 1e9

  // ═══ STEP 4: COMPUTE TP SIZE & REPLICAS ═══

  // First pass: estimate weight_gb_per_gpu to determine gpu_memory_utilization
  const estimated_tp_for_utilization = Math.max(1, Math.ceil(weight_gb / (gpu.vramGb * 0.90)))
  const estimated_weight_per_gpu = weight_gb / estimated_tp_for_utilization

  const { usable_gb, utilization } = computeUsableHBM(gpu.vramGb, estimated_weight_per_gpu)
  const tp_size = computeTensorParallelSize(weight_gb, usable_gb)
  const weight_gb_per_gpu = weight_gb / tp_size

  // Determine GPU count
  let gpu_count = req.gpu_count || tp_size

  // If user provided gpu_count < tp_size, warn and recommend minimum
  if (req.gpu_count && req.gpu_count < tp_size) {
    validation.warnings.push(
      `Requested ${req.gpu_count} GPUs but model requires minimum ${tp_size} GPUs (TP size). ` +
      `Recommending ${tp_size} GPUs.`
    )
    gpu_count = tp_size
  }

  let replicas = computeReplicas(gpu_count, tp_size)

  // Warn if only 1 replica (no fault tolerance)
  if (replicas === 1) {
    validation.warnings.push(
      'Only 1 replica - no fault tolerance. Consider adding more GPUs for redundancy.'
    )
  }

  // ═══ STEP 5: COMPUTE KV CACHE BUDGET ═══

  // Use existing KV cache engine for accurate category-specific calculation
  // For catalog models without HF config, we need to build a minimal config from model metadata
  let cfg: ExtractedConfig

  if (useHFConfig && req.hf_config) {
    cfg = extractConfig(req.hf_config as Record<string, unknown>)
  } else if (model) {
    // Model in catalog but no HF config - create minimal config
    // This is a fallback - ideally we'd fetch the real HF config
    const paramMatch = model.paramLabel.match(/(\d+)B/)
    const estimatedParams = paramMatch ? parseInt(paramMatch[1], 10) : 70

    // Rough estimates based on common architectures
    cfg = {
      model_type: model.id,
      L: estimatedParams < 20 ? 32 : estimatedParams < 100 ? 80 : 120,
      H_q: estimatedParams < 20 ? 32 : 64,
      H_kv: estimatedParams < 20 ? 32 : 8,  // Assume GQA for larger models
      d: 128,
      d_source: 'computed' as const,
      hidden_size: estimatedParams < 20 ? 4096 : estimatedParams < 100 ? 8192 : 16384,
      intermediate_size: estimatedParams < 20 ? 11008 : estimatedParams < 100 ? 28672 : 49152,
      vocab_size: 128000,
      B: 2,
      dtype: 'bfloat16',

      sliding_window: null,
      sliding_window_pattern: null,
      use_sliding_window: null,
      global_attn_every_n_layers: null,
      layer_types: null,
      max_window_layers: null,

      kv_lora_rank: null,
      qk_rope_head_dim: null,

      use_cla: null,
      cla_share_factor: null,

      ssm_cfg: null,
      mamba_d_state: null,
      mamba_d_conv: null,
      mamba_expand: null,

      attn_layer_period: null,
      attn_layer_offset: null,
      attention_layers_idx: null,

      block_types: null,
      attention_window_size: null,
      lru_width: null,
      conv1d_width: null,
      residual_in_fp32: null,

      is_moe: false,
      total_routed_experts: null,
      shared_experts: 0,
      active_routed_per_tok: null,
      total_experts: null,
      active_experts_per_tok: null,
      active_ratio: null,
      moe_intermediate_size: null,
      expert_layer_period: null,
      expert_layer_offset: 0,

      is_multimodal: false,
      mm_tokens_per_image: null,
      quantization_config: { type: 'none' }
    }
    console.log('⚠️ Using fallback config for catalog model (no HF config available)')
  } else {
    throw new Error('Cannot extract config - no HF config or catalog model available')
  }

  // Detect KV category using the proper detection engine
  // Model families are loaded from model-families.json in production
  // For now, pass empty object - detection will work from config fields
  const families: ModelFamilies = {}

  const detection = detectKVCategory(cfg, families)
  console.log('🔍 KV category detection result:', detection)

  // Use detected category (overrides the simple detection above)
  kv_category = detection.category
  kv_category_label = KV_CATEGORY_LABELS[detection.category]

  // Compute accurate KV cache per token using the real formula
  const kvResult = computeKVCacheResult(
    cfg,
    detection,
    {
      tp: tp_size,
      max_model_len: req.isl + req.osl,
      max_num_seqs: 256, // Will be overridden by vLLM config later
      gpu_memory_utilization: utilization,
      ISL: req.isl,
      OSL: req.osl,
      block_size: 128,
      kv_cache_dtype: req.kv_cache_precision || req.precision,
      mamba_ssm_cache_dtype: 'bfloat16'
    },
    families,
    req.precision
  )

  console.log('📊 KV cache result:', kvResult)

  // Calculate KV cache memory for the workload
  const total_context = req.isl + req.osl
  const kv_bytes_per_token = kvResult.kv_bytes_per_token
  const kv_gb_per_sequence = (kv_bytes_per_token * total_context) / 1e9

  console.log(`   KV: ${kv_bytes_per_token.toFixed(0)} bytes/token × ${total_context} tokens = ${kv_gb_per_sequence.toFixed(3)} GB/seq`)

  // Available memory per replica for KV cache
  const kv_budget_per_replica_gb = usable_gb - weight_gb_per_gpu

  // Calculate actual KV cache used by concurrent users
  const kv_cache_used_gb = (req.concurrent_users * kv_gb_per_sequence)

  // Check if we need more replicas to fit the KV cache workload
  const kv_cache_per_replica = kv_cache_used_gb / replicas
  if (kv_cache_per_replica > kv_budget_per_replica_gb) {
    // Need more replicas to distribute KV cache
    const required_replicas = Math.ceil(kv_cache_used_gb / kv_budget_per_replica_gb)
    const old_replicas = replicas
    replicas = required_replicas
    gpu_count = tp_size * replicas

    validation.warnings.push(
      `KV cache (${kv_cache_used_gb.toFixed(1)} GB) exceeds available memory. ` +
      `Increased replicas from ${old_replicas} to ${replicas} (${gpu_count} total GPUs).`
    )
  }

  // Calculate total KV budget and max sequences with final replica count
  const total_kv_budget_gb = kv_budget_per_replica_gb * replicas
  const max_sequences_from_memory = Math.floor(kv_budget_per_replica_gb / kv_gb_per_sequence)

  // ═══ STEP 6: BUILD MEMORY ANALYSIS ═══

  const memory_analysis: MemoryAnalysis = {
    weight_gb,
    weight_gb_per_gpu,
    usable_hbm_per_gpu: usable_gb,
    tp_size,
    replicas,
    kv_cache_budget_gb: total_kv_budget_gb,  // Available space
    kv_cache_used_gb,  // Actually consumed by workload
    max_sequences_from_memory,
    kv_category,        // e.g., "KV-1", "KV-2", "KV-3a"
    kv_category_label   // e.g., "Standard Dense (GQA / MHA / MQA)"
  }

  // ═══ STEP 7: COMPUTE VLLM CONFIG ═══

  const vllm_config = computeVLLMConfig(req, {
    tp_size,
    replicas,
    max_sequences_from_memory,
    gpu_memory_utilization: utilization
  })

  // ═══ STEP 8: CLASSIFY BOTTLENECK ═══

  const bottleneck_analysis = classifyBottleneck(req)

  // ═══ STEP 9: DETERMINE PARALLELISM STRATEGY ═══

  const parallelism_strategy = determineParallelismStrategy(
    tp_size,
    8,  // Default 8 GPUs per node (standard DGX configuration)
    req.network_topology || 'nvlink'
  )

  // ═══ STEP 10: OPTIONAL - LLMD CONFIG ═══

  let llmd_config = undefined
  if (req.enable_llmd) {
    // For llm-d config, we need model architecture details
    // This is a placeholder - real implementation needs actual layer/head counts
    const estimated_layers = params_billions < 20 ? 32 : params_billions < 100 ? 80 : 120
    const estimated_kv_heads = 8  // GQA typical
    const estimated_head_dim = 128

    llmd_config = computeLLMDConfig(
      req,
      tp_size,
      estimated_layers,
      estimated_kv_heads,
      estimated_head_dim
    )
  }

  // ═══ STEP 11: DIAGNOSTICS ═══

  const diagnostics = {
    nvidia_smi_watch: 'nvidia-smi dmon -s pucvmet -c 10',
    dcgm_metrics: [
      'DCGM_FI_DEV_GPU_UTIL',
      'DCGM_FI_DEV_MEM_COPY_UTIL',
      'DCGM_FI_DEV_FB_USED'
    ],
    vllm_metrics: [
      'vllm:gpu_kv_cache_usage',
      'vllm:gpu_prefix_cache_hit_rate',
      'vllm:avg_generation_throughput_toks_per_s'
    ]
  }

  // ═══ RETURN COMPLETE RESULT ═══

  return {
    memory_analysis,
    vllm_config,
    parallelism_strategy,
    bottleneck_analysis,
    llmd_config,
    diagnostics,
    // Include validation warnings in result
    warnings: validation.warnings
  }
}
