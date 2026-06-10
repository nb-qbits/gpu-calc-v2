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
    // Extract param count from HF config
    const {
      hidden_size = 4096,
      num_hidden_layers = 32,
      intermediate_size = 11008,
      vocab_size = 32000
    } = req.hf_config

    // Rough estimation formula
    const embedding_params = vocab_size * hidden_size
    const attention_params = 4 * hidden_size * hidden_size
    const ffn_params = 2 * hidden_size * intermediate_size
    const layer_params = attention_params + ffn_params
    const total_params = embedding_params + (num_hidden_layers * layer_params) + embedding_params

    params_billions = total_params / 1e9

    console.log(`📊 Estimated ${params_billions.toFixed(1)}B parameters from HF config`)
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

  // Simplified KV budget calculation with separate KV cache precision
  // TODO: Integrate with existing KV cache engine (kv-formulas.ts) for precise calculation
  // For now, use a conservative estimate adjusted by KV cache dtype

  // Determine KV cache precision (defaults to weight precision if not specified)
  const kv_cache_precision = req.kv_cache_precision || req.precision

  // KV cache estimation based on model architecture
  // KV bytes/token = 2 (K+V) × layers × kv_heads × head_dim × bytes_per_element / TP
  // For Llama 70B: 2 × 80 layers × 8 KV heads × 128 head_dim × 2 bytes (FP16) = 327,680 bytes/token
  // Divided by TP size: ~164 KB/token for TP=2

  // Rough estimation based on model size (more accurate than fixed 200 bytes)
  // Small models (7-13B): ~40 KB/token
  // Medium models (30-70B): ~160 KB/token
  // Large models (175B+): ~400 KB/token
  const kv_bytes_per_token_fp16 = params_billions < 20 ? 40000 :
                                    params_billions < 100 ? 160000 :
                                    400000

  // Adjust for KV cache precision
  const kv_precision_multiplier: Record<string, number> = {
    FP16: 1.0,  // 2 bytes per element
    FP8: 0.5    // 1 byte per element
  }

  // Adjust for tensor parallelism (KV cache is sharded across TP)
  const kv_bytes_per_token_estimate = (kv_bytes_per_token_fp16 * kv_precision_multiplier[kv_cache_precision]) / tp_size

  const total_context = req.isl + req.osl
  const kv_gb_per_sequence = (kv_bytes_per_token_estimate * total_context) / 1e9

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
