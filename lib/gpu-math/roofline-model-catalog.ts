// Source: llm-inference-planner/catalog/models.yaml — keep in sync manually
import type { HFModelConfig } from '@/lib/huggingface/fetch-config'
import type { RooflineModel } from './roofline-types'

function kv(layers: number, kvHeads: number, headDim: number, kvDtypeBytes: number): number {
  return 2 * layers * kvHeads * headDim * kvDtypeBytes
}

export const ROOFLINE_MODEL_CATALOG: RooflineModel[] = [
  {
    id: 'llama-3.1-8b',
    hfId: 'meta-llama/Llama-3.1-8B-Instruct',
    display_name: 'Llama 3.1 8B',
    is_moe: false,
    total_params: 8_030_000_000,
    active_params: 8_030_000_000,
    num_layers: 32,
    d_model: 4096,
    num_q_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(32, 8, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'llama-3.1-70b',
    hfId: 'meta-llama/Llama-3.1-70B-Instruct',
    display_name: 'Llama 3.1 70B',
    is_moe: false,
    total_params: 70_600_000_000,
    active_params: 70_600_000_000,
    num_layers: 80,
    d_model: 8192,
    num_q_heads: 64,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(80, 8, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'llama-3.3-70b',
    hfId: 'meta-llama/Llama-3.3-70B-Instruct',
    display_name: 'Llama 3.3 70B',
    is_moe: false,
    total_params: 70_600_000_000,
    active_params: 70_600_000_000,
    num_layers: 80,
    d_model: 8192,
    num_q_heads: 64,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(80, 8, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'llama-4-maverick',
    hfId: 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    display_name: 'Llama 4 Maverick',
    is_moe: true,
    total_params: 400_000_000_000,
    active_params: 52_000_000_000,
    num_experts: 128,
    experts_per_token: 8,
    num_layers: 48,
    d_model: 5120,
    num_q_heads: 40,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(48, 8, 128, 1),
    geometry_source: 'estimated',
  },
  {
    id: 'qwen3-8b',
    hfId: 'Qwen/Qwen3-8B',
    display_name: 'Qwen3-8B',
    is_moe: false,
    total_params: 8_190_000_000,
    active_params: 8_190_000_000,
    num_layers: 36,
    d_model: 4096,
    num_q_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(36, 8, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'qwen3-30b-a3b',
    hfId: 'Qwen/Qwen3-30B-A3B',
    display_name: 'Qwen3-30B-A3B',
    is_moe: true,
    total_params: 30_500_000_000,
    active_params: 3_300_000_000,
    num_experts: 128,
    experts_per_token: 8,
    num_layers: 48,
    d_model: 2048,
    num_q_heads: 32,
    num_kv_heads: 4,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(48, 4, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'mistral-small-24b',
    hfId: 'mistralai/Mistral-Small-Instruct-2409',
    display_name: 'Mistral Small 24B',
    is_moe: false,
    total_params: 23_570_000_000,
    active_params: 23_570_000_000,
    num_layers: 40,
    d_model: 5120,
    num_q_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    kv_dtype_bytes: 1,
    kv_bytes_per_token: kv(40, 8, 128, 1),
    geometry_source: 'known',
  },
  {
    id: 'gemma-2-9b',
    hfId: 'google/gemma-2-9b-it',
    display_name: 'Gemma 2 9B',
    is_moe: false,
    total_params: 9_240_000_000,
    active_params: 9_240_000_000,
    num_layers: 42,
    d_model: 3584,
    num_q_heads: 16,
    num_kv_heads: 8,
    head_dim: 256,
    kv_dtype_bytes: 2,
    kv_bytes_per_token: kv(42, 8, 256, 2),
    geometry_source: 'known',
    sliding_window: 4096,
    global_layer_every_n: 2,
  },
  {
    id: 'gpt-oss-20b',
    display_name: 'GPT-OSS 20B (MoE)',
    is_moe: true,
    total_params: 20_900_000_000,
    active_params: 3_610_000_000,
    num_experts: 32,
    experts_per_token: 4,
    num_layers: 24,
    d_model: 2880,
    num_q_heads: 64,
    num_kv_heads: 8,
    head_dim: 64,
    kv_dtype_bytes: 1,
    resident_weights_gb: 13.0,
    kv_bytes_per_token: kv(24, 8, 64, 1),
    geometry_source: 'known',
  },
]

export function getRooflineModelById(id: string): RooflineModel | undefined {
  return ROOFLINE_MODEL_CATALOG.find(m => m.id === id || m.hfId === id)
}

/**
 * Build a RooflineModel from a HuggingFace config.json.
 * Used when the user types a model ID not in the static catalog.
 * geometry_source is 'estimated' — confidence band widens to ±25%.
 */
export function hfConfigToRooflineModel(hfId: string, cfg: HFModelConfig): RooflineModel {
  const d_model      = cfg.hidden_size         ?? 4096
  const num_layers   = cfg.num_hidden_layers   ?? 32
  const H_q          = cfg.num_attention_heads ?? 32
  const H_kv         = cfg.num_key_value_heads ?? H_q
  const head_dim     = (cfg.head_dim as number | undefined) ?? Math.round(d_model / H_q)
  const ffn_dim      = cfg.intermediate_size   ?? (4 * d_model)
  const vocab_size   = cfg.vocab_size          ?? 32_000
  const num_experts  = cfg.num_local_experts   ?? 1
  const experts_per_token = cfg.num_experts_per_tok ?? 1
  const is_moe       = num_experts > 1

  // Attention params — GQA-correct: Q(d×H_q×head_dim) K(d×H_kv×head_dim) V(d×H_kv×head_dim) O(H_q×head_dim×d)
  const attn_per_layer = d_model * head_dim * (2 * H_q + 2 * H_kv)

  // FFN — SwiGLU uses 3 matrices (gate + up + down)
  const ffn_per_expert = 3 * d_model * ffn_dim
  const ffn_per_layer  = num_experts * ffn_per_expert

  // Embeddings (assume tied for simplicity)
  const embed = vocab_size * d_model

  const total_params = embed + num_layers * (attn_per_layer + ffn_per_layer)
  const active_params = is_moe
    ? embed + num_layers * (attn_per_layer + experts_per_token * ffn_per_expert)
    : total_params

  const kv_dtype_bytes = 2  // bf16 KV cache default
  const kv_bytes_per_token = kv(num_layers, H_kv, head_dim, kv_dtype_bytes)

  return {
    id: hfId,
    hfId,
    display_name: hfId.split('/').pop() ?? hfId,
    is_moe,
    total_params,
    active_params,
    num_layers,
    d_model,
    num_q_heads: H_q,
    num_kv_heads: H_kv,
    head_dim,
    kv_dtype_bytes,
    kv_bytes_per_token,
    geometry_source: 'estimated',
    ...(is_moe ? { num_experts, experts_per_token } : {}),
  }
}
