import {
  DeploymentParams,
  ExtractedConfig,
  GPUCountResult,
  MemoryBudget,
  ModelFamilies,
  RecurrentState,
} from './kv-types'
import { KVMemory3Cases } from './kv-types'

// vllm-versions.json shape
interface VLLMVersionConfig {
  version_min: string
  version_max: string | null
  act_coeff:   number
}

// ─── act_coeff from vLLM version ─────────────────────────────────────────────

function semverGte(a: string, b: string): boolean {
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  for (let i = 0; i < 3; i++) {
    const av = pa[i] ?? 0
    const bv = pb[i] ?? 0
    if (av > bv) return true
    if (av < bv) return false
  }
  return true
}

export function resolveActCoeff(
  vllmVersion: string,
  configs:     VLLMVersionConfig[]
): number {
  for (const c of configs) {
    const aboveMin = semverGte(vllmVersion, c.version_min)
    const belowMax = c.version_max == null || !semverGte(vllmVersion, c.version_max)
    if (aboveMin && (c.version_max == null || belowMax)) return c.act_coeff
  }
  return configs.at(-1)?.act_coeff ?? 0.11
}

// ─── Activation memory (MoE-aware) ───────────────────────────────────────────
// Uses per-GPU weight (already divided by TP) so the result is also per-GPU.

export function computeActivationMemory(
  weightPerGpuBytes: number,
  cfg:               ExtractedConfig,
  actCoeff:          number
): number {
  if (!cfg.is_moe || cfg.active_ratio == null) {
    return actCoeff * weightPerGpuBytes
  }

  // Partial MoE (Jamba-style): only some layers are MoE
  if (cfg.expert_layer_period != null) {
    const n_moe    = Math.floor(cfg.L / cfg.expert_layer_period)
    const moeFrac  = n_moe / cfg.L
    const effective = weightPerGpuBytes * ((1 - moeFrac) + moeFrac * cfg.active_ratio)
    return actCoeff * effective
  }

  // Full MoE (Mixtral, Qwen3, DeepSeek): all layers
  return actCoeff * weightPerGpuBytes * cfg.active_ratio
}

// ─── Safety buffer ────────────────────────────────────────────────────────────

const SAFETY_CUDA_GRAPHS_GB   = 2.0
const SAFETY_FRAGMENTATION_PCT = 0.03
const SAFETY_RUNTIME_BUFFERS_GB = 1.0
const SAFETY_DRIVER_OVERHEAD_GB = 0.5

export function computeSafetyBuffer(vramGb: number): number {
  return (
    SAFETY_CUDA_GRAPHS_GB    * 1e9 +
    SAFETY_FRAGMENTATION_PCT * vramGb * 1e9 +
    SAFETY_RUNTIME_BUFFERS_GB * 1e9 +
    SAFETY_DRIVER_OVERHEAD_GB * 1e9
  )
}

// ─── Overhead (logits buffer + CUDA context) ──────────────────────────────────

const FIXED_OVERHEAD_GB = 1.2

export function computeOverhead(cfg: ExtractedConfig, deploy: DeploymentParams): number {
  const B_logits = cfg.B
  // Logits buffer sharded by TP; freed after each decode step
  const logitsMemory = deploy.max_num_seqs * (cfg.vocab_size / deploy.tp) * B_logits
  return FIXED_OVERHEAD_GB * 1e9 + logitsMemory
}

// ─── Full memory budget ───────────────────────────────────────────────────────

export function computeMemoryBudget(
  weightBytes:    number,
  kvMemory:       KVMemory3Cases,
  recurrent:      RecurrentState | null,
  deploy:         DeploymentParams,
  cfg:            ExtractedConfig,
  vramGb:         number,
  actCoeff:       number
): MemoryBudget {
  const TP = deploy.tp

  // Weights sharded across TP GPUs
  const weight_per_gpu = weightBytes / TP

  // Activation peak scales with per-GPU weight
  const activation_per_gpu = computeActivationMemory(weight_per_gpu, cfg, actCoeff)

  // Recurrent state sharded across TP GPUs
  const recurrent_per_gpu = (recurrent?.total_state_memory_bytes ?? 0) / TP

  const safety  = computeSafetyBuffer(vramGb)
  const overhead = computeOverhead(cfg, deploy)

  const fixed_per_gpu =
    weight_per_gpu + activation_per_gpu + recurrent_per_gpu + safety + overhead

  return {
    weight_memory_total: weightBytes,
    weight_per_gpu,
    activation_per_gpu,
    recurrent_per_gpu,
    safety_buffer: safety,
    overhead,
    kv_memory: kvMemory,
    total_per_gpu_optimistic:   fixed_per_gpu + kvMemory.optimistic,
    total_per_gpu_expected:     fixed_per_gpu + kvMemory.expected,
    total_per_gpu_conservative: fixed_per_gpu + kvMemory.conservative,
  }
}

// ─── GPU count ────────────────────────────────────────────────────────────────

export function computeGPUCount(
  budget:  MemoryBudget,
  vramGb:  number,
  deploy:  DeploymentParams
): GPUCountResult {
  const TP     = deploy.tp
  const usable = vramGb * 1e9 * deploy.gpu_memory_utilization

  const totalGpus = (perGpu: number): number =>
    Math.max(TP, Math.ceil(perGpu / usable) * TP)

  const n_opt = totalGpus(budget.total_per_gpu_optimistic)
  const n_exp = totalGpus(budget.total_per_gpu_expected)
  const n_con = totalGpus(budget.total_per_gpu_conservative)

  const headroom = (perGpu: number, n: number): number =>
    (usable * (n / TP) - perGpu) / 1e9

  const h_opt = headroom(budget.total_per_gpu_optimistic,   n_opt)
  const h_exp = headroom(budget.total_per_gpu_expected,     n_exp)
  const h_con = headroom(budget.total_per_gpu_conservative, n_con)

  const safetyGb = budget.safety_buffer / 1e9
  const warnings: string[] = []

  if (h_exp < safetyGb) {
    warnings.push(
      `Expected scenario: only ${h_exp.toFixed(1)} GB headroom. ` +
      `Consider increasing GPU count or reducing concurrent users.`
    )
  }
  if (h_con < safetyGb) {
    warnings.push(
      `Conservative scenario: ${h_con.toFixed(1)} GB headroom — not production-safe at max context.`
    )
  }

  return {
    optimistic:   n_opt,
    expected:     n_exp,
    conservative: n_con,
    headroom_gb:  { optimistic: h_opt, expected: h_exp, conservative: h_con },
    tp_used:      TP,
    warnings,
  }
}

// ─── Auto TP resolution ───────────────────────────────────────────────────────
// Finds the smallest TP (1, 2, 4, 8) where the model fits in one TP group's
// worth of VRAM. Returns that TP value.

export function resolveAutoTP(
  cfg:         ExtractedConfig,
  weightBytes: number,
  kvMemory:    KVMemory3Cases,
  recurrent:   RecurrentState | null,
  vramGb:      number,
  baseParams:  Omit<DeploymentParams, 'tp'>,
  actCoeff:    number,
  families:    ModelFamilies
): number {
  const candidates: Array<1 | 2 | 4 | 8> = [1, 2, 4, 8]

  for (const tp of candidates) {
    // TP must divide query heads
    if (cfg.H_q % tp !== 0) continue

    const deploy: DeploymentParams = { ...baseParams, tp }
    const budget = computeMemoryBudget(
      weightBytes, kvMemory, recurrent, deploy, cfg, vramGb, actCoeff
    )
    const gpuCount = computeGPUCount(budget, vramGb, deploy)

    // Fits if expected scenario needs at most one TP group (tp GPUs)
    if (gpuCount.expected <= tp) return tp
  }

  return 8  // largest single-node TP
}
