// Constants from llm-inference-planner/planner/efficiency_constants.yaml
// Update both files if efficiency_constants.yaml changes.
import type { RooflineGpu, RooflineModel, Dtype, GpuArch } from './roofline-types'

const MFU_BASE: Partial<Record<GpuArch, Partial<Record<Dtype, number>>>> = {
  hopper:    { bf16: 0.55, fp16: 0.55, fp8: 0.53, mxfp4: 0.50 },
  ampere:    { bf16: 0.50, fp16: 0.50, fp8: 0.405, mxfp4: 0.45 },
  ada:       { bf16: 0.42, fp16: 0.42, fp8: 0.45, mxfp4: 0.45 },
  blackwell: { bf16: 0.58, fp8: 0.55, mxfp4: 0.55 },
}

const BW_BASE: Record<string, number> = { hbm: 0.39, gddr: 0.35 }

const SIZE_FLOOR  = 0.4824
const SIZE_SCALE  = 5_864_689_951
const ISL_FLOOR   = 0.392
const ISL_SCALE   = 512.0
const BATCH_FLOOR = 0.80
const BATCH_SCALE = 30.38
const MOE_FACTOR  = 0.80

export function mfuPrefill(model: RooflineModel, gpu: RooflineGpu, dtype: Dtype, isl: number): number {
  const archMap = MFU_BASE[gpu.arch]
  if (!archMap) return gpu.default_mfu_prefill

  const base = archMap[dtype] ?? archMap['bf16'] ?? gpu.default_mfu_prefill

  const fSize = SIZE_FLOOR + (1 - SIZE_FLOOR) * (1 - Math.exp(-model.active_params / SIZE_SCALE))
  const fIsl  = ISL_FLOOR  + (1 - ISL_FLOOR)  * (1 - Math.exp(-isl / ISL_SCALE))
  const fMoe  = model.is_moe ? MOE_FACTOR : 1.0

  return Math.max(0.08, Math.min(base, base * fSize * fIsl * fMoe))
}

export function bwEffDecode(gpu: RooflineGpu, effBatch: number): number {
  const base = BW_BASE[gpu.memory_type]
  if (base == null) return gpu.default_bw_efficiency_decode

  const gBatch = BATCH_FLOOR + (1 - BATCH_FLOOR) * (1 - Math.exp(-effBatch / BATCH_SCALE))
  return Math.min(base, base * gBatch)
}

export function bwEffPrefill(gpu: RooflineGpu): number {
  return BW_BASE[gpu.memory_type] ?? gpu.default_bw_efficiency_decode
}
