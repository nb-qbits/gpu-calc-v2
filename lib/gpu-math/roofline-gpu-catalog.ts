// Source: llm-inference-planner/catalog/gpus.yaml — keep in sync manually
import type { RooflineGpu } from './roofline-types'

export const ROOFLINE_GPU_CATALOG: RooflineGpu[] = [
  {
    id: 'h100_sxm',
    display_name: 'NVIDIA H100 SXM',
    arch: 'hopper',
    memory_type: 'hbm',
    mem_gb: 80,
    hbm_bandwidth_gbps: 3350,
    peak_flops: { fp16: 989, bf16: 989, fp8: 1979, mxfp4: 1979 },
    default_mfu_prefill: 0.40,
    default_bw_efficiency_decode: 0.70,
  },
  {
    id: 'h200_sxm',
    display_name: 'NVIDIA H200 SXM',
    arch: 'hopper',
    memory_type: 'hbm',
    mem_gb: 141,
    hbm_bandwidth_gbps: 4800,
    peak_flops: { fp16: 989, bf16: 989, fp8: 1979, mxfp4: 1979 },
    default_mfu_prefill: 0.40,
    default_bw_efficiency_decode: 0.70,
  },
  {
    id: 'a100_80gb_sxm',
    display_name: 'NVIDIA A100 80GB SXM4',
    arch: 'ampere',
    memory_type: 'hbm',
    mem_gb: 80,
    hbm_bandwidth_gbps: 2039,
    peak_flops: { fp16: 312, bf16: 312, fp8: 312, mxfp4: 312 },
    default_mfu_prefill: 0.45,
    default_bw_efficiency_decode: 0.70,
  },
  {
    id: 'l40s',
    display_name: 'NVIDIA L40S',
    arch: 'ada',
    memory_type: 'gddr',
    mem_gb: 48,
    hbm_bandwidth_gbps: 864,
    peak_flops: { fp16: 362, bf16: 362, fp8: 733, mxfp4: 733 },
    default_mfu_prefill: 0.40,
    default_bw_efficiency_decode: 0.65,
  },
  {
    id: 'l4',
    display_name: 'NVIDIA L4',
    arch: 'ada',
    memory_type: 'gddr',
    mem_gb: 24,
    hbm_bandwidth_gbps: 300,
    peak_flops: { fp16: 121, bf16: 121, fp8: 242, mxfp4: 242 },
    default_mfu_prefill: 0.40,
    default_bw_efficiency_decode: 0.65,
  },
]

export function getRooflineGpuById(id: string): RooflineGpu | undefined {
  return ROOFLINE_GPU_CATALOG.find(g => g.id === id)
}
