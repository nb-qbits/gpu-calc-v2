// Port of llm-inference-planner/planner/capacity.py
// V1 simplification: no anchor lookup. Confidence is medium (known geometry) or default (estimated).
import type {
  RooflineGpu, RooflineModel, WorkloadInputs,
  Traffic, KvBudget, TtftEstimate, CapacityEstimate,
  RooflineResult, BindingConstraint, ConfidenceLevel, Dtype,
} from './roofline-types'
import { mfuPrefill, bwEffDecode, bwEffPrefill } from './roofline-efficiency'

const DTYPE_BYTES: Record<string, number> = {
  fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, mxfp4: 0.5, int8: 1.0, int4: 0.5,
}
const HEADROOM: Record<string, number> = { realtime: 1.40, mixed: 1.25, batch: 1.10 }
export const CONFIDENCE_BAND: Record<ConfidenceLevel, number> = { high: 0.10, medium: 0.20, default: 0.25 }
const FIXED_OVERHEAD_BYTES = 0.5e9
const MAX_QUEUE_UTIL = 0.94
const LOW_KV_THRESHOLD = 4
const CHUNKED_PREFILL_ISL_THRESHOLD = 4096
const BATCH_EFFICIENCY = 0.70
// Engine throughput multiplier vs vLLM baseline.
// Derived from llm-inference-planner/planner/efficiency_constants.yaml engine_factor.
// Python capacity.py plan() does not apply this factor at call time — it is an offline
// calibration constant there. TypeScript applies it as a live ceiling multiplier, which
// is the physically correct behaviour (TRT-LLM genuinely achieves ~29% higher throughput).
const ENGINE_FACTOR: Record<string, number> = { vllm: 1.0, trtllm: 1.2936 }

function normalizeTraffic(rpd: number, peakMult: number, isl: number, osl: number): Traffic {
  const avg_rps = rpd / 86_400
  const peak_rps = avg_rps * peakMult
  return {
    requests_per_day: rpd,
    avg_rps,
    peak_rps,
    input_tps_avg: avg_rps * isl,
    output_tps_avg: avg_rps * osl,
    input_tps_peak: peak_rps * isl,
    output_tps_peak: peak_rps * osl,
    total_tokens_day: rpd * (isl + osl),
  }
}

type KvResult = KvBudget | { error: string }

function computeKvBudget(
  gpu: RooflineGpu, model: RooflineModel, dtype: string,
  isl: number, osl: number, gpuMemUtil: number, tp: number,
): KvResult {
  const dtypeBytes = DTYPE_BYTES[dtype] ?? 2.0
  const weightsBytes = model.resident_weights_gb != null
    ? model.resident_weights_gb * 1e9 / tp
    : model.total_params * dtypeBytes / tp

  const usableMem = gpu.mem_gb * 1e9 * gpuMemUtil
  if (weightsBytes > usableMem) {
    return {
      error: `Model requires ${(weightsBytes / 1e9).toFixed(1)} GB per GPU at ${dtype} ` +
             `(tp=${tp}), but ${gpu.display_name} has only ${(usableMem / 1e9).toFixed(1)} GB usable. ` +
             `Use larger tp or a GPU with more VRAM.`,
    }
  }

  const kvShardFactor = Math.min(tp, model.num_kv_heads)
  const kvCacheBudget = (usableMem - weightsBytes - FIXED_OVERHEAD_BYTES) * kvShardFactor
  if (kvCacheBudget <= 0) {
    return { error: `No KV cache budget remaining after weights + overhead on ${gpu.display_name}.` }
  }

  const maxKvTokens = kvCacheBudget / model.kv_bytes_per_token

  // Sliding-window models (Gemma 2/3/4): local layers cap KV at the window size;
  // global layers store the full sequence. effective_context_tokens is the
  // per-layer-average tokens stored per in-flight sequence.
  let effective_context_tokens: number
  if (model.sliding_window != null && model.global_layer_every_n != null) {
    const globalLayers = Math.floor(model.num_layers / model.global_layer_every_n)
    const localLayers  = model.num_layers - globalLayers
    effective_context_tokens = (
      globalLayers * (isl + osl) +
      localLayers  * Math.min(isl + osl, model.sliding_window)
    ) / model.num_layers
  } else {
    effective_context_tokens = isl + osl
  }

  const maxConcurrentSeqs = Math.max(1, Math.floor(maxKvTokens / effective_context_tokens))

  return {
    kv_bytes_per_token: model.kv_bytes_per_token,
    weights_resident_bytes: weightsBytes,
    usable_mem_bytes: usableMem,
    kv_cache_budget_bytes: usableMem - weightsBytes - FIXED_OVERHEAD_BYTES,
    max_kv_tokens: maxKvTokens,
    max_concurrent_seqs: maxConcurrentSeqs,
    effective_context_tokens,
  }
}

function computePrefillCeiling(
  gpu: RooflineGpu, model: RooflineModel, dtype: string,
  isl: number, mfu: number, bwEff: number, tp: number,
): number {
  const dtypeBytes = DTYPE_BYTES[dtype] ?? 2.0
  const qProjDim = model.num_q_heads * model.head_dim
  const flopsPerToken = 2 * model.active_params + 2 * model.num_layers * isl * qProjDim
  const peakFlops = gpu.peak_flops[dtype as Dtype] ?? gpu.peak_flops['bf16'] ?? 0
  const computeTps = peakFlops * 1e12 * mfu * tp / flopsPerToken
  const weightBytes = model.active_params * dtypeBytes
  const bwTps = isl * gpu.hbm_bandwidth_gbps * 1e9 * bwEff * tp / weightBytes
  return Math.min(computeTps, bwTps)
}

function computeDecodeCeiling(
  gpu: RooflineGpu, model: RooflineModel, dtype: string,
  batch: number, avgCtx: number, bwEff: number, tp: number, mfu: number,
): number {
  const dtypeBytes = DTYPE_BYTES[dtype] ?? 2.0
  const achievableBw = gpu.hbm_bandwidth_gbps * 1e9 * bwEff

  let weightBytesStep: number
  if (model.is_moe && model.num_experts && model.experts_per_token) {
    const r = model.experts_per_token / model.num_experts
    const distinctFrac = 1.0 - Math.pow(1.0 - r, batch)
    const denseParams = (model.active_params - r * model.total_params) / (1.0 - r)
    const expertPoolParams = model.total_params - denseParams
    weightBytesStep = (denseParams + distinctFrac * expertPoolParams) * dtypeBytes
  } else {
    weightBytesStep = model.active_params * dtypeBytes
  }

  const bytesPerStep = weightBytesStep / tp + batch * model.kv_bytes_per_token * avgCtx / tp
  const bwTps = batch * achievableBw / bytesPerStep
  const peakFlops = gpu.peak_flops[dtype as Dtype] ?? gpu.peak_flops['bf16'] ?? 0
  const computeTps = peakFlops * 1e12 * mfu * tp / (2 * model.active_params)
  return Math.min(bwTps, computeTps)
}

function estimateTtft(
  isl: number, prefillTpsPerReplica: number,
  utilization: number, sloMs: number,
): TtftEstimate {
  const rho = Math.min(utilization, MAX_QUEUE_UTIL)
  const computeS = isl / prefillTpsPerReplica
  const queueS = rho > 0 ? computeS * (rho / (1.0 - rho)) : 0.0
  const ttftMs = (computeS + queueS) * 1000.0
  const sloMet = ttftMs <= sloMs
  let breachReason: string | null = null
  if (!sloMet) {
    breachReason = queueS > computeS
      ? `Queue-bound: queuing (${(queueS * 1000).toFixed(0)} ms) > compute (${(computeS * 1000).toFixed(0)} ms). Add replicas or reduce peak load.`
      : `Compute-bound: ${isl}-token prefill takes ${(computeS * 1000).toFixed(0)} ms, exceeds SLO ${sloMs.toFixed(0)} ms.`
  }
  return {
    ttft_compute_ms: computeS * 1000,
    ttft_queue_ms: queueS * 1000,
    ttft_ms: ttftMs,
    utilization: rho,
    slo_ms: sloMs,
    slo_met: sloMet,
    slo_breach_reason: breachReason,
  }
}

function sizeReplicas(
  traffic: Traffic, prefillTps: number, decodeTps: number,
  maxConcurrentSeqs: number, trafficClass: string,
  isl: number, osl: number, effBatch: number,
) {
  const replicasPrefill    = Math.ceil(traffic.input_tps_peak  / prefillTps)
  const replicasDecode     = Math.ceil(traffic.output_tps_peak / decodeTps)
  const ttftRoughS         = isl / prefillTps
  const itlS               = effBatch / decodeTps
  const avgLatencyS        = ttftRoughS + osl * itlS
  const peakConcurrentReq  = traffic.peak_rps * avgLatencyS
  const replicasConcurrency = Math.ceil(peakConcurrentReq / maxConcurrentSeqs)

  const base = Math.max(replicasPrefill, replicasDecode, replicasConcurrency)
  let binding: BindingConstraint
  if      (base === replicasPrefill)    binding = 'prefill-bound'
  else if (base === replicasDecode)     binding = 'decode-bound'
  else                                  binding = 'kv-memory-bound'

  const headroom = HEADROOM[trafficClass] ?? 1.25
  return {
    replicasPrefill, replicasDecode, replicasConcurrency,
    binding, headroom, replicas: Math.ceil(base * headroom),
  }
}

export function runRooflinePlan(
  inputs: WorkloadInputs, model: RooflineModel, gpu: RooflineGpu,
): RooflineResult {
  const {
    dtype, tp, requests_per_day, peak_multiplier,
    isl, osl, ttft_slo_ms, traffic_class, gpu_mem_util,
    max_num_seqs,
    prefix_cache_len   = 0,
    prefix_cache_hit_rate = 0.0,
    runtime = 'vllm',
  } = inputs

  if (!(dtype in DTYPE_BYTES)) {
    return { ok: false, error: { type: 'unknown_dtype', message: `Unknown dtype: ${dtype}` } }
  }
  if (gpu.peak_flops[dtype as Dtype] == null) {
    return { ok: false, error: { type: 'unknown_dtype', message: `${gpu.display_name} does not support ${dtype}. Choose a supported precision for this GPU.` } }
  }

  // ── Prefix cache: reduce effective ISL for prefill compute only ──────────
  // Cached tokens skip recomputation but remain in KV memory — KV budget uses full ISL.
  const prefixCachedTokens = prefix_cache_len > 0 && prefix_cache_hit_rate > 0
    ? Math.floor(Math.min(prefix_cache_len, isl) * prefix_cache_hit_rate)
    : 0
  const effectiveIsl = Math.max(1, isl - prefixCachedTokens)

  // ── Engine factor ─────────────────────────────────────────────────────────
  const engineFactor = ENGINE_FACTOR[runtime] ?? 1.0

  // ── Traffic (uses effectiveIsl — cached tokens don't drive prefill compute) ──
  const traffic = normalizeTraffic(requests_per_day, peak_multiplier, effectiveIsl, osl)

  // ── KV budget (always uses full isl — prefix KV must stay resident) ──────
  const kvResult = computeKvBudget(gpu, model, dtype, isl, osl, gpu_mem_util, tp)
  if ('error' in kvResult) {
    return { ok: false, error: { type: 'insufficient_vram', message: kvResult.error } }
  }
  const kv = kvResult as KvBudget

  // ── max_num_seqs cap ──────────────────────────────────────────────────────
  // vLLM --max-num-seqs limits the scheduler batch independently of VRAM capacity.
  const effectiveMaxSeqs = max_num_seqs != null
    ? Math.min(kv.max_concurrent_seqs, max_num_seqs)
    : kv.max_concurrent_seqs

  const confidence: ConfidenceLevel = model.geometry_source === 'estimated' ? 'default' : 'medium'
  const bandFactor = CONFIDENCE_BAND[confidence]

  const mfu       = mfuPrefill(model, gpu, dtype as Dtype, effectiveIsl)
  const bwEffBase = bwEffPrefill(gpu)

  // ── Prefill ceiling — apply engine factor ─────────────────────────────────
  const pfillTpsRaw = computePrefillCeiling(gpu, model, dtype, effectiveIsl, mfu, bwEffBase, tp)
  const pfillTps    = pfillTpsRaw * engineFactor

  const avgCtx   = isl + Math.floor(osl / 2)
  const effBatch = Math.max(1, Math.floor(effectiveMaxSeqs * BATCH_EFFICIENCY))

  const dtypeBytes        = DTYPE_BYTES[dtype] ?? 2.0
  const weightBytesActive = model.active_params * dtypeBytes
  const kvBytesInflight   = kv.kv_bytes_per_token * avgCtx * effBatch
  const kvRatio           = kvBytesInflight / Math.max(weightBytesActive, 1)
  const decodeBwEff       = bwEffDecode(gpu, effBatch)

  // ── Decode ceiling — apply engine factor ──────────────────────────────────
  const decodeTpsRaw = computeDecodeCeiling(gpu, model, dtype, effBatch, avgCtx, decodeBwEff, tp, mfu)
  const decodeTps    = decodeTpsRaw * engineFactor

  const sz = sizeReplicas(traffic, pfillTps, decodeTps, effectiveMaxSeqs, traffic_class, effectiveIsl, osl, effBatch)

  const replicasLow  = Math.max(1, Math.ceil(sz.replicas * (1 - bandFactor)))
  const replicasHigh = Math.ceil(sz.replicas * (1 + bandFactor))

  const rho  = traffic.input_tps_peak / (sz.replicas * pfillTps)
  const ttft = estimateTtft(effectiveIsl, pfillTps, rho, ttft_slo_ms)

  const warnings: string[] = []
  if (kv.max_concurrent_seqs < LOW_KV_THRESHOLD)
    warnings.push(`Very limited KV cache: only ${kv.max_concurrent_seqs} concurrent sequence(s) per replica. Consider higher tp or a GPU with more VRAM.`)
  if (max_num_seqs != null && max_num_seqs < kv.max_concurrent_seqs)
    warnings.push(`max_num_seqs=${max_num_seqs} caps scheduler batch (KV budget allows ${kv.max_concurrent_seqs} seqs; scheduler cap is the binding limit).`)
  if (effectiveIsl >= CHUNKED_PREFILL_ISL_THRESHOLD)
    warnings.push(`ISL ${effectiveIsl} ≥ ${CHUNKED_PREFILL_ISL_THRESHOLD}: chunked prefill required (--enable-chunked-prefill for vLLM).`)
  if (model.geometry_source === 'estimated')
    warnings.push(`Model geometry estimated — verify layers, d_model, head config before production use.`)
  if (!ttft.slo_met && ttft.slo_breach_reason)
    warnings.push(`TTFT SLO breach: ${ttft.slo_breach_reason}`)
  if (confidence === 'default')
    warnings.push(`DEFAULT confidence (±25%): no anchor data. Validate on a live GPU before committing infrastructure.`)
  if (model.sliding_window != null && model.global_layer_every_n != null)
    warnings.push(
      `Sliding-window model: effective context per layer = ${kv.effective_context_tokens.toFixed(0)} tokens ` +
      `(${Math.floor(model.num_layers / model.global_layer_every_n)} global × ${isl + osl} + ` +
      `${model.num_layers - Math.floor(model.num_layers / model.global_layer_every_n)} local × min(${isl + osl}, ${model.sliding_window})). ` +
      `KV budget holds ${kv.max_concurrent_seqs} seqs vs ${Math.floor(kv.max_kv_tokens / (isl + osl))} without sliding-window correction.`
    )
  if (prefixCachedTokens > 0)
    warnings.push(
      `Prefix cache: ${prefix_cache_len} token prefix × ${(prefix_cache_hit_rate * 100).toFixed(0)}% hit rate ` +
      `→ ${prefixCachedTokens} tokens skipped per request → effective prefill ISL ${effectiveIsl} ` +
      `(KV budget still sized for full ISL ${isl}).`
    )
  if (runtime === 'trtllm')
    warnings.push(`TRT-LLM engine factor ${engineFactor.toFixed(4)}× applied to prefill and decode ceilings.`)
  const pfR = sz.replicasPrefill, dcR = sz.replicasDecode
  if (pfR > 0 && dcR > 0 && Math.max(pfR, dcR) < Math.min(pfR, dcR) * 2)
    warnings.push(`Balanced prefill/decode load (${pfR} vs ${dcR} replicas). Both phases share the GPU — actual need may be higher.`)

  const assumptions = [
    `GPU memory utilization: ${(gpu_mem_util * 100).toFixed(0)}%`,
    `MFU (prefill): ${mfu.toFixed(2)} (efficiency curve)`,
    `Bandwidth efficiency (decode): ${decodeBwEff.toFixed(2)}, KV ratio ${kvRatio.toFixed(1)}`,
    `Batch efficiency: ${(BATCH_EFFICIENCY * 100).toFixed(0)}% of max seqs (${effBatch}/${effectiveMaxSeqs})`,
    `Traffic headroom (${traffic_class}): ${sz.headroom.toFixed(2)}×`,
    `Tensor parallelism: tp=${tp} → ${sz.replicas} replicas × ${tp} = ${sz.replicas * tp} total GPUs`,
    `Avg context for decode: ${avgCtx} tokens (ISL + OSL/2)`,
    `Runtime: ${runtime} (engine factor ${engineFactor.toFixed(4)}×)`,
    `TTFT queue model: M/M/1 heuristic — validate on live GPU.`,
  ]

  const estimate: CapacityEstimate = {
    traffic,
    kv_budget: kv,
    prefill_tps_gpu: pfillTps,
    decode_tps_gpu: decodeTps,
    replicas: sz.replicas,
    replicas_low: replicasLow,
    replicas_high: replicasHigh,
    binding_constraint: sz.binding,
    ttft_estimate: ttft,
    confidence,
    assumptions,
    warnings,
    replicas_prefill: sz.replicasPrefill,
    replicas_decode: sz.replicasDecode,
    replicas_concurrency: sz.replicasConcurrency,
    mfu_used: mfu,
    decode_bw_eff_used: decodeBwEff,
    tp_used: tp,
    total_gpus: sz.replicas * tp,
    tpot_ms: decodeTps > 0 ? (effBatch / decodeTps) * 1000 : 0,
    eff_batch_used: effBatch,
    kv_ratio: kvRatio,
    headroom_factor: sz.headroom,
    runtime_used: runtime,
  }

  return { ok: true, estimate }
}
