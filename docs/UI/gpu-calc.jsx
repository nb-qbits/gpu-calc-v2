const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* ============================== Curated models ============================== */

const MODEL_PRESETS = [
  { id: 'meta-llama/Llama-3.1-8B-Instruct',  label: 'Llama 3.1 8B Instruct',  vendor: 'Meta',    params: 8 },
  { id: 'meta-llama/Llama-3.1-70B-Instruct', label: 'Llama 3.1 70B Instruct', vendor: 'Meta',    params: 70 },
  { id: 'google/gemma-2-9b-it',              label: 'Gemma 2 9B',             vendor: 'Google',  params: 9 },
  { id: 'google/gemma-2-27b-it',             label: 'Gemma 2 27B',            vendor: 'Google',  params: 27 },
  { id: 'mistralai/Mistral-7B-Instruct-v0.3',label: 'Mistral 7B',             vendor: 'Mistral', params: 7 },
  { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', label: 'Mixtral 8x7B',        vendor: 'Mistral', params: 47, activeParams: 13 },
  { id: 'Qwen/Qwen2.5-7B-Instruct',          label: 'Qwen 2.5 7B',            vendor: 'Qwen',    params: 7 },
];

/* Known GPUs */
const GPUS = [
  { id: 'H100_80GB',  label: 'NVIDIA H100 80GB',  mem: 80,  rate: 2.49 },
  { id: 'H200_141GB', label: 'NVIDIA H200 141GB', mem: 141, rate: 3.49 },
  { id: 'A100_80GB',  label: 'NVIDIA A100 80GB',  mem: 80,  rate: 1.79 },
  { id: 'A100_40GB',  label: 'NVIDIA A100 40GB',  mem: 40,  rate: 1.29 },
  { id: 'L40S_48GB',  label: 'NVIDIA L40S 48GB',  mem: 48,  rate: 1.15 },
  { id: 'MI300X',     label: 'AMD MI300X 192GB',  mem: 192, rate: 2.99 },
];

/* Quick presets — model architectures with their KV-cache variant labels */
const ARCH_HINTS = {
  // model_id_prefix: { kv_variant, kv_label }
  'meta-llama/Llama-3.1':  { kv_variant: 'GQA',  kv_label: 'GQA (8 KV heads)' },
  'meta-llama/Llama-3':    { kv_variant: 'GQA',  kv_label: 'GQA' },
  'google/gemma-2':        { kv_variant: 'GQA',  kv_label: 'GQA (sliding window)' },
  'google/gemma-3':        { kv_variant: 'GQA',  kv_label: 'GQA' },
  'mistralai/Mistral':     { kv_variant: 'GQA',  kv_label: 'GQA (8 KV heads)' },
  'mistralai/Mixtral':     { kv_variant: 'GQA',  kv_label: 'MoE · GQA' },
  'Qwen/Qwen2':            { kv_variant: 'GQA',  kv_label: 'GQA' },
  'deepseek':              { kv_variant: 'MLA',  kv_label: 'MLA (multi-head latent)' },
};

function archHintFor(id) {
  if (!id) return { kv_variant: 'GQA', kv_label: 'GQA (estimated)' };
  for (const [prefix, hint] of Object.entries(ARCH_HINTS)) {
    if (id.toLowerCase().startsWith(prefix.toLowerCase())) return hint;
  }
  return { kv_variant: 'GQA', kv_label: 'GQA (estimated)' };
}

/* ============================== Defaults ============================== */

const DEFAULTS = {
  model_id: 'meta-llama/Llama-3.1-8B-Instruct',
  gpu: 'H100_80GB',
  workload: {
    isl: 100,
    osl: 50,
    requests_per_day: 1_000_000,
    peak_multiplier: 3,
    concurrent_requests: null, // auto = req/day * peak / 86400
    active_request_ratio: 0.3,
    prefix_cache_hit_rate: 0,
    target_ttft_ms: 500,
    target_tpot_ms: 50,
  },
  memory: {
    weight_precision: 'BF16',
    weight_quantization: 'None',
    kv_precision: 'FP16',
    gpu_mem_utilization: 0.90,
    headroom_pct: 10,
  },
  parallelism: {
    tensor_parallel: 1,
    pipeline_parallel: 1,
    replicas: 1,
  },
  vllm: {
    block_size: 16,
    max_num_seqs: 256,
    max_num_batched_tokens: 16384,
    enable_prefix_caching: true,
    enable_chunked_prefill: 'auto',
  },
};

/* ============================== Helpers ============================== */

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function fmt(n) { return Number(n).toLocaleString('en-US'); }
function fmtBytes(gb) {
  if (gb >= 1024) return (gb / 1024).toFixed(2) + ' TB';
  if (gb >= 1) return gb.toFixed(gb < 10 ? 1 : 0) + ' GB';
  return (gb * 1024).toFixed(0) + ' MB';
}

function precisionBytes(precision) {
  return { 'BF16': 2, 'FP16': 2, 'FP8': 1, 'INT8': 1, 'INT4': 0.5 }[precision] || 2;
}

function parseHfModelId(rawId) {
  const id = rawId.trim();
  const moe = id.match(/(\d+)x(\d+(?:\.\d+)?)b\b/i);
  let params, activeParams;
  if (moe) {
    activeParams = parseFloat(moe[2]);
    params = activeParams * parseFloat(moe[1]);
  } else {
    const m = id.match(/(\d+(?:\.\d+)?)\s*b\b/i);
    params = m ? parseFloat(m[1]) : 8;
  }
  return { params, activeParams };
}

function computeAll(state) {
  const { model_id, gpu, workload, memory, parallelism, vllm, modelProfile } = state;
  const gpuSpec = GPUS.find(g => g.id === gpu) || GPUS[0];

  // Find params + arch
  const preset = MODEL_PRESETS.find(m => m.id === model_id);
  const guessed = parseHfModelId(model_id || '');
  const params = modelProfile.params || preset?.params || guessed.params;
  const activeParams = modelProfile.active_params || preset?.activeParams || guessed.activeParams || params;
  const arch = archHintFor(model_id);

  const weightBytes = precisionBytes(memory.weight_precision);
  const weightMemGB = activeParams * weightBytes;

  // Concurrent active requests
  // active_requests = (peak_rps × avg_request_duration) — approximated as users × ratio
  const avgRps = workload.requests_per_day / 86400;
  const peakRps = avgRps * workload.peak_multiplier;
  const avgDurationS = (workload.osl * (workload.target_tpot_ms || 50)) / 1000;
  const autoConcurrent = Math.max(1, Math.ceil(peakRps * avgDurationS));
  const concurrent = workload.concurrent_requests || autoConcurrent;

  // Model internals (auto-detect or fallback)
  const numAttentionHeads = modelProfile.num_attention_heads || 32;
  const numLayers = modelProfile.num_layers || Math.round(8 + Math.log2(activeParams) * 8);
  const numKvHeads = modelProfile.num_kv_heads || (arch.kv_variant === 'MHA' ? numAttentionHeads : 8);
  const headDim = modelProfile.head_dim || 128;
  const maxModelLen = modelProfile.max_position_embeddings || 8192;

  // KV variant classification
  let kvVariant = arch.kv_variant;
  let gqaRatio = 1;
  if (numAttentionHeads === numKvHeads) kvVariant = 'MHA';
  else if (numKvHeads === 1) kvVariant = 'MQA';
  else if (numKvHeads < numAttentionHeads) { kvVariant = 'GQA'; gqaRatio = numAttentionHeads / numKvHeads; }

  const kvBytes = precisionBytes(memory.kv_precision);
  const tp = parallelism.tensor_parallel === 'auto' ? 1 : Math.max(1, parseInt(parallelism.tensor_parallel) || 1);
  const kvBytesPerToken = (2 * numLayers * numKvHeads * headDim * kvBytes) / tp;
  const tokensPerReq = workload.isl + workload.osl;
  const effectiveTokens = tokensPerReq * (1 - workload.prefix_cache_hit_rate);
  const kvPerReqGB = (kvBytesPerToken * effectiveTokens) / (1024 ** 3);
  const kvTotalGB = kvPerReqGB * concurrent;

  // Equivalent comparisons for KV-architecture education
  const kvEquivMHA = (2 * numLayers * numAttentionHeads * headDim * kvBytes) / tp;
  const kvEquivMQA = (2 * numLayers * 1 * headDim * kvBytes) / tp;

  const actGB = (weightMemGB / tp) * 0.07;
  const totalGB = (weightMemGB / tp) + kvTotalGB + actGB;

  const usableMem = gpuSpec.mem * memory.gpu_mem_utilization * (1 - memory.headroom_pct / 100);

  // Likely / low / high GPU count
  function gpusFor({ peak_mult, isl_mult, prefix_hit, active_ratio }) {
    const _concurrent = workload.concurrent_requests
      || Math.max(1, Math.ceil(avgRps * peak_mult * avgDurationS * active_ratio / 0.3));
    const _tokens = (workload.isl * isl_mult + workload.osl) * (1 - prefix_hit);
    const _kvTotal = (kvBytesPerToken * _tokens * _concurrent) / (1024 ** 3);
    const _total = (weightMemGB / tp) + _kvTotal + actGB;
    return { gpus: Math.max(tp, Math.ceil(_total / usableMem)), totalGB: _total, kvTotal: _kvTotal, concurrent: _concurrent };
  }
  const sLow = gpusFor({ peak_mult: workload.peak_multiplier * 0.5, isl_mult: 0.7, prefix_hit: Math.max(workload.prefix_cache_hit_rate, 0.4), active_ratio: 0.15 });
  const sLikely = gpusFor({ peak_mult: workload.peak_multiplier, isl_mult: 1, prefix_hit: workload.prefix_cache_hit_rate, active_ratio: 0.3 });
  const sHigh = gpusFor({ peak_mult: workload.peak_multiplier * 1.5, isl_mult: 1.5, prefix_hit: workload.prefix_cache_hit_rate * 0.5, active_ratio: 0.6 });

  const gpus = sLikely.gpus;
  const utilization = clamp(totalGB / (gpus * gpuSpec.mem), 0, 0.99);
  const hourly = gpus * gpuSpec.rate;
  const monthly = Math.round(hourly * 730);

  // Sensitivity for sparkline: GPUs vs concurrent over 0.1x to 2x
  const sensitivity = [];
  for (let i = 1; i <= 20; i++) {
    const c = Math.max(1, Math.round(concurrent * i / 10));
    const tot = (weightMemGB / tp) + (kvPerReqGB * c) + actGB;
    sensitivity.push({ concurrent: c, gpus: Math.max(tp, Math.ceil(tot / usableMem)) });
  }

  // KV scenarios
  const kvScenarios = {
    prompt:   { tokens: workload.isl, gb: (kvBytesPerToken * workload.isl * concurrent) / (1024 ** 3) },
    typical:  { tokens: tokensPerReq, gb: kvTotalGB },
    worst:    { tokens: maxModelLen, gb: (kvBytesPerToken * maxModelLen * concurrent) / (1024 ** 3) },
  };

  // Constraint analysis
  const reservedFraction = 0.5;
  const memoryFit = totalGB <= gpus * usableMem ? 'ok' : 'bottleneck';
  const kvFit = kvTotalGB <= gpus * usableMem * 0.7 ? 'ok' : kvTotalGB <= gpus * usableMem * 0.9 ? 'watch' : 'bottleneck';
  const schedulerFit = concurrent <= vllm.max_num_seqs ? 'ok' : 'bottleneck';
  const maxBatched = (vllm.max_num_batched_tokens === 'auto' ? 16384 : (parseInt(vllm.max_num_batched_tokens) || 16384));
  const prefillsPerStep = Math.max(1, Math.floor(maxBatched / Math.max(1, workload.isl)));
  const batchedTokenFit = workload.isl <= maxBatched ? (prefillsPerStep >= 4 ? 'ok' : 'watch') : 'bottleneck';
  const tpFit = gpus >= tp ? 'ok' : 'bottleneck';
  // block_size waste
  const allocatedTokens = Math.ceil(tokensPerReq / vllm.block_size) * vllm.block_size;
  const blockOverhead = (allocatedTokens - tokensPerReq) / tokensPerReq;

  // Primary driver
  let primaryDriver = 'model_weight_fit';
  if (kvTotalGB > weightMemGB * 1.2) primaryDriver = 'kv_cache_pressure';
  if (memoryFit === 'bottleneck') primaryDriver = 'memory_overflow';
  if (kvFit === 'bottleneck') primaryDriver = 'kv_memory';
  if (schedulerFit === 'bottleneck') primaryDriver = 'max_num_seqs';
  if (batchedTokenFit === 'bottleneck') primaryDriver = 'max_num_batched_tokens';
  if (tpFit === 'bottleneck') primaryDriver = 'tp_minimum';

  // Range drivers analysis — sensitivity to each input
  const baseGpu = sLikely.gpus;
  const driverImpacts = [
    { name: 'Peak traffic factor', impact: Math.abs(sHigh.gpus - sLow.gpus), low: sLow.gpus, high: sHigh.gpus, reason: 'Bursty traffic drives more concurrent requests' },
    { name: 'ISL / prompt length', impact: Math.max(1, Math.round(Math.abs(workload.isl * 1.5 - workload.isl * 0.7) / Math.max(1, workload.isl) * baseGpu)), low: 'shorter prompts', high: 'longer prompts', reason: 'Longer prompts inflate KV cache linearly' },
    { name: 'Active-request ratio', impact: 2, low: '15%', high: '60%', reason: 'Concurrent users ≠ active inference requests' },
    { name: 'Prefix-cache hit rate', impact: Math.round(workload.prefix_cache_hit_rate * baseGpu) || 1, low: 'high cache hit', high: 'no cache hit', reason: 'Cache hits skip prompt KV allocation' },
  ];
  driverImpacts.sort((a, b) => b.impact - a.impact);

  return {
    params, activeParams,
    arch: { ...arch, kv_variant: kvVariant, gqa_ratio: gqaRatio },
    numLayers, numKvHeads, numAttentionHeads, headDim, maxModelLen,
    weightBytes, weightMemGB,
    kvBytes, kvBytesPerToken, kvPerReqGB, kvTotalGB,
    kvEquivMHA, kvEquivMQA,
    actGB, totalGB,
    concurrent, autoConcurrent, avgRps, peakRps,
    gpuSpec, gpus, gpu_low: sLow.gpus, gpu_high: sHigh.gpus, utilization, usableMem,
    hourly, monthly,
    sensitivity,
    kvScenarios,
    constraints: { memoryFit, kvFit, schedulerFit, batchedTokenFit, tpFit },
    primaryDriver,
    blockOverhead, allocatedTokens, prefillsPerStep,
    tp,
    driverImpacts,
  };
}

/* ============================== Animated number ============================== */

function useAnimatedNumber(target, duration = 600) {
  const [val, setVal] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    fromRef.current = val;
    startRef.current = performance.now();
    cancelAnimationFrame(rafRef.current);
    const tick = (t) => {
      const p = clamp((t - startRef.current) / duration, 0, 1);
      const e = 1 - Math.pow(1 - p, 5);
      setVal(fromRef.current + (target - fromRef.current) * e);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target]); // eslint-disable-line
  return val;
}

function Ticker({ value, digits = 0 }) {
  const animated = useAnimatedNumber(value);
  const str = fmt(Number(animated.toFixed(digits)));
  return <span className="num-roll">{str}</span>;
}

/* ============================== Icons ============================== */

function Icon({ name, size = 14 }) {
  const s = { width: size, height: size, strokeWidth: 1.6, stroke: 'currentColor', fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' };
  if (name === 'cpu') return (<svg viewBox="0 0 24 24" {...s}><rect x="6" y="6" width="12" height="12" rx="1.5"/><rect x="9" y="9" width="6" height="6"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/></svg>);
  if (name === 'gpu') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="7" width="18" height="11" rx="1.5"/><rect x="6" y="10" width="5" height="5" rx="0.5"/><rect x="13" y="10" width="5" height="5" rx="0.5"/><path d="M3 11h-1M3 14h-1M22 11h1M22 14h1"/></svg>);
  if (name === 'memory') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="7" width="18" height="10" rx="1.5"/><path d="M7 7v-2M11 7v-2M13 7v-2M17 7v-2M7 19v-2M11 19v-2M13 19v-2M17 19v-2"/></svg>);
  if (name === 'cache') return (<svg viewBox="0 0 24 24" {...s}><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>);
  if (name === 'cash') return (<svg viewBox="0 0 24 24" {...s}><rect x="2" y="6" width="20" height="12" rx="1.5"/><circle cx="12" cy="12" r="2.5"/><path d="M5 9.5v5M19 9.5v5"/></svg>);
  if (name === 'workload') return (<svg viewBox="0 0 24 24" {...s}><path d="M4 6h16M4 12h10M4 18h7"/></svg>);
  if (name === 'precision') return (<svg viewBox="0 0 24 24" {...s}><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>);
  if (name === 'hardware') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="4" width="18" height="6" rx="1"/><rect x="3" y="14" width="18" height="6" rx="1"/><circle cx="7" cy="7" r="0.6" fill="currentColor"/><circle cx="7" cy="17" r="0.6" fill="currentColor"/></svg>);
  if (name === 'parallel') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="4" width="6" height="16"/><rect x="11" y="4" width="2" height="16"/><rect x="15" y="4" width="6" height="16"/></svg>);
  if (name === 'engine') return (<svg viewBox="0 0 24 24" {...s}><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2"/></svg>);
  if (name === 'internals') return (<svg viewBox="0 0 24 24" {...s}><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 4v16M15 4v16M4 9h16M4 15h16"/></svg>);
  if (name === 'pencil') return (<svg viewBox="0 0 24 24" {...s}><path d="M4 20h4l10-10-4-4L4 16v4Z"/><path d="M13.5 6.5l4 4"/></svg>);
  if (name === 'chev') return (<svg viewBox="0 0 24 24" {...s}><polyline points="6 9 12 15 18 9"/></svg>);
  if (name === 'flip') return (<svg viewBox="0 0 24 24" {...s}><path d="M3 12a9 9 0 0 1 16-5.7"/><polyline points="19 3 19 7 15 7"/><path d="M21 12a9 9 0 0 1-16 5.7"/><polyline points="5 21 5 17 9 17"/></svg>);
  if (name === 'star') return (<svg viewBox="0 0 24 24" {...s}><polygon points="12 3 14.5 9.5 21.5 10 16 14.5 17.5 21.5 12 17.5 6.5 21.5 8 14.5 2.5 10 9.5 9.5" fill="currentColor" stroke="currentColor"/></svg>);
  if (name === 'star-o') return (<svg viewBox="0 0 24 24" {...s}><polygon points="12 3 14.5 9.5 21.5 10 16 14.5 17.5 21.5 12 17.5 6.5 21.5 8 14.5 2.5 10 9.5 9.5"/></svg>);
  if (name === 'arrow') return (<svg viewBox="0 0 24 24" {...s}><path d="M5 12h14"/><polyline points="13 6 19 12 13 18"/></svg>);
  if (name === 'check') return (<svg viewBox="0 0 24 24" {...s}><polyline points="5 13 9.5 17.5 19 7"/></svg>);
  if (name === 'check-circle') return (<svg viewBox="0 0 24 24" width={size} height={size}><circle cx="12" cy="12" r="10" fill="#3e8635"/><polyline points="7 12.5 10.5 16 17 9" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>);
  if (name === 'warn') return (<svg viewBox="0 0 24 24" {...s}><path d="M12 3 2 20h20Z"/><path d="M12 10v5"/><circle cx="12" cy="18" r="0.7" fill="currentColor"/></svg>);
  if (name === 'external') return (<svg viewBox="0 0 24 24" {...s}><path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></svg>);
  if (name === 'copy') return (<svg viewBox="0 0 24 24" {...s}><rect x="8" y="8" width="13" height="13" rx="1.5"/><path d="M5 16V5a1 1 0 0 1 1-1h11"/></svg>);
  if (name === 'download') return (<svg viewBox="0 0 24 24" {...s}><path d="M12 4v12"/><polyline points="7 11 12 16 17 11"/><path d="M5 20h14"/></svg>);
  if (name === 'sheet') return (<svg viewBox="0 0 24 24" {...s}><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M4 9h16M4 15h16M10 3v18"/></svg>);
  if (name === 'cli') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="4" width="18" height="16" rx="1.5"/><polyline points="7 9 11 13 7 17"/><line x1="13" y1="17" x2="17" y2="17"/></svg>);
  if (name === 'hf') return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="#FFD21E"/>
      <circle cx="11" cy="14" r="2" fill="#3B2D2A"/>
      <circle cx="21" cy="14" r="2" fill="#3B2D2A"/>
      <path d="M9 19c1 3 4 4 7 4s6-1 7-4" stroke="#3B2D2A" strokeWidth="1.6" fill="none" strokeLinecap="round"/>
      <circle cx="6" cy="18" r="2.2" fill="#FF8C8C"/>
      <circle cx="26" cy="18" r="2.2" fill="#FF8C8C"/>
    </svg>
  );
  if (name === 'lock') return (<svg viewBox="0 0 24 24" {...s}><rect x="4" y="11" width="16" height="10" rx="1.5"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>);
  if (name === 'eye') return (<svg viewBox="0 0 24 24" {...s}><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12Z"/><circle cx="12" cy="12" r="3"/></svg>);
  if (name === 'eye-off') return (<svg viewBox="0 0 24 24" {...s}><path d="M3 3l18 18"/><path d="M10.6 6.1A11 11 0 0 1 12 6c7 0 11 6 11 6a17 17 0 0 1-3.2 3.8"/><path d="M6.6 6.6A17 17 0 0 0 1 12s4 7 11 7c1.7 0 3.2-.4 4.6-1"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>);
  if (name === 'key') return (<svg viewBox="0 0 24 24" {...s}><circle cx="8" cy="14" r="4"/><path d="M11 11l9-9"/><path d="M16 6l2 2"/><path d="M19 3l2 2"/></svg>);
  return null;
}

/* ============================== HuggingFace fetch ============================== */

async function fetchHfModel(modelId, token) {
  const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
  // Don't encode slashes — HF model IDs are path segments (org/name)
  const encId = modelId.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`https://huggingface.co/api/models/${encId}`, { headers });
  if (!res.ok) throw new Error(`HF ${res.status}`);
  const data = await res.json();
  let cfg = null;
  try {
    const cfgRes = await fetch(`https://huggingface.co/${encId}/raw/main/config.json`, { headers });
    if (cfgRes.ok) cfg = await cfgRes.json();
  } catch (e) {}
  return { data, cfg };
}

function profileFromHf({ data, cfg }) {
  const profile = {};
  if (cfg) {
    profile.hidden_size = cfg.hidden_size;
    profile.num_layers = cfg.num_hidden_layers;
    profile.num_attention_heads = cfg.num_attention_heads;
    profile.num_kv_heads = cfg.num_key_value_heads || cfg.num_attention_heads;
    profile.head_dim = cfg.head_dim || (cfg.hidden_size && cfg.num_attention_heads ? Math.round(cfg.hidden_size / cfg.num_attention_heads) : null);
    profile.vocab_size = cfg.vocab_size;
    profile.max_position_embeddings = cfg.max_position_embeddings;
    profile.architectures = cfg.architectures;
    profile.kv_variant = (profile.num_kv_heads && profile.num_attention_heads && profile.num_kv_heads < profile.num_attention_heads) ? 'GQA' : 'MHA';
  }
  // Params from siblings (safetensors index totals if available)
  if (data?.safetensors?.total) {
    profile.params = Math.round(data.safetensors.total / 1e9);
    profile.params_exact = data.safetensors.total;
  }
  return profile;
}

/* ============================== Main App ============================== */

function App() {
  const [modelId, setModelId] = useState(DEFAULTS.model_id);
  const [modelInput, setModelInput] = useState(DEFAULTS.model_id);
  const [gpu, setGpu] = useState(DEFAULTS.gpu);
  const [workload, setWorkload] = useState(DEFAULTS.workload);
  const [memory, setMemory] = useState(DEFAULTS.memory);
  const [parallelism, setParallelism] = useState(DEFAULTS.parallelism);
  const [vllm, setVllm] = useState(DEFAULTS.vllm);
  const [modelProfile, setModelProfile] = useState({});
  const [hfState, setHfState] = useState('idle'); // idle | loading | ok | err
  const [hfErr, setHfErr] = useState('');
  const [hfToken, setHfToken] = useState(() => {
    try { return localStorage.getItem('gpucalc.hfToken') || ''; } catch (e) { return ''; }
  });
  const [tokenOpen, setTokenOpen] = useState(false);
  const [showToken, setShowToken] = useState(false);
  const [openAcc, setOpenAcc] = useState(null);
  const [starred, setStarred] = useState(false);
  const [toast, setToast] = useState('');

  useEffect(() => {
    try {
      if (hfToken) localStorage.setItem('gpucalc.hfToken', hfToken);
      else localStorage.removeItem('gpucalc.hfToken');
    } catch (e) {}
  }, [hfToken]);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 2200);
  };

  const loadModel = useCallback(async (id) => {
    if (!id) return;
    setModelId(id);
    setHfState('loading');
    setHfErr('');
    try {
      const result = await fetchHfModel(id, hfToken);
      const profile = profileFromHf(result);
      setModelProfile(profile);
      setHfState('ok');
    } catch (e) {
      setHfErr(String(e.message || e));
      setHfState('err');
      // Fallback to guessed profile
      setModelProfile({});
    }
  }, [hfToken]);

  // Auto-fetch on initial mount with default model
  useEffect(() => { loadModel(DEFAULTS.model_id); }, []); // eslint-disable-line

  const state = { model_id: modelId, gpu, workload, memory, parallelism, vllm, modelProfile };
  const est = useMemo(() => computeAll(state), [modelId, gpu, workload, memory, parallelism, vllm, modelProfile]);

  // Did user edit any workload assumption?
  const usingDefaults = JSON.stringify(workload) === JSON.stringify(DEFAULTS.workload)
    && JSON.stringify(memory) === JSON.stringify(DEFAULTS.memory);

  // Build JSON request payload
  const apiPayload = useMemo(() => ({
    model: { model_id: modelId, max_model_len: modelProfile.max_position_embeddings || 'auto_from_model_config' },
    workload: {
      isl_tokens: workload.isl,
      osl_tokens: workload.osl,
      prefix_cache_hit_rate: workload.prefix_cache_hit_rate,
      requests_per_day: workload.requests_per_day,
      peak_multiplier: workload.peak_multiplier,
      target_ttft_ms: workload.target_ttft_ms,
      target_tpot_ms: workload.target_tpot_ms,
    },
    memory: {
      weight_precision: memory.weight_precision.toLowerCase(),
      kv_cache_precision: memory.kv_precision.toLowerCase(),
      gpu_memory_utilization: memory.gpu_mem_utilization,
    },
    hardware: { gpu_type: gpu },
    parallelism: { tensor_parallel_size: parallelism.tensor_parallel },
    engine: {
      runtime: 'vllm',
      block_size: vllm.block_size,
      max_num_seqs: vllm.max_num_seqs,
      max_num_batched_tokens: vllm.max_num_batched_tokens,
      enable_prefix_caching: vllm.enable_prefix_caching,
      enable_chunked_prefill: vllm.enable_chunked_prefill,
    },
  }), [modelId, gpu, workload, memory, parallelism, vllm, modelProfile]);

  const cliCommand = `gpu-calc estimate \\
  --model ${modelId} \\
  --gpu ${gpu} \\
  --isl ${workload.isl} --osl ${workload.osl} \\
  --requests-per-day ${workload.requests_per_day} \\
  --peak ${workload.peak_multiplier} \\
  --weight-precision ${memory.weight_precision.toLowerCase()} \\
  --kv-precision ${memory.kv_precision.toLowerCase()}`;

  const copy = (txt, label) => {
    navigator.clipboard.writeText(txt).then(() => showToast(label + ' copied to clipboard'));
  };

  const resetAll = () => {
    setWorkload(DEFAULTS.workload);
    setMemory(DEFAULTS.memory);
    setParallelism(DEFAULTS.parallelism);
    setVllm(DEFAULTS.vllm);
    setGpu(DEFAULTS.gpu);
    setModelInput(DEFAULTS.model_id);
    loadModel(DEFAULTS.model_id);
  };

  return (
    <>
      {/* Page header */}
      <div className="page-hdr">
        <div className="breadcrumb"><a href="#">Estimate</a><span className="sep">/</span><span>Quick estimate</span></div>
        <div className="page-title-row">
          <h1 className="page-title">Quick estimate</h1>
          <span className="page-tagline">Start with just a model name. We fill the rest.</span>
          <div className="page-actions">
            <button className={`star-btn ${starred ? 'starred' : ''}`} onClick={() => setStarred(s => !s)} title="Add to favorites" aria-label="Favorite">
              <Icon name={starred ? 'star' : 'star-o'} size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="content">
        {/* Input strip */}
        <div className="card content-row" style={{marginBottom: 14}}>
          <div className="input-strip">
            <div className="input-strip-row">
              <div className="input-field">
                <label className="input-field-label">
                  <span className="hf-mini"><Icon name="hf" size={14} /></span>
                  Model <span className="hint">— Hugging Face ID</span>
                  {hfState === 'loading' && <span className="hf-state loading"><span className="spin"></span>fetching</span>}
                  {hfState === 'ok' && <span className="hf-state ok"><Icon name="check" size={11} />auto-detected</span>}
                  {hfState === 'err' && <span className="hf-state err" title={hfErr}>using estimates</span>}
                </label>
                <div className="input-control">
                  <input
                    type="text"
                    value={modelInput}
                    onChange={(e) => setModelInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') loadModel(modelInput); }}
                    placeholder="meta-llama/Llama-3.1-8B-Instruct"
                    list="model-presets"
                  />
                </div>
                <datalist id="model-presets">
                  {MODEL_PRESETS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                </datalist>
              </div>
              <div className="input-field">
                <label className="input-field-label">GPU target</label>
                <div className="input-control select">
                  <select value={gpu} onChange={(e) => setGpu(e.target.value)}>
                    {GPUS.map(g => <option key={g.id} value={g.id}>{g.label}</option>)}
                  </select>
                </div>
              </div>
              <div className="input-action">
                <button className="btn primary" onClick={() => loadModel(modelInput)}>
                  Calculate <Icon name="arrow" size={14} />
                </button>
              </div>
              <div className="input-action">
                <button className="btn subtle" onClick={() => setTokenOpen(o => !o)}>
                  <Icon name="key" size={13} />
                  {hfToken ? 'Token set' : 'Add HF token'}
                </button>
              </div>
            </div>
            {tokenOpen && (
              <div className="hf-token-row">
                <div className="hf-token-toggle">
                  <Icon name="key" size={12} /> HuggingFace access token <span style={{color:'var(--text-3)'}}>· optional, for gated/private models</span>
                </div>
                <div className="hf-token-field">
                  <input
                    type={showToken ? 'text' : 'password'}
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
                    value={hfToken}
                    onChange={(e) => setHfToken(e.target.value)}
                    autoComplete="off"
                    spellCheck="false"
                  />
                  <button className="hf-eye" onClick={() => setShowToken(s => !s)} title={showToken ? 'Hide' : 'Show'}>
                    <Icon name={showToken ? 'eye-off' : 'eye'} size={14} />
                  </button>
                  <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" style={{fontSize:'11px'}}>
                    Get a token <Icon name="external" size={11} />
                  </a>
                </div>
                <div className="hf-token-note">
                  <Icon name="lock" size={11} /> Stored in this browser only — never sent anywhere except huggingface.co.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Warning */}
        {usingDefaults && (
          <div className="warn-strip">
            <span className="ic"><Icon name="warn" size={16} /></span>
            <span className="body"><strong>Based on default workload</strong> — ISL {workload.isl}, OSL {workload.osl}, {memory.kv_precision} KV cache, {(memory.gpu_mem_utilization*100).toFixed(0)}% GPU util, {workload.prefix_cache_hit_rate*100}% prefix-cache hit. Edit assumptions below to make this realistic.</span>
            <a className="anchor" href="#assumptions" onClick={(e) => { e.preventDefault(); setOpenAcc('workload'); document.getElementById('assumptions')?.scrollIntoView({behavior:'smooth', block:'start'}); }}>Customize →</a>
          </div>
        )}

        {/* 4 result tiles */}
        <div className="hero-grid" style={{marginBottom: 16}}>
          <ResultTile
            primary
            icon="gpu"
            label={<HelpTip wide text="How many GPUs you need to serve this workload. Shown as 'likely' with a range, because real traffic isn't deterministic — peak hours, longer prompts, and lower cache hits all swing the answer.">GPUs required</HelpTip>}
            value={
              <>
                <Ticker value={est.gpus} />
                <span className="unit" style={{whiteSpace:'nowrap'}}>likely</span>
              </>
            }
            sub={<><strong style={{color:'#fff'}}>Range: {est.gpu_low}–{est.gpu_high}</strong> · {est.gpuSpec.label.replace('NVIDIA ','')} · {est.concurrent} active req</>}
            spark={<Sparkline data={est.sensitivity} currentIdx={9} />}
            back={
              <>
                <div className="stat-back-title">GPU count math</div>
                <div className="stat-formula">
                  <div>Likely: ⌈ {fmtBytes(est.totalGB)} ÷ {est.usableMem.toFixed(0)} GB ⌉ = <strong>{est.gpus} GPUs</strong></div>
                  <div>Low (P25 workload): <strong>{est.gpu_low}</strong></div>
                  <div>High (P95 workload): <strong>{est.gpu_high}</strong></div>
                  <div className="eq">Usable / GPU = {est.gpuSpec.mem} × {(memory.gpu_mem_utilization*100).toFixed(0)}% × {(100-memory.headroom_pct)}% = <strong>{est.usableMem.toFixed(0)} GB</strong></div>
                </div>
              </>
            }
          />
          <ResultTile
            accent="accent-blue"
            icon="memory"
            label={<HelpTip wide text="Memory needed to hold the model's parameters (its 'brain') in GPU memory. Equals active parameter count × bytes per number. Smaller precisions like FP8 or INT4 cut this in half or more.">Weight memory</HelpTip>}
            value={<><Ticker value={est.weightMemGB} digits={est.weightMemGB < 10 ? 1 : 0} /><span className="unit">GB</span></>}
            sub={`${est.activeParams}B × ${est.weightBytes} bytes · ${memory.weight_precision}`}
            back={
              <>
                <div className="stat-back-title">Weight memory math</div>
                <div className="stat-formula">
                  <div>activeParams × bytes_per_param</div>
                  <div>= <span className="t">{est.activeParams}</span>B × <span className="t">{est.weightBytes}</span> bytes</div>
                  <div className="eq">= <strong>{est.weightMemGB.toFixed(1)} GB</strong></div>
                  {est.activeParams !== est.params && <div style={{color:'var(--text-3)', marginTop:6}}>MoE: {est.params}B total / {est.activeParams}B active per token</div>}
                </div>
              </>
            }
          />
          <ResultTile
            accent="accent-purple"
            icon="cache"
            label={<HelpTip wide text="Memory used to remember past tokens during generation. Grows linearly with sequence length and number of concurrent requests — often the biggest swing factor in GPU sizing.">KV cache / request</HelpTip>}
            value={<><Ticker value={est.kvPerReqGB * 1024} digits={est.kvPerReqGB * 1024 < 10 ? 2 : 0} /><span className="unit">MB</span></>}
            sub={<>{(est.kvBytesPerToken / 1024).toFixed(0)} KB/token · {(workload.isl + workload.osl)} tokens{est.arch.gqa_ratio > 1 ? ` · ${est.arch.gqa_ratio}× smaller than MHA` : ''}</>}
            chip={`${est.arch.kv_variant}${est.numKvHeads !== est.numAttentionHeads ? ` · ${est.numKvHeads}/${est.numAttentionHeads} KV heads` : ''}`}
            chipClass={`stat-chip ${est.arch.kv_variant.toLowerCase()}`}
            back={
              <>
                <div className="stat-back-title">KV cache math</div>
                <div className="stat-formula">
                  <div>2 × layers × kv_heads × head_dim × bytes / TP</div>
                  <div>= 2 × <span className="t">{est.numLayers}</span> × <span className="t">{est.numKvHeads}</span> × <span className="t">{est.headDim}</span> × <span className="t">{est.kvBytes}</span> / <span className="t">{est.tp}</span></div>
                  <div>= <strong>{(est.kvBytesPerToken / 1024).toFixed(1)} KB/token</strong></div>
                  <div style={{marginTop:8, opacity:.85}}>vs MHA: {(est.kvEquivMHA/1024).toFixed(0)} KB · vs MQA: {(est.kvEquivMQA/1024).toFixed(0)} KB</div>
                  <div className="eq">× <span className="t">{(workload.isl + workload.osl)}</span> × (1 − <span className="t">{workload.prefix_cache_hit_rate}</span>) = <strong>{(est.kvPerReqGB * 1024).toFixed(1)} MB / req</strong></div>
                </div>
              </>
            }
          />
          <ResultTile
            accent="accent-gold"
            icon="cash"
            label={<HelpTip text="Estimated monthly compute cost: GPU count × hourly rate × 730 hours per month (assumes 24/7 uptime).">Monthly cost</HelpTip>}
            value={<><span className="currency">$</span><Ticker value={est.monthly} /></>}
            sub={`${gpuSpecOf(gpu).rate.toFixed(2)}/gpu-hr · 730 hrs`}
            back={
              <>
                <div className="stat-back-title">Monthly cost math</div>
                <div className="stat-formula">
                  <div><span className="t">{est.gpus}</span> GPUs × <span className="t">${gpuSpecOf(gpu).rate.toFixed(2)}</span>/hr × <span className="t">730</span> hrs/mo</div>
                  <div className="eq">= <strong>${fmt(est.monthly)} / month</strong></div>
                </div>
              </>
            }
          />
        </div>

        {/* Insight strips: KV scenarios + Why this GPU count */}
        <div className="content-row" style={{gridTemplateColumns:'minmax(0, 1.4fr) minmax(0, 1fr)', display:'grid', gap:16, marginBottom: 16}}>
          <section className="card">
            <div className="card-hdr" style={{padding:'16px 20px 8px', display:'flex', alignItems:'center', gap:10, flexWrap:'wrap'}}>
              <h2 className="card-title" style={{fontFamily:'var(--display)', fontSize:16, fontWeight:500, margin:0}}>
                <HelpTip wide text="KV cache stores the keys and values from previous tokens so the model doesn't have to re-process them. Memory grows linearly with the number of tokens and the number of active requests. These three scenarios show how that memory pressure changes from short prompts to a full-context request.">KV cache scenarios</HelpTip>
              </h2>
              <span className="page-tagline">how memory grows with context</span>
            </div>
            <div style={{padding:'4px 20px 18px'}}>
              <div className="kv-scen">
                <ScenarioTile label={<HelpTip text="Just the prompt's KV cache — the request hasn't generated any tokens yet.">Prompt only</HelpTip>} sub={`ISL = ${workload.isl} tokens`} gb={est.kvScenarios.prompt.gb} color="var(--c-cyan)" />
                <ScenarioTile label={<HelpTip text="Full KV cache for the average request: prompt plus generated output.">Typical request</HelpTip>} sub={`ISL + OSL = ${(workload.isl + workload.osl)} tokens`} gb={est.kvScenarios.typical.gb} color="var(--c-purple)" highlight />
                <ScenarioTile label={<HelpTip wide text="If every active request used the full context window the model supports. This is the upper bound — rarely actual, but useful to know your ceiling.">Worst-case context</HelpTip>} sub={`max_model_len = ${fmt(est.maxModelLen)} tokens`} gb={est.kvScenarios.worst.gb} color="var(--c-orange)" />
              </div>
              <div className="kv-insight">
                <Icon name="warn" size={13} />
                <span>
                  <HelpTip wide text="GQA (Grouped-Query Attention) shares KV heads across multiple query heads, so the KV cache is smaller than full MHA. MQA is the extreme: one KV head per layer. MLA (DeepSeek) compresses KV into a latent vector — even smaller.">
                    <strong>{est.arch.kv_variant} detected</strong>
                  </HelpTip>
                  {est.arch.gqa_ratio > 1 && <> · {est.numKvHeads} KV heads / {est.numAttentionHeads} attention heads · KV cache is <strong>{est.arch.gqa_ratio}× smaller</strong> than full MHA</>}
                  {est.arch.kv_variant === 'MHA' && <> · full multi-head attention · KV cache is at its maximum</>}
                  {est.arch.kv_variant === 'MQA' && <> · single KV head · most aggressive KV compression</>}
                </span>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-hdr" style={{padding:'16px 20px 8px'}}>
              <h2 className="card-title" style={{fontFamily:'var(--display)', fontSize:16, fontWeight:500, margin:0}}>
                <HelpTip wide text="The GPU count isn't a single calculation — it's the worst-case of several constraints. Each row shows whether one of those constraints is comfortably met (OK), getting close (Watch), or actively forcing more GPUs (Bottleneck).">Why this GPU count?</HelpTip>
              </h2>
            </div>
            <div style={{padding:'4px 20px 18px'}}>
              <ConstraintRow label="Memory fit" tip="All model weights, KV cache, and activations fit within the GPU memory budget (90% × headroom)." status={est.constraints.memoryFit} detail={`${est.totalGB.toFixed(1)} / ${(est.gpus * est.usableMem).toFixed(0)} GB usable`} />
              <ConstraintRow label="KV cache fit" tip="How much of usable GPU memory the KV cache occupies. Watch when KV climbs past 70% — long prompts make this the dominant cost." status={est.constraints.kvFit} detail={`${est.kvTotalGB.toFixed(1)} GB (${((est.kvTotalGB/est.totalGB)*100).toFixed(0)}% of memory)`} />
              <ConstraintRow label="Scheduler · max_num_seqs" tip="vLLM's per-engine-step concurrency cap. If your active requests exceed it, new requests queue up and TTFT climbs." status={est.constraints.schedulerFit} detail={`${est.concurrent} / ${vllm.max_num_seqs} sequences`} />
              <ConstraintRow label="Batched tokens · prefills/step" tip="How many prompts can be prefilled in a single engine step. Long prompts (high ISL) reduce this and slow TTFT." status={est.constraints.batchedTokenFit} detail={`${est.prefillsPerStep} prefills/step`} />
              <ConstraintRow label="Tensor parallel minimum" tip="Tensor parallelism (TP) splits the model across N GPUs. You always need at least N GPUs per replica." status={est.constraints.tpFit} detail={`${est.gpus} ≥ TP ${est.tp}`} />
              <div className="driver-row">
                <div className="driver-label">Bottleneck</div>
                <div className="driver-value">{est.primaryDriver === 'model_weight_fit' ? 'none — driven by model weights' : est.primaryDriver.replace(/_/g,' ')}</div>
              </div>
            </div>
          </section>
        </div>

        {/* Range drivers strip */}
        <section className="card" style={{marginBottom: 16}}>
          <div className="card-hdr" style={{padding:'14px 20px 6px', display:'flex', alignItems:'center', gap:10, flexWrap:'wrap'}}>
            <h2 className="card-title" style={{fontFamily:'var(--display)', fontSize:15, fontWeight:500, margin:0, display:'inline-flex', alignItems:'center', gap:6}}>
              <Icon name="parallel" size={14} />
              <HelpTip wide text="Real traffic isn't deterministic — peak hours, longer prompts, and lower cache hits all swing the GPU count. These four factors explain the gap between the low and high estimates, ranked by how much each one moves the needle.">Range drivers</HelpTip>
            </h2>
            <span className="page-tagline">what shifts the answer between {est.gpu_low} and {est.gpu_high} GPUs</span>
          </div>
          <div style={{padding:'8px 20px 16px'}}>
            <div className="driver-list">
              {est.driverImpacts.map((d, i) => (
                <div className="driver-item" key={i}>
                  <div className="driver-rank">{i+1}</div>
                  <div className="driver-text">
                    <div className="driver-name">{d.name}</div>
                    <div className="driver-reason">{d.reason}</div>
                  </div>
                  <div className="driver-impact"><div className="driver-bar" style={{width: Math.min(100, d.impact / Math.max(1, est.driverImpacts[0].impact) * 100) + '%'}}></div></div>
                  <div className="driver-range">{d.low} → {d.high}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Layout + breakdown */}
        <div className="content-row" style={{gridTemplateColumns: 'minmax(0, 1.5fr) minmax(0, 1fr)', display: 'grid', gap: 16}}>
          <section className="card layout-card">
            <div className="card-hdr">
              <h2 className="card-title">Memory layout</h2>
              <div className="right">
                <span className="status ok"><span className="dot"></span>Fits in {est.gpus} GPU{est.gpus !== 1 ? 's' : ''}</span>
              </div>
            </div>
            <div className="layout-body">
              {Array.from({ length: Math.min(est.gpus, 4) }).map((_, gi) => {
                const w = est.weightMemGB / est.gpus;
                const k = est.kvTotalGB / est.gpus;
                const a = est.actGB / est.gpus;
                const tot = est.gpuSpec.mem;
                const wp = (w / tot) * 100;
                const kp = (k / tot) * 100;
                const ap = (a / tot) * 100;
                const free = 100 - wp - kp - ap;
                const util = wp + kp + ap;
                return (
                  <div className="gpu-bar" key={gi}>
                    <div className="gpu-bar-hdr">
                      <div className="gpu-bar-name"><span className="idx">#{gi+1}</span>{est.gpuSpec.label}</div>
                      <div className="gpu-bar-meta"><span className="util">{util.toFixed(0)}%</span> used · {(w+k+a).toFixed(1)} / {tot} GB</div>
                    </div>
                    <div className="stack">
                      <div className="stack-seg weights" style={{width: wp + '%'}}>{wp > 12 && <span className="lbl">Weights</span>}</div>
                      <div className="stack-seg kv" style={{width: kp + '%'}}>{kp > 12 && <span className="lbl">KV cache</span>}</div>
                      <div className="stack-seg act" style={{width: ap + '%'}}>{ap > 12 && <span className="lbl">Activations</span>}</div>
                      <div className="stack-seg headroom" style={{width: Math.max(0, free) + '%'}}>{free > 12 && <span className="lbl">Free</span>}</div>
                    </div>
                  </div>
                );
              })}
              {est.gpus > 4 && (
                <div style={{textAlign:'center', fontFamily:'var(--mono)', fontSize:'12px', color:'var(--text-3)', padding:'8px'}}>
                  + {est.gpus - 4} more {est.gpuSpec.label}
                </div>
              )}
              <div className="legend">
                <span><span className="sw weights"></span>Weights — {fmtBytes(est.weightMemGB)}</span>
                <span><span className="sw kv"></span>KV cache — {fmtBytes(est.kvTotalGB)} ({est.concurrent} reqs)</span>
                <span><span className="sw act"></span>Activations — {fmtBytes(est.actGB)}</span>
                <span><span className="sw headroom"></span>Headroom — {fmtBytes(est.gpus * est.gpuSpec.mem - est.totalGB)}</span>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-hdr" style={{padding: '16px 20px 4px'}}>
              <h2 className="card-title" style={{margin: 0, fontFamily: 'var(--display)', fontSize: 16, fontWeight: 500}}>Breakdown</h2>
            </div>
            <div style={{padding: '8px 0 0'}}>
              <table className="br-table">
                <thead>
                  <tr><th>Item</th><th className="num">Value</th></tr>
                </thead>
                <tbody>
                  <tr><td>Model params (active)</td><td className="num">{est.activeParams}B</td></tr>
                  <tr><td>Layers / KV heads / head dim</td><td className="num">{est.numLayers} / {est.numKvHeads} / {est.headDim}</td></tr>
                  <tr><td>Weight precision</td><td className="num">{memory.weight_precision}</td></tr>
                  <tr><td>KV precision</td><td className="num">{memory.kv_precision}</td></tr>
                  <tr><td>KV bytes/token</td><td className="num">{(est.kvBytesPerToken / 1024).toFixed(1)} KB</td></tr>
                  <tr><td>Tokens/req (ISL+OSL)</td><td className="num">{workload.isl + workload.osl}</td></tr>
                  <tr><td>Concurrent requests</td><td className="num">{est.concurrent}</td></tr>
                  <tr><td>Hourly rate</td><td className="num">${est.hourly.toFixed(2)}/hr</td></tr>
                  <tr className="total"><td>Monthly estimate</td><td className="num">${fmt(est.monthly)}</td></tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>

        {/* Assumptions accordions */}
        <div className="assumptions-hdr" id="assumptions" style={{marginTop: 24}}>
          <h2>Want to change assumptions?</h2>
          <span className="sub">Tap any section to edit · all values auto-update the estimate</span>
          <span style={{marginLeft: 'auto'}}>
            <button className="btn link" onClick={resetAll}>Reset to defaults</button>
          </span>
        </div>
        <div className="assumptions-grid">
          <Accordion
            id="workload"
            open={openAcc === 'workload'}
            onToggle={() => setOpenAcc(openAcc === 'workload' ? null : 'workload')}
            icon="workload"
            title="Workload"
            summary={[
              ['ISL', workload.isl],
              ['OSL', workload.osl],
              ['Req/day', fmt(workload.requests_per_day)],
              ['Peak', workload.peak_multiplier + '×'],
              ['Active', est.concurrent + (workload.concurrent_requests ? '' : ' (auto)')],
              ['Active ratio', (workload.active_request_ratio*100) + '%'],
              ['Prefix-hit', (workload.prefix_cache_hit_rate*100) + '%'],
            ]}
          >
            <WorkloadEditor workload={workload} setWorkload={setWorkload} autoConcurrent={est.autoConcurrent} />
          </Accordion>

          <Accordion
            id="precision"
            open={openAcc === 'precision'}
            onToggle={() => setOpenAcc(openAcc === 'precision' ? null : 'precision')}
            icon="precision"
            title="Precision & memory"
            summary={[
              ['Weights', memory.weight_precision],
              ['KV', memory.kv_precision],
              ['GPU mem util', (memory.gpu_mem_utilization*100).toFixed(0) + '%'],
              ['Headroom', memory.headroom_pct + '%'],
              ['Quant', memory.weight_quantization],
            ]}
          >
            <PrecisionEditor memory={memory} setMemory={setMemory} />
          </Accordion>

          <Accordion
            id="hardware"
            open={openAcc === 'hardware'}
            onToggle={() => setOpenAcc(openAcc === 'hardware' ? null : 'hardware')}
            icon="hardware"
            title="Hardware"
            summary={[
              ['GPU', est.gpuSpec.label],
              ['VRAM/GPU', est.gpuSpec.mem + ' GB'],
              ['$/gpu-hr', '$' + est.gpuSpec.rate.toFixed(2)],
            ]}
          >
            <div className="fields">
              <div className="field">
                <label className="field-label">GPU type</label>
                <div className="field-input select">
                  <select value={gpu} onChange={(e) => setGpu(e.target.value)}>
                    {GPUS.map(g => <option key={g.id} value={g.id}>{g.label} · ${g.rate.toFixed(2)}/hr</option>)}
                  </select>
                </div>
              </div>
            </div>
          </Accordion>

          <Accordion
            id="parallelism"
            open={openAcc === 'parallelism'}
            onToggle={() => setOpenAcc(openAcc === 'parallelism' ? null : 'parallelism')}
            icon="parallel"
            title={<>Parallelism <span className="badge">advanced</span></>}
            summary={[
              ['Tensor parallel', parallelism.tensor_parallel],
              ['Pipeline parallel', parallelism.pipeline_parallel],
              ['Replicas', parallelism.replicas],
            ]}
          >
            <ParallelEditor parallelism={parallelism} setParallelism={setParallelism} />
          </Accordion>

          <Accordion
            id="vllm"
            open={openAcc === 'vllm'}
            onToggle={() => setOpenAcc(openAcc === 'vllm' ? null : 'vllm')}
            icon="engine"
            title={<>vLLM engine <span className="badge">advanced</span></>}
            summary={[
              ['block_size', vllm.block_size],
              ['max_num_seqs', vllm.max_num_seqs],
              ['prefix_caching', vllm.enable_prefix_caching ? 'on' : 'off'],
              ['chunked_prefill', String(vllm.enable_chunked_prefill)],
            ]}
          >
            <VllmEditor vllm={vllm} setVllm={setVllm} />
          </Accordion>

          <Accordion
            id="internals"
            open={openAcc === 'internals'}
            onToggle={() => setOpenAcc(openAcc === 'internals' ? null : 'internals')}
            icon="internals"
            title={<>Model internals <span className="badge">auto-detected</span></>}
            summary={[
              ['Hidden size', modelProfile.hidden_size || '—'],
              ['Layers', modelProfile.num_layers || est.numLayers + ' (est)'],
              ['Attn heads', modelProfile.num_attention_heads || '—'],
              ['KV heads', modelProfile.num_kv_heads || est.numKvHeads + ' (est)'],
              ['Head dim', modelProfile.head_dim || est.headDim + ' (est)'],
            ]}
          >
            <InternalsView profile={modelProfile} hfState={hfState} est={est} modelId={modelId} />
          </Accordion>
        </div>

        {/* Footer action row */}
        <div className="action-row" style={{marginTop: 24}}>
          <div className="left">
            <Icon name="check-circle" size={16} /> Estimate updates live as you edit · last computed just now
          </div>
          <div className="right">
            <button className="btn subtle" onClick={() => copy(JSON.stringify(apiPayload, null, 2), 'API request')}>
              <Icon name="copy" size={13} /> Copy API request
            </button>
            <button className="btn subtle" onClick={() => copy(cliCommand, 'CLI command')}>
              <Icon name="cli" size={13} /> Copy CLI
            </button>
            <button className="btn secondary" onClick={() => exportSheet(state, est, apiPayload, showToast)}>
              <Icon name="sheet" size={13} /> Export to Sheets
            </button>
            <button className="btn primary">
              Save estimate <Icon name="arrow" size={13} />
            </button>
          </div>
        </div>

        {/* API request preview */}
        <details style={{marginTop: 18}}>
          <summary style={{fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-3)', cursor: 'pointer', padding: '8px 0'}}>
            ▸ Preview API request body
          </summary>
          <pre className="copy-code"><code>{JSON.stringify(apiPayload, null, 2)}</code></pre>
        </details>
      </div>

      {toast && (
        <div className="toast">
          <span className="ic"><Icon name="check-circle" size={16} /></span>
          <span>{toast}</span>
        </div>
      )}
    </>
  );
}

function gpuSpecOf(id) { return GPUS.find(g => g.id === id) || GPUS[0]; }

/* ============================== Result tile ============================== */

function ResultTile({ icon, label, value, sub, chip, chipClass, back, primary, accent, spark }) {
  const [flipped, setFlipped] = useState(false);
  return (
    <div
      className={`stat-tile ${primary ? 'primary' : ''} ${accent || ''} ${flipped ? 'flipped' : ''}`}
      onClick={() => setFlipped(f => !f)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped(f => !f); } }}
    >
      <div className="stat-front">
        <div className="stat-icon-wrap"><Icon name={icon} size={16} /></div>
        {spark && <div className="stat-spark">{spark}</div>}
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
        {sub && <div className="stat-sub">{sub}</div>}
        {chip && <span className={chipClass || 'stat-chip'}>{chip}</span>}
        <div className="stat-flip-hint"><Icon name="flip" size={10} /> see math</div>
      </div>
      <div className="stat-back">
        {back}
        <div className="stat-flip-hint" style={{position:'static',marginTop:'auto',alignSelf:'flex-end'}}><Icon name="flip" size={10} /> flip back</div>
      </div>
    </div>
  );
}

function ScenarioTile({ label, sub, gb, color, highlight }) {
  const display = gb < 1 ? (gb * 1024).toFixed(0) + ' MB' : gb < 100 ? gb.toFixed(gb < 10 ? 1 : 0) + ' GB' : gb.toFixed(0) + ' GB';
  return (
    <div className={`scen-tile ${highlight ? 'highlight' : ''}`} style={{ borderLeftColor: color }}>
      <div className="scen-label">{label}</div>
      <div className="scen-value" style={{ color }}>{display}</div>
      <div className="scen-sub">{sub}</div>
    </div>
  );
}

function ConstraintRow({ label, tip, status, detail }) {
  const cls = status === 'ok' ? 'ok' : status === 'watch' ? 'watch' : 'bottleneck';
  const txt = status === 'ok' ? 'OK' : status === 'watch' ? 'Watch' : 'Bottleneck';
  return (
    <div className="constraint-row">
      <span className={`con-pip ${cls}`}></span>
      <div className="con-label">{tip ? <HelpTip text={tip}>{label}</HelpTip> : label}</div>
      <div className="con-detail">{detail}</div>
      <span className={`con-status ${cls}`}>{txt}</span>
    </div>
  );
}

/* ============================== Tooltip ============================== */

function HelpTip({ text, children, side = 'top', wide = false }) {
  return (
    <span className="helptip">
      {children}
      <span className="helptip-trigger" tabIndex={0} aria-label="Help">
        <svg width="11" height="11" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.2"/>
          <path d="M6 6.5a2 2 0 1 1 2.7 1.9c-.5.2-.7.5-.7 1V10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none"/>
          <circle cx="8" cy="12" r="0.6" fill="currentColor"/>
        </svg>
        <span className={`helptip-body ${side} ${wide ? 'wide' : ''}`}>{text}</span>
      </span>
    </span>
  );
}

/* ============================== Sparkline ============================== */

function Sparkline({ data, currentIdx, width = 80, height = 32 }) {
  if (!data || !data.length) return null;
  const maxG = Math.max(...data.map(d => d.gpus));
  const minG = Math.min(...data.map(d => d.gpus));
  const range = Math.max(1, maxG - minG);
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((d.gpus - minG) / range) * (height - 6) - 3;
    return [x, y];
  });
  const linePath = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  const cur = pts[currentIdx] || pts[pts.length - 1];
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <path className="area" d={areaPath} />
      <path className="line" d={linePath} />
      <circle className="dot" cx={cur[0]} cy={cur[1]} r="2.5" />
    </svg>
  );
}

/* ============================== Accordion ============================== */

function Accordion({ id, open, onToggle, icon, title, summary, children }) {
  return (
    <div className={`acc ${open ? 'open' : ''}`} id={id}>
      <button className="acc-hdr" onClick={onToggle}>
        <div className="acc-icon"><Icon name={icon} size={16} /></div>
        <div>
          <div className="acc-title">{title}</div>
          <div className="acc-summary">
            {summary.map(([k, v], i) => (
              <span className="pair" key={i}><span className="k">{k}:</span> <span className="v">{v}</span></span>
            ))}
          </div>
        </div>
        <span className="acc-edit"><Icon name="pencil" size={11} /> {open ? 'Done' : 'Edit'}</span>
        <span className="acc-chev"><Icon name="chev" size={14} /></span>
      </button>
      <div className="acc-body">
        <div className="acc-body-inner">
          <div className="acc-body-content">{children}</div>
        </div>
      </div>
    </div>
  );
}

/* ============================== Editors ============================== */

function WorkloadEditor({ workload, setWorkload, autoConcurrent }) {
  const set = (k, v) => setWorkload({ ...workload, [k]: v });
  return (
    <div className="fields">
      <div className="field">
        <label className="field-label">Input sequence length (ISL)</label>
        <div className="field-input">
          <input type="number" min="1" value={workload.isl} onChange={(e) => set('isl', Math.max(1, +e.target.value || 1))} />
          <span className="field-suffix">tokens</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Output sequence length (OSL)</label>
        <div className="field-input">
          <input type="number" min="1" value={workload.osl} onChange={(e) => set('osl', Math.max(1, +e.target.value || 1))} />
          <span className="field-suffix">tokens</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Requests per day</label>
        <div className="field-input">
          <input type="number" min="1" value={workload.requests_per_day} onChange={(e) => set('requests_per_day', Math.max(1, +e.target.value || 1))} />
          <span className="field-suffix">req/day</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Peak multiplier</label>
        <div className="field-input">
          <input type="number" min="1" step="0.1" value={workload.peak_multiplier} onChange={(e) => set('peak_multiplier', Math.max(1, +e.target.value || 1))} />
          <span className="field-suffix">×</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Concurrent active requests</label>
        <div className="field-input">
          <input type="number" min="0" placeholder={`auto = ${autoConcurrent}`} value={workload.concurrent_requests || ''} onChange={(e) => set('concurrent_requests', +e.target.value || null)} />
          <span className="field-suffix">req</span>
        </div>
        <div className="field-help">{workload.concurrent_requests ? 'manual override' : 'auto from req/day × peak ÷ 86400 s'}</div>
      </div>
      <div className="field">
        <label className="field-label">Active-request ratio</label>
        <div className="field-input">
          <input type="number" min="0" max="1" step="0.05" value={workload.active_request_ratio} onChange={(e) => set('active_request_ratio', clamp(+e.target.value || 0, 0, 1))} />
          <span className="field-suffix">0–1</span>
        </div>
        <div className="field-help">Fraction of concurrent users actively generating tokens</div>
      </div>
      <div className="field">
        <label className="field-label">Prefix-cache hit rate</label>
        <div className="field-input">
          <input type="number" min="0" max="1" step="0.05" value={workload.prefix_cache_hit_rate} onChange={(e) => set('prefix_cache_hit_rate', clamp(+e.target.value || 0, 0, 1))} />
          <span className="field-suffix">0–1</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Target TTFT</label>
        <div className="field-input">
          <input type="number" value={workload.target_ttft_ms} onChange={(e) => set('target_ttft_ms', +e.target.value || 0)} />
          <span className="field-suffix">ms</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Target TPOT</label>
        <div className="field-input">
          <input type="number" value={workload.target_tpot_ms} onChange={(e) => set('target_tpot_ms', +e.target.value || 0)} />
          <span className="field-suffix">ms/tok</span>
        </div>
      </div>
    </div>
  );
}

function PrecisionEditor({ memory, setMemory }) {
  const set = (k, v) => setMemory({ ...memory, [k]: v });
  return (
    <div className="fields">
      <div className="field">
        <label className="field-label">Weight precision</label>
        <div className="pill-row">
          {['BF16','FP16','FP8','INT8','INT4'].map(p => (
            <button key={p} className={`pill ${memory.weight_precision === p ? 'active' : ''}`} onClick={() => set('weight_precision', p)}>{p}</button>
          ))}
        </div>
      </div>
      <div className="field">
        <label className="field-label">KV cache precision</label>
        <div className="pill-row">
          {['FP16','BF16','FP8','INT8'].map(p => (
            <button key={p} className={`pill ${memory.kv_precision === p ? 'active' : ''}`} onClick={() => set('kv_precision', p)}>{p}</button>
          ))}
        </div>
      </div>
      <div className="field">
        <label className="field-label">Weight quantization</label>
        <div className="field-input select">
          <select value={memory.weight_quantization} onChange={(e) => set('weight_quantization', e.target.value)}>
            {['None','AWQ','GPTQ','FP8','BitsAndBytes'].map(q => <option key={q} value={q}>{q}</option>)}
          </select>
        </div>
      </div>
      <div className="field">
        <label className="field-label">GPU memory utilization</label>
        <div className="field-input">
          <input type="range" min="0.5" max="0.99" step="0.01" value={memory.gpu_mem_utilization} onChange={(e) => set('gpu_mem_utilization', +e.target.value)} style={{padding: '8px 10px'}} />
          <span className="field-suffix">{(memory.gpu_mem_utilization*100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="field">
        <label className="field-label">Headroom reserved</label>
        <div className="field-input">
          <input type="number" min="0" max="50" value={memory.headroom_pct} onChange={(e) => set('headroom_pct', clamp(+e.target.value || 0, 0, 50))} />
          <span className="field-suffix">%</span>
        </div>
      </div>
    </div>
  );
}

function ParallelEditor({ parallelism, setParallelism }) {
  const set = (k, v) => setParallelism({ ...parallelism, [k]: v });
  return (
    <div className="fields">
      <div className="field">
        <label className="field-label">Tensor parallel size</label>
        <div className="field-input">
          <input type="text" value={parallelism.tensor_parallel} onChange={(e) => set('tensor_parallel', e.target.value)} placeholder="auto" />
        </div>
        <div className="field-help">Set to <code>auto</code> or a power of 2 (1, 2, 4, 8)</div>
      </div>
      <div className="field">
        <label className="field-label">Pipeline parallel size</label>
        <div className="field-input">
          <input type="number" min="1" value={parallelism.pipeline_parallel} onChange={(e) => set('pipeline_parallel', Math.max(1, +e.target.value || 1))} />
        </div>
      </div>
      <div className="field">
        <label className="field-label">Replicas</label>
        <div className="field-input">
          <input type="number" min="1" value={parallelism.replicas} onChange={(e) => set('replicas', Math.max(1, +e.target.value || 1))} />
        </div>
      </div>
    </div>
  );
}

function VllmEditor({ vllm, setVllm }) {
  const set = (k, v) => setVllm({ ...vllm, [k]: v });
  return (
    <div className="fields">
      <div className="field">
        <label className="field-label">block_size</label>
        <div className="field-input">
          <input type="number" min="1" value={vllm.block_size} onChange={(e) => set('block_size', Math.max(1, +e.target.value || 16))} />
        </div>
      </div>
      <div className="field">
        <label className="field-label">max_num_seqs</label>
        <div className="field-input">
          <input type="number" min="1" value={vllm.max_num_seqs} onChange={(e) => set('max_num_seqs', Math.max(1, +e.target.value || 256))} />
        </div>
      </div>
      <div className="field">
        <label className="field-label">max_num_batched_tokens</label>
        <div className="field-input">
          <input type="text" value={vllm.max_num_batched_tokens} onChange={(e) => set('max_num_batched_tokens', e.target.value)} />
        </div>
      </div>
      <div className="field">
        <label className="field-label">enable_prefix_caching</label>
        <div className="pill-row">
          <button className={`pill ${vllm.enable_prefix_caching ? 'active' : ''}`} onClick={() => set('enable_prefix_caching', true)}>On</button>
          <button className={`pill ${!vllm.enable_prefix_caching ? 'active' : ''}`} onClick={() => set('enable_prefix_caching', false)}>Off</button>
        </div>
      </div>
      <div className="field">
        <label className="field-label">enable_chunked_prefill</label>
        <div className="pill-row">
          {['auto', true, false].map(v => (
            <button key={String(v)} className={`pill ${vllm.enable_chunked_prefill === v ? 'active' : ''}`} onClick={() => set('enable_chunked_prefill', v)}>{v === true ? 'On' : v === false ? 'Off' : 'Auto'}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

function InternalsView({ profile, hfState, est, modelId }) {
  const hasReal = profile && profile.num_layers;
  return (
    <>
      {!hasReal && hfState !== 'loading' && (
        <div style={{padding: '10px 14px', background: 'var(--warn-bg)', border: '1px solid #f0ab00', borderRadius: 4, marginBottom: 16, fontSize: 13, color: '#5c4a00'}}>
          <Icon name="warn" size={13} /> &nbsp; Couldn't auto-detect from HuggingFace. Using approximations. Verify on the model card or add an access token if the repo is gated.{' '}
          <a href={`https://huggingface.co/${modelId}/blob/main/config.json`} target="_blank" rel="noreferrer">View config.json <Icon name="external" size={11} /></a>
        </div>
      )}
      <table className="br-table">
        <thead>
          <tr><th>Field</th><th className="num">Detected</th><th className="num">Used in math</th></tr>
        </thead>
        <tbody>
          <tr><td>Parameters</td><td className="num">{profile.params || '—'}{profile.params ? ' B' : ''}</td><td className="num">{est.activeParams} B</td></tr>
          <tr><td>Hidden size</td><td className="num">{profile.hidden_size || '—'}</td><td className="num">—</td></tr>
          <tr><td>Layers</td><td className="num">{profile.num_layers || '—'}</td><td className="num">{est.numLayers}</td></tr>
          <tr><td>Attention heads</td><td className="num">{profile.num_attention_heads || '—'}</td><td className="num">—</td></tr>
          <tr><td>KV heads</td><td className="num">{profile.num_kv_heads || '—'}</td><td className="num">{est.numKvHeads}</td></tr>
          <tr><td>Head dim</td><td className="num">{profile.head_dim || '—'}</td><td className="num">{est.headDim}</td></tr>
          <tr><td>Max position embeddings</td><td className="num">{profile.max_position_embeddings ? fmt(profile.max_position_embeddings) : '—'}</td><td className="num">—</td></tr>
          <tr><td>Architecture</td><td className="num">{profile.architectures?.[0] || '—'}</td><td className="num">{est.arch.kv_variant}</td></tr>
        </tbody>
      </table>
    </>
  );
}

/* ============================== Export to Sheets ============================== */

function exportSheet(state, est, apiPayload, showToast) {
  // Build a CSV the user can paste into Sheets
  const rows = [
    ['gpu.calc · Quick Estimate'],
    ['Generated', new Date().toISOString()],
    [''],
    ['INPUTS'],
    ['Model', state.model_id],
    ['GPU', state.gpu],
    ['ISL (tokens)', state.workload.isl],
    ['OSL (tokens)', state.workload.osl],
    ['Requests/day', state.workload.requests_per_day],
    ['Peak multiplier', state.workload.peak_multiplier],
    ['Concurrent requests', est.concurrent],
    ['Prefix-cache hit rate', state.workload.prefix_cache_hit_rate],
    ['Weight precision', state.memory.weight_precision],
    ['KV precision', state.memory.kv_precision],
    ['GPU mem utilization', state.memory.gpu_mem_utilization],
    [''],
    ['MODEL INTERNALS'],
    ['Params (active, B)', est.activeParams],
    ['Layers', est.numLayers],
    ['KV heads', est.numKvHeads],
    ['Head dim', est.headDim],
    [''],
    ['RESULTS'],
    ['Weight memory (GB)', est.weightMemGB.toFixed(2)],
    ['KV bytes/token (KB)', (est.kvBytesPerToken/1024).toFixed(2)],
    ['KV cache per req (MB)', (est.kvPerReqGB*1024).toFixed(2)],
    ['KV cache total (GB)', est.kvTotalGB.toFixed(2)],
    ['Activation overhead (GB)', est.actGB.toFixed(2)],
    ['Total VRAM (GB)', est.totalGB.toFixed(2)],
    ['GPUs required', est.gpus],
    ['Utilization', (est.utilization*100).toFixed(0) + '%'],
    ['Hourly cost ($/hr)', est.hourly.toFixed(2)],
    ['Monthly cost ($)', est.monthly],
  ];
  const csv = rows.map(r => r.map(c => {
    const s = String(c == null ? '' : c);
    return s.includes(',') || s.includes('"') ? `"${s.replace(/"/g,'""')}"` : s;
  }).join(',')).join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `gpucalc-${state.model_id.split('/').pop()}-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast && showToast('Sheet (CSV) downloaded — open in Google Sheets');
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
