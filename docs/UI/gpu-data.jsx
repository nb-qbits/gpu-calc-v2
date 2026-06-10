// ─────────────────────────────────────────────────────────────────────────
// Model catalogue. Numbers are real-ish architectural facts pulled from
// model cards. Used to compute weight + KV-cache footprint.
//
// kv_heads × head_dim is the per-token KV state. GQA models (Llama 3+,
// Mistral, Qwen 2.5) collapse this dramatically vs full multi-head.
// ─────────────────────────────────────────────────────────────────────────
const MODELS = [
  { id: 'gemma-2-2b',  family: 'Google',  name: 'Gemma 2',  size: '2B',  paramsB: 2.6,  layers: 26, kvHeads: 4,  headDim: 256 },
  { id: 'gemma-2-9b',  family: 'Google',  name: 'Gemma 2',  size: '9B',  paramsB: 9.2,  layers: 42, kvHeads: 8,  headDim: 256 },
  { id: 'gemma-2-27b', family: 'Google',  name: 'Gemma 2',  size: '27B', paramsB: 27.2, layers: 46, kvHeads: 16, headDim: 128 },
  { id: 'gemma-3-12b', family: 'Google',  name: 'Gemma 3',  size: '12B', paramsB: 12.2, layers: 48, kvHeads: 8,  headDim: 256, badge: 'New' },
  { id: 'gemma-3-27b', family: 'Google',  name: 'Gemma 3',  size: '27B', paramsB: 27.4, layers: 62, kvHeads: 16, headDim: 128, badge: 'New' },
  { id: 'llama-3-8b',  family: 'Meta',    name: 'Llama 3',  size: '8B',  paramsB: 8.0,  layers: 32, kvHeads: 8,  headDim: 128 },
  { id: 'llama-3-70b', family: 'Meta',    name: 'Llama 3',  size: '70B', paramsB: 70.6, layers: 80, kvHeads: 8,  headDim: 128, popular: true },
  { id: 'llama-31-8b', family: 'Meta',    name: 'Llama 3.1',size: '8B',  paramsB: 8.0,  layers: 32, kvHeads: 8,  headDim: 128, popular: true },
  { id: 'llama-31-70b',family: 'Meta',    name: 'Llama 3.1',size: '70B', paramsB: 70.6, layers: 80, kvHeads: 8,  headDim: 128 },
  { id: 'mistral-7b',  family: 'Mistral', name: 'Mistral',  size: '7B',  paramsB: 7.2,  layers: 32, kvHeads: 8,  headDim: 128 },
  { id: 'mixtral-8x7b',family: 'Mistral', name: 'Mixtral 8x7B', size: '47B', paramsB: 46.7, layers: 32, kvHeads: 8, headDim: 128, moe: true, activeB: 12.9 },
  { id: 'qwen-25-7b',  family: 'Qwen',    name: 'Qwen 2.5', size: '7B',  paramsB: 7.6,  layers: 28, kvHeads: 4,  headDim: 128 },
  { id: 'nemotron-340',family: 'NVIDIA',  name: 'Nemotron', size: '340B',paramsB: 340.0,layers: 96, kvHeads: 8,  headDim: 128 },
];

const FAMILIES = ['All', 'Meta', 'Google', 'Mistral', 'NVIDIA', 'Qwen'];

// ─────────────────────────────────────────────────────────────────────────
// GPU catalogue. Cloud price = on-demand hyperscaler average ($/hr).
// Capex = 1× card street price (ignoring chassis/networking).
// ─────────────────────────────────────────────────────────────────────────
const GPUS = [
  { id: 'a10',     name: 'NVIDIA A10',     vramGB: 24,  hourly: 0.75, capex: 3200,  tier: 'workstation' },
  { id: 'l4',      name: 'NVIDIA L4',      vramGB: 24,  hourly: 0.80, capex: 2800,  tier: 'workstation' },
  { id: 'l40s',    name: 'NVIDIA L40S',    vramGB: 48,  hourly: 1.45, capex: 8400,  tier: 'inference' },
  { id: 'a100-40', name: 'NVIDIA A100',    vramGB: 40,  hourly: 1.75, capex: 10500, tier: 'datacenter', variant: '40GB' },
  { id: 'a100-80', name: 'NVIDIA A100',    vramGB: 80,  hourly: 2.20, capex: 15500, tier: 'datacenter', variant: '80GB' },
  { id: 'h100',    name: 'NVIDIA H100',    vramGB: 80,  hourly: 3.40, capex: 30000, tier: 'flagship',   variant: 'SXM' },
  { id: 'h200',    name: 'NVIDIA H200',    vramGB: 141, hourly: 4.65, capex: 38000, tier: 'flagship',   variant: 'SXM' },
  { id: 'b200',    name: 'NVIDIA B200',    vramGB: 192, hourly: 6.20, capex: 52000, tier: 'frontier',   variant: 'SXM' },
];

// ─────────────────────────────────────────────────────────────────────────
// Sizing math.
//
//   weights_GB        = params × bytes_per_param
//   kv_per_token_GB   = 2 (K+V) × layers × kv_heads × head_dim × kv_bytes / 1e9
//   kv_total_GB       = kv_per_token × context × concurrent_users
//   overhead          = 15% activation buffer
//   total_VRAM        = (weights + kv_total) × 1.15
//
// Returns weight/kv breakdown and a sorted list of viable GPU configs.
// ─────────────────────────────────────────────────────────────────────────
const PRECISIONS = {
  fp16: { label: 'FP16', bytes: 2.0,  kvBytes: 2 },
  int8: { label: 'INT8', bytes: 1.0,  kvBytes: 1 },
  int4: { label: 'INT4', bytes: 0.5,  kvBytes: 1 },
};

function computeRequirements({ model, precision, contextTokens, concurrentUsers }) {
  const p = PRECISIONS[precision];
  const effParams = model.moe ? model.paramsB : model.paramsB; // MoE: full weights resident
  const weightsGB = effParams * p.bytes;

  const kvPerTokenBytes = 2 * model.layers * model.kvHeads * model.headDim * p.kvBytes;
  const kvPerUserGB = (kvPerTokenBytes * contextTokens) / 1e9;
  const kvTotalGB = kvPerUserGB * concurrentUsers;

  const subtotal = weightsGB + kvTotalGB;
  const overheadGB = subtotal * 0.15;
  const totalGB = subtotal + overheadGB;

  // GPU configs: smallest count that fits, up to 8 cards.
  const configs = GPUS.map((gpu) => {
    const count = Math.ceil(totalGB / gpu.vramGB);
    const fits = count <= 8;
    // require power of 2 for multi-GPU (tensor parallel reality)
    const validCount = count === 1 || count === 2 || count === 4 || count === 8;
    return {
      gpu,
      count,
      fits: fits && validCount,
      utilization: Math.min(1, totalGB / (count * gpu.vramGB)),
      hourly: gpu.hourly * count,
      capex: gpu.capex * count,
      yearlyCloud: gpu.hourly * count * 24 * 365,
    };
  });

  const viable = configs.filter((c) => c.fits).sort((a, b) => a.hourly - b.hourly);
  const cheapest = viable[0];
  const fastest = viable.slice().sort((a, b) => {
    // proxy: flagship > datacenter > inference > workstation, then VRAM
    const order = { frontier: 0, flagship: 1, datacenter: 2, inference: 3, workstation: 4 };
    return order[a.gpu.tier] - order[b.gpu.tier] || b.gpu.vramGB - a.gpu.vramGB;
  })[0];

  return {
    weightsGB,
    kvPerUserGB,
    kvTotalGB,
    overheadGB,
    totalGB,
    configs: viable,
    cheapest,
    fastest,
  };
}

const CONCURRENCY_PRESETS = [
  { id: 'team',    label: '<10',  value: 8,    sub: 'Small team' },
  { id: 'dept',    label: '30',   value: 30,   sub: 'Department' },
  { id: 'org',     label: '100',  value: 100,  sub: 'Organization' },
  { id: 'plat',    label: '500',  value: 500,  sub: 'Platform' },
  { id: 'ent',     label: '1K+',  value: 1500, sub: 'Enterprise' },
];

const CONTEXT_PRESETS = [
  { id: 'short',  label: 'Short',     tokens: 8192,    sub: '8K' },
  { id: 'medium', label: 'Medium',    tokens: 65536,   sub: '64K' },
  { id: 'long',   label: 'Long',      tokens: 131072,  sub: '128K' },
  { id: 'huge',   label: 'Very long', tokens: 1048576, sub: '1M' },
];

const DEPLOYMENTS = [
  { id: 'cloud',  label: 'Cloud only', sub: 'Pay per hour, no hardware' },
  { id: 'onprem', label: 'On-prem',    sub: 'Own hardware, capex model' },
  { id: 'hybrid', label: 'Hybrid',     sub: 'Own base, rent burst' },
];

// ─────────────────────────────────────────────────────────────────────────
// Formatters
// ─────────────────────────────────────────────────────────────────────────
function fmtGB(gb) {
  if (gb < 1) return `${(gb * 1024).toFixed(0)} MB`;
  if (gb < 10) return `${gb.toFixed(1)} GB`;
  if (gb < 1000) return `${gb.toFixed(0)} GB`;
  return `${(gb / 1024).toFixed(2)} TB`;
}
function fmtMoney(n, opts = {}) {
  const { compact = false } = opts;
  if (n < 0.01) return '$0';
  if (n < 1) return `$${n.toFixed(2)}`;
  if (compact && n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (compact && n >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}
function fmtTokens(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return `${n}`;
}

Object.assign(window, {
  MODELS, FAMILIES, GPUS, PRECISIONS,
  CONCURRENCY_PRESETS, CONTEXT_PRESETS, DEPLOYMENTS,
  computeRequirements, fmtGB, fmtMoney, fmtTokens,
});
