import { ModelSpec } from './models';
import { GpuSpec } from './gpus';

export type Precision = 'fp16' | 'int8' | 'int4' | 'mixed';
export type DeploymentType = 'cloud' | 'onprem' | 'hybrid';
export type ContextLength = 'short' | 'medium' | 'long' | 'verylong';
export type TensorParallelism = 'auto' | 1 | 2 | 4 | 8;

export const CONTEXT_TOKENS: Record<ContextLength, number> = {
  short:    8_192,
  medium:   65_536,
  long:     131_072,
  verylong: 1_048_576,
};

export const CONTEXT_LABELS: Record<ContextLength, string> = {
  short:    'Short',
  medium:   'Medium',
  long:     'Long',
  verylong: 'Very long',
};

export const CONTEXT_SUBLABELS: Record<ContextLength, string> = {
  short:    '8K tokens',
  medium:   '64K tokens',
  long:     '128K tokens',
  verylong: '1M tokens',
};

// Bytes per weight and per KV element for each precision
export const PRECISION_BYTES: Record<Precision, { weight: number; kv: number; label: string; desc: string }> = {
  fp16:  { weight: 2,   kv: 2,   label: 'FP16',  desc: '2 bytes/param'        },
  int8:  { weight: 1,   kv: 1,   label: 'INT8',  desc: '1 byte/param'         },
  int4:  { weight: 0.5, kv: 0.5, label: 'INT4',  desc: '0.5 bytes/param'      },
  mixed: { weight: 1,   kv: 1.5, label: 'Mixed', desc: 'INT8 weights + FP16 KV' },
};

export interface QuickEstimateInput {
  model: ModelSpec;
  concurrentUsers: number;
  contextLength: ContextLength;
  precision: Precision;
  deploymentType: DeploymentType;
  gpu: GpuSpec;
  tensorParallelism: TensorParallelism;
  memUtilization: number; // 0.50 – 0.95
}

export interface SensitivityItem {
  label: string;
  usersCapacity: number;
  changePct: number;
  color: string;
}

export interface QuickEstimateResult {
  // Memory
  weightsGb: number;
  kvBytesPerToken: number;
  kvTotalGb: number;
  totalContextTokens: number;
  activationGb: number;
  totalVramGb: number;
  // Scaling
  gpusPerReplica: number;
  replicasNeeded: number;
  totalGpus: number;
  // Throughput (approx)
  ttftMs: number;
  tpotMs: number;
  systemTps: number; // total tokens/sec for all users
  // Cost
  costPerHour: number;
  costPerDay: number;
  costPerMonth: number;
  // On-prem
  hardwareCost: number;
  electricityPerMonth: number;
  breakEvenMonths: number;
  // Meta
  cloudAvailabilityPct: number;
  tpuAvailabilityPct: number;
  idleServerCostsPerMonth: number;
  // Sensitivity
  sensitivity: SensitivityItem[];
}

function resolveTP(tp: TensorParallelism, gpusNeeded: number): number {
  if (tp !== 'auto') return tp;
  // Auto: smallest power-of-2 that is <= gpus needed
  for (const t of [1, 2, 4, 8]) {
    if (t >= gpusNeeded) return t;
  }
  return 8;
}

export function runQuickEstimate(input: QuickEstimateInput): QuickEstimateResult {
  const { model, concurrentUsers, contextLength, precision, gpu, tensorParallelism, memUtilization } = input;
  const contextTokens = CONTEXT_TOKENS[contextLength];
  const { weight: bytesPerWeight, kv: bytesPerKV } = PRECISION_BYTES[precision];

  // --- Memory ---
  const weightsGb = model.paramsBillions * model.activeFraction * bytesPerWeight;
  // KV cache: 2 × layers × hiddenSize × bytesPerKVElem per token
  const kvBytesPerToken = 2 * model.numLayers * model.hiddenSize * bytesPerKV;
  const totalContextTokens = concurrentUsers * contextTokens;
  const kvTotalGb = (kvBytesPerToken * totalContextTokens) / 1e9;
  const activationGb = weightsGb * 0.2;
  const totalVramGb = weightsGb + kvTotalGb + activationGb;

  // --- Scaling ---
  const effectiveVramGb = gpu.vramGb * memUtilization;
  const minGpus = Math.ceil(totalVramGb / effectiveVramGb);
  const tp = resolveTP(tensorParallelism, minGpus);
  const gpusPerReplica = Math.max(tp, Math.ceil(minGpus / tp) * tp);
  const replicasNeeded = Math.max(2, 1); // always recommend ≥2 for HA
  const totalGpus = gpusPerReplica * replicasNeeded;

  // --- Throughput (roofline approximations) ---
  // TPOT: KV-read dominated at average context position
  const kvBytesPerStep = kvBytesPerToken * concurrentUsers * (contextTokens / 2);
  const kvBytesPerGpuPerStep = kvBytesPerStep / gpusPerReplica;
  const tpotMs = (kvBytesPerGpuPerStep / (gpu.bandwidthTbps * 1e12 * 0.5)) * 1000;

  // TTFT: compute-dominated prefill (calibrated to ~20% of theoretical FLOPS peak)
  const prefillFlops = 12 * model.paramsBillions * 1e9 * contextTokens;
  const ttftMs = prefillFlops / (gpu.tflops * 1e12 * gpusPerReplica * 0.19) * 1000;

  // System throughput
  const systemTps = Math.round((concurrentUsers * 1000) / tpotMs);

  // --- Cost ---
  const costPerHour = totalGpus * gpu.pricePerHour;
  const costPerDay = costPerHour * 24;
  const costPerMonth = costPerHour * 730; // avg hours/month
  const idleServerCostsPerMonth = costPerMonth * 0.30; // 30% idle assumption

  // --- On-prem ---
  const hardwareCost = totalGpus * gpu.hardwareCostPerGpu;
  const PUE = 1.4; // power usage effectiveness
  const electricityRate = 0.12; // $/kWh
  const electricityPerMonth = (totalGpus * gpu.powerWatts * PUE * 24 * 30) / 1000 * electricityRate;
  const breakEvenMonths = Math.round(hardwareCost / costPerMonth * 10) / 10;

  // --- Sensitivity analysis ---
  const sensitivity: SensitivityItem[] = buildSensitivity(input, concurrentUsers, gpusPerReplica, totalVramGb, gpu);

  return {
    weightsGb, kvBytesPerToken, kvTotalGb, totalContextTokens,
    activationGb, totalVramGb,
    gpusPerReplica, replicasNeeded, totalGpus,
    ttftMs, tpotMs, systemTps,
    costPerHour, costPerDay, costPerMonth,
    hardwareCost, electricityPerMonth, breakEvenMonths,
    cloudAvailabilityPct: gpu.cloudAvailabilityPct,
    tpuAvailabilityPct: gpu.tpuAvailabilityPct,
    idleServerCostsPerMonth,
    sensitivity,
  };
}

function buildSensitivity(
  input: QuickEstimateInput,
  baseUsers: number,
  gpusPerReplica: number,
  baseVramGb: number,
  gpu: GpuSpec
): SensitivityItem[] {
  const items: SensitivityItem[] = [];
  const { precision, model } = input;

  // For each alternative precision, compute users that fit in the same VRAM budget
  const hardwareBudgetGb = gpusPerReplica * gpu.vramGb * input.memUtilization;

  function usersForPrecision(p: Precision): number {
    const { weight: bw, kv: bkv } = PRECISION_BYTES[p];
    const wGb = model.paramsBillions * model.activeFraction * bw;
    const actGb = wGb * 0.2;
    const tokensAvail = (hardwareBudgetGb - wGb - actGb) * 1e9 /
      (2 * model.numLayers * model.hiddenSize * bkv * CONTEXT_TOKENS[input.contextLength]);
    return Math.max(0, Math.floor(tokensAvail));
  }

  const alternatives: Array<{ label: string; key: Precision; color: string }> = [
    { label: 'Use INT8',  key: 'int8',  color: '#ec7a08' },
    { label: 'Use INT4',  key: 'int4',  color: '#0066cc' },
    { label: 'Use Mixed', key: 'mixed', color: '#3e8635' },
  ];

  for (const alt of alternatives) {
    if (alt.key === precision) continue;
    const users = usersForPrecision(alt.key);
    const pct = Math.round(((users - baseUsers) / baseUsers) * 100);
    items.push({ label: alt.label, usersCapacity: users, changePct: pct, color: alt.color });
  }

  // NeMo framework: ~25% overhead reduction (software optimization)
  const nemoUsers = Math.round(baseUsers * 1.25);
  items.push({
    label: 'Use NVIDIA NeMo',
    usersCapacity: nemoUsers,
    changePct: 25,
    color: '#009596',
  });

  return items.sort((a, b) => b.usersCapacity - a.usersCapacity);
}
