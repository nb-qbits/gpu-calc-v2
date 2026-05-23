"use client";

import React, { useState, useMemo, useRef, useCallback, useEffect } from "react";
import {
  Alert,
  Button,
  Card, CardBody, CardTitle,
  DescriptionList, DescriptionListDescription, DescriptionListGroup, DescriptionListTerm,
  ExpandableSection,
  Flex, FlexItem,
  FormSelect, FormSelectOption,
  Grid, GridItem,
  PageSection,
  Progress,
  Slider,
  Spinner,
  Stack, StackItem,
  Text, TextContent, TextVariants,
  TextInput,
  Tile,
  Title,
  ToggleGroup, ToggleGroupItem,
} from "@patternfly/react-core";
import {
  ArrowLeftIcon,
  BoltIcon,
  ChartBarIcon,
  CogIcon,
  CubeIcon,
  DollarSignIcon,
  MemoryIcon,
  MicrochipIcon,
  PencilAltIcon,
  ServerIcon,
  TrendUpIcon,
} from "@patternfly/react-icons";

import {
  MODEL_CATALOG,
  GPU_CATALOG,
  DEFAULT_GPU,
  CONTEXT_TOKENS,
  CONTEXT_LABELS,
  CONTEXT_SUBLABELS,
  contextToDeployParams,
  extractConfig,
  detectKVCategory,
  computeKVCacheResult,
  computeKVMemory,
  computeRecurrentState,
  computeMemoryBudget,
  computeGPUCount,
  resolveAutoTP,
  resolveActCoeff,
  estimateWeightMemoryBytes,
  KV_CATEGORY_LABELS,
  type ContextLength,
  type DeploymentType,
  type TensorParallelism,
  type ExtractedConfig,
  type EngineResult,
  type WeightMemorySource,
  type ModelFamilies,
} from "@/lib/gpu-math";
import type { GpuSpec } from "@/lib/gpu-math/gpus";
import type { ModelSpec } from "@/lib/gpu-math/models";

import modelFamiliesJson from "@/lib/data/model-families.json";
import vllmVersionsJson  from "@/lib/data/vllm-versions.json";

const MODEL_FAMILIES = modelFamiliesJson as ModelFamilies;
const VLLM_VERSION   = "0.17.0"; // default vLLM version assumption

// ─── constants ────────────────────────────────────────────────────────────────

const NEW_MODEL_IDS = new Set(
  MODEL_CATALOG.filter((m) => m.isNew).map((m) => m.id)
);

const USER_PRESETS = [
  { label: "<10",  value: 9,    sub: "Small team"  },
  { label: "30",   value: 30,   sub: "Department"  },
  { label: "100",  value: 100,  sub: "Org"         },
  { label: "500",  value: 500,  sub: "Platform"    },
  { label: "1K+",  value: 1000, sub: "Enterprise"  },
] as const;

const DEPLOYMENT_OPTIONS = [
  { key: "cloud",  label: "Cloud only",   sub: "Pay per hour, no hardware" },
  { key: "onprem", label: "On-prem only", sub: "Own hardware, capex model" },
  { key: "hybrid", label: "Hybrid",       sub: "Own base, rent burst"      },
] as const;

type FetchStatus = "idle" | "loading" | "success" | "gated" | "not_found" | "network_error";

// ─── helpers ──────────────────────────────────────────────────────────────────

function fmtGb(gb: number): string {
  if (gb >= 1000) return `${(gb / 1000).toFixed(1)} TB`;
  if (gb >= 1)    return `${gb.toFixed(1)} GB`;
  return `${(gb * 1024).toFixed(0)} MB`;
}

function fmtMoney(usd: number): string {
  if (usd >= 1_000_000) return `$${(usd / 1_000_000).toFixed(1)}M`;
  if (usd >= 1_000)     return `$${Math.round(usd / 1_000)}k`;
  return `$${usd.toFixed(0)}`;
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

function fmtBytes(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(2)} MB`;
  if (bytes >= 1_024)     return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${bytes.toFixed(0)} B`;
}

function fmtMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms.toFixed(0)}ms`;
}

// ─── Engine runner ────────────────────────────────────────────────────────────
// Pure function — no side effects. Called in useMemo whenever inputs change.

interface EngineInput {
  extractedConfig: ExtractedConfig
  weightBytes:     number
  weightSource:    WeightMemorySource
  concurrentUsers: number
  contextLength:   ContextLength
  tp:              number
  blockSize:       16 | 32 | 64 | 128
  kvCacheDtype:    "auto" | "fp8"
  gpu:             GpuSpec
  memUtilization:  number
}

function runEngine(input: EngineInput): EngineResult {
  const {
    extractedConfig: cfg, weightBytes, weightSource,
    concurrentUsers, contextLength, tp,
    blockSize, kvCacheDtype, gpu, memUtilization,
  } = input;

  const { ISL, OSL, max_model_len } = contextToDeployParams(CONTEXT_TOKENS[contextLength]);
  const actCoeff = resolveActCoeff(VLLM_VERSION, vllmVersionsJson.configs);
  const weightDtype = cfg.quantization_config?.type === "fp8" ? "float8_e4m3fn" : cfg.dtype;

  const baseParams = {
    max_model_len, max_num_seqs: concurrentUsers,
    gpu_memory_utilization: memUtilization,
    ISL, OSL, block_size: blockSize,
    kv_cache_dtype: kvCacheDtype === "fp8" ? "fp8" : undefined,
    mamba_ssm_cache_dtype: "float32" as const,
  };

  const detection = detectKVCategory(cfg, MODEL_FAMILIES);

  // Resolve TP — auto mode tries 1/2/4/8 and picks the minimum that fits
  let resolvedTP = tp;
  let autoTPUsed = false;
  if (tp === 0) {
    const kvResult0 = computeKVCacheResult(cfg, detection, { ...baseParams, tp: 1 }, MODEL_FAMILIES, weightDtype);
    const kvMem0    = computeKVMemory(kvResult0, cfg, { ...baseParams, tp: 1 }, MODEL_FAMILIES);
    const rec0      = computeRecurrentState(cfg, { ...baseParams, tp: 1 }, MODEL_FAMILIES);
    resolvedTP = resolveAutoTP(cfg, weightBytes, kvMem0, rec0, gpu.vramGb, baseParams, actCoeff, MODEL_FAMILIES);
    autoTPUsed = true;
  }

  const deploy = { ...baseParams, tp: resolvedTP };
  const tpValidation = { // inline — validateTP is also called inside computeKVCacheResult
    is_valid: true, hard_reject: false, warn_kv_split: false,
    kv_heads_per_gpu: Math.max(1, Math.floor(cfg.H_kv / resolvedTP)),
    kv_replication: resolvedTP > cfg.H_kv,
    kv_tp_mode: (resolvedTP > cfg.H_kv ? "replicated" : "sharded") as "replicated" | "sharded",
    tp_inflection: cfg.H_kv,
    tp_gives_kv_benefit: resolvedTP <= cfg.H_kv,
  };

  const kv       = computeKVCacheResult(cfg, detection, deploy, MODEL_FAMILIES, weightDtype);
  const kvMemory = computeKVMemory(kv, cfg, deploy, MODEL_FAMILIES);
  const recurrent = computeRecurrentState(cfg, deploy, MODEL_FAMILIES);

  const finalWeightBytes = weightBytes > 0
    ? weightBytes
    : estimateWeightMemoryBytes(cfg);

  const budget   = computeMemoryBudget(finalWeightBytes, kvMemory, recurrent, deploy, cfg, gpu.vramGb, actCoeff);
  const gpuCount = computeGPUCount(budget, gpu.vramGb, deploy);

  return {
    weight_bytes: finalWeightBytes, weight_source: weightSource,
    weight_confidence: weightBytes > 0 ? "exact" : "medium",
    weight_warnings: [],
    kv, kv_memory: kvMemory,
    recurrent: recurrent ?? null,
    budget, gpu_count: gpuCount,
    resolved_tp: resolvedTP, tp_validation: tpValidation,
    auto_tp_used: autoTPUsed,
  };
}

// Derives legacy-compatible QuickEstimateResult shape from EngineResult
// so all existing UI components render without changes.
interface LegacyResult {
  weightsGb:         number
  kvBytesPerToken:   number
  kvTotalGb:         number
  totalContextTokens: number
  activationGb:      number
  totalVramGb:       number
  gpusPerReplica:    number
  replicasNeeded:    number
  totalGpus:         number
  ttftMs:            number
  tpotMs:            number
  systemTps:         number
  costPerHour:       number
  costPerDay:        number
  costPerMonth:      number
  hardwareCost:      number
  electricityPerMonth: number
  breakEvenMonths:   number
  cloudAvailabilityPct: number
  tpuAvailabilityPct: number
  idleServerCostsPerMonth: number
  sensitivity:       SensitivityItem[]
  // New fields exposed to the flip card back-faces
  kvCategory:        string
  kvFormula:         string
  kvCategoryLabel:   string
  kvOptimisticGb:    number
  kvExpectedGb:      number
  kvConservativeGb:  number
  resolvedTP:        number
  autoTPUsed:        boolean
  weightSource:      WeightMemorySource
  engineWarnings:    string[]
}

interface SensitivityItem {
  label:         string
  usersCapacity: number
  changePct:     number
  color:         string
}

function toLegacy(
  engine: EngineResult,
  gpu: GpuSpec,
  concurrentUsers: number,
  contextLength: ContextLength,
  deploymentType: DeploymentType,
): LegacyResult {
  const b = engine.budget;
  const g = engine.gpu_count;

  const weightsGb     = b.weight_memory_total / 1e9;
  const kvExpectedGb  = b.kv_memory.expected / 1e9;
  const activationGb  = b.activation_per_gpu / 1e9;
  const totalVramGb   = b.total_per_gpu_expected / 1e9;

  const gpusPerReplica = g.expected;
  const totalGpus      = gpusPerReplica * 2; // always recommend ≥2 replicas for HA

  const contextTokens  = CONTEXT_TOKENS[contextLength];
  const totalContextTokens = concurrentUsers * contextTokens;

  // Throughput — roofline approximation using corrected per-GPU weight
  const weightPerGpuGb    = b.weight_per_gpu / 1e9;
  const kvBytesPerStep    = engine.kv.kv_bytes_per_token * concurrentUsers * (contextTokens / 2);
  const kvBytesPerGpuStep = kvBytesPerStep / gpusPerReplica;
  const tpotMs = (kvBytesPerGpuStep / (gpu.memoryBandwidthGbps * 1e9 * 0.5)) * 1000;

  const prefillFlops = 12 * (b.weight_memory_total / 1e9) * 1e9 * contextTokens;
  const ttftMs = prefillFlops / (gpu.tflops * 1e12 * gpusPerReplica * 0.19) * 1000;
  const systemTps = Math.round((concurrentUsers * 1000) / Math.max(tpotMs, 0.001));

  // Cost
  const costPerHour  = totalGpus * gpu.pricePerHour;
  const costPerMonth = costPerHour * 730;
  const isOnPrem     = deploymentType !== "cloud";
  const PUE = 1.4, electricityRate = 0.12;
  const hardwareCost        = totalGpus * gpu.hardwareCostPerGpu;
  const electricityPerMonth = (totalGpus * gpu.powerWatts * PUE * 24 * 30) / 1000 * electricityRate;
  const breakEvenMonths     = costPerMonth > 0
    ? Math.round(hardwareCost / (isOnPrem ? hardwareCost / 36 + electricityPerMonth : costPerMonth) * 10) / 10
    : 0;

  // Sensitivity — quantisation alternatives
  const sensitivity: SensitivityItem[] = [
    {
      label: "Use FP8 KV cache",
      usersCapacity: Math.round(concurrentUsers * 1.8),
      changePct: 80,
      color: "#0066cc",
    },
    {
      label: "Reduce context 50%",
      usersCapacity: Math.round(concurrentUsers * 1.9),
      changePct: 90,
      color: "#ec7a08",
    },
    {
      label: "Add 1 GPU per replica",
      usersCapacity: Math.round(concurrentUsers * 1.35),
      changePct: 35,
      color: "#3e8635",
    },
  ].sort((a, b) => b.usersCapacity - a.usersCapacity);

  const engineWarnings = [
    ...engine.kv.warnings,
    ...engine.gpu_count.warnings,
    ...engine.weight_warnings,
  ];

  return {
    weightsGb,
    kvBytesPerToken:    engine.kv.kv_bytes_per_token,
    kvTotalGb:          kvExpectedGb,
    totalContextTokens,
    activationGb,
    totalVramGb,
    gpusPerReplica,
    replicasNeeded:     2,
    totalGpus,
    ttftMs,
    tpotMs,
    systemTps,
    costPerHour,
    costPerDay:         costPerHour * 24,
    costPerMonth,
    hardwareCost,
    electricityPerMonth,
    breakEvenMonths,
    cloudAvailabilityPct: gpu.cloudAvailabilityPct,
    tpuAvailabilityPct:   gpu.tpuAvailabilityPct,
    idleServerCostsPerMonth: costPerMonth * 0.30,
    sensitivity,
    kvCategory:         engine.kv.kv_category,
    kvFormula:          engine.kv.formula,
    kvCategoryLabel:    engine.kv.kv_category_label,
    kvOptimisticGb:     engine.budget.kv_memory.optimistic / 1e9,
    kvExpectedGb:       engine.budget.kv_memory.expected / 1e9,
    kvConservativeGb:   engine.budget.kv_memory.conservative / 1e9,
    resolvedTP:         engine.resolved_tp,
    autoTPUsed:         engine.auto_tp_used,
    weightSource:       engine.weight_source,
    engineWarnings,
  };
}

// ─── state ────────────────────────────────────────────────────────────────────

interface PageState {
  view:              "estimate" | "results";
  selectedModelId:   string;
  hfModelId:         string;        // custom HF input
  concurrentUsers:   number;
  customUsersText:   string;
  contextLength:     ContextLength;
  deploymentType:    DeploymentType;
  tensorParallelism: TensorParallelism;
  memUtilization:    number;
  blockSize:         16 | 32 | 64 | 128;
  kvCacheDtype:      "auto" | "fp8";
  advancedOpen:      boolean;
  controlsVisible:   boolean;
  expandedSections:  Record<string, boolean>;
  gpuId:             string;
  expandedTile:      "users" | "context" | null;
  // HF fetch state
  fetchStatus:       FetchStatus;
  extractedConfig:   ExtractedConfig | null;
  weightBytes:       number;
  weightSource:      WeightMemorySource;
  fetchWarnings:     string[];
  showHfToken:       boolean;
  hfToken:           string;
}

const DEFAULT_STATE: PageState = {
  view:              "estimate",
  selectedModelId:   "llama-3.1-8b",
  hfModelId:         "",
  concurrentUsers:   10,
  customUsersText:   "",
  contextLength:     "short",
  deploymentType:    "onprem",
  tensorParallelism: "auto",
  memUtilization:    0.9,
  blockSize:         16,
  kvCacheDtype:      "auto",
  advancedOpen:      false,
  controlsVisible:   true,
  expandedSections: {
    modelWeights: true, kvCache: true, vramBudget: true,
    precision: false, gpuCapability: false, throughput: false,
    cost: false, sensitivity: true,
  },
  gpuId:             "h100-sxm-80gb",
  expandedTile:      null,
  fetchStatus:       "idle",
  extractedConfig:   null,
  weightBytes:       0,
  weightSource:      "estimated",
  fetchWarnings:     [],
  showHfToken:       false,
  hfToken:           "",
};

// ─── FlipCard ─────────────────────────────────────────────────────────────────

function FlipCard({
  front, back, height = 180, frontClassName = "", backClassName = "",
}: {
  front: React.ReactNode;
  back: React.ReactNode;
  height?: number | string;
  frontClassName?: string;
  backClassName?: string;
}) {
  const [flipped, setFlipped] = useState(false);
  const toggle = () => setFlipped((f) => !f);
  return (
    <div
      className={`rh-flip-card${flipped ? " rh-flip-card--flipped" : ""}`}
      style={{ height }}
      onClick={toggle}
      role="button"
      tabIndex={0}
      aria-pressed={flipped}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      }}
    >
      <div className="rh-flip-card__inner">
        <div className={`rh-flip-card__face rh-flip-card__face--front ${frontClassName}`}>
          {front}
        </div>
        <div className={`rh-flip-card__face rh-flip-card__face--back ${backClassName}`}>
          {back}
        </div>
      </div>
    </div>
  );
}

// ─── MemoryGrid ───────────────────────────────────────────────────────────────

function MemoryGrid({ weightsGb, kvGb, totalGb }: {
  weightsGb: number;
  kvGb: number;
  totalGb: number;
}) {
  const COLS = 22, ROWS = 9, CELLS = COLS * ROWS;
  const safeTotal = totalGb > 0 ? totalGb : 1;
  const weightCells = Math.round((weightsGb / safeTotal) * CELLS);
  const kvCells = Math.min(CELLS - weightCells, Math.round((kvGb / safeTotal) * CELLS));
  return (
    <div className="rh-mem-grid">
      {Array.from({ length: CELLS }, (_, i) => {
        let bg: string;
        if (i < weightCells)               bg = "var(--rh-red)";
        else if (i < weightCells + kvCells) bg = "var(--rh-red-muted, rgba(238,0,0,0.22))";
        else                               bg = "var(--rh-gray-20)";
        return <div key={i} className="rh-mem-cell" style={{ background: bg }} />;
      })}
    </div>
  );
}

// ─── LiveEstimatePanel ────────────────────────────────────────────────────────

function LiveEstimatePanel({
  result, state, selectedGpu, onGpuChange, fetchStatus,
}: {
  result:       LegacyResult | null;
  state:        PageState;
  selectedGpu:  GpuSpec;
  onGpuChange:  (gpuId: string) => void;
  fetchStatus:  FetchStatus;
}) {
  const isOnPrem = state.deploymentType !== "cloud";
  const gpuLabel = selectedGpu.name.replace("NVIDIA ", "");

  const monoSm: React.CSSProperties = {
    fontFamily: "var(--font-mono)", fontSize: "0.72rem", lineHeight: 1.8,
  };
  const label: React.CSSProperties = {
    fontFamily: "var(--font-mono)", fontSize: "0.6rem",
    letterSpacing: "0.07em", opacity: 0.65, marginBottom: 4, display: "block",
  };
  const flipHint: React.CSSProperties = {
    marginTop: "auto", paddingTop: 4, textAlign: "right",
    fontFamily: "var(--font-mono)", fontSize: "0.6rem", opacity: 0.5,
  };

  const onPremMonthly = result ? result.hardwareCost / 36 + result.electricityPerMonth : 0;
  const displayMonthly = result
    ? (isOnPrem ? onPremMonthly : result.costPerMonth)
    : 0;
  const ratePerGpuHr = result
    ? (isOnPrem ? onPremMonthly / (result.totalGpus * 730) : selectedGpu.pricePerHour)
    : 0;
  const deployLabel = isOnPrem ? "ON-PREM" : "CLOUD";

  const totalVramAvailable = result ? result.totalGpus * selectedGpu.vramGb : 0;
  const utilPct = totalVramAvailable > 0 && result
    ? Math.round((result.totalVramGb / totalVramAvailable) * 100)
    : 0;
  const freeGb = Math.max(0, totalVramAvailable - (result?.totalVramGb ?? 0));

  // Non-success placeholder states — always keep the sidebar visible
  if (!result) {
    let statusIcon = "var(--rh-gray-40)";
    let statusLabel = "LIVE ESTIMATE";
    let body: React.ReactNode;

    if (fetchStatus === "loading") {
      statusLabel = "FETCHING MODEL…";
      body = <Spinner size="lg" />;
    } else if (fetchStatus === "gated") {
      statusIcon = "#f0ab00";
      statusLabel = "GATED MODEL";
      body = (
        <div style={{ textAlign: "center", padding: "0 12px" }}>
          <div style={{ fontSize: "2rem", marginBottom: 8 }}>🔒</div>
          <div style={{ fontFamily: "var(--font-body)", fontSize: "0.82rem", color: "var(--rh-gray-60)", lineHeight: 1.5 }}>
            This model requires a HuggingFace access token.
          </div>
          <div style={{ marginTop: 8, fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "#795600", background: "#fff8e6", border: "1px solid #f0ab00", borderRadius: 6, padding: "6px 10px" }}>
            Enter your token in the form on the left and retry.
          </div>
        </div>
      );
    } else if (fetchStatus === "not_found") {
      statusIcon = "var(--rh-red)";
      statusLabel = "MODEL NOT FOUND";
      body = (
        <div style={{ textAlign: "center", padding: "0 12px" }}>
          <div style={{ fontSize: "2rem", marginBottom: 8 }}>❓</div>
          <div style={{ fontFamily: "var(--font-body)", fontSize: "0.82rem", color: "var(--rh-gray-60)", lineHeight: 1.5 }}>
            Model not found on HuggingFace. Check the model ID and try again.
          </div>
        </div>
      );
    } else if (fetchStatus === "network_error") {
      statusIcon = "var(--rh-red)";
      statusLabel = "NETWORK ERROR";
      body = (
        <div style={{ textAlign: "center", padding: "0 12px" }}>
          <div style={{ fontSize: "2rem", marginBottom: 8 }}>⚠️</div>
          <div style={{ fontFamily: "var(--font-body)", fontSize: "0.82rem", color: "var(--rh-gray-60)", lineHeight: 1.5 }}>
            Could not reach HuggingFace. Check your connection and try again.
          </div>
        </div>
      );
    } else {
      // idle
      body = <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>Select a model to begin</span>;
    }

    return (
      <div className="rh-qe-sidebar">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", color: "var(--rh-gray-60)" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: statusIcon, display: "inline-block" }} />
            {statusLabel}
          </span>
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--rh-gray-40)" }}>
          {body}
        </div>
        <div style={{ marginTop: "auto", paddingTop: 12, borderTop: "1px solid var(--rh-gray-20)" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)", letterSpacing: "0.07em", marginBottom: 4 }}>GPU</div>
          <FormSelect value={state.gpuId} onChange={(_e, v) => onGpuChange(v)} aria-label="Select GPU" style={{ fontSize: "0.8rem" }}>
            {GPU_CATALOG.map((g) => <FormSelectOption key={g.id} value={g.id} label={g.name} />)}
          </FormSelect>
        </div>
      </div>
    );
  }

  return (
    <div className="rh-qe-sidebar">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", color: "var(--rh-gray-60)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--rh-red)", display: "inline-block" }} />
          LIVE ESTIMATE
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)" }}>
          {result.autoTPUsed ? `auto TP=${result.resolvedTP}` : `TP=${result.resolvedTP}`}
        </span>
      </div>

      {/* KV category badge */}
      <div style={{ marginBottom: 8, fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-60)", background: "var(--rh-gray-10)", borderRadius: 4, padding: "3px 8px", width: "fit-content" }}>
        {result.kvCategoryLabel}
      </div>

      {/* Cost card */}
      <FlipCard height={140} frontClassName="rh-flip-card__face--red" backClassName="rh-flip-card__face--dark"
        front={
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.08em", opacity: 0.8 }}>ESTIMATED MONTHLY</span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", background: "rgba(255,255,255,0.25)", borderRadius: 3, padding: "1px 6px" }}>{deployLabel}</span>
            </div>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "2.4rem", lineHeight: 1.1, marginTop: 4 }}>
              <span style={{ fontSize: "1.1rem", verticalAlign: "super", marginRight: 1 }}>$</span>
              {Math.round(displayMonthly).toLocaleString()}
            </div>
            <div style={{ fontSize: "0.7rem", opacity: 0.8, marginTop: 4 }}>
              {result.totalGpus} × {gpuLabel} · ${ratePerGpuHr.toFixed(2)}/hr · 730 hrs/mo
            </div>
            <div style={flipHint}>↺ SEE MATH</div>
          </>
        }
        back={
          <>
            <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>MONTHLY COST MATH</span>
            <div style={monoSm}>
              {result.totalGpus} × <span style={{ color: "var(--rh-red)" }}>${ratePerGpuHr.toFixed(2)}</span>/gpu-hr × 730 hrs
              <br />= <span style={{ color: "var(--rh-red)", fontWeight: 700 }}>{fmtMoney(displayMonthly)}</span> / month
            </div>
            <div style={{ marginTop: 8, fontSize: "0.62rem", color: "var(--rh-gray-40)" }}>
              {isOnPrem ? "On-prem · amortized 36 months + electricity" : "Cloud on-demand · 100% uptime"}
            </div>
            <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
          </>
        }
      />

      {/* GPU count + VRAM */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
        <FlipCard height={155}
          front={
            <>
              <span style={label}>GPU COUNT</span>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.4rem", lineHeight: 1.1 }}>
                {result.totalGpus}<span style={{ fontWeight: 400, fontSize: "0.75rem", marginLeft: 4 }}>× {gpuLabel}</span>
              </div>
              <div style={{ fontSize: "0.68rem", color: "var(--rh-gray-60)", marginTop: 2 }}>{utilPct}% utilized</div>
              <div style={flipHint}>↺ SEE MATH</div>
            </>
          }
          back={
            <>
              <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>GPU COUNT MATH</span>
              <div style={monoSm}>
                [{Math.round(result.totalVramGb)} ÷ {selectedGpu.vramGb}GB]<br />
                = <span style={{ color: "var(--rh-red)", fontWeight: 700 }}>{result.totalGpus} GPUs</span>
              </div>
              <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
            </>
          }
        />
        <FlipCard height={155}
          front={
            <>
              <span style={label}>TOTAL VRAM</span>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.4rem", lineHeight: 1.1 }}>
                {Math.round(result.totalVramGb)}<span style={{ fontWeight: 400, fontSize: "0.9rem", marginLeft: 4 }}>GB</span>
              </div>
              <div style={{ fontSize: "0.68rem", color: "var(--rh-gray-60)", marginTop: 2 }}>model + KV cache</div>
              <div style={flipHint}>↺ SEE MATH</div>
            </>
          }
          back={
            <>
              <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>VRAM MATH</span>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", lineHeight: 1.55 }}>
                Weights: {fmtGb(result.weightsGb)}<br />
                KV cache: {fmtGb(result.kvTotalGb)}<br />
                Activation: {fmtGb(result.activationGb)}<br />
                = <span style={{ color: "var(--rh-red)", fontWeight: 700 }}>{fmtGb(result.totalVramGb)}</span>
              </div>
              <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
            </>
          }
        />
      </div>

      {/* Memory layout */}
      <div className="rh-qe-mem-wrap" style={{ marginTop: 8, flex: 1, display: "flex", flexDirection: "column" }}>
        <FlipCard height="100%"
          front={
            <>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={label}>MEMORY LAYOUT</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)" }}>
                  {result.totalGpus} × {gpuLabel} {selectedGpu.vramGb}GB
                </span>
              </div>
              <MemoryGrid weightsGb={result.weightsGb} kvGb={result.kvTotalGb} totalGb={totalVramAvailable} />
              <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: "0.65rem", color: "var(--rh-gray-60)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-red)", display: "inline-block" }} /> Weights
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-red-muted, rgba(238,0,0,0.22))", display: "inline-block" }} /> KV cache
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-gray-20)", display: "inline-block" }} /> Free
                </span>
              </div>
              <div style={flipHint}>↺ SEE MATH</div>
            </>
          }
          back={
            <>
              <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>PER-GPU SPLIT</span>
              <div style={{ fontSize: "0.85rem", lineHeight: 2.1 }}>
                <span style={{ color: "var(--rh-red)", fontWeight: 500 }}>{fmtGb(result.weightsGb / result.totalGpus)}</span> weights<br />
                <span style={{ color: "var(--rh-red-dark)", fontWeight: 500 }}>{fmtGb(result.kvTotalGb / result.totalGpus)}</span> KV cache<br />
                <span style={{ color: "var(--rh-gray-60)", fontWeight: 500 }}>{fmtGb(freeGb / result.totalGpus)}</span> free headroom
              </div>
              {state.concurrentUsers > 0 && (
                <div style={{ marginTop: 8, fontSize: "0.65rem", color: "var(--rh-gray-40)" }}>
                  Each user adds ~{fmtGb(result.kvTotalGb / state.concurrentUsers / result.totalGpus)} of KV cache.
                </div>
              )}
              <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
            </>
          }
        />
      </div>

      {/* GPU selector */}
      <div style={{ marginTop: "auto", paddingTop: 12, borderTop: "1px solid var(--rh-gray-20)" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)", letterSpacing: "0.07em", marginBottom: 4 }}>GPU</div>
        <FormSelect value={state.gpuId} onChange={(_e, v) => onGpuChange(v)} aria-label="Select GPU" style={{ fontSize: "0.8rem" }}>
          {GPU_CATALOG.map((g) => <FormSelectOption key={g.id} value={g.id} label={g.name} />)}
        </FormSelect>
      </div>
    </div>
  );
}

// ─── ConfigPanel ──────────────────────────────────────────────────────────────

function ConfigPanel({
  state, setState, onLoadModel,
}: {
  state:        PageState;
  setState:     React.Dispatch<React.SetStateAction<PageState>>;
  onLoadModel:  (hfId: string) => void;
}) {
  const galleryRef   = useRef<HTMLDivElement>(null);
  const presetMatch  = USER_PRESETS.find((p) => p.value === state.concurrentUsers);
  const contextLabel = CONTEXT_LABELS[state.contextLength];
  const contextSub   = CONTEXT_SUBLABELS[state.contextLength];

  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId);

  const contextBars: Record<ContextLength, string> = {
    short: "—", medium: "——", long: "———", verylong: "————",
  };

  function toggleTile(tile: "users" | "context") {
    setState((s) => ({ ...s, expandedTile: s.expandedTile === tile ? null : tile }));
  }

  function handleModelSelect(model: ModelSpec) {
    setState((s) => ({ ...s, selectedModelId: model.id, hfModelId: "" }));
    onLoadModel(model.hfId);
  }

  function handleCustomLoad() {
    const id = state.hfModelId.trim();
    if (!id || !id.includes("/")) return;
    setState((s) => ({ ...s, selectedModelId: "" }));
    onLoadModel(id);
  }

  const tpValue = state.tensorParallelism === "auto" ? 0 : state.tensorParallelism;

  return (
    <div>
      {/* Hero */}
      <div style={{ marginBottom: "2.5rem" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", color: "var(--rh-gray-60)", marginBottom: "1rem" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--rh-red)", display: "inline-block" }} />
          QUICK ESTIMATE · ~30 SECONDS
        </div>
        <h1 style={{ fontFamily: "var(--font-display)", fontSize: "clamp(2rem, 4vw, 2.75rem)", fontWeight: 700, lineHeight: 1.15, margin: 0, marginBottom: "0.75rem" }}>
          Size your <span style={{ color: "var(--rh-red)" }}>LLM</span> deployment.
        </h1>
        <p style={{ fontSize: "1rem", color: "var(--rh-gray-60)", margin: 0, lineHeight: 1.5 }}>
          Pick a model, tell us your load. The estimate fills in on the right — tap any tile to flip and see the math.
        </p>
      </div>

      {/* Step 01: Model */}
      <div style={{ marginBottom: "2.5rem" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "1rem" }}>
            <span className="rh-step-number">01</span>
            <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>What model are you serving?</h2>
          </div>
          {state.fetchStatus === "success" && (
            <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.7rem", fontWeight: 500, color: "var(--rh-red)" }}>
              <span style={{ fontSize: "0.8rem" }}>✓</span> SET
            </span>
          )}
        </div>
        <p style={{ fontSize: "0.875rem", color: "var(--rh-gray-60)", margin: "0 0 1rem" }}>
          Choose from common models, or paste any HuggingFace model ID below.
        </p>

        {/* Model gallery */}
        <div className="rh-model-gallery" ref={galleryRef}>
          {MODEL_CATALOG.map((model) => {
            const selected = model.id === state.selectedModelId && state.fetchStatus === "success";
            return (
              <div
                key={model.id}
                className={`rh-model-card${selected ? " rh-model-card--selected" : ""}`}
                onClick={() => handleModelSelect(model)}
                role="button" tabIndex={0} aria-pressed={selected}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleModelSelect(model); }
                }}
              >
                {selected && <span style={{ position: "absolute", top: 7, right: 8, color: "var(--rh-red)", fontSize: "0.85rem" }}>✓</span>}
                {state.fetchStatus === "loading" && model.id === state.selectedModelId && (
                  <span style={{ position: "absolute", top: 7, right: 8 }}><Spinner size="sm" /></span>
                )}
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.55rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 4, textTransform: "uppercase" }}>
                  {model.vendor}
                </div>
                <div style={{ fontWeight: 500, fontSize: "0.85rem", lineHeight: 1.3 }}>{model.name}</div>
                <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
                  <span style={{ fontSize: "0.7rem", color: "var(--rh-gray-60)" }}>{model.paramLabel}</span>
                  {model.isNew && <span className="rh-badge-new">NEW</span>}
                  {model.tags?.map((t) => (
                    <span key={t} style={{ fontSize: "0.6rem", background: "var(--rh-gray-10)", border: "1px solid var(--rh-gray-20)", borderRadius: 3, padding: "1px 4px", color: "var(--rh-gray-60)" }}>{t}</span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected status */}
        {state.fetchStatus === "success" && selectedModel && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: "0.75rem", background: "var(--rh-gray-10)", border: "1px solid var(--rh-gray-20)", borderRadius: 20, padding: "6px 12px", fontSize: "0.8rem", width: "fit-content" }}>
            <span style={{ color: "var(--rh-red)", fontWeight: 500 }}>✓</span>
            <span>
              Selected: <strong>{selectedModel.name}</strong>
              {state.extractedConfig && (
                <span style={{ color: "var(--rh-gray-60)" }}>
                  {" "}· {KV_CATEGORY_LABELS[state.extractedConfig ? "KV-1" : "KV-1"]}
                </span>
              )}
            </span>
          </div>
        )}

        {/* Fetch error alerts */}
        {state.fetchStatus === "not_found" && (
          <Alert variant="danger" isInline title="Model not found" style={{ marginTop: "0.75rem" }}>
            Could not find this model on HuggingFace. Check the model ID and try again.
          </Alert>
        )}
        {state.fetchStatus === "network_error" && (
          <Alert variant="danger" isInline title="Network error" style={{ marginTop: "0.75rem" }}>
            Could not reach HuggingFace. Check your connection and try again.
          </Alert>
        )}

        {/* HuggingFace custom input */}
        <div style={{ marginTop: "1rem", background: "var(--pf-v5-global--BackgroundColor--100, #fff)", border: "1px solid var(--rh-gray-20)", borderRadius: 8, padding: "14px 16px" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-60)", marginBottom: 8 }}>
            OR PASTE A HUGGINGFACE MODEL ID
          </div>
          <div className="rh-hf-row" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <TextInput
              value={state.hfModelId}
              onChange={(_e, v) => setState((s) => ({ ...s, hfModelId: v }))}
              onKeyDown={(e) => { if (e.key === "Enter") handleCustomLoad(); }}
              placeholder="meta-llama/Llama-3.1-70B-Instruct"
              aria-label="HuggingFace model ID"
              style={{ flex: 1, fontFamily: "var(--font-mono)", fontSize: "0.82rem", background: "white" }}
            />
            <Button
              variant="secondary"
              onClick={handleCustomLoad}
              isDisabled={!state.hfModelId.trim().includes("/")}
              isLoading={state.fetchStatus === "loading" && !!state.hfModelId}
              style={{ flexShrink: 0 }}
            >
              Load
            </Button>
            <a href="https://huggingface.co/models" target="_blank" rel="noopener noreferrer" style={{ fontSize: "0.8rem", color: "var(--rh-red)", whiteSpace: "nowrap", flexShrink: 0 }}>
              Browse on HF →
            </a>
          </div>

          {/* HF token — only shown after a 403 */}
          {state.showHfToken && (
            <div style={{ marginTop: 12, padding: "10px 12px", background: "#fff8e6", border: "1px solid #f0ab00", borderRadius: 6 }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 600, marginBottom: 6, color: "#795600" }}>
                This model is gated. Enter your HuggingFace access token to continue.
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <TextInput
                  type="password"
                  value={state.hfToken}
                  onChange={(_e, v) => setState((s) => ({ ...s, hfToken: v }))}
                  placeholder="hf_..."
                  aria-label="HuggingFace access token"
                  style={{ flex: 1, fontSize: "0.82rem", fontFamily: "var(--font-mono)" }}
                />
                <Button variant="secondary" onClick={handleCustomLoad} isDisabled={!state.hfToken}>
                  Retry
                </Button>
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--rh-gray-60)", marginTop: 6 }}>
                Your token is sent directly to HuggingFace and never stored.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Step 02: Load profile */}
      <div style={{ marginBottom: "2.5rem" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "1rem" }}>
            <span className="rh-step-number">02</span>
            <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>Load profile</h2>
          </div>
          <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: "0.7rem", fontWeight: 500, color: "var(--rh-red)" }}>
            <span style={{ fontSize: "0.8rem" }}>✓</span> SET
          </span>
        </div>
        <p style={{ fontSize: "0.875rem", color: "var(--rh-gray-60)", margin: "0 0 1rem" }}>
          Tap the pencil to adjust either value.
        </p>

        <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
          {/* Concurrent users tile */}
          <div className={`rh-load-tile${state.expandedTile === "users" ? " rh-load-tile--active" : ""}`} onClick={() => toggleTile("users")}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 4 }}>CONCURRENT USERS</div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.75rem", lineHeight: 1 }}>{state.concurrentUsers}</div>
                <div style={{ fontSize: "0.72rem", color: "var(--rh-gray-60)", marginTop: 4 }}>{presetMatch?.sub ?? "custom"}</div>
              </div>
              <button style={{ background: state.expandedTile === "users" ? "#fff0f0" : "var(--rh-gray-10)", border: "1px solid var(--rh-gray-20)", borderRadius: "50%", width: 28, height: 28, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }} aria-label="Edit concurrent users" onClick={(e) => { e.stopPropagation(); toggleTile("users"); }}>
                <PencilAltIcon style={{ fontSize: "0.7rem", color: state.expandedTile === "users" ? "var(--rh-red)" : "var(--rh-gray-60)" }} />
              </button>
            </div>
            {state.expandedTile === "users" && (
              <div style={{ marginTop: 14, borderTop: "1px dashed var(--rh-gray-20)", paddingTop: 12 }} onClick={(e) => e.stopPropagation()}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 8 }}>
                  {USER_PRESETS.map((p) => (
                    <div key={p.value} className={`rh-load-tile-option${state.concurrentUsers === p.value ? " rh-load-tile-option--selected" : ""}`} onClick={() => setState((s) => ({ ...s, concurrentUsers: p.value, customUsersText: "" }))}>
                      <div style={{ fontWeight: 700, fontSize: "1rem" }}>{p.label}</div>
                      <div style={{ fontSize: "0.65rem", color: "var(--rh-gray-60)", marginTop: 1 }}>{p.sub}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--rh-gray-60)", flexShrink: 0 }}>Or type exact:</span>
                  <TextInput type="number" placeholder="e.g. 75" value={state.customUsersText}
                    onChange={(_e, v) => { const n = parseInt(v, 10); setState((s) => ({ ...s, customUsersText: v, concurrentUsers: isNaN(n) || n <= 0 ? s.concurrentUsers : n })); }}
                    style={{ flex: 1, fontSize: "0.82rem" }} aria-label="Custom concurrent users" />
                </div>
              </div>
            )}
          </div>

          {/* Conversation length tile */}
          <div className={`rh-load-tile${state.expandedTile === "context" ? " rh-load-tile--active" : ""}`} onClick={() => toggleTile("context")}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 4 }}>CONVERSATION LENGTH</div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.75rem", lineHeight: 1 }}>{contextSub.replace(" tokens", "")}</div>
                <div style={{ fontSize: "0.72rem", color: "var(--rh-gray-60)", marginTop: 4 }}>{contextLabel.toLowerCase()}</div>
              </div>
              <button style={{ background: state.expandedTile === "context" ? "#fff0f0" : "var(--rh-gray-10)", border: "1px solid var(--rh-gray-20)", borderRadius: "50%", width: 28, height: 28, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }} aria-label="Edit conversation length" onClick={(e) => { e.stopPropagation(); toggleTile("context"); }}>
                <PencilAltIcon style={{ fontSize: "0.7rem", color: state.expandedTile === "context" ? "var(--rh-red)" : "var(--rh-gray-60)" }} />
              </button>
            </div>
            {state.expandedTile === "context" && (
              <div style={{ marginTop: 14, borderTop: "1px dashed var(--rh-gray-20)", paddingTop: 12 }} onClick={(e) => e.stopPropagation()}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                    <div key={cl} className={`rh-load-tile-option${state.contextLength === cl ? " rh-load-tile-option--selected" : ""}`} onClick={() => setState((s) => ({ ...s, contextLength: cl }))}>
                      <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{CONTEXT_LABELS[cl]}</div>
                      <div style={{ fontSize: "0.65rem", color: "var(--rh-gray-60)", marginTop: 1 }}>{CONTEXT_SUBLABELS[cl]}</div>
                      <div style={{ marginTop: 4, fontSize: "0.6rem", color: state.contextLength === cl ? "var(--rh-red)" : "var(--rh-gray-40)", letterSpacing: 2 }}>{contextBars[cl]}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Deployment type */}
        <div style={{ marginTop: "1.25rem" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 8 }}>DEPLOYMENT TYPE</div>
          <div style={{ display: "flex", gap: 8 }}>
            {DEPLOYMENT_OPTIONS.map((dt) => (
              <button key={dt.key} onClick={() => setState((s) => ({ ...s, deploymentType: dt.key }))} style={{ padding: "7px 14px", border: `1.5px solid ${state.deploymentType === dt.key ? "var(--rh-red)" : "var(--rh-gray-20)"}`, borderRadius: 5, background: state.deploymentType === dt.key ? "var(--rh-red-50)" : "var(--pf-v5-global--BackgroundColor--100)", color: state.deploymentType === dt.key ? "var(--rh-red)" : "var(--rh-gray-60)", fontWeight: state.deploymentType === dt.key ? 600 : 400, cursor: "pointer", fontSize: "0.8rem", transition: "all 0.12s" }}>
                {dt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="rh-qe-footer">
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.8rem", color: "var(--rh-gray-60)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: state.fetchStatus === "success" ? "var(--rh-red)" : "var(--rh-gray-40)", display: "inline-block" }} />
          {state.fetchStatus === "success" ? "Ready · estimate is live on the right" : "Select a model to see the estimate"}
        </div>
        <Button variant="primary" onClick={() => setState((s) => ({ ...s, view: "results" }))} isDisabled={state.fetchStatus !== "success"} style={{ fontSize: "0.9rem", padding: "10px 20px" }}>
          See full breakdown →
        </Button>
      </div>
    </div>
  );
}

// ─── ResultsView helpers ──────────────────────────────────────────────────────

function SummaryCard({ icon, title, value, sub, isDanger }: {
  icon: React.ReactNode; title: string; value: string; sub: string; isDanger?: boolean;
}) {
  return (
    <Card isFlat isFullHeight>
      <CardBody>
        <Stack>
          <StackItem><Text component={TextVariants.small} className="pf-v5-u-color-200">{icon} {title}</Text></StackItem>
          <StackItem><Title headingLevel="h3" size="2xl" className={isDanger ? "pf-v5-u-primary-color-100" : ""}>{value}</Title></StackItem>
          <StackItem><Text component={TextVariants.small} className="pf-v5-u-color-200">{sub}</Text></StackItem>
        </Stack>
      </CardBody>
    </Card>
  );
}

function ResultSection({ id, icon, title, subtitle, expanded, onToggle, children }: {
  id: string; icon: React.ReactNode; title: string; subtitle: string;
  expanded: boolean; onToggle: (id: string, open: boolean) => void; children: React.ReactNode;
}) {
  return (
    <Card isFlat className="pf-v5-u-mb-sm">
      <ExpandableSection
        toggleContent={
          <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
            <FlexItem><span className="pf-v5-u-primary-color-100">{icon}</span></FlexItem>
            <FlexItem>
              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>{title}</Text>
              <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>{subtitle}</Text>
            </FlexItem>
          </Flex>
        }
        isExpanded={expanded}
        onToggle={(_e, open) => onToggle(id, open)}
      >
        <CardBody>{children}</CardBody>
      </ExpandableSection>
    </Card>
  );
}

// ─── ResultsView ──────────────────────────────────────────────────────────────

function ResultsView({ state, setState, result, selectedGpu }: {
  state:       PageState;
  setState:    React.Dispatch<React.SetStateAction<PageState>>;
  result:      LegacyResult;
  selectedGpu: GpuSpec;
}) {
  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId);
  const modelLabel    = selectedModel?.name ?? state.hfModelId;
  const isOnPrem      = state.deploymentType !== "cloud";

  function toggleSection(id: string, open: boolean) {
    setState((s) => ({ ...s, expandedSections: { ...s.expandedSections, [id]: open } }));
  }

  const maxSensBar = Math.max(...result.sensitivity.map((s) => s.usersCapacity), state.concurrentUsers, 1);

  return (
    <PageSection>
      <Flex justifyContent={{ default: "justifyContentSpaceBetween" }} alignItems={{ default: "alignItemsCenter" }} className="pf-v5-u-mb-md">
        <FlexItem>
          <Button variant="link" icon={<ArrowLeftIcon />} onClick={() => setState((s) => ({ ...s, view: "estimate" }))}>
            Back to estimate
          </Button>
        </FlexItem>
        <FlexItem>
          <Button variant="secondary" icon={<CogIcon />} onClick={() => setState((s) => ({ ...s, controlsVisible: !s.controlsVisible }))}>
            {state.controlsVisible ? "Hide controls" : "Show controls"}
          </Button>
        </FlexItem>
      </Flex>

      <Title headingLevel="h1" size="xl" className="pf-v5-u-mb-lg">
        GPU requirements for {modelLabel}
      </Title>

      {result.engineWarnings.length > 0 && (
        <Alert variant="warning" isInline title="Estimation notes" className="pf-v5-u-mb-md">
          <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
            {result.engineWarnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </Alert>
      )}

      <Grid hasGutter>
        <GridItem span={12} lg={state.controlsVisible ? 8 : 12}>
          <Stack hasGutter>
            <StackItem>
              <Grid hasGutter>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<ServerIcon />} title="GPUs needed (expected)" value={`${result.gpusPerReplica}`} sub={`× ${selectedGpu.name.replace("NVIDIA ", "")}`} isDanger />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<MicrochipIcon />} title="GPU availability" value={`${result.cloudAvailabilityPct}%`} sub={`TPU availability: ${result.tpuAvailabilityPct}%`} />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<DollarSignIcon />} title="Monthly cost" value={fmtMoney(isOnPrem ? result.hardwareCost / 36 + result.electricityPerMonth : result.costPerMonth)} sub={isOnPrem ? "/mo on-prem" : "/mo cloud"} />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<TrendUpIcon />} title="Idle server costs" value={fmtMoney(result.idleServerCostsPerMonth)} sub="30-min+ idle" />
                </GridItem>
              </Grid>
            </StackItem>

            {/* Model weights flip card */}
            <StackItem>
              <FlipCard height={210}
                front={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><ServerIcon /></span></FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>Model weights</Text>
                        <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>How much memory the model needs</Text>
                      </FlexItem>
                    </Flex>
                    <DescriptionList columnModifier={{ default: "3Col" }}>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Total weight memory</DescriptionListTerm>
                        <DescriptionListDescription><span className="pf-v5-u-font-weight-bold">{fmtGb(result.weightsGb)}</span></DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Per GPU (TP={result.resolvedTP})</DescriptionListTerm>
                        <DescriptionListDescription><span className="pf-v5-u-font-weight-bold">{fmtGb(result.weightsGb / result.resolvedTP)}</span></DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Source</DescriptionListTerm>
                        <DescriptionListDescription><span className="pf-v5-u-font-weight-bold" style={{ textTransform: "capitalize" }}>{result.weightSource.replace(/_/g, " ")}</span></DescriptionListDescription>
                      </DescriptionListGroup>
                    </DescriptionList>
                    <div className="rh-flip-card__hint">↻ Flip for formula</div>
                  </>
                }
                back={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><ServerIcon /></span></FlexItem>
                      <FlexItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Weight memory</Text></FlexItem>
                    </Flex>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm">
                      Total: {fmtGb(result.weightsGb)} (from {result.weightSource.replace(/_/g, " ")})
                    </Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">
                      Per GPU: {fmtGb(result.weightsGb)} ÷ {result.resolvedTP} = {fmtGb(result.weightsGb / result.resolvedTP)}
                    </Text>
                    <div className="rh-flip-card__hint">↻ Flip back</div>
                  </>
                }
              />
            </StackItem>

            {/* KV cache flip card */}
            <StackItem>
              <FlipCard height={280}
                front={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><MemoryIcon /></span></FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>KV cache</Text>
                        <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>
                          {result.kvCategoryLabel}
                        </Text>
                      </FlexItem>
                    </Flex>
                    <DescriptionList columnModifier={{ default: "3Col" }}>
                      <DescriptionListGroup>
                        <DescriptionListTerm>KV/token</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtBytes(result.kvBytesPerToken)}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Expected total</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtGb(result.kvExpectedGb)}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Scenario range</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtGb(result.kvOptimisticGb)} – {fmtGb(result.kvConservativeGb)}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                    </DescriptionList>
                    <div className="rh-flip-card__hint">↻ Flip for formula</div>
                  </>
                }
                back={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><MemoryIcon /></span></FlexItem>
                      <FlexItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">KV cache formula · {result.kvCategory}</Text></FlexItem>
                    </Flex>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm">{result.kvFormula}</Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">= {fmtBytes(result.kvBytesPerToken)}/token</Text>
                    <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "12ch" }} className="pf-v5-u-mt-sm">
                      <DescriptionListGroup>
                        <DescriptionListTerm>Optimistic</DescriptionListTerm>
                        <DescriptionListDescription>{fmtGb(result.kvOptimisticGb)} (prompt only)</DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Expected</DescriptionListTerm>
                        <DescriptionListDescription>{fmtGb(result.kvExpectedGb)} (prompt + reply)</DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Conservative</DescriptionListTerm>
                        <DescriptionListDescription>{fmtGb(result.kvConservativeGb)} (full context window)</DescriptionListDescription>
                      </DescriptionListGroup>
                    </DescriptionList>
                    <div className="rh-flip-card__hint">↻ Flip back</div>
                  </>
                }
              />
            </StackItem>

            {/* VRAM budget */}
            <StackItem>
              <ResultSection id="vramBudget" icon={<CubeIcon />} title="VRAM budget breakdown" subtitle="Total per-GPU memory" expanded={state.expandedSections.vramBudget} onToggle={toggleSection}>
                <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "22ch" }}>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Model weights (per GPU)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.weightsGb / result.resolvedTP)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>KV cache — expected</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.kvTotalGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Activation memory</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.activationGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm><span className="pf-v5-u-font-weight-bold">Total per GPU</span></DescriptionListTerm>
                    <DescriptionListDescription><span className="pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">{fmtGb(result.totalVramGb)}</span></DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </ResultSection>
            </StackItem>

            {/* Throughput */}
            <StackItem>
              <ResultSection id="throughput" icon={<TrendUpIcon />} title="Throughput" subtitle="Estimated serving performance" expanded={state.expandedSections.throughput} onToggle={toggleSection}>
                <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "20ch" }}>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Prefill (TTFT)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtMs(result.ttftMs)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Decode (TPOT)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtMs(result.tpotMs)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Tokens/sec (system)</DescriptionListTerm>
                    <DescriptionListDescription>{result.systemTps.toLocaleString()}</DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </ResultSection>
            </StackItem>

            {/* Cost */}
            <StackItem>
              <ResultSection id="cost" icon={<DollarSignIcon />} title="Cost" subtitle="Detailed cost breakdown" expanded={state.expandedSections.cost} onToggle={toggleSection}>
                <Grid hasGutter>
                  <GridItem span={12} md={6}>
                    <Card isFlat>
                      <CardTitle>GPU costs</CardTitle>
                      <CardBody>
                        <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "12ch" }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Per hour</DescriptionListTerm>
                            <DescriptionListDescription>${result.costPerHour.toFixed(2)}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Per month</DescriptionListTerm>
                            <DescriptionListDescription>${Math.round(result.costPerMonth).toLocaleString()}</DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>
                  <GridItem span={12} md={6}>
                    <Card isFlat>
                      <CardTitle>On-prem estimate</CardTitle>
                      <CardBody>
                        <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "14ch" }}>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Hardware</DescriptionListTerm>
                            <DescriptionListDescription>{fmtMoney(result.hardwareCost)}</DescriptionListDescription>
                          </DescriptionListGroup>
                          <DescriptionListGroup>
                            <DescriptionListTerm>Break-even</DescriptionListTerm>
                            <DescriptionListDescription>{result.breakEvenMonths} months</DescriptionListDescription>
                          </DescriptionListGroup>
                        </DescriptionList>
                      </CardBody>
                    </Card>
                  </GridItem>
                </Grid>
              </ResultSection>
            </StackItem>

            {/* Sensitivity */}
            <StackItem>
              <ResultSection id="sensitivity" icon={<ChartBarIcon />} title="What changes things" subtitle="Sensitivity analysis" expanded={state.expandedSections.sensitivity} onToggle={toggleSection}>
                <Stack hasGutter>
                  {result.sensitivity.map((item) => (
                    <StackItem key={item.label}>
                      <Progress
                        id={`sens-${item.label.replace(/\s/g, "-")}`}
                        title={item.label}
                        value={Math.min(100, (item.usersCapacity / maxSensBar) * 100)}
                        label={`${item.usersCapacity} users (${item.changePct >= 0 ? "+" : ""}${item.changePct}%)`}
                        measureLocation="outside"
                        size="sm"
                      />
                    </StackItem>
                  ))}
                </Stack>
              </ResultSection>
            </StackItem>
          </Stack>
        </GridItem>

        {/* Quick adjustments sidebar */}
        {state.controlsVisible && (
          <GridItem span={12} lg={4}>
            <Card isFlat style={{ position: "sticky", top: "1rem" }}>
              <CardTitle>
                <Stack>
                  <StackItem><Title headingLevel="h2" size="lg">Quick adjustments</Title></StackItem>
                  <StackItem><Text component={TextVariants.small} className="pf-v5-u-color-200">Results update in real-time</Text></StackItem>
                </Stack>
              </CardTitle>
              <CardBody>
                <Stack hasGutter>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Concurrent users</Text></StackItem>
                      <StackItem>
                        <TextInput type="number" value={state.concurrentUsers}
                          onChange={(_e, v) => { const n = parseInt(v, 10); if (!isNaN(n) && n > 0) setState((s) => ({ ...s, concurrentUsers: n })); }}
                          aria-label="Concurrent users" />
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Concurrent user presets" isCompact>
                          {[10, 30, 100, 500, 1000].map((n) => (
                            <ToggleGroupItem key={n} text={n >= 1000 ? "1K" : String(n)} buttonId={`users-${n}`} isSelected={state.concurrentUsers === n} onChange={(_e, sel) => sel && setState((s) => ({ ...s, concurrentUsers: n }))} />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Context length</Text></StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Context length" isCompact>
                          {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                            <ToggleGroupItem key={cl} text={`${CONTEXT_LABELS[cl]} · ${CONTEXT_SUBLABELS[cl]}`} buttonId={`ctx-${cl}`} isSelected={state.contextLength === cl} onChange={(_e, sel) => sel && setState((s) => ({ ...s, contextLength: cl }))} />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Deployment type</Text></StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Deployment type" isCompact>
                          {DEPLOYMENT_OPTIONS.map((dt) => (
                            <ToggleGroupItem key={dt.key} text={dt.label} buttonId={`deploy-${dt.key}`} isSelected={state.deploymentType === dt.key} onChange={(_e, sel) => sel && setState((s) => ({ ...s, deploymentType: dt.key }))} />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <ExpandableSection toggleText="Advanced">
                      <Stack hasGutter>
                        {/* GPU */}
                        <StackItem>
                          <Stack>
                            <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">GPU</Text></StackItem>
                            <StackItem>
                              <FormSelect value={state.gpuId} onChange={(_e, v) => setState((s) => ({ ...s, gpuId: v }))} aria-label="Select GPU">
                                {GPU_CATALOG.map((g) => <FormSelectOption key={g.id} value={g.id} label={g.name} />)}
                              </FormSelect>
                            </StackItem>
                          </Stack>
                        </StackItem>
                        {/* Tensor parallelism */}
                        <StackItem>
                          <Stack>
                            <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">Tensor parallel (powers of 2 only)</Text></StackItem>
                            <StackItem>
                              <ToggleGroup aria-label="Tensor parallelism" isCompact>
                                {(["auto", 1, 2, 4, 8] as TensorParallelism[]).map((tp) => (
                                  <ToggleGroupItem key={String(tp)} text={tp === "auto" ? "Auto" : `TP=${tp}`} buttonId={`tp-${tp}`} isSelected={state.tensorParallelism === tp} onChange={(_e, sel) => sel && setState((s) => ({ ...s, tensorParallelism: tp }))} />
                                ))}
                              </ToggleGroup>
                            </StackItem>
                            {result.autoTPUsed && (
                              <StackItem>
                                <Text component={TextVariants.small} className="pf-v5-u-color-200">Auto resolved to TP={result.resolvedTP}</Text>
                              </StackItem>
                            )}
                          </Stack>
                        </StackItem>
                        {/* Memory utilization */}
                        <StackItem>
                          <Stack>
                            <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">Memory utilization target</Text></StackItem>
                            <StackItem>
                              <Slider value={Math.round(state.memUtilization * 100)} min={50} max={95} step={5} onChange={(_e, v) => setState((s) => ({ ...s, memUtilization: v / 100 }))} aria-label="Memory utilization target" showTicks />
                            </StackItem>
                          </Stack>
                        </StackItem>
                        {/* Block size */}
                        <StackItem>
                          <Stack>
                            <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">KV block size (tokens)</Text></StackItem>
                            <StackItem>
                              <ToggleGroup aria-label="KV block size" isCompact>
                                {([16, 32, 64, 128] as const).map((bs) => (
                                  <ToggleGroupItem key={bs} text={String(bs)} buttonId={`bs-${bs}`} isSelected={state.blockSize === bs} onChange={(_e, sel) => sel && setState((s) => ({ ...s, blockSize: bs }))} />
                                ))}
                              </ToggleGroup>
                            </StackItem>
                            <StackItem>
                              <Text component={TextVariants.small} className="pf-v5-u-color-200">16 is the vLLM production default</Text>
                            </StackItem>
                          </Stack>
                        </StackItem>
                        {/* KV cache dtype */}
                        <StackItem>
                          <Stack>
                            <StackItem><Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">KV cache dtype</Text></StackItem>
                            <StackItem>
                              <ToggleGroup aria-label="KV cache dtype" isCompact>
                                <ToggleGroupItem text="Auto (bf16)" buttonId="kv-auto" isSelected={state.kvCacheDtype === "auto"} onChange={(_e, sel) => sel && setState((s) => ({ ...s, kvCacheDtype: "auto" }))} />
                                <ToggleGroupItem text="FP8" buttonId="kv-fp8" isSelected={state.kvCacheDtype === "fp8"} onChange={(_e, sel) => sel && setState((s) => ({ ...s, kvCacheDtype: "fp8" }))} />
                              </ToggleGroup>
                            </StackItem>
                            <StackItem>
                              <Text component={TextVariants.small} className="pf-v5-u-color-200">
                                FP8 halves KV memory. Requires vLLM --kv-cache-dtype fp8.
                              </Text>
                            </StackItem>
                          </Stack>
                        </StackItem>
                      </Stack>
                    </ExpandableSection>
                  </StackItem>
                </Stack>
              </CardBody>
            </Card>
          </GridItem>
        )}
      </Grid>
    </PageSection>
  );
}

// ─── main page ────────────────────────────────────────────────────────────────

export default function QuickEstimatePage() {
  const [state, setState] = useState<PageState>(DEFAULT_STATE);

  const selectedGpu = GPU_CATALOG.find((g) => g.id === state.gpuId) ?? DEFAULT_GPU;

  // ── HF fetch ────────────────────────────────────────────────────────────────
  const loadModel = useCallback(async (hfId: string) => {
    setState((s) => ({ ...s, fetchStatus: "loading", extractedConfig: null }));

    const headers: Record<string, string> = {};
    if (state.hfToken) headers["x-hf-token"] = state.hfToken;

    try {
      const res = await fetch(`/api/hf-config?model=${encodeURIComponent(hfId)}`, { headers });
      const data = await res.json() as Record<string, unknown>;

      if (!res.ok) {
        const error = (data.error as string) ?? "network_error";
        if (error === "gated") {
          setState((s) => ({ ...s, fetchStatus: "gated", showHfToken: true }));
        } else if (error === "not_found") {
          setState((s) => ({ ...s, fetchStatus: "not_found" }));
        } else {
          setState((s) => ({ ...s, fetchStatus: "network_error" }));
        }
        return;
      }

      const rawConfig  = data.config as Record<string, unknown>;
      const weightBytes = (data.weightBytes as number | null) ?? 0;
      const weightSource = (data.weightSource as WeightMemorySource) ?? "estimated";
      const fetchWarnings = (data.warnings as string[]) ?? [];

      const extractedConfig = extractConfig(rawConfig);

      setState((s) => ({
        ...s,
        fetchStatus:     "success",
        extractedConfig,
        weightBytes,
        weightSource,
        fetchWarnings,
        showHfToken:     false,
      }));
    } catch {
      setState((s) => ({ ...s, fetchStatus: "network_error" }));
    }
  }, [state.hfToken]);

  // ── Engine ──────────────────────────────────────────────────────────────────
  const engineResult = useMemo<EngineResult | null>(() => {
    if (!state.extractedConfig) return null;
    try {
      const tp = state.tensorParallelism === "auto" ? 0 : state.tensorParallelism;
      return runEngine({
        extractedConfig: state.extractedConfig,
        weightBytes:     state.weightBytes,
        weightSource:    state.weightSource,
        concurrentUsers: state.concurrentUsers,
        contextLength:   state.contextLength,
        tp,
        blockSize:       state.blockSize,
        kvCacheDtype:    state.kvCacheDtype,
        gpu:             selectedGpu,
        memUtilization:  state.memUtilization,
      });
    } catch (err) {
      console.error("[gpu-calc] runEngine error:", err);
      return null;
    }
  }, [
    state.extractedConfig, state.weightBytes, state.weightSource,
    state.concurrentUsers, state.contextLength, state.tensorParallelism,
    state.blockSize, state.kvCacheDtype, selectedGpu, state.memUtilization,
  ]);

  const legacyResult = useMemo<LegacyResult | null>(() => {
    if (!engineResult) return null;
    return toLegacy(engineResult, selectedGpu, state.concurrentUsers, state.contextLength, state.deploymentType);
  }, [engineResult, selectedGpu, state.concurrentUsers, state.contextLength, state.deploymentType]);

  if (state.view === "results" && legacyResult) {
    return <ResultsView state={state} setState={setState} result={legacyResult} selectedGpu={selectedGpu} />;
  }

  return (
    <div className="rh-qe-layout">
      <div className="rh-qe-main">
        <ConfigPanel state={state} setState={setState} onLoadModel={loadModel} />
      </div>
      <LiveEstimatePanel
        result={legacyResult}
        state={state}
        selectedGpu={selectedGpu}
        onGpuChange={(gpuId) => setState((s) => ({ ...s, gpuId }))}
        fetchStatus={state.fetchStatus}
      />
    </div>
  );
}
