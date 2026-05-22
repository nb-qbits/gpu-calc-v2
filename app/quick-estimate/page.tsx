"use client";

import React, { useState, useMemo, useRef } from "react";
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
  runQuickEstimate,
  PRECISION_BYTES,
  CONTEXT_TOKENS,
  CONTEXT_LABELS,
  CONTEXT_SUBLABELS,
  type Precision,
  type ContextLength,
  type DeploymentType,
  type TensorParallelism,
  type QuickEstimateResult,
} from "@/lib/gpu-math";
import type { GpuSpec } from "@/lib/gpu-math/gpus";
import type { ModelSpec } from "@/lib/gpu-math/models";

// ─── constants ────────────────────────────────────────────────────────────────

const NEW_MODEL_IDS = new Set([
  "gemma-3-12b", "gemma-3-27b",
  "gemma-4-2b",  "gemma-4-9b", "gemma-4-27b",
]);

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

// ─── state ────────────────────────────────────────────────────────────────────

interface PageState {
  view: "estimate" | "results";
  selectedModelId: string;
  concurrentUsers: number;
  customUsersText: string;
  contextLength: ContextLength;
  deploymentType: DeploymentType;
  precision: Precision;
  tensorParallelism: TensorParallelism;
  memUtilization: number;
  advancedOpen: boolean;
  controlsVisible: boolean;
  expandedSections: Record<string, boolean>;
  gpuId: string;
  expandedTile: "users" | "context" | null;
  hfModelId: string;
}

const DEFAULT_STATE: PageState = {
  view: "estimate",
  selectedModelId: "gemma-2-27b",
  concurrentUsers: 10,
  customUsersText: "",
  contextLength: "short",
  deploymentType: "onprem",
  precision: "fp16",
  tensorParallelism: "auto",
  memUtilization: 0.9,
  advancedOpen: false,
  controlsVisible: true,
  expandedSections: {
    modelWeights: true, kvCache: true, vramBudget: true,
    precision: false, gpuCapability: false, throughput: false,
    cost: false, sensitivity: true,
  },
  gpuId: "h100",
  expandedTile: null,
  hfModelId: "",
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
  const COLS = 22;
  const ROWS = 9;
  const CELLS = COLS * ROWS;
  const safeTotal = totalGb > 0 ? totalGb : 1;
  const weightCells = Math.round((weightsGb / safeTotal) * CELLS);
  const kvCells = Math.min(
    CELLS - weightCells,
    Math.round((kvGb / safeTotal) * CELLS),
  );
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
  result, state, selectedModel, selectedGpu, onGpuChange,
}: {
  result: QuickEstimateResult;
  state: PageState;
  selectedModel: ModelSpec;
  selectedGpu: GpuSpec;
  onGpuChange: (gpuId: string) => void;
}) {
  const isOnPrem = state.deploymentType !== "cloud";
  const onPremMonthly = result.hardwareCost / 36 + result.electricityPerMonth;
  const displayMonthly = isOnPrem ? onPremMonthly : result.costPerMonth;
  const ratePerGpuHr = isOnPrem
    ? onPremMonthly / (result.totalGpus * 730)
    : selectedGpu.pricePerHour;
  const deployLabel = isOnPrem ? "ON-PREM" : "CLOUD";
  const gpuLabel = selectedGpu.name.replace("NVIDIA ", "");
  const totalVramAvailable = result.totalGpus * selectedGpu.vramGb;
  const utilPct = totalVramAvailable > 0
    ? Math.round((result.totalVramGb / totalVramAvailable) * 100)
    : 0;
  const freeGb = Math.max(0, totalVramAvailable - result.totalVramGb);

  const monoSm: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
    fontSize: "0.72rem",
    lineHeight: 1.8,
  };

  const label: React.CSSProperties = {
    fontFamily: "var(--font-mono)",
    fontSize: "0.6rem",
    letterSpacing: "0.07em",
    opacity: 0.65,
    marginBottom: 4,
    display: "block",
  };

  const flipHint: React.CSSProperties = {
    marginTop: "auto",
    paddingTop: 4,
    textAlign: "right",
    fontFamily: "var(--font-mono)",
    fontSize: "0.6rem",
    opacity: 0.5,
  };

  return (
    <div className="rh-qe-sidebar">
      {/* header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", color: "var(--rh-gray-60)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--rh-red)", display: "inline-block" }} />
          LIVE ESTIMATE
        </span>
        <span className="rh-qe-tap-hint" style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)" }}>
          tap a tile to see the math
        </span>
      </div>

      {/* Cost card */}
      <FlipCard
        height={140}
        frontClassName="rh-flip-card__face--red"
        backClassName="rh-flip-card__face--dark"
        front={
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.08em", opacity: 0.8 }}>
                ESTIMATED MONTHLY
              </span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", background: "rgba(255,255,255,0.25)", borderRadius: 3, padding: "1px 6px" }}>
                {deployLabel}
              </span>
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
              <br />
              = <span style={{ color: "var(--rh-red)", fontWeight: 700 }}>{fmtMoney(displayMonthly)}</span> / month
            </div>
            <div style={{ marginTop: 8, fontSize: "0.62rem", color: "var(--rh-gray-40)" }}>
              {isOnPrem ? "On-prem pricing · amortized 36 months + electricity" : "Cloud on-demand pricing · assumes 100% uptime"}
            </div>
            <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
          </>
        }
      />

      {/* GPU count + VRAM */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
        <FlipCard
          height={120}
          front={
            <>
              <span style={label}>GPU COUNT</span>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.4rem", lineHeight: 1.1 }}>
                {result.totalGpus}
                <span style={{ fontWeight: 400, fontSize: "0.75rem", marginLeft: 4 }}>× {gpuLabel}</span>
              </div>
              <div style={{ fontSize: "0.68rem", color: "var(--rh-gray-60)", marginTop: 2 }}>
                {utilPct}% utilized
              </div>
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
        <FlipCard
          height={120}
          front={
            <>
              <span style={label}>TOTAL VRAM</span>
              <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.4rem", lineHeight: 1.1 }}>
                {Math.round(result.totalVramGb)}
                <span style={{ fontWeight: 400, fontSize: "0.9rem", marginLeft: 4 }}>GB</span>
              </div>
              <div style={{ fontSize: "0.68rem", color: "var(--rh-gray-60)", marginTop: 2 }}>
                model + KV cache
              </div>
              <div style={flipHint}>↺ SEE MATH</div>
            </>
          }
          back={
            <>
              <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>VRAM MATH</span>
              <div style={monoSm}>
                Weights: {fmtGb(result.weightsGb)}<br />
                KV: {fmtGb(result.kvTotalGb)}<br />
                = <span style={{ color: "var(--rh-red)", fontWeight: 700 }}>{fmtGb(result.totalVramGb)}</span>
              </div>
              <div style={{ ...flipHint, color: "var(--rh-gray-40)", opacity: 1 }}>↺ FLIP BACK</div>
            </>
          }
        />
      </div>

      {/* Memory layout — grows to fill remaining vertical space */}
      <div className="rh-qe-mem-wrap" style={{ marginTop: 8, flex: 1, display: "flex", flexDirection: "column" }}>
        <FlipCard
          height="100%"
          front={
            <>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={label}>MEMORY LAYOUT</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)" }}>
                  {result.totalGpus} × {gpuLabel} {selectedGpu.vramGb}GB
                </span>
              </div>
              <MemoryGrid
                weightsGb={result.weightsGb}
                kvGb={result.kvTotalGb}
                totalGb={totalVramAvailable}
              />
              <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: "0.65rem", color: "var(--rh-gray-60)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-red)", display: "inline-block" }} />
                  Weights
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-red-muted, rgba(238,0,0,0.22))", display: "inline-block" }} />
                  KV cache
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--rh-gray-20)", display: "inline-block" }} />
                  Free
                </span>
              </div>
              <div style={flipHint}>↺ SEE MATH</div>
            </>
          }
          back={
            <>
              <span style={{ ...label, color: "var(--rh-gray-40)", opacity: 1 }}>PER-GPU SPLIT</span>
              <div style={{ fontSize: "0.85rem", lineHeight: 2.1 }}>
                <span style={{ color: "var(--rh-red)", fontWeight: 500 }}>
                  {fmtGb(result.weightsGb / result.totalGpus)}
                </span>{" "}weights<br />
                <span style={{ color: "var(--rh-red-dark)", fontWeight: 500 }}>
                  {fmtGb(result.kvTotalGb / result.totalGpus)}
                </span>{" "}KV cache<br />
                <span style={{ color: "var(--rh-gray-60)", fontWeight: 500 }}>
                  {fmtGb(freeGb / result.totalGpus)}
                </span>{" "}free headroom
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

      {/* GPU selector — pinned to bottom */}
      <div style={{ marginTop: "auto", paddingTop: 12, borderTop: "1px solid var(--rh-gray-20)" }}>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--rh-gray-40)", letterSpacing: "0.07em", marginBottom: 4 }}>
          GPU
        </div>
        <FormSelect
          value={state.gpuId}
          onChange={(_e, v) => onGpuChange(v)}
          aria-label="Select GPU"
          style={{ fontSize: "0.8rem" }}
        >
          {GPU_CATALOG.map((g) => (
            <FormSelectOption key={g.id} value={g.id} label={g.name} />
          ))}
        </FormSelect>
      </div>
    </div>
  );
}

// ─── ConfigPanel ──────────────────────────────────────────────────────────────

function ConfigPanel({
  state, setState,
}: {
  state: PageState;
  setState: React.Dispatch<React.SetStateAction<PageState>>;
}) {
  const galleryRef = useRef<HTMLDivElement>(null);

  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId)!;
  const presetMatch = USER_PRESETS.find((p) => p.value === state.concurrentUsers);
  const contextLabel = CONTEXT_LABELS[state.contextLength];
  const contextSub = CONTEXT_SUBLABELS[state.contextLength];

  const contextBars: Record<ContextLength, string> = {
    short:    "—",
    medium:   "——",
    long:     "———",
    verylong: "————",
  };

  function toggleTile(tile: "users" | "context") {
    setState((s) => ({ ...s, expandedTile: s.expandedTile === tile ? null : tile }));
  }

  return (
    <div>
      {/* Hero */}
      <div style={{ marginBottom: "2.5rem" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 6,
          fontFamily: "var(--font-mono)", fontSize: "0.65rem",
          letterSpacing: "0.08em", color: "var(--rh-gray-60)",
          marginBottom: "1rem",
        }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--rh-red)", display: "inline-block" }} />
          QUICK ESTIMATE · ~30 SECONDS
        </div>
        <h1 style={{
          fontFamily: "var(--font-display)",
          fontSize: "clamp(2rem, 4vw, 2.75rem)",
          fontWeight: 700,
          lineHeight: 1.15,
          margin: 0,
          marginBottom: "0.75rem",
        }}>
          Size your{" "}
          <span style={{ color: "var(--rh-red)" }}>LLM</span>
          {" "}deployment.
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
            <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>
              What model are you serving?
            </h2>
          </div>
          <span style={{
            display: "flex", alignItems: "center", gap: 4,
            fontSize: "0.7rem", fontWeight: 500, color: "var(--rh-red)",
          }}>
            <span style={{ fontSize: "0.8rem" }}>✓</span> SET
          </span>
        </div>
        <p style={{ fontSize: "0.875rem", color: "var(--rh-gray-60)", margin: "0 0 1rem" }}>
          Sliding gallery of common models, or paste any HuggingFace model ID.
        </p>

        {/* Gallery */}
        <div className="rh-model-gallery" ref={galleryRef}>
          {MODEL_CATALOG.map((model) => {
            const selected = model.id === state.selectedModelId;
            const isNew = NEW_MODEL_IDS.has(model.id);
            return (
              <div
                key={model.id}
                className={`rh-model-card${selected ? " rh-model-card--selected" : ""}`}
                onClick={() => setState((s) => ({ ...s, selectedModelId: model.id }))}
                role="button"
                tabIndex={0}
                aria-pressed={selected}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setState((s) => ({ ...s, selectedModelId: model.id }));
                  }
                }}
              >
                {selected && (
                  <span style={{ position: "absolute", top: 7, right: 8, color: "var(--rh-red)", fontSize: "0.85rem" }}>✓</span>
                )}
                <div style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.55rem",
                  letterSpacing: "0.07em",
                  color: "var(--rh-gray-40)",
                  marginBottom: 4,
                  textTransform: "uppercase",
                }}>
                  {model.vendor}
                </div>
                <div style={{ fontWeight: 500, fontSize: "0.85rem", lineHeight: 1.3 }}>
                  {model.name}
                </div>
                <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ fontSize: "0.7rem", color: "var(--rh-gray-60)" }}>
                    {model.paramsBillions}B
                  </span>
                  {isNew && <span className="rh-badge-new">NEW</span>}
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          marginTop: "0.75rem",
          background: "var(--rh-gray-10)",
          border: "1px solid var(--rh-gray-20)",
          borderRadius: 20,
          padding: "6px 12px",
          fontSize: "0.8rem",
          width: "fit-content",
        }}>
          <span style={{ color: "var(--rh-red)", fontWeight: 500 }}>✓</span>
          <span>
            Selected: <strong>{selectedModel.name}</strong>{" "}
            <span style={{ color: "var(--rh-gray-60)" }}>
              · {selectedModel.paramsBillions}B params
            </span>
          </span>
          <button
            onClick={() => galleryRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })}
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "var(--rh-red)", fontSize: "0.75rem", fontWeight: 500,
              padding: "0 4px", textDecoration: "underline",
            }}
          >
            change
          </button>
        </div>

        {/* HuggingFace input */}
        <div style={{
          marginTop: "1rem",
          background: "var(--pf-v5-global--BackgroundColor--100, #fff)",
          border: "1px solid var(--rh-gray-20)",
          borderRadius: 8,
          padding: "14px 16px",
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-60)", marginBottom: 8 }}>
            OR PASTE A HUGGINGFACE MODEL ID
          </div>
          <div className="rh-hf-row" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <TextInput
              value={state.hfModelId}
              onChange={(_e, v) => setState((s) => ({ ...s, hfModelId: v }))}
              placeholder="meta-llama/Llama-3.1-70B-Instruct"
              aria-label="HuggingFace model ID"
              style={{ flex: 1, fontFamily: "var(--font-mono)", fontSize: "0.82rem", background: "white" }}
            />
            <Button variant="secondary" isDisabled style={{ flexShrink: 0 }}>
              Load
            </Button>
            <a
              href="https://huggingface.co/models"
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "0.8rem", color: "var(--rh-red)", whiteSpace: "nowrap", flexShrink: 0 }}
            >
              Browse on HF →
            </a>
          </div>
          <div style={{ marginTop: 8 }}>
            <button style={{
              background: "none", border: "none", cursor: "pointer",
              fontSize: "0.72rem", color: "var(--rh-gray-60)", padding: 0,
            }}>
              🔑 Add access token · <span style={{ color: "var(--rh-gray-40)" }}>optional, for gated/private models</span>
            </button>
          </div>
        </div>
      </div>

      {/* Step 02: Load profile */}
      <div style={{ marginBottom: "2.5rem" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "1rem" }}>
            <span className="rh-step-number">02</span>
            <h2 style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", fontWeight: 700, margin: 0 }}>
              Load profile
            </h2>
          </div>
          <span style={{
            display: "flex", alignItems: "center", gap: 4,
            fontSize: "0.7rem", fontWeight: 500, color: "var(--rh-red)",
          }}>
            <span style={{ fontSize: "0.8rem" }}>✓</span> SET
          </span>
        </div>
        <p style={{ fontSize: "0.875rem", color: "var(--rh-gray-60)", margin: "0 0 1rem" }}>
          Tap the pencil to adjust either value.
        </p>

        <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
          {/* Concurrent users tile */}
          <div
            className={`rh-load-tile${state.expandedTile === "users" ? " rh-load-tile--active" : ""}`}
            onClick={() => toggleTile("users")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 4 }}>
                  CONCURRENT USERS
                </div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.75rem", lineHeight: 1 }}>
                  {state.concurrentUsers}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--rh-gray-60)", marginTop: 4 }}>
                  {presetMatch?.sub ?? "custom"}
                </div>
              </div>
              <button
                style={{
                  background: state.expandedTile === "users" ? "#fff0f0" : "var(--rh-gray-10)",
                  border: "1px solid var(--rh-gray-20)",
                  borderRadius: "50%", width: 28, height: 28,
                  cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0,
                }}
                aria-label="Edit concurrent users"
                onClick={(e) => { e.stopPropagation(); toggleTile("users"); }}
              >
                <PencilAltIcon style={{ fontSize: "0.7rem", color: state.expandedTile === "users" ? "var(--rh-red)" : "var(--rh-gray-60)" }} />
              </button>
            </div>

            {state.expandedTile === "users" && (
              <div style={{ marginTop: 14, borderTop: "1px dashed var(--rh-gray-20)", paddingTop: 12 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 8 }}>
                  {USER_PRESETS.map((p) => (
                    <div
                      key={p.value}
                      className={`rh-load-tile-option${state.concurrentUsers === p.value ? " rh-load-tile-option--selected" : ""}`}
                      onClick={() => setState((s) => ({ ...s, concurrentUsers: p.value, customUsersText: "" }))}
                    >
                      <div style={{ fontWeight: 700, fontSize: "1rem" }}>{p.label}</div>
                      <div style={{ fontSize: "0.65rem", color: "var(--rh-gray-60)", marginTop: 1 }}>{p.sub}</div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                  <span style={{ fontSize: "0.75rem", color: "var(--rh-gray-60)", flexShrink: 0 }}>Or type exact:</span>
                  <TextInput
                    type="number"
                    placeholder="e.g. 75"
                    value={state.customUsersText}
                    onChange={(_e, v) => {
                      const n = parseInt(v, 10);
                      setState((s) => ({
                        ...s,
                        customUsersText: v,
                        concurrentUsers: isNaN(n) || n <= 0 ? s.concurrentUsers : n,
                      }));
                    }}
                    style={{ flex: 1, fontSize: "0.82rem" }}
                    aria-label="Custom concurrent users"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Conversation length tile */}
          <div
            className={`rh-load-tile${state.expandedTile === "context" ? " rh-load-tile--active" : ""}`}
            onClick={() => toggleTile("context")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 4 }}>
                  CONVERSATION LENGTH
                </div>
                <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.75rem", lineHeight: 1 }}>
                  {contextSub.replace(" tokens", "")}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--rh-gray-60)", marginTop: 4 }}>
                  {contextLabel.toLowerCase()}
                </div>
              </div>
              <button
                style={{
                  background: state.expandedTile === "context" ? "#fff0f0" : "var(--rh-gray-10)",
                  border: "1px solid var(--rh-gray-20)",
                  borderRadius: "50%", width: 28, height: 28,
                  cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0,
                }}
                aria-label="Edit conversation length"
                onClick={(e) => { e.stopPropagation(); toggleTile("context"); }}
              >
                <PencilAltIcon style={{ fontSize: "0.7rem", color: state.expandedTile === "context" ? "var(--rh-red)" : "var(--rh-gray-60)" }} />
              </button>
            </div>

            {state.expandedTile === "context" && (
              <div style={{ marginTop: 14, borderTop: "1px dashed var(--rh-gray-20)", paddingTop: 12 }}
                onClick={(e) => e.stopPropagation()}
              >
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                    <div
                      key={cl}
                      className={`rh-load-tile-option${state.contextLength === cl ? " rh-load-tile-option--selected" : ""}`}
                      onClick={() => setState((s) => ({ ...s, contextLength: cl }))}
                    >
                      <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{CONTEXT_LABELS[cl]}</div>
                      <div style={{ fontSize: "0.65rem", color: "var(--rh-gray-60)", marginTop: 1 }}>
                        {CONTEXT_SUBLABELS[cl]}
                      </div>
                      <div style={{
                        marginTop: 4, fontSize: "0.6rem",
                        color: state.contextLength === cl ? "var(--rh-red)" : "var(--rh-gray-40)",
                        letterSpacing: 2,
                      }}>
                        {contextBars[cl]}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Deployment type */}
        <div style={{ marginTop: "1.25rem" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.07em", color: "var(--rh-gray-40)", marginBottom: 8 }}>
            DEPLOYMENT TYPE
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {DEPLOYMENT_OPTIONS.map((dt) => (
              <button
                key={dt.key}
                onClick={() => setState((s) => ({ ...s, deploymentType: dt.key }))}
                style={{
                  padding: "7px 14px",
                  border: `1.5px solid ${state.deploymentType === dt.key ? "var(--rh-red)" : "var(--rh-gray-20)"}`,
                  borderRadius: 5,
                  background: state.deploymentType === dt.key ? "var(--rh-red-50)" : "var(--pf-v5-global--BackgroundColor--100)",
                  color: state.deploymentType === dt.key ? "var(--rh-red)" : "var(--rh-gray-60)",
                  fontWeight: state.deploymentType === dt.key ? 600 : 400,
                  cursor: "pointer",
                  fontSize: "0.8rem",
                  transition: "all 0.12s",
                }}
              >
                {dt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="rh-qe-footer">
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.8rem", color: "var(--rh-gray-60)" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--rh-red)", display: "inline-block" }} />
          Ready · estimate is live on the right
        </div>
        <Button
          variant="primary"
          onClick={() => setState((s) => ({ ...s, view: "results" }))}
          style={{ fontSize: "0.9rem", padding: "10px 20px" }}
        >
          See full breakdown →
        </Button>
      </div>
    </div>
  );
}

// ─── ResultsView helpers ──────────────────────────────────────────────────────

function SummaryCard({ icon, title, value, sub, isDanger }: {
  icon: React.ReactNode;
  title: string;
  value: string;
  sub: string;
  isDanger?: boolean;
}) {
  return (
    <Card isFlat isFullHeight>
      <CardBody>
        <Stack>
          <StackItem>
            <Text component={TextVariants.small} className="pf-v5-u-color-200">
              {icon} {title}
            </Text>
          </StackItem>
          <StackItem>
            <Title headingLevel="h3" size="2xl"
              className={isDanger ? "pf-v5-u-primary-color-100" : ""}
            >
              {value}
            </Title>
          </StackItem>
          <StackItem>
            <Text component={TextVariants.small} className="pf-v5-u-color-200">{sub}</Text>
          </StackItem>
        </Stack>
      </CardBody>
    </Card>
  );
}

function ResultSection({ id, icon, title, subtitle, expanded, onToggle, children }: {
  id: string;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  expanded: boolean;
  onToggle: (id: string, open: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <Card isFlat className="pf-v5-u-mb-sm">
      <ExpandableSection
        toggleContent={
          <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
            <FlexItem><span className="pf-v5-u-primary-color-100">{icon}</span></FlexItem>
            <FlexItem>
              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>
                {title}
              </Text>
              <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>
                {subtitle}
              </Text>
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

function ResultsView({ state, setState }: {
  state: PageState;
  setState: React.Dispatch<React.SetStateAction<PageState>>;
}) {
  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId)!;
  const selectedGpu = GPU_CATALOG.find((g) => g.id === state.gpuId) ?? DEFAULT_GPU;

  const result: QuickEstimateResult = useMemo(
    () => runQuickEstimate({
      model: selectedModel,
      concurrentUsers: state.concurrentUsers,
      contextLength: state.contextLength,
      precision: state.precision,
      deploymentType: state.deploymentType,
      gpu: selectedGpu,
      tensorParallelism: state.tensorParallelism,
      memUtilization: state.memUtilization,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedModel, state.concurrentUsers, state.contextLength, state.precision,
      state.deploymentType, selectedGpu, state.tensorParallelism, state.memUtilization]
  );

  const { precision } = state;
  const { weight: bytesPerWeight } = PRECISION_BYTES[precision];
  const deployLabel = state.deploymentType === "cloud" ? "cloud" : state.deploymentType === "onprem" ? "on-prem" : "hybrid";

  function toggleSection(id: string, open: boolean) {
    setState((s) => ({ ...s, expandedSections: { ...s.expandedSections, [id]: open } }));
  }

  const precisionVariants = (["fp16", "int8", "int4", "mixed"] as Precision[]).map((p) => {
    const { weight: bw, kv: bkv, label } = PRECISION_BYTES[p];
    const wGb = selectedModel.paramsBillions * selectedModel.activeFraction * bw;
    const kvGb = (2 * selectedModel.numLayers * selectedModel.hiddenSize * bkv *
      state.concurrentUsers * CONTEXT_TOKENS[state.contextLength]) / 1e9;
    const actGb = wGb * 0.2;
    const kvKb = (2 * selectedModel.numLayers * selectedModel.hiddenSize * bkv) / 1024;
    return { key: p, label, total: wGb + kvGb + actGb, kvKb };
  });

  const maxSensBar = Math.max(...result.sensitivity.map((s) => s.usersCapacity), state.concurrentUsers, 1);

  return (
    <PageSection>
      <Flex
        justifyContent={{ default: "justifyContentSpaceBetween" }}
        alignItems={{ default: "alignItemsCenter" }}
        className="pf-v5-u-mb-md"
      >
        <FlexItem>
          <Button variant="link" icon={<ArrowLeftIcon />}
            onClick={() => setState((s) => ({ ...s, view: "estimate" }))}
          >
            Back to estimate
          </Button>
        </FlexItem>
        <FlexItem>
          <Button variant="secondary" icon={<CogIcon />}
            onClick={() => setState((s) => ({ ...s, controlsVisible: !s.controlsVisible }))}
          >
            {state.controlsVisible ? "Hide controls" : "Show controls"}
          </Button>
        </FlexItem>
      </Flex>

      <Title headingLevel="h1" size="xl" className="pf-v5-u-mb-lg">
        GPU requirements for {selectedModel.name}
      </Title>

      <Grid hasGutter>
        <GridItem span={12} lg={state.controlsVisible ? 8 : 12}>
          <Stack hasGutter>
            <StackItem>
              <Grid hasGutter>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<ServerIcon />} title="Replicas needed"
                    value={`${result.replicasNeeded}+`} sub={`${result.gpusPerReplica} × GPU`} isDanger />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<MicrochipIcon />} title="GPU availability"
                    value={`${result.cloudAvailabilityPct}%`} sub={`TPU availability: ${result.tpuAvailabilityPct}%`} />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<DollarSignIcon />} title="Server costs"
                    value={fmtMoney(result.costPerMonth)} sub={`/mo ${deployLabel}`} />
                </GridItem>
                <GridItem span={6} lg={3}>
                  <SummaryCard icon={<TrendUpIcon />} title="Idle server costs"
                    value={fmtMoney(result.idleServerCostsPerMonth)} sub="30-min+ idle" />
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
                        <DescriptionListTerm>Parameters</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{selectedModel.paramsBillions}B</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>{PRECISION_BYTES[precision].label} ({bytesPerWeight}B)</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtGb(result.weightsGb)}</span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Active params</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">
                            {(selectedModel.paramsBillions * selectedModel.activeFraction).toFixed(1)}B
                          </span>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                    </DescriptionList>
                    <div className="rh-flip-card__hint">↻ Flip for formula</div>
                  </>
                }
                back={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><ServerIcon /></span></FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Calculation</Text>
                      </FlexItem>
                    </Flex>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm">
                      {selectedModel.paramsBillions}B params × {bytesPerWeight} bytes ({PRECISION_BYTES[precision].label})
                    </Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">
                      = {fmtGb(result.weightsGb)}
                    </Text>
                    <Text component={TextVariants.small} className="pf-v5-u-color-200">
                      Active: {(selectedModel.paramsBillions * selectedModel.activeFraction).toFixed(1)}B × {selectedModel.activeFraction < 1 ? `${(selectedModel.activeFraction * 100).toFixed(0)}% (MoE)` : "100% (dense)"}
                    </Text>
                    <div className="rh-flip-card__hint">↻ Flip back</div>
                  </>
                }
              />
            </StackItem>

            {/* KV cache flip card */}
            <StackItem>
              <FlipCard height={250}
                front={
                  <>
                    <Flex alignItems={{ default: "alignItemsCenter" }} spaceItems={{ default: "spaceItemsSm" }}>
                      <FlexItem><span className="pf-v5-u-primary-color-100"><MemoryIcon /></span></FlexItem>
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold" style={{ display: "inline" }}>KV cache</Text>
                        <Text component={TextVariants.small} className="pf-v5-u-color-200 pf-v5-u-ml-sm" style={{ display: "inline" }}>Memory for attention</Text>
                      </FlexItem>
                    </Flex>
                    <DescriptionList columnModifier={{ default: "3Col" }}>
                      <DescriptionListGroup>
                        <DescriptionListTerm>KV total</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtGb(result.kvTotalGb)}</span>
                          <br />
                          <Text component={TextVariants.small} className="pf-v5-u-color-200">{fmtBytes(result.kvBytesPerToken)}/token</Text>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Total context</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{fmtTokens(result.totalContextTokens)}</span>
                          <br />
                          <Text component={TextVariants.small} className="pf-v5-u-color-200">
                            {state.concurrentUsers} × {fmtTokens(CONTEXT_TOKENS[state.contextLength])}
                          </Text>
                        </DescriptionListDescription>
                      </DescriptionListGroup>
                      <DescriptionListGroup>
                        <DescriptionListTerm>Layers</DescriptionListTerm>
                        <DescriptionListDescription>
                          <span className="pf-v5-u-font-weight-bold">{selectedModel.numLayers}</span>
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
                      <FlexItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">KV cache formula</Text>
                      </FlexItem>
                    </Flex>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm">
                      2 × {selectedModel.numLayers} layers × {selectedModel.hiddenSize} hidden × {PRECISION_BYTES[precision].kv}B
                    </Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">
                      = {fmtBytes(result.kvBytesPerToken)} per token
                    </Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm">
                      {state.concurrentUsers} users × {fmtTokens(CONTEXT_TOKENS[state.contextLength])} tokens = {fmtTokens(result.totalContextTokens)}
                    </Text>
                    <Text component={TextVariants.p} className="pf-v5-u-font-family-mono pf-v5-u-font-size-sm pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">
                      = {fmtGb(result.kvTotalGb)} total KV cache
                    </Text>
                    <div className="rh-flip-card__hint">↻ Flip back</div>
                  </>
                }
              />
            </StackItem>

            {/* VRAM budget */}
            <StackItem>
              <ResultSection id="vramBudget" icon={<CubeIcon />}
                title="VRAM budget breakdown" subtitle="Total per-replica memory"
                expanded={state.expandedSections.vramBudget} onToggle={toggleSection}>
                <DescriptionList isHorizontal horizontalTermWidthModifier={{ default: "20ch" }}>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Model weights ({PRECISION_BYTES[precision].label})</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.weightsGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>KV cache ({state.concurrentUsers} users)</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.kvTotalGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm>Activation overhead</DescriptionListTerm>
                    <DescriptionListDescription>{fmtGb(result.activationGb)}</DescriptionListDescription>
                  </DescriptionListGroup>
                  <DescriptionListGroup>
                    <DescriptionListTerm><span className="pf-v5-u-font-weight-bold">Total (per replica)</span></DescriptionListTerm>
                    <DescriptionListDescription>
                      <span className="pf-v5-u-font-weight-bold pf-v5-u-primary-color-100">{fmtGb(result.totalVramGb)}</span>
                    </DescriptionListDescription>
                  </DescriptionListGroup>
                </DescriptionList>
              </ResultSection>
            </StackItem>

            {/* Precision */}
            <StackItem>
              <ResultSection id="precision" icon={<BoltIcon />}
                title="Precision & memory" subtitle="Quantization options"
                expanded={state.expandedSections.precision} onToggle={toggleSection}>
                <Flex spaceItems={{ default: "spaceItemsMd" }}>
                  {precisionVariants.map((pv) => {
                    const sel = precision === pv.key;
                    return (
                      <FlexItem key={pv.key}>
                        <Tile title={pv.label} isSelected={sel}
                          onClick={() => setState((s) => ({ ...s, precision: pv.key }))} isStacked
                        >
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.small} className="pf-v5-u-color-200">
                                {fmtGb(pv.total)}
                              </Text>
                            </StackItem>
                            {sel && (
                              <StackItem>
                                <Text component={TextVariants.small} className="pf-v5-u-primary-color-100">✓ Selected</Text>
                              </StackItem>
                            )}
                          </Stack>
                        </Tile>
                      </FlexItem>
                    );
                  })}
                </Flex>
              </ResultSection>
            </StackItem>

            {/* Throughput */}
            <StackItem>
              <ResultSection id="throughput" icon={<TrendUpIcon />}
                title="Throughput" subtitle="Estimated serving performance"
                expanded={state.expandedSections.throughput} onToggle={toggleSection}>
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
              <ResultSection id="cost" icon={<DollarSignIcon />}
                title="Cost" subtitle="Detailed cost breakdown"
                expanded={state.expandedSections.cost} onToggle={toggleSection}>
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
              <ResultSection id="sensitivity" icon={<ChartBarIcon />}
                title="What changes things" subtitle="Sensitivity analysis"
                expanded={state.expandedSections.sensitivity} onToggle={toggleSection}>
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
                  <StackItem>
                    <Text component={TextVariants.small} className="pf-v5-u-color-200">Results update in real-time</Text>
                  </StackItem>
                </Stack>
              </CardTitle>
              <CardBody>
                <Stack hasGutter>
                  <StackItem>
                    <Stack>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">Model</Text>
                      </StackItem>
                      <StackItem>
                        <FormSelect value={state.selectedModelId}
                          onChange={(_e, v) => setState((s) => ({ ...s, selectedModelId: v }))}
                          aria-label="Select model"
                        >
                          {MODEL_CATALOG.map((m) => (
                            <FormSelectOption key={m.id} value={m.id} label={m.name} />
                          ))}
                        </FormSelect>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Concurrent users</Text>
                      </StackItem>
                      <StackItem>
                        <TextInput type="number" value={state.concurrentUsers}
                          onChange={(_e, v) => {
                            const n = parseInt(v, 10);
                            if (!isNaN(n) && n > 0) setState((s) => ({ ...s, concurrentUsers: n }));
                          }}
                          aria-label="Concurrent users"
                        />
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Concurrent user presets" isCompact>
                          {[10, 30, 100, 500, 1000].map((n) => (
                            <ToggleGroupItem key={n}
                              text={n >= 1000 ? "1K" : String(n)}
                              buttonId={`users-${n}`}
                              isSelected={state.concurrentUsers === n}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, concurrentUsers: n }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Context length</Text>
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Context length" isCompact>
                          {(["short", "medium", "long", "verylong"] as ContextLength[]).map((cl) => (
                            <ToggleGroupItem key={cl}
                              text={`${CONTEXT_LABELS[cl]} · ${CONTEXT_SUBLABELS[cl]}`}
                              buttonId={`ctx-${cl}`}
                              isSelected={state.contextLength === cl}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, contextLength: cl }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <Stack hasGutter>
                      <StackItem>
                        <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold">Deployment type</Text>
                      </StackItem>
                      <StackItem>
                        <ToggleGroup aria-label="Deployment type" isCompact>
                          {DEPLOYMENT_OPTIONS.map((dt) => (
                            <ToggleGroupItem key={dt.key}
                              text={dt.label}
                              buttonId={`deploy-${dt.key}`}
                              isSelected={state.deploymentType === dt.key}
                              onChange={(_e, sel) => sel && setState((s) => ({ ...s, deploymentType: dt.key }))}
                            />
                          ))}
                        </ToggleGroup>
                      </StackItem>
                    </Stack>
                  </StackItem>
                  <StackItem>
                    <ExpandableSection toggleText="Advanced">
                      <Stack hasGutter>
                        <StackItem>
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">GPU</Text>
                            </StackItem>
                            <StackItem>
                              <FormSelect value={state.gpuId}
                                onChange={(_e, v) => setState((s) => ({ ...s, gpuId: v }))}
                                aria-label="Select GPU"
                              >
                                {GPU_CATALOG.map((g) => (
                                  <FormSelectOption key={g.id} value={g.id} label={g.name} />
                                ))}
                              </FormSelect>
                            </StackItem>
                          </Stack>
                        </StackItem>
                        <StackItem>
                          <Stack>
                            <StackItem>
                              <Text component={TextVariants.p} className="pf-v5-u-font-weight-bold pf-v5-u-mb-xs">Memory utilization target</Text>
                            </StackItem>
                            <StackItem>
                              <Slider
                                value={Math.round(state.memUtilization * 100)}
                                min={50} max={95} step={5}
                                onChange={(_e, v) => setState((s) => ({ ...s, memUtilization: v / 100 }))}
                                aria-label="Memory utilization target"
                                showTicks
                              />
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

  const selectedModel = MODEL_CATALOG.find((m) => m.id === state.selectedModelId) ?? MODEL_CATALOG[0];
  const selectedGpu   = GPU_CATALOG.find((g) => g.id === state.gpuId) ?? DEFAULT_GPU;

  const result = useMemo(
    () => runQuickEstimate({
      model: selectedModel,
      concurrentUsers: state.concurrentUsers,
      contextLength: state.contextLength,
      precision: state.precision,
      deploymentType: state.deploymentType,
      gpu: selectedGpu,
      tensorParallelism: state.tensorParallelism,
      memUtilization: state.memUtilization,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedModel, state.concurrentUsers, state.contextLength, state.precision,
      state.deploymentType, selectedGpu, state.tensorParallelism, state.memUtilization]
  );

  if (state.view === "results") {
    return <ResultsView state={state} setState={setState} />;
  }

  return (
    <div className="rh-qe-layout">
      <div className="rh-qe-main">
        <ConfigPanel state={state} setState={setState} />
      </div>
      <LiveEstimatePanel
        result={result}
        state={state}
        selectedModel={selectedModel}
        selectedGpu={selectedGpu}
        onGpuChange={(gpuId) => setState((s) => ({ ...s, gpuId }))}
      />
    </div>
  );
}
