'use client';

import * as React from 'react';
import { Label, Accordion, AccordionItem, AccordionToggle, AccordionContent } from '@patternfly/react-core';
import MicrochipIcon from '@patternfly/react-icons/dist/esm/icons/microchip-icon';
import MemoryIcon from '@patternfly/react-icons/dist/esm/icons/memory-icon';
import ClockIcon from '@patternfly/react-icons/dist/esm/icons/clock-icon';
import TachometerAltIcon from '@patternfly/react-icons/dist/esm/icons/tachometer-alt-icon';
import EyeIcon from '@patternfly/react-icons/dist/esm/icons/eye-icon';
import EyeSlashIcon from '@patternfly/react-icons/dist/esm/icons/eye-slash-icon';
import ExclamationTriangleIcon from '@patternfly/react-icons/dist/esm/icons/exclamation-triangle-icon';
import CheckCircleIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import InfoCircleIcon from '@patternfly/react-icons/dist/esm/icons/info-circle-icon';

import styles from './AdvancedEstimate.module.css';
import { MODEL_CATALOG } from '@/lib/gpu-math/models';
import { GPU_CATALOG, GPU_OPTIONS_ADV } from '@/lib/gpu-math/gpus';
import { fetchModelConfig } from '@/lib/huggingface/fetch-config';
import { useGpuSizer } from '@/contexts/GpuSizerContext';
import { GpuChipLoader } from '@/components/GpuChipLoader/GpuChipLoader';

const MODEL_OPTIONS = MODEL_CATALOG.map(m => m.hfId);

// ─── FlipTile (reused from Quick Estimate pattern) ───────────────────────────

function FlipTile({ dark = false, front, back }: {
  dark?: boolean; front: React.ReactNode; back: React.ReactNode;
}) {
  const [flipped, setFlipped] = React.useState(false);
  return (
    <div
      className={`${styles.flip} ${dark ? styles.tileDark : ''} ${flipped ? styles.flipped : ''}`}
      role="button" tabIndex={0} aria-pressed={flipped}
      onClick={() => setFlipped(f => !f)}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped(f => !f); } }}
    >
      <div className={`${styles.flipFace} ${styles.flipFront}`}>
        {front}
        <span className={styles.seeMath}>↻ see math</span>
      </div>
      <div className={`${styles.flipFace} ${styles.flipBack}`}>
        {back}
        <span className={styles.seeMath}>↻ flip back</span>
      </div>
    </div>
  );
}

// ─── useCountUp (reused from Quick Estimate) ─────────────────────────────────

function useCountUp(target: number, duration = 750, decimals = 0) {
  const [val, setVal] = React.useState(target);
  const raf = React.useRef<number>(0);
  React.useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVal(target); return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 4);
      setVal(target * e);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

// ─── Friendly error messages ────────────────────────────────────────────────

function friendlyErrorTitle(code: string | null): string {
  switch (code) {
    case 'GPU_SIZER_TIMEOUT': return 'Request timed out';
    case 'GPU_SIZER_NO_CONFIGURATION': return 'No valid configuration found';
    case 'GPU_SIZER_UNAVAILABLE': return 'Sizing service unavailable';
    case 'GPU_SIZER_AUTH_FAILED': return 'Service authentication error';
    case 'GPU_SIZER_NOT_CONFIGURED': return 'Service not configured';
    case 'GPU_SIZER_INVALID_RESPONSE': return 'Unexpected response';
    case 'INVALID_REQUEST': return 'Invalid input';
    case 'NETWORK_ERROR': return 'Connection error';
    default: return 'Something went wrong';
  }
}

function friendlyErrorMessage(code: string | null, raw: string): string {
  switch (code) {
    case 'GPU_SIZER_TIMEOUT':
      return 'The sizing engine took too long to respond. This can happen with very large models or complex configurations.';
    case 'GPU_SIZER_NO_CONFIGURATION':
      return 'This model and GPU combination doesn’t have a valid sizing configuration. The engine couldn’t find a workable setup.';
    case 'GPU_SIZER_UNAVAILABLE':
      return 'The GPU sizing service is temporarily unreachable. This is usually a transient issue.';
    case 'GPU_SIZER_AUTH_FAILED':
      return 'The sizing service rejected our credentials. Please contact your administrator.';
    case 'GPU_SIZER_NOT_CONFIGURED':
      return 'The sizing service hasn’t been set up yet. Please contact your administrator.';
    case 'GPU_SIZER_INVALID_RESPONSE':
      return 'The sizing engine returned an unexpected response format.';
    case 'INVALID_REQUEST':
      return 'Some input values are missing or invalid. Please check your model name and parameters.';
    case 'NETWORK_ERROR':
      return 'Could not connect to the sizing service. Please check your internet connection.';
    default:
      return raw;
  }
}

function friendlyErrorHint(code: string | null): string {
  switch (code) {
    case 'GPU_SIZER_TIMEOUT':
      return 'Try again, or try a smaller model or simpler configuration.';
    case 'GPU_SIZER_NO_CONFIGURATION':
      return 'Try a different GPU system, or reduce the input token length (ISL).';
    case 'GPU_SIZER_UNAVAILABLE':
      return 'Wait a moment and try again.';
    case 'NETWORK_ERROR':
      return 'Check your connection and try again.';
    case 'INVALID_REQUEST':
      return 'Make sure the model name is a valid Hugging Face ID (e.g. meta-llama/Llama-3.1-70B-Instruct).';
    default:
      return 'If this persists, try a different model or GPU combination.';
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function AdvancedEstimate() {
  // Input state
  const [model, setModel] = React.useState('meta-llama/Llama-3.1-70B-Instruct');
  const [gpuSystem, setGpuSystem] = React.useState(
    GPU_OPTIONS_ADV.find(g => g.systemId === 'h200_sxm')?.systemId ?? GPU_OPTIONS_ADV[0]?.systemId ?? ''
  );
  const [isl, setIsl] = React.useState(2048);
  const [osl, setOsl] = React.useState(128);
  const [ttft, setTtft] = React.useState(1000);

  // HF token
  const [hfToken, setHfToken] = React.useState('');
  const [hfReveal, setHfReveal] = React.useState(false);
  const [showHfSection, setShowHfSection] = React.useState(false);

  // Model status + HF config
  const [modelStatus, setModelStatus] = React.useState<'idle' | 'fetching' | 'catalog' | 'fetched' | 'error'>('idle');

  // GPU sizer (persistent across navigation)
  const { isLoading, result, error, errorCode, elapsed, startSizing } = useGpuSizer();

  // Additional constraints accordion
  const [expanded, setExpanded] = React.useState<string[]>(['perf']);

  // Live pricing
  const [livePricing, setLivePricing] = React.useState<Record<string, number>>({});

  // Load HF token from localStorage
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('hf_token');
      if (saved?.startsWith('hf_')) {
        setHfToken(saved);
        setShowHfSection(true);
      }
    }
  }, []);

  // Model status check + fetch HF config
  React.useEffect(() => {
    const timer = setTimeout(() => {
      if (!model.includes('/')) { setModelStatus('idle'); return; }
      const inCatalog = MODEL_CATALOG.some(m => m.hfId === model);
      setModelStatus(inCatalog ? 'catalog' : 'fetching');
      fetchModelConfig(model, hfToken).then(r => {
        if (r.success && r.config) {
          setModelStatus(inCatalog ? 'catalog' : 'fetched');
        } else {
          if (!inCatalog) setModelStatus('error');
        }
      });
    }, 500);
    return () => clearTimeout(timer);
  }, [model, hfToken]);

  // Fetch live pricing
  React.useEffect(() => {
    const fetchPricing = async () => {
      try {
        const res = await fetch('/api/v1/gpus?live_pricing=true');
        if (!res.ok) return;
        const data = await res.json();
        if (!data?.data?.gpus) return;
        const prices: Record<string, number> = {};
        for (const g of data.data.gpus) {
          if (g.live_pricing?.onDemand?.median) {
            const shortName = g.name.replace(/NVIDIA\s+/i, '').replace(/AMD\s+/i, '').split(' ')[0];
            prices[shortName] = g.live_pricing.onDemand.median;
          }
        }
        setLivePricing(prices);
      } catch { /* ignore */ }
    };
    fetchPricing();
  }, []);

  // Save HF token
  const handleTokenChange = (val: string) => {
    setHfToken(val);
    if (typeof window !== 'undefined' && val.startsWith('hf_')) {
      localStorage.setItem('hf_token', val);
    }
  };

  // Get GPU + model spec from catalog
  const currentGpuOption = GPU_OPTIONS_ADV.find(g => g.systemId === gpuSystem) || GPU_OPTIONS_ADV[0];
  const gpuSpec = GPU_CATALOG.find(g => g.id === currentGpuOption.id);
  const catalogModel = MODEL_CATALOG.find(m => m.hfId === model);

  const handleCalculate = () => {
    startSizing({ model_path: model, system: gpuSystem, isl, osl, ttft });
  };

  // Local memory analysis using the same engine as Quick Estimate
  // Animated values
  const gpuCount = useCountUp(result?.recommendation.gpusNeeded ?? 0);
  const ttftMs = useCountUp(result?.performance.ttftLatencyMs ?? 0, 750, 0);
  const tpsVal = useCountUp(result?.throughput.tokensPerSecond ?? 0, 750, 0);
  const memVal = useCountUp(result?.memory.value ?? 0, 750, 1);

  // Cost calculations
  const gpuShortName = currentGpuOption.label.replace(/NVIDIA\s+/i, '').replace(/AMD\s+/i, '').split(' ')[0];
  const livePrice = livePricing[gpuShortName];
  const hwCost = gpuSpec?.hardware_cost_usd ?? 30000;
  const pricePerHour = livePrice ?? hwCost / (36 * 730);
  const numGpus = result?.recommendation.totalGpus ?? 0;
  const monthlyCost = numGpus * pricePerHour * 730;

  return (
    <div className={styles.page}>
      {/* ─── Header ─── */}
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>GPU requirement sizing</h1>
        <p className={styles.subtitle}>
          Start with just a model name — we fill the rest, then let you tune every assumption.
        </p>
      </div>

      {/* ─── Input card ─── */}
      <div className={`${styles.card} ${styles.inputCard}`}>
        {/* Model + GPU row */}
        <div className={styles.inputGrid}>
          <div>
            <label className={styles.fieldLabel}>
              Model — Hugging Face ID
              <StatusChip status={modelStatus} />
            </label>
            <div className={styles.modelInputWrapper}>
              <input
                type="text"
                className={styles.modelInput}
                value={model}
                onChange={e => setModel(e.target.value)}
                list="model-options"
                placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct"
              />
              <datalist id="model-options">
                {MODEL_OPTIONS.map(m => <option key={m} value={m} />)}
              </datalist>
              <div className={styles.autoChipWrapper}>
                {modelStatus === 'catalog' && (
                  <Label color="green" isCompact icon={<CheckCircleIcon />}>In catalog</Label>
                )}
                {modelStatus === 'fetching' && (
                  <Label color="blue" isCompact>Fetching...</Label>
                )}
                {modelStatus === 'fetched' && (
                  <Label color="cyan" isCompact icon={<CheckCircleIcon />}>From HuggingFace</Label>
                )}
                {modelStatus === 'error' && (
                  <Label color="red" isCompact icon={<ExclamationTriangleIcon />}>Not found</Label>
                )}
              </div>
            </div>
            <div className={styles.helperText}>
              Popular models: Llama 3.1, Mistral, Qwen 2.5, Gemma 2 — type to autocomplete
            </div>
          </div>

          <div>
            <label className={styles.fieldLabel}>GPU system</label>
            <select
              className={styles.gpuSelect}
              value={gpuSystem}
              onChange={e => setGpuSystem(e.target.value)}
            >
              {GPU_OPTIONS_ADV.map(g => (
                <option key={g.systemId} value={g.systemId}>{g.label}</option>
              ))}
            </select>
            {gpuSpec && (
              <div className={styles.helperText}>
                {gpuSpec.vram_gb} GB · {gpuSpec.memory_bandwidth_tbps} TB/s · {gpuSpec.tflops_bf16} TFLOPS
              </div>
            )}
          </div>
        </div>

        {/* Calculate button */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 16, marginTop: 16 }}>
          <button
            className={styles.calcBtn}
            onClick={handleCalculate}
            disabled={isLoading || !model.includes('/')}
          >
            {isLoading ? 'Calculating...' : 'Calculate GPU Requirement'}
          </button>
        </div>
      </div>

      {/* ─── Default assumptions info strip ─── */}
      <div className={styles.infoStrip}>
        <InfoCircleIcon style={{ color: '#0066cc', flexShrink: 0 }} />
        <span>
          Based on your configuration — ISL {isl.toLocaleString()}, OSL {osl}, TTFT target {(ttft / 1000).toFixed(1)}s.
          &nbsp;
          <button
            type="button"
            onClick={() => setExpanded(expanded.includes('customize') ? expanded.filter(e => e !== 'customize') : [...expanded, 'customize'])}
            style={{ background: 'none', border: 0, color: '#0066cc', fontWeight: 600, cursor: 'pointer', fontSize: 14, padding: 0 }}
          >
            Customize? (edit fields below)
          </button>
        </span>
      </div>

      {/* ─── Customization section (ISL/OSL/TTFT + HF token + constraints) ─── */}
      {expanded.includes('customize') && (
        <div className={`${styles.card}`} style={{ marginBottom: 16 }}>
          <div className={styles.cardBody}>
            <div className={styles.paramGrid}>
              <div>
                <label className={styles.fieldLabel}>Avg input tokens (ISL)</label>
                <input
                  type="number"
                  className={styles.paramInput}
                  value={isl}
                  onChange={e => setIsl(Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Avg output tokens (OSL)</label>
                <input
                  type="number"
                  className={styles.paramInput}
                  value={osl}
                  onChange={e => setOsl(Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                />
              </div>
              <div>
                <label className={styles.fieldLabel}>Max TTFT (seconds)</label>
                <input
                  type="number"
                  className={styles.paramInput}
                  value={ttft / 1000}
                  onChange={e => {
                    const sec = parseFloat(e.target.value);
                    if (!isNaN(sec) && sec > 0) setTtft(Math.round(sec * 1000));
                  }}
                  min={0.1}
                  step={0.1}
                />
              </div>
            </div>

            {/* HF Token */}
            <div className={styles.hfSection}>
              <div className={styles.fieldLabel} style={{ marginBottom: 8 }}>
                Hugging Face token (optional — for gated or private models)
              </div>
              <div className={styles.hfRow}>
                <input
                  type={hfReveal ? 'text' : 'password'}
                  className={styles.hfInput}
                  value={hfToken}
                  onChange={e => handleTokenChange(e.target.value)}
                  placeholder="hf_..."
                />
                <button
                  type="button"
                  onClick={() => setHfReveal(!hfReveal)}
                  style={{ background: 'none', border: '1px solid #d2d2d2', borderRadius: 4, padding: '8px 10px', cursor: 'pointer' }}
                  aria-label={hfReveal ? 'Hide token' : 'Show token'}
                >
                  {hfReveal ? <EyeSlashIcon /> : <EyeIcon />}
                </button>
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener"
                  style={{ color: '#0066cc', fontWeight: 500, whiteSpace: 'nowrap', fontSize: 14 }}
                >
                  Get a token
                </a>
              </div>
              <div className={styles.hfNote}>
                Stored in this browser only — never sent to our servers.
              </div>
            </div>

            {/* Additional constraints */}
            <Accordion style={{ marginTop: 12 }}>
              <AccordionItem>
                <AccordionToggle
                  id="constraints-toggle"
                  onClick={() => setExpanded(
                    expanded.includes('constraints') ? expanded.filter(e => e !== 'constraints') : [...expanded, 'constraints']
                  )}
                  isExpanded={expanded.includes('constraints')}
                >
                  Additional constraints
                </AccordionToggle>
                <AccordionContent isHidden={!expanded.includes('constraints')}>
                  <div className={styles.paramGrid} style={{ marginTop: 8 }}>
                    <div>
                      <label className={styles.fieldLabel}>Batch size (optional)</label>
                      <input type="number" className={styles.paramInput} placeholder="Auto" min={1} />
                    </div>
                    <div>
                      <label className={styles.fieldLabel}>Tokens/sec per user (optional)</label>
                      <input type="number" className={styles.paramInput} placeholder="Auto" min={1} step={0.1} />
                    </div>
                    <div>
                      <label className={styles.fieldLabel}>E2E latency ms (optional)</label>
                      <input type="number" className={styles.paramInput} placeholder="Auto" min={1} />
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </div>
      )}

      {/* ─── Loading ─── */}
      {isLoading && (
        <div className={styles.card}>
          <GpuChipLoader elapsed={elapsed} />
        </div>
      )}

      {/* ─── Error ─── */}
      {error && (
        <div className={styles.errorWrap}>
          <div className={styles.errorTitle}>
            <ExclamationTriangleIcon /> {friendlyErrorTitle(errorCode)}
          </div>
          <div className={styles.errorMsg}>{friendlyErrorMessage(errorCode, error)}</div>
          <div className={styles.errorHint}>{friendlyErrorHint(errorCode)}</div>
        </div>
      )}

      {/* ─── Result tiles ─── */}
      {result && (
        <>
          <div className={styles.tilesGrid}>
            {/* GPUs Required */}
            <FlipTile
              dark
              front={
                <>
                  <span className={styles.tileLabel}><MicrochipIcon /> GPUs required</span>
                  <span className={styles.tileValue}>
                    {gpuCount}<span className={styles.tileUnit}>× {currentGpuOption.label}</span>
                  </span>
                  <span className={styles.tileSub}>
                    TP {result.recommendation.tensorParallelSize} · PP {result.recommendation.pipelineParallelSize} · DP {result.recommendation.dataParallelSize} · {result.performance.concurrency} concurrent users
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>GPU topology</div>
                  <div className={styles.formula}>
                    tensor parallel = <span className={styles.em}>{result.recommendation.tensorParallelSize}</span><br />
                    pipeline parallel = <span className={styles.em}>{result.recommendation.pipelineParallelSize}</span><br />
                    data parallel = <span className={styles.em}>{result.recommendation.dataParallelSize}</span><br />
                    total = TP×PP×DP = <span className={styles.em}>{result.recommendation.totalGpus} GPUs</span>
                  </div>
                </>
              }
            />

            {/* TTFT */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><ClockIcon /> TTFT</span>
                  <span className={styles.tileValue}>
                    {ttftMs}<span className={styles.tileUnit}>ms</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.performance.ttftLatencyMs <= ttft ? (
                      <Label color="green" isCompact icon={<CheckCircleIcon />}>meets target</Label>
                    ) : (
                      <Label color="orange" isCompact icon={<ExclamationTriangleIcon />}>above target</Label>
                    )}
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Time to first token</div>
                  <div className={styles.formula}>
                    target: <span className={styles.em}>{ttft.toLocaleString()} ms</span><br />
                    estimated: <span className={styles.em}>{result.performance.ttftLatencyMs.toFixed(1)} ms</span><br />
                    headroom: <span className={styles.em}>{(ttft - result.performance.ttftLatencyMs).toFixed(1)} ms</span><br />
                    TPOT: <span className={styles.em}>{result.performance.tpotMs.toFixed(1)} ms</span>
                  </div>
                </>
              }
            />

            {/* Throughput */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><TachometerAltIcon /> Throughput</span>
                  <span className={styles.tileValue}>
                    {tpsVal}<span className={styles.tileUnit}>tok/s</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.throughput.tokensPerSecondPerGpu.toFixed(1)}/GPU · {result.throughput.tokensPerSecondPerUser.toFixed(2)}/user
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Throughput breakdown</div>
                  <div className={styles.formula}>
                    total: <span className={styles.em}>{result.throughput.tokensPerSecond.toFixed(1)} tok/s</span><br />
                    per GPU: <span className={styles.em}>{result.throughput.tokensPerSecondPerGpu.toFixed(1)} tok/s</span><br />
                    per user: <span className={styles.em}>{result.throughput.tokensPerSecondPerUser.toFixed(2)} tok/s</span><br />
                    concurrency: <span className={styles.em}>{result.performance.concurrency}</span>
                  </div>
                </>
              }
            />

            {/* Est. Memory */}
            <FlipTile
              front={
                <>
                  <span className={styles.tileLabel}><MemoryIcon /> Est. memory</span>
                  <span className={styles.tileValue}>
                    {memVal}<span className={styles.tileUnit}>GB</span>
                  </span>
                  <span className={styles.tileSub}>
                    {result.recommendation.totalGpus === 1
                      ? '1 GPU per model instance'
                      : `${result.recommendation.totalGpus} GPUs per model instance`}
                  </span>
                </>
              }
              back={
                <>
                  <div className={styles.backTitle}>Memory estimate</div>
                  <div className={styles.formula}>
                    memory: <span className={styles.em}>{result.memory.value.toFixed(1)} {result.memory.unit}</span><br />
                    scope: <span className={styles.em}>{result.memory.scope}</span><br />
                    GPUs: <span className={styles.em}>{result.recommendation.totalGpus}</span><br />
                    ~{(result.memory.value / result.recommendation.totalGpus).toFixed(1)} GB/GPU
                  </div>
                </>
              }
            />
          </div>

          {/* ─── Estimated serving performance ─── */}
          <div className={styles.card} style={{ marginBottom: 24 }}>
            <Accordion>
              <AccordionItem>
                <AccordionToggle
                  id="perf-toggle"
                  onClick={() => setExpanded(
                    expanded.includes('perf') ? expanded.filter(e => e !== 'perf') : [...expanded, 'perf']
                  )}
                  isExpanded={expanded.includes('perf')}
                >
                  <span style={{ fontWeight: 600 }}>Estimated serving performance</span>
                </AccordionToggle>
                <AccordionContent isHidden={!expanded.includes('perf')}>
                  <div className={styles.cardBody}>
                    <div className={styles.paramGrid}>
                      <div>
                        <div className={styles.fieldLabel}>Request latency</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {(result.performance.requestLatencyMs / 1000).toFixed(1)}s
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          End-to-end for {osl} output tokens
                        </div>
                      </div>
                      <div>
                        <div className={styles.fieldLabel}>Concurrency</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {result.performance.concurrency}
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          Concurrent users supported
                        </div>
                      </div>
                      <div>
                        <div className={styles.fieldLabel}>TPOT</div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700 }}>
                          {result.performance.tpotMs.toFixed(1)} ms
                        </div>
                        <div style={{ fontSize: 13, color: '#3c3f42', marginTop: 4 }}>
                          Time per output token
                        </div>
                      </div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>

          {/* Memory layout bar hidden — local estimates don't match API internals */}

          {/* ─── Warnings ─── */}
          {result.warnings.length > 0 && (
            <div className={styles.warningsList}>
              {result.warnings.map((w, i) => (
                <div key={i} className={styles.warningItem}>
                  <ExclamationTriangleIcon style={{ color: '#f0ab00', flexShrink: 0, marginTop: 1 }} />
                  <span>{w.message}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Status chip ─────────────────────────────────────────────────────────────

function StatusChip({ status }: { status: string }) {
  if (status === 'idle') return null;
  return null; // Chip is rendered inside the input wrapper instead
}
