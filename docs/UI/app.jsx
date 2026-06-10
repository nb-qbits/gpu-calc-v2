// ─────────────────────────────────────────────────────────────────────────
// GPU Calc — Quick Estimate
// Live-updating split-screen: configure on the left, results on the right.
// ─────────────────────────────────────────────────────────────────────────
const { useState, useMemo, useEffect, useRef } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#B4FE2C",
  "theme": "dark",
  "layout": "split",
  "showAlternatives": true,
  "showCostOutlook": true
}/*EDITMODE-END*/;

// ─────────────────────────────────────────────────────────────────────────
// Animated number — smoothly tweens to its target so live recalcs feel
// like physical instrumentation, not jumpy text.
// ─────────────────────────────────────────────────────────────────────────
function AnimNumber({ value, fmt = (n) => n.toFixed(0), duration = 280 }) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const toRef = useRef(value);
  const startRef = useRef(performance.now());
  const rafRef = useRef(null);

  useEffect(() => {
    fromRef.current = display;
    toRef.current = value;
    startRef.current = performance.now();
    cancelAnimationFrame(rafRef.current);
    const tick = (t) => {
      const k = Math.min(1, (t - startRef.current) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      const cur = fromRef.current + (toRef.current - fromRef.current) * eased;
      setDisplay(cur);
      if (k < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration]);

  return <>{fmt(display)}</>;
}

// ─────────────────────────────────────────────────────────────────────────
// Filter chip row + model card grid
// ─────────────────────────────────────────────────────────────────────────
function ModelGrid({ selectedId, onSelect }) {
  const [filter, setFilter] = useState('All');
  const visible = filter === 'All' ? MODELS : MODELS.filter((m) => m.family === filter);

  return (
    <div>
      <div className="chip-row">
        {FAMILIES.map((f) => (
          <button
            key={f}
            className={`chip ${filter === f ? 'on' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>
      <div className="model-grid">
        {visible.map((m) => {
          const active = m.id === selectedId;
          return (
            <button
              key={m.id}
              className={`model-card ${active ? 'on' : ''}`}
              onClick={() => onSelect(m.id)}
            >
              <div className="model-card-top">
                <span className="model-name">{m.name}</span>
                {m.badge && <span className="model-badge">{m.badge}</span>}
                {m.popular && !m.badge && <span className="model-dot" title="Popular" />}
              </div>
              <div className="model-card-mid">
                <span className="model-size">{m.size}</span>
                {m.moe && <span className="model-moe">MoE</span>}
              </div>
              <div className="model-card-bot">
                <span className="model-family">{m.family}</span>
                <span className="model-layers">{m.layers}L</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Segment selector (for context, precision, deployment)
// ─────────────────────────────────────────────────────────────────────────
function Segment({ options, value, onChange, render }) {
  return (
    <div className="segment">
      {options.map((opt) => {
        const id = opt.id ?? opt;
        const active = (opt.id ?? opt) === value;
        return (
          <button
            key={id}
            className={`seg-btn ${active ? 'on' : ''}`}
            onClick={() => onChange(id)}
          >
            {render ? render(opt) : opt.label}
          </button>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section header — small numbered eyebrow + title
// ─────────────────────────────────────────────────────────────────────────
function Step({ num, title, hint, children }) {
  return (
    <section className="step">
      <header className="step-h">
        <span className="step-num">{String(num).padStart(2, '0')}</span>
        <span className="step-title">{title}</span>
        {hint && <span className="step-hint">{hint}</span>}
      </header>
      <div className="step-body">{children}</div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// VRAM breakdown bar — weights + KV cache + overhead, stacked horizontally,
// against the recommended GPU's total VRAM.
// ─────────────────────────────────────────────────────────────────────────
function VRAMBar({ result, gpuTotalGB }) {
  const { weightsGB, kvTotalGB, overheadGB, totalGB } = result;
  const denom = Math.max(totalGB, gpuTotalGB || totalGB);
  const wPct = (weightsGB / denom) * 100;
  const kPct = (kvTotalGB / denom) * 100;
  const oPct = (overheadGB / denom) * 100;
  const usedPct = ((totalGB) / denom) * 100;
  return (
    <div className="vram-bar-wrap">
      <div className="vram-bar">
        <div className="vram-seg seg-weights" style={{ width: `${wPct}%` }} title={`Weights ${fmtGB(weightsGB)}`} />
        <div className="vram-seg seg-kv"      style={{ width: `${kPct}%` }} title={`KV cache ${fmtGB(kvTotalGB)}`} />
        <div className="vram-seg seg-oh"      style={{ width: `${oPct}%` }} title={`Overhead ${fmtGB(overheadGB)}`} />
        {gpuTotalGB && (
          <div className="vram-mark" style={{ left: `${Math.min(100, usedPct)}%` }} />
        )}
      </div>
      <div className="vram-legend">
        <span className="lg lg-w"><i /> Weights <b>{fmtGB(weightsGB)}</b></span>
        <span className="lg lg-k"><i /> KV cache <b>{fmtGB(kvTotalGB)}</b></span>
        <span className="lg lg-o"><i /> Overhead <b>{fmtGB(overheadGB)}</b></span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Right-pane result card. The hero number is total VRAM; below it the
// recommended GPU config, alternatives, and cost outlook.
// ─────────────────────────────────────────────────────────────────────────
function ResultPanel({ state, result, tweaks }) {
  const cheapest = result.cheapest;
  const fastest = result.fastest;
  const showAlts = tweaks.showAlternatives;
  const showCost = tweaks.showCostOutlook;

  const altList = result.configs
    .filter((c) => c.gpu.id !== cheapest?.gpu.id || c.count !== cheapest?.count)
    .slice(0, 3);

  const deploy = state.deployment;

  return (
    <aside className="result">
      <div className="result-head">
        <div className="result-eyebrow">
          <span className="dot pulse" />
          Live estimate
        </div>
        <div className="result-hint">
          {MODELS.find((m) => m.id === state.modelId).name} {MODELS.find((m) => m.id === state.modelId).size}
          {' · '}
          {state.concurrentUsers.toLocaleString()} users
          {' · '}
          {fmtTokens(state.contextTokens)} ctx
          {' · '}
          {PRECISIONS[state.precision].label}
        </div>
      </div>

      <div className="hero">
        <div className="hero-label">Total VRAM required</div>
        <div className="hero-num">
          <span className="hero-val"><AnimNumber value={result.totalGB} fmt={(n) => n < 10 ? n.toFixed(1) : n.toFixed(0)} /></span>
          <span className="hero-unit">GB</span>
        </div>
        <VRAMBar result={result} gpuTotalGB={cheapest ? cheapest.gpu.vramGB * cheapest.count : null} />
      </div>

      <div className="rec">
        <div className="rec-eyebrow">Recommended config</div>
        {cheapest ? (
          <div className="rec-card">
            <div className="rec-card-left">
              <div className="rec-count">{cheapest.count}×</div>
              <div>
                <div className="rec-name">
                  {cheapest.gpu.name}
                  {cheapest.gpu.variant && <span className="rec-variant"> {cheapest.gpu.variant}</span>}
                </div>
                <div className="rec-sub">
                  {cheapest.gpu.vramGB} GB each · {fmtGB(cheapest.gpu.vramGB * cheapest.count)} pooled
                </div>
              </div>
            </div>
            <div className="rec-card-right">
              <div className="util-ring">
                <svg viewBox="0 0 36 36" width="48" height="48">
                  <circle cx="18" cy="18" r="15.9" fill="none" stroke="currentColor" strokeOpacity="0.12" strokeWidth="3" />
                  <circle
                    cx="18" cy="18" r="15.9" fill="none"
                    stroke="currentColor" strokeWidth="3" strokeLinecap="round"
                    strokeDasharray={`${cheapest.utilization * 100}, 100`}
                    transform="rotate(-90 18 18)"
                    className="util-stroke"
                  />
                </svg>
                <span className="util-pct"><AnimNumber value={cheapest.utilization * 100} fmt={(n) => `${n.toFixed(0)}%`} /></span>
              </div>
              <div className="util-label">Utilization</div>
            </div>
          </div>
        ) : (
          <div className="rec-card rec-card-empty">
            <div>
              <div className="rec-name">Exceeds single-node config</div>
              <div className="rec-sub">Try INT8/INT4 or shorter context — or move to multi-node.</div>
            </div>
          </div>
        )}
      </div>

      <div className="costs">
        <div className={`cost-card ${deploy === 'cloud' ? 'primary' : ''}`}>
          <div className="cost-label">Cloud, on-demand</div>
          <div className="cost-val">
            <AnimNumber value={cheapest ? cheapest.hourly : 0} fmt={(n) => `$${n.toFixed(2)}`} />
            <span className="cost-unit">/hr</span>
          </div>
          <div className="cost-sub">
            <AnimNumber value={cheapest ? cheapest.yearlyCloud : 0} fmt={(n) => fmtMoney(n, { compact: true })} /> /yr 24×7
          </div>
        </div>
        <div className={`cost-card ${deploy === 'onprem' ? 'primary' : ''}`}>
          <div className="cost-label">On-prem capex</div>
          <div className="cost-val">
            <AnimNumber value={cheapest ? cheapest.capex : 0} fmt={(n) => fmtMoney(n, { compact: true })} />
            <span className="cost-unit"> GPUs</span>
          </div>
          <div className="cost-sub">~{fmtMoney(cheapest ? cheapest.capex * 0.6 : 0, { compact: true })} infra · 3yr life</div>
        </div>
        <div className={`cost-card ${deploy === 'hybrid' ? 'primary' : ''}`}>
          <div className="cost-label">Hybrid breakeven</div>
          <div className="cost-val">
            <AnimNumber
              value={cheapest ? (cheapest.capex / cheapest.hourly) / 24 / 30 : 0}
              fmt={(n) => `${n.toFixed(1)}mo`}
            />
          </div>
          <div className="cost-sub">when capex beats cloud burn</div>
        </div>
      </div>

      {showAlts && (
        <div className="alts">
          <div className="alts-head">
            <span className="alts-label">Alternatives</span>
            <span className="alts-sub">cheapest first</span>
          </div>
          <div className="alts-list">
            {altList.map((c) => (
              <div className="alt-row" key={c.gpu.id + c.count}>
                <span className="alt-count">{c.count}×</span>
                <span className="alt-name">
                  {c.gpu.name}
                  {c.gpu.variant && <span className="alt-variant"> {c.gpu.variant}</span>}
                </span>
                <span className="alt-vram">{c.gpu.vramGB}GB</span>
                <span className="alt-bar">
                  <span className="alt-bar-fill" style={{ width: `${c.utilization * 100}%` }} />
                </span>
                <span className="alt-price">${c.hourly.toFixed(2)}/hr</span>
              </div>
            ))}
            {altList.length === 0 && (
              <div className="alt-empty">No alternative configs found for these constraints.</div>
            )}
          </div>
        </div>
      )}

      {showCost && cheapest && (
        <div className="outlook">
          <div className="outlook-head">3-year cost outlook</div>
          <CostOutlook cheapest={cheapest} deployment={state.deployment} />
        </div>
      )}

      <button className="cta">
        Get full estimate
        <span className="cta-arrow" aria-hidden>→</span>
      </button>
    </aside>
  );
}

// 3-year stacked cost line — cloud (linear) vs on-prem (capex + maintenance).
function CostOutlook({ cheapest, deployment }) {
  const months = 36;
  const cloudPerMonth = cheapest.hourly * 24 * 30;
  const onpremMonth0 = cheapest.capex;
  const onpremMaintPerMonth = cheapest.capex * 0.015; // 1.5%/mo opex

  const cloud = Array.from({ length: months + 1 }, (_, i) => cloudPerMonth * i);
  const onprem = Array.from({ length: months + 1 }, (_, i) => onpremMonth0 + onpremMaintPerMonth * i);
  const maxY = Math.max(cloud[months], onprem[months]) * 1.05;

  const W = 320, H = 84, P = 4;
  const x = (i) => P + (i / months) * (W - P * 2);
  const y = (v) => H - P - (v / maxY) * (H - P * 2);

  const cloudPath = cloud.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const onpremPath = onprem.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

  // intersection
  let cross = null;
  for (let i = 1; i <= months; i++) {
    if ((cloud[i - 1] < onprem[i - 1]) !== (cloud[i] < onprem[i])) { cross = i; break; }
  }

  return (
    <div className="outlook-body">
      <svg className="outlook-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id="cloudFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="currentColor" stopOpacity="0.18" />
            <stop offset="1" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${cloudPath} L${x(months)},${H - P} L${x(0)},${H - P} Z`} fill="url(#cloudFill)" className="outlook-cloud-fill" />
        <path d={cloudPath} className="outlook-cloud" />
        <path d={onpremPath} className="outlook-onprem" />
        {cross && (
          <g>
            <line x1={x(cross)} y1={P} x2={x(cross)} y2={H - P} className="outlook-cross-line" />
            <circle cx={x(cross)} cy={y(cloud[cross])} r="2.5" className="outlook-cross-dot" />
          </g>
        )}
      </svg>
      <div className="outlook-legend">
        <span className="ol ol-cloud"><i />Cloud {fmtMoney(cloud[months], { compact: true })}</span>
        <span className="ol ol-onprem"><i />On-prem {fmtMoney(onprem[months], { compact: true })}</span>
        {cross && <span className="ol ol-cross">Crossover ~mo {cross}</span>}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// App
// ─────────────────────────────────────────────────────────────────────────
function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  const [modelId, setModelId] = useState('llama-31-70b');
  const [concurrentUsers, setConcurrentUsers] = useState(100);
  const [concurrencyPreset, setConcurrencyPreset] = useState('org');
  const [contextPreset, setContextPreset] = useState('medium');
  const [precision, setPrecision] = useState('fp16');
  const [deployment, setDeployment] = useState('onprem');

  const model = useMemo(() => MODELS.find((m) => m.id === modelId), [modelId]);
  const contextTokens = CONTEXT_PRESETS.find((c) => c.id === contextPreset).tokens;

  const result = useMemo(() => computeRequirements({
    model, precision, contextTokens, concurrentUsers,
  }), [model, precision, contextTokens, concurrentUsers]);

  const state = { modelId, concurrentUsers, contextTokens, precision, deployment };

  // ── Theme application ────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.dataset.theme = t.theme;
    document.documentElement.style.setProperty('--accent', t.accent);
  }, [t.theme, t.accent]);

  function pickConcurrency(presetId) {
    const p = CONCURRENCY_PRESETS.find((c) => c.id === presetId);
    setConcurrencyPreset(presetId);
    setConcurrentUsers(p.value);
  }

  return (
    <div className={`app layout-${t.layout}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none">
              <rect x="3" y="3" width="18" height="18" rx="4" stroke="currentColor" strokeWidth="1.5" />
              <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" />
              <rect x="2" y="9"  width="2" height="2" fill="currentColor" />
              <rect x="2" y="13" width="2" height="2" fill="currentColor" />
              <rect x="20" y="9" width="2" height="2" fill="currentColor" />
              <rect x="20" y="13" width="2" height="2" fill="currentColor" />
              <rect x="9"  y="2" width="2" height="2" fill="currentColor" />
              <rect x="13" y="2" width="2" height="2" fill="currentColor" />
              <rect x="9"  y="20" width="2" height="2" fill="currentColor" />
              <rect x="13" y="20" width="2" height="2" fill="currentColor" />
            </svg>
          </span>
          <span className="brand-name">GPU Calc</span>
        </div>
        <nav className="nav">
          <a href="#" className="nav-link">Home</a>
          <a href="#" className="nav-link on">Quick Estimate</a>
          <a href="#" className="nav-link">Advanced</a>
          <a href="#" className="nav-link">GPU Explorer</a>
          <a href="#" className="nav-link">Hybrid Savings</a>
          <a href="#" className="nav-link">Routing</a>
        </nav>
        <div className="topbar-right">
          <span className="kbd">⌘K</span>
          <button className="ghost-btn">Docs</button>
        </div>
      </header>

      <main className="canvas" data-screen-label="01 Quick Estimate">
        <section className="config">
          <div className="config-head">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              Quick estimate
            </div>
            <h1 className="title">Configure your <em>LLM deployment</em></h1>
            <p className="lede">Pick a model, your concurrency, and how much context each session needs. We estimate VRAM, recommend GPUs, and price three deployment shapes — live, as you choose.</p>
          </div>

          <Step num={1} title="What model are you serving?" hint={`${MODELS.length} open models`}>
            <ModelGrid selectedId={modelId} onSelect={setModelId} />
          </Step>

          <Step num={2} title="How many people at the same time?" hint="Peak concurrent users — not total">
            <div className="conc-row">
              <Segment
                options={CONCURRENCY_PRESETS}
                value={concurrencyPreset}
                onChange={pickConcurrency}
                render={(o) => (
                  <span className="seg-stack">
                    <span className="seg-big">{o.label}</span>
                    <span className="seg-sub">{o.sub}</span>
                  </span>
                )}
              />
              <div className="exact">
                <span className="exact-label">Exact</span>
                <input
                  className="exact-input"
                  type="number"
                  min="1"
                  value={concurrentUsers}
                  onChange={(e) => {
                    const v = Math.max(1, parseInt(e.target.value || '1', 10));
                    setConcurrentUsers(v);
                    setConcurrencyPreset('custom');
                  }}
                />
                <span className="exact-unit">users</span>
              </div>
            </div>
          </Step>

          <Step num={3} title="Context per user" hint="Conversation memory per session">
            <Segment
              options={CONTEXT_PRESETS}
              value={contextPreset}
              onChange={setContextPreset}
              render={(o) => (
                <span className="seg-stack">
                  <span className="seg-big">{o.label}</span>
                  <span className="seg-sub">{o.sub} tokens</span>
                </span>
              )}
            />
          </Step>

          <Step num={4} title="Weight precision" hint="Lower precision = less VRAM, slight quality loss">
            <Segment
              options={Object.entries(PRECISIONS).map(([id, p]) => ({ id, label: p.label, bytes: p.bytes }))}
              value={precision}
              onChange={setPrecision}
              render={(o) => (
                <span className="seg-stack">
                  <span className="seg-big">{o.label}</span>
                  <span className="seg-sub">{o.bytes} byte/param</span>
                </span>
              )}
            />
          </Step>

          <Step num={5} title="Where do you want to run this?" hint="Sets the cost model default">
            <Segment
              options={DEPLOYMENTS}
              value={deployment}
              onChange={setDeployment}
              render={(o) => (
                <span className="seg-stack">
                  <span className="seg-big">{o.label}</span>
                  <span className="seg-sub">{o.sub}</span>
                </span>
              )}
            />
          </Step>

          <div className="footer-note">
            <span className="note-dot" />
            Numbers update live. Hit <em>Get full estimate</em> in the result panel to lock these into the Advanced Calculator with detailed throughput, KV-paging, and procurement scenarios.
          </div>
        </section>

        <ResultPanel state={state} result={result} tweaks={t} />
      </main>

      <TweaksPanel>
        <TweakSection label="Theme" />
        <TweakRadio
          label="Mode"
          value={t.theme}
          options={['dark', 'light']}
          onChange={(v) => setTweak('theme', v)}
        />
        <TweakColor
          label="Accent"
          value={t.accent}
          options={['#B4FE2C', '#7CE7FF', '#FF8C5C', '#C4A6FF', '#FFE066']}
          onChange={(v) => setTweak('accent', v)}
        />
        <TweakSection label="Layout" />
        <TweakRadio
          label="Composition"
          value={t.layout}
          options={['split', 'stacked', 'dense']}
          onChange={(v) => setTweak('layout', v)}
        />
        <TweakSection label="Result panel" />
        <TweakToggle
          label="Show alternatives"
          value={t.showAlternatives}
          onChange={(v) => setTweak('showAlternatives', v)}
        />
        <TweakToggle
          label="Show 3-yr outlook"
          value={t.showCostOutlook}
          onChange={(v) => setTweak('showCostOutlook', v)}
        />
      </TweaksPanel>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
