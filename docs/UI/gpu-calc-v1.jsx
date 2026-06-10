const { useState, useEffect, useRef, useMemo, useLayoutEffect } = React;

/* ============================== Data ============================== */

const MODELS = [
  { id: 'llama3-70b',  name: 'Llama 3 70B',  params: 70, vendor: 'Meta',    tag: 'popular' },
  { id: 'llama31-70b', name: 'Llama 3.1 70B',params: 70, vendor: 'Meta',    tag: 'popular' },
  { id: 'llama3-8b',   name: 'Llama 3 8B',   params: 8,  vendor: 'Meta' },
  { id: 'llama31-8b',  name: 'Llama 3.1 8B', params: 8,  vendor: 'Meta' },
  { id: 'gemma3-27b',  name: 'Gemma 3 27B',  params: 27, vendor: 'Google',  tag: 'new' },
  { id: 'gemma3-12b',  name: 'Gemma 3 12B',  params: 12, vendor: 'Google',  tag: 'new' },
  { id: 'gemma2-27b',  name: 'Gemma 2 27B',  params: 27, vendor: 'Google' },
  { id: 'gemma2-9b',   name: 'Gemma 2 9B',   params: 9,  vendor: 'Google' },
  { id: 'gemma2-2b',   name: 'Gemma 2 2B',   params: 2,  vendor: 'Google' },
  { id: 'mistral-7b',  name: 'Mistral 7B',   params: 7,  vendor: 'Mistral' },
  { id: 'mixtral-8x7b',name: 'Mixtral 8x7B', params: 47, activeParams: 13, vendor: 'Mistral', tag: 'moe' },
  { id: 'qwen25-7b',   name: 'Qwen 2.5 7B',  params: 7,  vendor: 'Qwen' },
  { id: 'nemotron-340b', name: 'Nemotron 340B', params: 340, vendor: 'NVIDIA', tag: 'frontier' },
];

const USER_PRESETS = [
  { value: 5,    label: '<10',  desc: 'Small team' },
  { value: 30,   label: '30',   desc: 'Department' },
  { value: 100,  label: '100',  desc: 'Org' },
  { value: 500,  label: '500',  desc: 'Platform' },
  { value: 1500, label: '1K+',  desc: 'Enterprise' },
];

const CONTEXT_PRESETS = [
  { value: 8,    label: 'Short',     tokens: '8K',   bars: 1 },
  { value: 64,   label: 'Medium',    tokens: '64K',  bars: 2 },
  { value: 128,  label: 'Long',      tokens: '128K', bars: 3 },
  { value: 1000, label: 'Very long', tokens: '1M',   bars: 4 },
];

const DEPLOY_RATE = { cloud: 2.49, 'on-prem': 1.05, hybrid: 1.62 };
const DEPLOY_LABELS = { cloud: 'Cloud', 'on-prem': 'On-prem', hybrid: 'Hybrid' };

/* ============================== Helpers ============================== */

function estimate({ model, users, contextK, deployment }) {
  if (!model) return null;
  const activeParams = model.activeParams ?? model.params;
  const modelGB = activeParams * 2;
  const kvCacheGB = activeParams * (contextK / 1024) * users * 0.6;
  const totalGB = modelGB + kvCacheGB;
  const gpuMem = 80;
  const gpus = Math.max(1, Math.ceil(totalGB / gpuMem));
  const utilization = Math.min(0.99, totalGB / (gpus * gpuMem));
  const monthly = Math.round(gpus * DEPLOY_RATE[deployment] * 730);
  const hourly = gpus * DEPLOY_RATE[deployment];
  return { activeParams, modelGB, kvCacheGB, totalGB, gpus, utilization, monthly, hourly, gpuMem };
}

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function fmt(n) { return n.toLocaleString('en-US'); }

function parseHfModelId(rawId) {
  const id = rawId.trim();
  if (!id) return null;
  // Try to find size hint in name. Patterns: 70B, 7B, 3.5B, 8x7B
  const moe = id.match(/(\d+)x(\d+)b\b/i);
  let params, activeParams;
  if (moe) {
    activeParams = parseFloat(moe[2]);
    params = activeParams * parseFloat(moe[1]); // total params
  } else {
    const m = id.match(/(\d+(?:\.\d+)?)\s*b\b/i);
    params = m ? parseFloat(m[1]) : 7;
  }
  const parts = id.split('/');
  const vendor = parts.length > 1 ? parts[0] : 'HuggingFace';
  const name = parts.length > 1 ? parts.slice(1).join('/') : id;
  return {
    id: 'hf:' + id,
    name,
    vendor,
    params,
    activeParams,
    tag: moe ? 'moe' : 'custom',
    hfId: id,
    isHf: true,
  };
}

function hexToRgbTriplet(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return `${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}`;
}

/* ============================== Ticker ============================== */

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

function Ticker({ value, prefix = '', suffix = '', digits = 0 }) {
  const animated = useAnimatedNumber(value);
  const str = animated.toFixed(digits);
  const chars = (prefix + fmt(Number(str)) + suffix).split('');
  return (
    <span className="ticker">
      {chars.map((c, i) => {
        if (/[0-9]/.test(c)) {
          const d = parseInt(c, 10);
          return (
            <span className="digit" key={i}>
              <span className="col" style={{ transform: `translateY(-${d}em)` }}>
                {[0,1,2,3,4,5,6,7,8,9].map((n) => <span key={n}>{n}</span>)}
              </span>
            </span>
          );
        }
        return <span className="sep" key={i}>{c}</span>;
      })}
    </span>
  );
}

/* ============================== Icons ============================== */

function Icon({ name, size = 16 }) {
  const s = { width: size, height: size, strokeWidth: 1.7, stroke: 'currentColor', fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' };
  if (name === 'people') return (<svg viewBox="0 0 24 24" {...s}><circle cx="9" cy="8" r="3"/><path d="M3 20a6 6 0 0 1 12 0"/><circle cx="17" cy="9" r="2.5"/><path d="M15 20a5 5 0 0 1 6.5-4.5"/></svg>);
  if (name === 'clock') return (<svg viewBox="0 0 24 24" {...s}><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15.5 14"/></svg>);
  if (name === 'pencil') return (<svg viewBox="0 0 24 24" {...s}><path d="M4 20h4l10-10-4-4L4 16v4Z"/><path d="M13.5 6.5l4 4"/></svg>);
  if (name === 'flip') return (<svg viewBox="0 0 24 24" {...s}><path d="M3 12a9 9 0 0 1 16-5.7"/><polyline points="19 3 19 7 15 7"/><path d="M21 12a9 9 0 0 1-16 5.7"/><polyline points="5 21 5 17 9 17"/></svg>);
  if (name === 'sparkle') return (<svg viewBox="0 0 24 24" {...s}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg>);
  if (name === 'arrow') return (<svg viewBox="0 0 24 24" {...s}><path d="M5 12h14"/><polyline points="13 6 19 12 13 18"/></svg>);
  if (name === 'external') return (<svg viewBox="0 0 24 24" {...s}><path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></svg>);
  if (name === 'gpu') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="6" width="18" height="12" rx="1.5"/><rect x="6" y="9" width="5" height="6" rx="0.5"/><rect x="13" y="9" width="5" height="6" rx="0.5"/><path d="M3 10h-1M3 14h-1M22 10h1M22 14h1"/></svg>);
  if (name === 'memory') return (<svg viewBox="0 0 24 24" {...s}><rect x="3" y="7" width="18" height="10" rx="1.5"/><path d="M7 7v-2M11 7v-2M13 7v-2M17 7v-2M7 19v-2M11 19v-2M13 19v-2M17 19v-2"/></svg>);
  if (name === 'cash') return (<svg viewBox="0 0 24 24" {...s}><rect x="2" y="6" width="20" height="12" rx="1.5"/><circle cx="12" cy="12" r="2.5"/><path d="M5 9.5v5M19 9.5v5"/></svg>);
  if (name === 'key') return (<svg viewBox="0 0 24 24" {...s}><circle cx="8" cy="14" r="4"/><path d="M11 11l9-9"/><path d="M16 6l2 2"/><path d="M19 3l2 2"/></svg>);
  if (name === 'lock') return (<svg viewBox="0 0 24 24" {...s}><rect x="4" y="11" width="16" height="10" rx="1.5"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>);
  if (name === 'eye') return (<svg viewBox="0 0 24 24" {...s}><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12Z"/><circle cx="12" cy="12" r="3"/></svg>);
  if (name === 'eye-off') return (<svg viewBox="0 0 24 24" {...s}><path d="M3 3l18 18"/><path d="M10.6 6.1A11 11 0 0 1 12 6c7 0 11 6 11 6a17 17 0 0 1-3.2 3.8"/><path d="M6.6 6.6A17 17 0 0 0 1 12s4 7 11 7c1.7 0 3.2-.4 4.6-1"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>);
  if (name === 'huggingface') return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="#FFD21E"/>
      <circle cx="11" cy="14" r="2" fill="#3B2D2A"/>
      <circle cx="21" cy="14" r="2" fill="#3B2D2A"/>
      <path d="M9 19c1 3 4 4 7 4s6-1 7-4" stroke="#3B2D2A" strokeWidth="1.6" fill="none" strokeLinecap="round"/>
      <circle cx="6" cy="18" r="2.2" fill="#FF8C8C"/>
      <circle cx="26" cy="18" r="2.2" fill="#FF8C8C"/>
    </svg>
  );
  return null;
}

/* ============================== Tweak defaults ============================== */

const TWEAK_DEFAULS = /*EDITMODE-BEGIN*/{
  "accent": "#ee0000",
  "highlight": "#ee0000",
  "ambient": true
}/*EDITMODE-END*/;

/* ============================== Main App ============================== */

function App() {
  const [t, setTweak] = window.useTweaks(TWEAK_DEFAULS);

  useEffect(() => {
    const a = t.accent || '#4f46e5';
    document.documentElement.style.setProperty('--accent', a);
    document.documentElement.style.setProperty('--accent-rgb', hexToRgbTriplet(a));
    const h = t.highlight || '#10b981';
    document.documentElement.style.setProperty('--highlight', h);
    document.documentElement.style.setProperty('--highlight-rgb', hexToRgbTriplet(h));
    const amb = document.querySelector('.ambient');
    if (amb) amb.style.opacity = t.ambient ? 1 : 0;
  }, [t.accent, t.highlight, t.ambient]);

  const [modelId, setModelId] = useState(null);
  const [hfModel, setHfModel] = useState(null);
  const [users, setUsers] = useState(null);
  const [customUsers, setCustomUsers] = useState('');
  const [contextK, setContextK] = useState(64);
  const [deployment, setDeployment] = useState('on-prem');
  const [openTile, setOpenTile] = useState(null); // 'users' | 'context' | null
  const [flashId, setFlashId] = useState(null);

  // Resolve current model
  const model = useMemo(() => {
    if (hfModel && modelId === hfModel.id) return hfModel;
    return MODELS.find(m => m.id === modelId);
  }, [modelId, hfModel]);

  const effectiveUsers = customUsers ? Math.max(1, parseInt(customUsers, 10) || 0) : users;
  const est = useMemo(() => estimate({
    model, users: effectiveUsers || 30, contextK, deployment,
  }), [model, effectiveUsers, contextK, deployment]);

  const ready = !!model && !!effectiveUsers;

  const selectModel = (m) => {
    setModelId(m.id);
    setFlashId(m.id);
    setTimeout(() => setFlashId(null), 500);
  };

  const [hfToken, setHfToken] = useState('');

  const submitHf = (id, token) => {
    const parsed = parseHfModelId(id);
    if (!parsed) return;
    setHfModel(parsed);
    setModelId(parsed.id);
    if (token) setHfToken(token);
  };

  return (
    <div className="app">
      <main className="form-col">
        <Hero />

        <Section
          num="01"
          title="What model are you serving?"
          help="Sliding gallery of common models, or paste any HuggingFace model ID."
          state={model ? 'done' : 'next'}
        >
          <ModelMarquee selectedId={modelId} onSelect={selectModel} flashId={flashId} />
          {model && (
            <div className="selected-pill">
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                <circle cx="7" cy="7" r="7" fill="var(--accent)"/>
                <polyline points="3.5 7.2 6 9.5 10.5 4.8" fill="none" stroke="white" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Selected: <strong>{model.name}</strong> · {model.params}B params{model.isHf ? ' · from HuggingFace' : ''}</span>
              <button className="change-btn" onClick={() => { setModelId(null); setHfModel(null); }}>change</button>
            </div>
          )}
          <HuggingFaceInput onSubmit={submitHf} active={!!hfModel && modelId === hfModel?.id} hfId={hfModel?.hfId} />
        </Section>

        <Section
          num="02"
          title="Load profile"
          help="Tap the pencil to adjust either value."
          state={effectiveUsers ? 'done' : !model ? 'open' : 'next'}
        >
          <div className="load-row">
            <ExpanderTile
              icon="people"
              label="Concurrent users"
              value={effectiveUsers ? `${fmt(effectiveUsers)}` : 'Select'}
              valueDesc={effectiveUsers ? labelForUsers(effectiveUsers) : 'tap to set'}
              accent={openTile === 'users'}
              empty={!effectiveUsers}
              onToggle={() => setOpenTile(openTile === 'users' ? null : 'users')}
            >
              <UsersPicker
                users={users}
                customUsers={customUsers}
                setUsers={(v) => { setUsers(v); setCustomUsers(''); }}
                setCustomUsers={(v) => { setCustomUsers(v); setUsers(null); }}
              />
            </ExpanderTile>
            <ExpanderTile
              icon="clock"
              label="Conversation length"
              value={CONTEXT_PRESETS.find(c => c.value === contextK)?.tokens || `${contextK}K`}
              valueDesc={CONTEXT_PRESETS.find(c => c.value === contextK)?.label.toLowerCase() || 'custom'}
              accent={openTile === 'context'}
              onToggle={() => setOpenTile(openTile === 'context' ? null : 'context')}
            >
              <ContextPicker contextK={contextK} setContextK={setContextK} />
            </ExpanderTile>
          </div>
        </Section>

        <div className="formfoot">
          <div className="summary">
            {ready ? (<><span style={{color:'var(--highlight)'}}>●</span> Ready · estimate is live on the right</>) : 'Pick a model and your load to see the estimate'}
          </div>
          <button className="cta" disabled={!ready}>
            See full breakdown <Icon name="arrow" size={16} />
          </button>
        </div>
      </main>

      <aside className="rail-col">
        <Rail est={est} model={model} users={effectiveUsers} contextK={contextK}
          deployment={deployment} setDeployment={setDeployment} />
      </aside>

      <window.TweaksPanel title="Tweaks">
        <window.TweakSection label="Look">
          <window.TweakColor label="Accent" value={t.accent}
            options={["#ee0000", "#0066cc", "#37a3a3", "#5e40be"]}
            onChange={(v) => setTweak('accent', v)} />
          <window.TweakColor label="Highlight (cost)" value={t.highlight}
            options={["#ee0000", "#a60000", "#151515", "#5e40be"]}
            onChange={(v) => setTweak('highlight', v)} />
          <window.TweakToggle label="Ambient grid" value={t.ambient}
            onChange={(v) => setTweak('ambient', v)} />
        </window.TweakSection>
      </window.TweaksPanel>
    </div>
  );
}

function labelForUsers(n) {
  if (n < 10) return 'small team';
  if (n < 50) return 'department';
  if (n < 200) return 'org';
  if (n < 1000) return 'platform';
  return 'enterprise';
}

/* ============================== Hero ============================== */

function Hero() {
  return (
    <div className="hero">
      <div className="kicker"><span className="dotpulse"></span>Quick estimate · ~30 seconds</div>
      <h1>Size your <em>LLM</em> deployment.</h1>
      <p>Pick a model, tell us your load. The estimate fills in on the right — tap any tile to flip and see the math.</p>
    </div>
  );
}

/* ============================== Section wrapper ============================== */

function Section({ num, title, help, state, children }) {
  const done = state === 'done';
  return (
    <section className={`section ${state}`}>
      <div className="section-head">
        <div className="section-num">{num}</div>
        <div className="section-title">{title}</div>
        <div className={`section-status ${done ? 'done' : ''}`}>
          {done ? (
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <rect x="0.5" y="0.5" width="13" height="13" rx="3.5" fill="var(--accent)" stroke="var(--accent)" />
              <polyline points="3.5 7.2 6 9.5 10.5 4.8" fill="none" stroke="white" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
              <rect x="0.5" y="0.5" width="13" height="13" rx="3.5" fill="transparent" stroke="var(--line-2)" />
            </svg>
          )}
          <span>{done ? 'set' : state === 'next' ? 'next' : 'optional'}</span>
        </div>
      </div>
      {help && <div className="section-help">{help}</div>}
      {children}
    </section>
  );
}

/* ============================== Model marquee ============================== */

function ModelMarquee({ selectedId, onSelect, flashId }) {
  // Triple to guarantee no visible seam at any zoom
  const dup = [...MODELS, ...MODELS, ...MODELS];
  const paused = !!selectedId;
  return (
    <div className={`marquee ${paused ? 'paused' : ''}`}>
      <div className="marquee-track">
        {dup.map((m, i) => {
          const selected = selectedId === m.id;
          return (
            <button
              key={i}
              className={`mchip ${selected ? 'selected' : ''} ${flashId === m.id ? 'flash' : ''}`}
              onClick={() => onSelect(m)}
            >
              <div className="mchip-vendor">{m.vendor}</div>
              <div className="mchip-name">{m.name}</div>
              <div className="mchip-foot">
                <span className="mchip-size">{m.params}<span className="unit">B</span></span>
                {m.tag && <span className={`mchip-tag ${m.tag}`}>{m.tag === 'moe' ? `MoE · ${m.activeParams}B` : m.tag}</span>}
              </div>
              {selected && <span className="mchip-check" aria-hidden="true">
                <svg width="14" height="14" viewBox="0 0 14 14">
                  <circle cx="7" cy="7" r="7" fill="var(--accent)"/>
                  <polyline points="3.5 7.2 6 9.5 10.5 4.8" fill="none" stroke="white" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ============================== HuggingFace input ============================== */

function HuggingFaceInput({ onSubmit, active, hfId }) {
  const [value, setValue] = useState(hfId || '');
  const [token, setToken] = useState(() => {
    try { return localStorage.getItem('gpucalc.hfToken') || ''; } catch (e) { return ''; }
  });
  const [showToken, setShowToken] = useState(false);
  const [tokenOpen, setTokenOpen] = useState(false);
  useEffect(() => { if (hfId) setValue(hfId); }, [hfId]);
  useEffect(() => {
    try {
      if (token) localStorage.setItem('gpucalc.hfToken', token);
      else localStorage.removeItem('gpucalc.hfToken');
    } catch (e) {}
  }, [token]);

  const submit = () => {
    if (value.trim()) onSubmit(value.trim(), token.trim() || undefined);
  };
  const tokenValid = token.trim().startsWith('hf_') && token.trim().length > 6;
  return (
    <div className={`hf-row ${active ? 'active' : ''}`}>
      <div className="hf-top">
        <div className="hf-logo"><Icon name="huggingface" size={22} /></div>
        <div className="hf-fieldwrap">
          <label className="hf-label">Or paste a HuggingFace model ID</label>
          <input
            className="hf-input"
            type="text"
            placeholder="meta-llama/Llama-3.1-70B-Instruct"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
          />
        </div>
        <button className="hf-load" onClick={submit} disabled={!value.trim()}>
          {active ? 'Loaded' : 'Load'}
        </button>
        <a className="hf-link" href={value.trim() ? `https://huggingface.co/${value.trim()}` : 'https://huggingface.co/models'} target="_blank" rel="noreferrer">
          Browse on HF <Icon name="external" size={12} />
        </a>
      </div>
      <div className="hf-token-row">
        <button className="hf-token-toggle" onClick={() => setTokenOpen(o => !o)}>
          <Icon name="key" size={12} />
          <span>{tokenOpen ? 'Hide' : 'Add'} access token</span>
          <span className="hf-token-hint">{token ? (tokenValid ? '· set' : '· check format') : '· optional, for gated/private models'}</span>
        </button>
        {tokenOpen && (
          <div className="hf-token-field">
            <input
              className="hf-input hf-token-input"
              type={showToken ? 'text' : 'password'}
              placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              autoComplete="off"
              spellCheck="false"
            />
            <button className="hf-eye" onClick={() => setShowToken(s => !s)} title={showToken ? 'Hide token' : 'Show token'} aria-label={showToken ? 'Hide token' : 'Show token'}>
              <Icon name={showToken ? 'eye-off' : 'eye'} size={14} />
            </button>
            <a className="hf-link sm" href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
              Get a token <Icon name="external" size={11} />
            </a>
          </div>
        )}
        {tokenOpen && (
          <div className="hf-token-note">
            <Icon name="lock" size={11} /> Stored in this browser only — never sent anywhere. Use a read-only token.
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================== Load row (users + context) ============================== */

function ExpanderTile({ icon, label, value, valueDesc, accent, empty, onToggle, children }) {
  return (
    <div className={`load-tile ${accent ? 'expanded' : ''} ${empty ? 'empty' : ''}`}>
      <button className="load-tile-head" onClick={onToggle}>
        <div className="load-tile-icon"><Icon name={icon} size={18} /></div>
        <div className="load-tile-text">
          <div className="load-tile-label">{label}</div>
          <div className="load-tile-value">{value} <span className="load-tile-desc">· {valueDesc}</span></div>
        </div>
        <span className="load-tile-pencil" aria-hidden="true"><Icon name="pencil" size={14} /></span>
      </button>
      <div className="load-tile-body" data-open={accent}>
        <div className="load-tile-body-inner">{children}</div>
      </div>
    </div>
  );
}

function UsersPicker({ users, customUsers, setUsers, setCustomUsers }) {
  return (
    <>
      <div className="chip-row">
        {USER_PRESETS.map(p => (
          <button
            key={p.value}
            className={`chip ${users === p.value ? 'active' : ''}`}
            onClick={() => setUsers(p.value)}
          >
            <span className="chip-big">{p.label}</span>
            <span className="chip-small">{p.desc}</span>
          </button>
        ))}
      </div>
      <div className="users-extra">
        <label>Or type exact:</label>
        <input
          type="number"
          min="1"
          placeholder="e.g. 75"
          value={customUsers}
          onChange={(e) => setCustomUsers(e.target.value)}
        />
      </div>
    </>
  );
}

function ContextPicker({ contextK, setContextK }) {
  return (
    <div className="chip-row">
      {CONTEXT_PRESETS.map(c => (
        <button
          key={c.value}
          className={`chip ${contextK === c.value ? 'active' : ''}`}
          onClick={() => setContextK(c.value)}
        >
          <span className="chip-big">{c.label}</span>
          <span className="chip-small">{c.tokens}</span>
          <span className="chip-bars" aria-hidden="true">
            {[1,2,3,4].map(i => <span key={i} className={`b ${i <= c.bars ? 'on' : ''}`}></span>)}
          </span>
        </button>
      ))}
    </div>
  );
}

/* ============================== Flip card ============================== */

function FlipCard({ className = '', highlight = false, accent = false, front, back, footer }) {
  const [flipped, setFlipped] = useState(false);
  return (
    <div className={`flip ${className} ${highlight ? 'highlight' : ''} ${accent ? 'accent' : ''} ${flipped ? 'flipped' : ''}`}
         onClick={() => setFlipped(f => !f)}
         role="button" tabIndex={0}
         onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped(f => !f); } }}>
      <div className="flip-inner">
        <div className="flip-face flip-front">
          {front}
          <span className="flip-hint" aria-hidden="true"><Icon name="flip" size={11} /> see math</span>
        </div>
        <div className="flip-face flip-back">
          {back}
          <span className="flip-hint" aria-hidden="true"><Icon name="flip" size={11} /> flip back</span>
        </div>
      </div>
    </div>
  );
}

/* ============================== Right rail ============================== */

function Rail({ est, model, users, contextK, deployment, setDeployment }) {
  if (!model) return (
    <div className="rail-inner">
      <div className="rail-head">
        <div className="label"><span className="live"></span>Live estimate</div>
        <div className="ts">updates as you pick</div>
      </div>
      <div className="empty">
        <div className="ring"><span className="glyph"><Icon name="sparkle" size={26} /></span></div>
        <h3>Pick a model to start</h3>
        <p>The estimate fills in as you make choices. Tap any tile to flip it and see the math.</p>
      </div>
    </div>
  );

  const activeParams = est.activeParams;
  const kvPerUser = (activeParams * (contextK / 1024) * 0.6).toFixed(2);

  return (
    <div className="rail-inner">
      <div className="rail-head">
        <div className="label"><span className="live"></span>Live estimate</div>
        <div className="ts">tap a tile to see the math</div>
      </div>

      {/* Hero — cost, highlight color */}
      <FlipCard
        className="stat-hero"
        highlight
        front={
          <>
            <div className="slabel">
              <Icon name="cash" size={13} />
              <span>Estimated monthly</span>
              <DeployToggle deployment={deployment} setDeployment={setDeployment} />
            </div>
            <div className="sval">
              <span className="currency">$</span><Ticker value={est.monthly} />
            </div>
            <div className="strend">{est.gpus} × H100 · ${est.hourly.toFixed(2)}/hr · 730 hrs/mo</div>
          </>
        }
        back={
          <>
            <div className="back-title">Monthly cost math</div>
            <div className="back-formula">
              <div className="line"><span className="t">{est.gpus}</span> × <span className="t">${DEPLOY_RATE[deployment].toFixed(2)}</span>/gpu-hr × <span className="t">730</span> hrs</div>
              <div className="line eq">= <strong>${fmt(est.monthly)}</strong> / month</div>
            </div>
            <div className="back-note">
              <span>{DEPLOY_LABELS[deployment]} pricing</span>
              <span>· assumes 100% uptime</span>
            </div>
          </>
        }
      />

      {/* Two flip tiles — GPUs and VRAM */}
      <div className="stats-row">
        <FlipCard
          className="stat-small"
          front={
            <>
              <div className="slabel"><Icon name="gpu" size={13} /><span>GPUs needed</span></div>
              <div className="sval"><Ticker value={est.gpus} /><span className="unit">× H100</span></div>
              <div className="strend">{(est.utilization * 100).toFixed(0)}% utilized</div>
            </>
          }
          back={
            <>
              <div className="back-title">GPU count math</div>
              <div className="back-formula small">
                <div className="line">⌈Total VRAM ÷ 80GB⌉</div>
                <div className="line">= ⌈<span className="t">{est.totalGB.toFixed(0)}</span> ÷ 80⌉</div>
                <div className="line eq">= <strong>{est.gpus} GPUs</strong></div>
              </div>
            </>
          }
        />
        <FlipCard
          className="stat-small"
          front={
            <>
              <div className="slabel"><Icon name="memory" size={13} /><span>Total VRAM</span></div>
              <div className="sval"><Ticker value={est.totalGB} digits={est.totalGB < 10 ? 1 : 0} /><span className="unit">GB</span></div>
              <div className="strend">model + KV cache</div>
            </>
          }
          back={
            <>
              <div className="back-title">VRAM math</div>
              <div className="back-formula small">
                <div className="line">Weights: <span className="t">{activeParams}</span>B × 2 = <strong>{est.modelGB.toFixed(0)} GB</strong></div>
                <div className="line">KV: <span className="t">{activeParams}</span> × <span className="t">{contextK}</span>K × <span className="t">{users}</span> users</div>
                <div className="line indent">≈ <strong>{est.kvCacheGB.toFixed(0)} GB</strong></div>
                <div className="line eq">= <strong>{est.totalGB.toFixed(0)} GB</strong> total</div>
              </div>
            </>
          }
        />
      </div>

      {/* GPU memory grid (also flips) */}
      <FlipCard
        className="gpu-block"
        front={
          <>
            <div className="gpu-block-head">
              <div className="gpu-block-title">Memory layout</div>
              <div className="gpu-block-meta">{est.gpus} × H100 80GB</div>
            </div>
            <div className="gpus">
              {Array.from({ length: Math.min(est.gpus, 4) }).map((_, gi) => {
                const totalCells = 80;
                const perGpuModel = est.modelGB / est.gpus;
                const perGpuKv = est.kvCacheGB / est.gpus;
                const modelCells = Math.round((perGpuModel / est.gpuMem) * totalCells);
                const kvCells = Math.round((perGpuKv / est.gpuMem) * totalCells);
                return (
                  <div className="gpu" key={gi}>
                    <div className="memgrid">
                      {Array.from({ length: totalCells }).map((_, i) => {
                        let cls = 'memcell';
                        if (i < modelCells) cls += ' model';
                        else if (i < modelCells + kvCells) cls += ' kv';
                        return <div key={i} className={cls}></div>;
                      })}
                    </div>
                  </div>
                );
              })}
              {est.gpus > 4 && <div className="gpu-more">+{est.gpus - 4} more</div>}
            </div>
            <div className="memlegend">
              <span><span className="sw model"></span>Weights</span>
              <span><span className="sw kv"></span>KV cache</span>
              <span><span className="sw free"></span>Free</span>
            </div>
          </>
        }
        back={
          <>
            <div className="back-title">Per-GPU split</div>
            <div className="back-formula small">
              <div className="line"><span className="t">{(est.modelGB / est.gpus).toFixed(1)}</span> GB weights</div>
              <div className="line"><span className="t">{(est.kvCacheGB / est.gpus).toFixed(1)}</span> GB KV cache</div>
              <div className="line"><span className="t">{(est.gpuMem - est.totalGB / est.gpus).toFixed(1)}</span> GB free headroom</div>
            </div>
            <div className="back-note">Each user adds ~{kvPerUser} GB of KV cache.</div>
          </>
        }
      />
    </div>
  );
}

function DeployToggle({ deployment, setDeployment }) {
  const opts = ['cloud', 'on-prem', 'hybrid'];
  const cycle = () => {
    const i = opts.indexOf(deployment);
    setDeployment(opts[(i + 1) % opts.length]);
  };
  return (
    <button className="deploy-toggle" onClick={(e) => { e.stopPropagation(); cycle(); }} title="Switch pricing mode">
      {DEPLOY_LABELS[deployment]}
    </button>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
