'use client';

import * as React from 'react';
import { Popover, Button } from '@patternfly/react-core';
import OutlinedQuestionCircleIcon from '@patternfly/react-icons/dist/esm/icons/outlined-question-circle-icon';
import styles from './QuickEstimate.module.css';

/* ----------------------------------------------------------------------------
   Term — a "?" popover for jargon. Plain-language explanations so business
   users aren't lost. Drop <Term k="kvCache" /> next to any label.
   ---------------------------------------------------------------------------- */

export const GLOSSARY: Record<string, { title: string; body: string }> = {
  kvCache: {
    title: 'KV cache',
    body: 'As the model generates text it remembers every previous token as "key/value" vectors. That memory is the KV cache. It grows with prompt length and the number of users active at once — usually the biggest driver of GPU memory after the model weights themselves.',
  },
  kvPerReq: {
    title: 'KV cache / request',
    body: 'Memory one in-flight request holds for its tokens. Multiply by the number of concurrent requests to get the total KV cache the GPUs must hold.',
  },
  weightMemory: {
    title: 'Weight memory',
    body: 'The model parameters loaded into GPU memory. Roughly parameters × bytes-per-number — e.g. an 8B model in BF16 (2 bytes) needs about 16 GB.',
  },
  gqa: {
    title: 'GQA · grouped-query attention',
    body: 'A memory-saving attention design where several query heads share one key/value head. Fewer KV heads means a smaller KV cache — here 8 KV heads vs 32 attention heads, about 4× smaller than full multi-head attention (MHA).',
  },
  isl: {
    title: 'ISL · input sequence length',
    body: 'How many tokens the prompt contains on average. Longer prompts use more KV cache per request.',
  },
  osl: {
    title: 'OSL · output sequence length',
    body: 'How many tokens the model generates per response on average.',
  },
  concurrent: {
    title: 'Concurrent requests',
    body: 'How many requests are being processed at the same instant — not total daily traffic. Derived from requests/day, the peak multiplier, and how long each request runs. This sets the batch size the GPUs must hold in memory at once.',
  },
  maxNumSeqs: {
    title: 'max_num_seqs',
    body: 'The vLLM scheduler limit on how many sequences run in a batch together. If concurrent requests exceed it, requests queue and latency rises.',
  },
  batchedTokens: {
    title: 'Batched tokens / step',
    body: 'Tokens the engine processes in a single forward pass (prompt "prefill" + ongoing "decode"). If prefills overflow the budget the engine splits them across steps (chunked prefill).',
  },
  tensorParallel: {
    title: 'Tensor parallel size (TP)',
    body: 'How many GPUs a single model copy is split across. TP = 1 means the whole model fits on one GPU; larger models need TP = 2, 4, 8…',
  },
  rangeDrivers: {
    title: 'Range drivers',
    body: 'The estimate is a range, not a single number, because real traffic varies. Range drivers are the assumptions that move the GPU count the most — tune these first to make the estimate match your reality.',
  },
  worstCase: {
    title: 'Worst-case context',
    body: 'KV cache if every active request filled the model\'s entire context window (max_model_len) at the same time. A safety ceiling — most workloads never reach it.',
  },
  prefixCache: {
    title: 'Prefix-cache hit rate',
    body: 'Share of requests that reuse an already-computed prompt prefix (e.g. a shared system prompt). Higher hit rates reuse KV cache and cut memory.',
  },
  gpuUtil: {
    title: 'GPU memory utilization',
    body: 'Fraction of each GPU\'s memory vLLM is allowed to use for weights + KV cache. The rest is headroom for activations and fragmentation — typically 90%.',
  },
  selfHosted: {
    title: 'Self-hosted (hardware only)',
    body: 'Hardware amortization only — excludes power, cooling, staff, and networking. Typical full TCO adds 40–80% to this number.',
  },
  cloudPricing: {
    title: 'Cloud pricing',
    body: 'GPU instance pricing only — excludes egress, storage, and managed service fees.',
  },
  kvCategory: {
    title: 'KV cache category',
    body: 'Different model architectures use different KV cache strategies. KV-1 = standard dense (GQA/MHA/MQA), KV-2 = MLA (Multi-head Latent Attention), KV-3a = sliding window, KV-4 = cross-layer sharing, KV-5b = SSM-based (state-space models). Fewer KV heads = smaller KV cache per request.',
  },
};

export function Term({ k }: { k: keyof typeof GLOSSARY }) {
  const g = GLOSSARY[k];
  if (!g) return null;
  return (
    <Popover headerContent={g.title} bodyContent={g.body} maxWidth="320px">
      <button type="button" className={styles.termBtn} aria-label={`What is ${g.title}?`}>
        <OutlinedQuestionCircleIcon style={{ width: 13, height: 13 }} />
      </button>
    </Popover>
  );
}

/* ----------------------------------------------------------------------------
   useCountUp — animate a number from 0 → target on mount (and on change).
   Respects prefers-reduced-motion.
   ---------------------------------------------------------------------------- */

export function useCountUp(target: number, duration = 750, decimals = 0) {
  const [val, setVal] = React.useState(target);
  const raf = React.useRef<number>(0);
  React.useEffect(() => {
    if (typeof window !== 'undefined' &&
        window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVal(target);
      return;
    }
    const from = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 4); // easeOutQuart
      setVal(from + (target - from) * e);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

/* ----------------------------------------------------------------------------
   FlipTile — front face + math back face. Robust opacity+rotate flip
   (no bare backface-visibility dependency). Click or Enter/Space to flip.
   ---------------------------------------------------------------------------- */

export function FlipTile({
  dark = false,
  sparkline,
  front,
  back,
}: {
  dark?: boolean;
  sparkline?: React.ReactNode;
  front: React.ReactNode;
  back: React.ReactNode;
}) {
  const [flipped, setFlipped] = React.useState(false);
  return (
    <div
      className={`${styles.flip} ${dark ? styles.tileDark : ''} ${flipped ? styles.flipped : ''}`}
      role="button"
      tabIndex={0}
      aria-pressed={flipped}
      onClick={() => setFlipped((f) => !f)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setFlipped((f) => !f); }
      }}
    >
      <div className={`${styles.flipFace} ${styles.flipFront}`}>
        {sparkline ? <div className={styles.sparkline}>{sparkline}</div> : null}
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

/* ----------------------------------------------------------------------------
   Sparkline — tiny GPU-count-vs-concurrency line for the dark hero tile.
   ---------------------------------------------------------------------------- */

export function Sparkline({
  points,
  currentX,
  width = 96,
  height = 36,
  stroke = 'rgba(255,255,255,0.85)',
}: {
  points: [number, number][];
  currentX?: number;
  width?: number;
  height?: number;
  stroke?: string;
}) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const nx = (x: number) => ((x - minX) / (maxX - minX || 1)) * (width - 2) + 1;
  const ny = (y: number) => height - 2 - ((y - minY) / (maxY - minY || 1)) * (height - 4);
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${nx(p[0]).toFixed(1)},${ny(p[1]).toFixed(1)}`).join(' ');

  // Find current point
  const currentPoint = currentX !== undefined ? points.find(p => p[0] >= currentX) || points[points.length - 1] : null;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={d} fill="none" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={nx(p[0])}
            cy={ny(p[1])}
            r={currentPoint && p[0] === currentPoint[0] ? 3 : 1.4}
            fill={currentPoint && p[0] === currentPoint[0] ? stroke : stroke}
            opacity={currentPoint && p[0] === currentPoint[0] ? 1 : 0.7}
          />
        ))}
      </svg>
      <div style={{
        position: 'absolute',
        bottom: '-18px',
        left: 0,
        fontSize: '9px',
        color: 'rgba(255,255,255,0.5)',
        fontFamily: 'var(--mono)',
        letterSpacing: '0.03em',
        textTransform: 'uppercase'
      }}>
        GPUs vs. concurrency
      </div>
      {currentPoint && (
        <div style={{
          position: 'absolute',
          top: '-14px',
          right: 0,
          fontSize: '10px',
          color: 'rgba(255,255,255,0.65)',
          fontFamily: 'var(--mono)',
          whiteSpace: 'nowrap'
        }}>
          now · {currentPoint[0]} → {currentPoint[1]} GPU
        </div>
      )}
    </div>
  );
}
