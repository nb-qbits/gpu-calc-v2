'use client'

import * as React from 'react'
import { Alert, Spinner } from '@patternfly/react-core'
import { MODEL_CATALOG } from '@/lib/gpu-math/models'
import { GPU_OPTIONS_KV } from '@/lib/gpu-math/gpus'
import { formatBytes } from '@/lib/utils/format'
import { useCountUp } from '@/app/quick-estimate/quickEstimateHelpers'
import type { KvCacheCalcResult } from '@/lib/api/kv-cache-calc'
import styles from './KvCacheCalc.module.css'

const MODEL_OPTIONS = MODEL_CATALOG.map(m => m.hfId)

const BREAKDOWN_COLORS: Record<string, string> = {
  kv: '#0066cc',
  weights: '#5e40be',
  activations: '#f0ab00',
  runtime: '#3e8635',
  comm: '#009596',
}

export default function KvCacheCalc() {
  const [model, setModel] = React.useState(MODEL_CATALOG[0]?.hfId ?? '')
  const [system, setSystem] = React.useState(GPU_OPTIONS_KV[0]?.systemId ?? '')
  const [loading, setLoading] = React.useState(false)
  const [result, setResult] = React.useState<KvCacheCalcResult | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [flipped, setFlipped] = React.useState<Record<string, boolean>>({})

  const catalogMatch = MODEL_CATALOG.find(m => m.hfId === model)

  async function handleCalculate() {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/v1/kv-cache-calc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: model, system }),
      })

      const data = await res.json()

      if (data.status === 'failed') {
        setError(data.error?.message ?? 'An unexpected error occurred')
        return
      }

      setResult(data as KvCacheCalcResult)
    } catch {
      setError('Failed to connect to the server. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function toggleFlip(id: string) {
    setFlipped(prev => ({ ...prev, [id]: !prev[id] }))
  }

  const animKv = useCountUp(result?.kvCache.totalBytes ?? 0, 800)
  const animPerToken = useCountUp(result?.kvCache.perTokenBytes ?? 0, 800)
  const animTokens = useCountUp(result?.kvCache.totalTokens ?? 0, 800)
  const animGpuCap = useCountUp(result?.gpuCapacity.totalBytes ?? 0, 800)

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>KV cache calculator</h1>
        <p className={styles.subtitle}>
          Calculate KV cache memory requirements for any model on supported GPU systems
        </p>
      </div>

      {/* Input card */}
      <div className={styles.inputCard}>
        <div className={styles.inputRow}>
          <div className={styles.field}>
            <label htmlFor="kv-model" className={styles.fieldLabel}>Model</label>
            <div className={styles.modelInputWrapper}>
              <input
                type="text"
                id="kv-model"
                list="kv-models"
                value={model}
                onChange={e => setModel(e.target.value)}
                placeholder="Type model name or select from dropdown..."
                className={styles.modelInput}
              />
              <datalist id="kv-models">
                {MODEL_OPTIONS.map(m => <option key={m} value={m} />)}
              </datalist>
              {model && (
                <span className={`${styles.modelChip} ${catalogMatch ? styles.modelChipCatalog : styles.modelChipCustom}`}>
                  {catalogMatch ? 'In catalog' : 'Custom model'}
                </span>
              )}
            </div>
          </div>

          <div className={styles.field}>
            <label htmlFor="kv-gpu" className={styles.fieldLabel}>GPU system</label>
            <select
              id="kv-gpu"
              value={system}
              onChange={e => setSystem(e.target.value)}
              className={styles.gpuSelect}
            >
              {GPU_OPTIONS_KV.map(g => (
                <option key={g.systemId} value={g.systemId}>{g.label}</option>
              ))}
            </select>
          </div>

          <button
            type="button"
            className={styles.calcBtn}
            onClick={handleCalculate}
            disabled={loading || !model.trim()}
          >
            {loading ? 'Calculating…' : 'Calculate'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className={styles.errorAlert}>
          <Alert variant="danger" title="Calculation failed" isInline>
            {error}
          </Alert>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className={styles.loadingWrap}>
          <Spinner size="lg" />
          <span>Calculating KV cache requirements…</span>
        </div>
      )}

      {/* Placeholder */}
      {!loading && !result && !error && (
        <div className={styles.placeholder}>
          Select a model and GPU system, then click Calculate
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Tiles */}
          <div className={styles.tilesGrid}>
            {/* Total KV cache */}
            <TileCard
              id="total"
              dark
              label="Total KV cache"
              value={formatBytes(animKv)}
              sub={`${result.kvCache.totalTokens.toLocaleString()} tokens capacity`}
              flipped={flipped.total ?? false}
              onFlip={() => toggleFlip('total')}
              backContent={
                <>
                  <div className={styles.backTitle}>KV cache detail</div>
                  <BackRow label="Raw bytes" value={result.kvCache.totalBytes.toLocaleString()} dark />
                  <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} dark />
                  <BackRow label="Per token" value={`${result.kvCache.perTokenBytes.toLocaleString()} B`} dark />
                  <BackRow label="Source" value={result.metadata.source} dark />
                </>
              }
            />

            {/* Per token */}
            <TileCard
              id="pertoken"
              label="Per token"
              value={formatBytes(animPerToken)}
              sub="KV cache memory per token"
              flipped={flipped.pertoken ?? false}
              onFlip={() => toggleFlip('pertoken')}
              backContent={
                <>
                  <div className={styles.backTitle}>Per-token detail</div>
                  <BackRow label="Bytes/token" value={result.kvCache.perTokenBytes.toLocaleString()} />
                  <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} />
                </>
              }
            />

            {/* Token capacity */}
            <TileCard
              id="tokens"
              label="Token capacity"
              value={animTokens.toLocaleString()}
              sub="Max tokens in KV cache"
              flipped={flipped.tokens ?? false}
              onFlip={() => toggleFlip('tokens')}
              backContent={
                <>
                  <div className={styles.backTitle}>Capacity detail</div>
                  <BackRow label="Total tokens" value={result.kvCache.totalTokens.toLocaleString()} />
                  <BackRow label="KV size" value={formatBytes(result.kvCache.totalBytes)} />
                  <BackRow label="Per token" value={`${result.kvCache.perTokenBytes.toLocaleString()} B`} />
                </>
              }
            />

            {/* GPU capacity */}
            <TileCard
              id="gpu"
              label="GPU memory"
              value={formatBytes(animGpuCap)}
              sub={`${result.metadata.system} total capacity`}
              flipped={flipped.gpu ?? false}
              onFlip={() => toggleFlip('gpu')}
              backContent={
                <>
                  <div className={styles.backTitle}>GPU memory detail</div>
                  <BackRow label="Total GPU" value={formatBytes(result.gpuCapacity.totalBytes)} />
                  <BackRow label="KV cache" value={formatBytes(result.kvCache.totalBytes)} />
                  <BackRow label="KV % of GPU" value={`${((result.kvCache.totalBytes / result.gpuCapacity.totalBytes) * 100).toFixed(1)}%`} />
                </>
              }
            />
          </div>

          {/* Memory breakdown bar */}
          <MemoryBreakdownSection result={result} />

          {/* Metadata */}
          <div className={styles.configSection}>
            <div className={styles.configTitle}>Request details</div>
            <div className={styles.configGrid}>
              <ConfigItem label="Model" value={result.metadata.modelPath} />
              <ConfigItem label="GPU system" value={result.metadata.system} />
              <ConfigItem label="Source" value={result.metadata.source} />
              <ConfigItem label="Response time" value={`${result.metadata.durationMs}ms`} />
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Sub-components ─────────────────────────────────────────────────────────

interface TileCardProps {
  id: string
  dark?: boolean
  label: string
  value: string
  sub: string
  flipped: boolean
  onFlip: () => void
  backContent: React.ReactNode
}

function TileCard({ dark, label, value, sub, flipped, onFlip, backContent }: TileCardProps) {
  return (
    <div
      className={`${styles.tileCard} ${dark ? styles.tileDark : ''}`}
      role="button"
      tabIndex={0}
      aria-pressed={flipped}
      onClick={() => { if (!flipped) onFlip() }}
      onKeyDown={e => {
        if ((e.key === 'Enter' || e.key === ' ') && !flipped) {
          e.preventDefault()
          onFlip()
        }
      }}
    >
      {!flipped ? (
        <>
          <div className={styles.tileLabel}>{label}</div>
          <div className={styles.tileValue}>{value}</div>
          <div className={styles.tileSub}>{sub}</div>
          <span className={styles.seeMath}>&#8635; details</span>
        </>
      ) : (
        <>
          {backContent}
          <span
            className={styles.seeMath}
            role="button"
            tabIndex={0}
            onClick={e => { e.stopPropagation(); onFlip() }}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); onFlip() } }}
          >
            &#8635; flip back
          </span>
        </>
      )}
    </div>
  )
}

function BackRow({ label, value }: { label: string; value: string; dark?: boolean }) {
  return (
    <div className={styles.backRow}>
      <span className={styles.backRowLabel}>{label}</span>
      <span className={styles.backRowValue}>{value}</span>
    </div>
  )
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.configItem}>
      <span className={styles.configLabel}>{label}</span>
      <span className={styles.configValue}>{value}</span>
    </div>
  )
}

function MemoryBreakdownSection({ result }: { result: KvCacheCalcResult }) {
  const segments = [
    { label: 'KV cache', bytes: result.kvCache.totalBytes, color: BREAKDOWN_COLORS.kv },
    { label: 'Weights', bytes: result.memoryBreakdown.weightsBytes, color: BREAKDOWN_COLORS.weights },
    { label: 'Activations', bytes: result.memoryBreakdown.activationsBytes, color: BREAKDOWN_COLORS.activations },
    { label: 'Runtime', bytes: result.memoryBreakdown.runtimeOverheadBytes, color: BREAKDOWN_COLORS.runtime },
  ]

  if (result.memoryBreakdown.commOverheadBytes > 0) {
    segments.push({ label: 'Comm', bytes: result.memoryBreakdown.commOverheadBytes, color: BREAKDOWN_COLORS.comm })
  }

  const totalUsed = segments.reduce((s, seg) => s + seg.bytes, 0)
  const gpuTotal = result.gpuCapacity.totalBytes
  const free = Math.max(0, gpuTotal - totalUsed)

  if (totalUsed === 0) return null

  return (
    <div className={styles.breakdownSection}>
      <div className={styles.breakdownTitle}>GPU memory breakdown</div>
      <div className={styles.memoryBar}>
        {segments.map((seg, i) => {
          const pct = gpuTotal > 0 ? (seg.bytes / gpuTotal) * 100 : (seg.bytes / totalUsed) * 100
          if (pct < 0.5) return null
          return (
            <div
              key={i}
              className={styles.memorySegment}
              style={{ width: `${pct}%`, background: seg.color }}
              title={`${seg.label}: ${formatBytes(seg.bytes)} (${pct.toFixed(1)}%)`}
            >
              {pct > 8 ? seg.label : ''}
            </div>
          )
        })}
        {free > 0 && gpuTotal > 0 && (
          <div
            className={styles.memorySegment}
            style={{ width: `${(free / gpuTotal) * 100}%`, background: '#e0e0e0', color: '#3c3f42' }}
            title={`Free: ${formatBytes(free)}`}
          >
            {(free / gpuTotal) * 100 > 8 ? 'Free' : ''}
          </div>
        )}
      </div>
      <div className={styles.breakdownLegend}>
        {segments.map((seg, i) => (
          <span key={i}>
            <span className={styles.legendDot} style={{ background: seg.color }} />
            {seg.label}: {formatBytes(seg.bytes)}
          </span>
        ))}
        {free > 0 && gpuTotal > 0 && (
          <span>
            <span className={styles.legendDot} style={{ background: '#e0e0e0' }} />
            Free: {formatBytes(free)}
          </span>
        )}
      </div>
    </div>
  )
}
