'use client'

import * as React from 'react'
import { Switch } from '@patternfly/react-core'
import { useCountUp } from '@/app/quick-estimate/quickEstimateHelpers'
import { FRONTIER_MODELS } from '@/lib/pricing/frontier-models'
import { getCloudRate, getOwnedRate } from '@/lib/pricing/gpu-rates'
import { DEFAULT_TIERS } from '@/lib/routing/tier-defaults'
import { type TierState, computeAllTiers, computeCostVolumePoints, findBreakeven } from '@/lib/routing/calc'
import { GPU_CATALOG } from '@/lib/gpu-math/gpus'
import { MODEL_CATALOG } from '@/lib/gpu-math/models'
import CostVolumeChart from './CostVolumeChart'
import CompareTable from './CompareTable'
import styles from './routing.module.css'

function fmtCost(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${Math.round(v / 1_000)}K`
  return `$${v.toFixed(0)}`
}

function fmtNum(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return `${v}`
}

function initTiers(): TierState[] {
  return DEFAULT_TIERS.map(d => ({
    id: d.id,
    pct: d.defaultPct,
    tokensIn: d.defaultTokensIn,
    tokensOut: d.defaultTokensOut,
    frontierModelId: d.defaultFrontierModelId,
    ossModelId: d.defaultOssModelId,
    gpuType: d.defaultGpuType,
    gpuPerReplica: d.defaultGpuPerReplica,
    capacityPerReplica: d.defaultCapacityPerReplica,
  }))
}

const OSS_MODELS = MODEL_CATALOG.filter(m =>
  ['Meta', 'Mistral', 'Google', 'Qwen', 'DeepSeek'].includes(m.vendor) &&
  !m.tags?.includes('Vision')
)

const GPU_OPTIONS = GPU_CATALOG.filter(g =>
  ['h100-80gb', 'h200-141gb', 'a100-80gb', 'a100-40gb', 'l40s-48gb', 'b200-192gb', 'mi300x-192gb'].includes(g.id)
)

export default function RoutingEconomics() {
  const [users, setUsers] = React.useState(1000)
  const [queriesPerUserPerDay, setQueriesPerUserPerDay] = React.useState(100)
  const [concurrencyPct, setConcurrencyPct] = React.useState(10)
  const [mode, setMode] = React.useState<'cloud' | 'owned'>('owned')
  const [inputsOpen, setInputsOpen] = React.useState(false)
  const [compareOpen, setCompareOpen] = React.useState(false)
  const [flipped, setFlipped] = React.useState<Record<string, boolean>>({})
  const [tiers, setTiers] = React.useState<TierState[]>(initTiers)
  const [livePricing, setLivePricing] = React.useState<Record<string, number>>({})

  React.useEffect(() => {
    const fetchPricing = async () => {
      try {
        const response = await fetch('/api/v1/gpus?live_pricing=true')
        const data = await response.json()
        if (data.status === 'success' && data.data?.gpus) {
          const pricing: Record<string, number> = {}
          data.data.gpus.forEach((gpu: { name: string; live_pricing?: { onDemand?: { median?: number } } }) => {
            if (gpu.live_pricing?.onDemand?.median) {
              pricing[gpu.name] = gpu.live_pricing.onDemand.median
            }
          })
          setLivePricing(pricing)
        }
      } catch {
        // Use fallback rates
      }
    }
    fetchPricing()
  }, [])

  const getRate = React.useCallback(
    (gpuId: string) => mode === 'cloud' ? getCloudRate(gpuId, livePricing) : getOwnedRate(gpuId),
    [mode, livePricing],
  )

  const result = React.useMemo(
    () => computeAllTiers(tiers, users, queriesPerUserPerDay, concurrencyPct, FRONTIER_MODELS, getRate),
    [tiers, users, queriesPerUserPerDay, concurrencyPct, getRate],
  )

  const volumePoints = React.useMemo(
    () => computeCostVolumePoints(tiers, users, queriesPerUserPerDay, concurrencyPct, FRONTIER_MODELS, getRate),
    [tiers, users, queriesPerUserPerDay, concurrencyPct, getRate],
  )

  const breakeven = React.useMemo(() => findBreakeven(volumePoints), [volumePoints])

  const animSavings = useCountUp(result.savings, 800)
  const animFrontier = useCountUp(result.totalFrontier, 800)
  const animSelfHosted = useCountUp(result.totalSelfHosted, 800)

  function handlePctChange(changedId: string, newPct: number) {
    setTiers(prev => {
      const clamped = Math.max(0, Math.min(100, newPct))
      const others = prev.filter(t => t.id !== changedId)
      const othersTotal = others.reduce((s, t) => s + t.pct, 0)
      const remaining = 100 - clamped

      return prev.map(t => {
        if (t.id === changedId) return { ...t, pct: clamped }
        if (othersTotal > 0) {
          return { ...t, pct: Math.round((t.pct / othersTotal) * remaining) }
        }
        return { ...t, pct: Math.round(remaining / others.length) }
      })
    })
  }

  function updateTier(id: string, patch: Partial<TierState>) {
    setTiers(prev => prev.map(t => t.id === id ? { ...t, ...patch } : t))
  }

  function toggleFlip(id: string) {
    setFlipped(prev => ({ ...prev, [id]: !prev[id] }))
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.pageTitle}>Routing economics</h1>
        <p className={styles.subtitle}>
          Compare frontier API costs vs self-hosted open models across traffic tiers
        </p>
      </div>

      {/* Input chips */}
      <div className={styles.inputChips}>
        <button type="button" className={styles.chipPill} onClick={() => setInputsOpen(o => !o)}>
          <span className={styles.chipLabel}>Users</span>
          <span className={styles.chipValue}>{fmtNum(users)}</span>
        </button>
        <button type="button" className={styles.chipPill} onClick={() => setInputsOpen(o => !o)}>
          <span className={styles.chipLabel}>Queries/user/day</span>
          <span className={styles.chipValue}>{queriesPerUserPerDay}</span>
        </button>
        <button type="button" className={styles.chipPill} onClick={() => setInputsOpen(o => !o)}>
          <span className={styles.chipLabel}>Peak concurrency</span>
          <span className={styles.chipValue}>{concurrencyPct}%</span>
        </button>
        <button type="button" className={styles.chipEditBtn} onClick={() => setInputsOpen(o => !o)} aria-label="Edit inputs">
          {inputsOpen ? '▲' : '▼'}
        </button>
      </div>

      {/* Chip editor — collapsible */}
      <div className={`${styles.chipEditor} ${inputsOpen ? '' : styles.chipEditorClosed}`}>
        <div className={styles.chipEditorInner}>
          <div className={styles.chipEditorGrid}>
            <div className={styles.chipField}>
              <label className={styles.chipFieldLabel}>Users</label>
              <input
                type="number" className={styles.chipInput} value={users} min={1}
                onChange={e => setUsers(Math.max(1, parseInt(e.target.value) || 1))}
              />
            </div>
            <div className={styles.chipField}>
              <label className={styles.chipFieldLabel}>Queries / user / day</label>
              <input
                type="number" className={styles.chipInput} value={queriesPerUserPerDay} min={1}
                onChange={e => setQueriesPerUserPerDay(Math.max(1, parseInt(e.target.value) || 1))}
              />
            </div>
            <div className={styles.chipField}>
              <label className={styles.chipFieldLabel}>Peak concurrency %</label>
              <input
                type="number" className={styles.chipInput} value={concurrencyPct} min={1} max={100}
                onChange={e => setConcurrencyPct(Math.max(1, Math.min(100, parseInt(e.target.value) || 1)))}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Tier tiles — conditional render, no 3D flip */}
      <div className={styles.tiersGrid}>
        {tiers.map((tier, i) => {
          const config = DEFAULT_TIERS[i]
          const tierResult = result.tiers[i]
          const isFlipped = flipped[tier.id] ?? false
          const frontierModel = FRONTIER_MODELS.find(m => m.id === tier.frontierModelId)
          const ossModel = OSS_MODELS.find(m => m.id === tier.ossModelId)

          return (
            <div key={tier.id} className={styles.tierCard}>
              <div
                className={styles.tileCard}
                style={{ borderTop: `4px solid ${config.color}` }}
                role="button"
                tabIndex={0}
                aria-pressed={isFlipped}
                onClick={() => {
                  if (!isFlipped) toggleFlip(tier.id)
                }}
                onKeyDown={e => {
                  if ((e.key === 'Enter' || e.key === ' ') && !isFlipped) {
                    e.preventDefault()
                    toggleFlip(tier.id)
                  }
                }}
              >
                {!isFlipped ? (
                  /* ---- Front face ---- */
                  <>
                    <div className={styles.tileLabel}>
                      <span className={styles.tileDot} style={{ background: config.color }} />
                      {config.label}
                      <span className={styles.tilePct} style={{ background: config.color }}>
                        {tier.pct}%
                      </span>
                    </div>
                    <div className={styles.tileValue}>{fmtCost(tierResult?.frontierAnnual ?? 0)}</div>
                    <div className={styles.tileSub}>
                      frontier API &middot; {frontierModel?.name ?? tier.frontierModelId}
                    </div>
                    <div className={styles.tileSub}>
                      {tier.tokensIn} in / {tier.tokensOut} out tokens per query
                    </div>
                    <span className={styles.seeMath}>&#8635; configure</span>
                  </>
                ) : (
                  /* ---- Back face (config form) ---- */
                  <>
                    <div className={styles.backTitle}>{config.label} — configuration</div>

                    <div className={styles.backField}>
                      <label className={styles.backFieldLabel}>Frontier model</label>
                      <select
                        className={styles.backSelect}
                        value={tier.frontierModelId}
                        onClick={e => e.stopPropagation()}
                        onChange={e => { e.stopPropagation(); updateTier(tier.id, { frontierModelId: e.target.value }) }}
                      >
                        {FRONTIER_MODELS.map(m => (
                          <option key={m.id} value={m.id}>{m.name} — ${m.pricePerMInput}/${m.pricePerMOutput}</option>
                        ))}
                      </select>
                    </div>

                    <div className={styles.backField}>
                      <label className={styles.backFieldLabel}>OSS model</label>
                      <select
                        className={styles.backSelect}
                        value={tier.ossModelId}
                        onClick={e => e.stopPropagation()}
                        onChange={e => { e.stopPropagation(); updateTier(tier.id, { ossModelId: e.target.value }) }}
                      >
                        {OSS_MODELS.map(m => (
                          <option key={m.id} value={m.id}>{m.name}</option>
                        ))}
                      </select>
                    </div>

                    <div className={styles.backField}>
                      <label className={styles.backFieldLabel}>GPU type</label>
                      <select
                        className={styles.backSelect}
                        value={tier.gpuType}
                        onClick={e => e.stopPropagation()}
                        onChange={e => { e.stopPropagation(); updateTier(tier.id, { gpuType: e.target.value }) }}
                      >
                        {GPU_OPTIONS.map(g => (
                          <option key={g.id} value={g.id}>{g.display_name}</option>
                        ))}
                      </select>
                    </div>

                    <div className={styles.backRow}>
                      <div className={styles.backField}>
                        <label className={styles.backFieldLabel}>GPUs / replica</label>
                        <input
                          type="number" className={styles.backInput} value={tier.gpuPerReplica} min={1}
                          onClick={e => e.stopPropagation()}
                          onChange={e => { e.stopPropagation(); updateTier(tier.id, { gpuPerReplica: Math.max(1, parseInt(e.target.value) || 1) }) }}
                        />
                      </div>
                      <div className={styles.backField}>
                        <label className={styles.backFieldLabel}>Capacity / replica</label>
                        <input
                          type="number" className={styles.backInput} value={tier.capacityPerReplica} min={1}
                          onClick={e => e.stopPropagation()}
                          onChange={e => { e.stopPropagation(); updateTier(tier.id, { capacityPerReplica: Math.max(1, parseInt(e.target.value) || 1) }) }}
                        />
                      </div>
                    </div>

                    <div className={styles.backRow} style={{ marginTop: 8 }}>
                      <div className={styles.backField}>
                        <label className={styles.backFieldLabel}>Tokens in</label>
                        <input
                          type="number" className={styles.backInput} value={tier.tokensIn} min={1}
                          onClick={e => e.stopPropagation()}
                          onChange={e => { e.stopPropagation(); updateTier(tier.id, { tokensIn: Math.max(1, parseInt(e.target.value) || 1) }) }}
                        />
                      </div>
                      <div className={styles.backField}>
                        <label className={styles.backFieldLabel}>Tokens out</label>
                        <input
                          type="number" className={styles.backInput} value={tier.tokensOut} min={1}
                          onClick={e => e.stopPropagation()}
                          onChange={e => { e.stopPropagation(); updateTier(tier.id, { tokensOut: Math.max(1, parseInt(e.target.value) || 1) }) }}
                        />
                      </div>
                    </div>

                    <span
                      className={styles.seeMath}
                      role="button"
                      tabIndex={0}
                      onClick={e => { e.stopPropagation(); toggleFlip(tier.id) }}
                      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); toggleFlip(tier.id) } }}
                    >
                      &#8635; flip back
                    </span>
                  </>
                )}
              </div>

              {/* Traffic slider */}
              <div className={styles.tierSlider}>
                <input
                  type="range" className={styles.tierSliderInput}
                  min={0} max={100} value={tier.pct}
                  onChange={e => handlePctChange(tier.id, parseInt(e.target.value))}
                  style={{
                    background: `linear-gradient(to right, ${config.color} ${tier.pct}%, var(--gc-bg-3, #e0e0e0) ${tier.pct}%)`,
                  }}
                />
                <span className={styles.tierSliderLabel}>{tier.pct}%</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Hero metrics */}
      <div className={styles.heroMetrics}>
        <div className={`${styles.heroCard} ${styles.heroDark}`}>
          <div className={styles.heroLabel}>Annual savings</div>
          <div className={styles.heroValue}>
            {result.savings >= 0 ? '' : '−'}{fmtCost(Math.abs(animSavings))}
          </div>
          {result.savingsPct > 0 && (
            <span className={styles.heroSavingsPct}>{result.savingsPct.toFixed(0)}% less</span>
          )}
          <div className={styles.heroSub}>
            {result.savings > 0
              ? 'Self-hosted saves vs frontier API'
              : 'Frontier API is cheaper at this scale'}
          </div>
        </div>

        <div className={styles.heroCard}>
          <div className={styles.heroLabel}>Frontier API cost</div>
          <div className={styles.heroValue}>{fmtCost(animFrontier)}</div>
          <div className={styles.heroSub}>
            {fmtNum(users * queriesPerUserPerDay)} queries/day &times; 365
          </div>
        </div>

        <div className={styles.heroCard}>
          <div className={styles.heroLabel}>Self-hosted cost</div>
          <div className={styles.heroValue}>{fmtCost(animSelfHosted)}</div>
          <div className={styles.heroSub}>
            {result.tiers.reduce((s, t) => s + t.gpuCount, 0)} GPUs total &middot;{' '}
            {mode === 'cloud' ? 'cloud rental' : 'hardware amortization'}
          </div>
          <div className={styles.modeToggle}>
            <Switch
              id="cost-mode"
              label="Cloud"
              labelOff="Owned"
              isChecked={mode === 'cloud'}
              onChange={(_e, checked) => setMode(checked ? 'cloud' : 'owned')}
            />
          </div>
        </div>
      </div>

      {/* Cost vs volume chart */}
      <CostVolumeChart
        points={volumePoints}
        breakeven={breakeven}
        currentDailyQueries={users * queriesPerUserPerDay}
      />

      {/* Compare table */}
      <button
        type="button" className={styles.compareToggle}
        onClick={() => setCompareOpen(o => !o)}
      >
        <span className={`${styles.compareArrow} ${compareOpen ? styles.compareArrowOpen : ''}`}>&#9654;</span>
        {compareOpen ? 'Hide' : 'Show'} model comparison
      </button>

      <div className={`${styles.compareWrap} ${compareOpen ? '' : styles.compareWrapClosed}`}>
        <div className={styles.compareInner}>
          <CompareTable
            tierResults={result.tiers}
            tiers={tiers}
            tierConfigs={DEFAULT_TIERS}
            frontierModels={FRONTIER_MODELS}
            totalFrontier={result.totalFrontier}
            totalSelfHosted={result.totalSelfHosted}
          />
        </div>
      </div>
    </div>
  )
}
