'use client'

import { type VolumePoint } from '@/lib/routing/calc'
import styles from './routing.module.css'

function formatAxisCost(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`
  return `$${v.toFixed(0)}`
}

function formatAxisQueries(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return `${v}`
}

interface Props {
  points: VolumePoint[]
  breakeven: number | null
  currentDailyQueries: number
}

const W = 780
const H = 230
const PAD = { top: 12, right: 16, bottom: 34, left: 64 }
const PW = W - PAD.left - PAD.right
const PH = H - PAD.top - PAD.bottom

export default function CostVolumeChart({ points, breakeven, currentDailyQueries }: Props) {
  if (points.length < 2) return null

  const maxQ = Math.max(...points.map(p => p.dailyQueries))
  const maxCost = Math.max(...points.map(p => Math.max(p.frontier, p.selfHosted)))
  const safeMaxCost = maxCost || 1

  const nx = (q: number) => PAD.left + (q / (maxQ || 1)) * PW
  const ny = (c: number) => PAD.top + PH - (c / safeMaxCost) * PH

  const frontierPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${nx(p.dailyQueries).toFixed(1)},${ny(p.frontier).toFixed(1)}`)
    .join(' ')
  const selfHostedPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${nx(p.dailyQueries).toFixed(1)},${ny(p.selfHosted).toFixed(1)}`)
    .join(' ')

  const lastPt = points[points.length - 1]
  const firstPt = points[0]
  const frontierFill = `${frontierPath} L${nx(lastPt.dailyQueries).toFixed(1)},${ny(0).toFixed(1)} L${nx(firstPt.dailyQueries).toFixed(1)},${ny(0).toFixed(1)} Z`
  const selfHostedFill = `${selfHostedPath} L${nx(lastPt.dailyQueries).toFixed(1)},${ny(0).toFixed(1)} L${nx(firstPt.dailyQueries).toFixed(1)},${ny(0).toFixed(1)} Z`

  const yTicks = 5
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) => (safeMaxCost / yTicks) * i)

  const xTicks = 5
  const xTickValues = Array.from({ length: xTicks + 1 }, (_, i) => (maxQ / xTicks) * i)

  return (
    <div className={styles.chartWrap}>
      <div className={styles.chartTitle}>Cost vs. volume</div>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        style={{ display: 'block', maxWidth: W }}
      >
        <defs>
          <linearGradient id="frontierGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--gc-c-orange, #b8390e)" stopOpacity="0.12" />
            <stop offset="100%" stopColor="var(--gc-c-orange, #b8390e)" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="selfHostedGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--gc-success, #3e8635)" stopOpacity="0.12" />
            <stop offset="100%" stopColor="var(--gc-success, #3e8635)" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {yTickValues.map((v, i) => (
          <line
            key={`yg-${i}`}
            x1={PAD.left} y1={ny(v)} x2={W - PAD.right} y2={ny(v)}
            stroke="var(--gc-border, #d2d2d2)" strokeWidth="0.5" strokeDasharray="2,3"
          />
        ))}

        <path d={frontierFill} fill="url(#frontierGrad)" />
        <path d={selfHostedFill} fill="url(#selfHostedGrad)" />

        <path d={frontierPath} fill="none" stroke="var(--gc-c-orange, #b8390e)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d={selfHostedPath} fill="none" stroke="var(--gc-success, #3e8635)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

        {points.map((p, i) => (
          <g key={i}>
            <circle cx={nx(p.dailyQueries)} cy={ny(p.frontier)} r={p.dailyQueries === currentDailyQueries ? 4 : 2}
              fill="var(--gc-c-orange, #b8390e)" />
            <circle cx={nx(p.dailyQueries)} cy={ny(p.selfHosted)} r={p.dailyQueries === currentDailyQueries ? 4 : 2}
              fill="var(--gc-success, #3e8635)" />
          </g>
        ))}

        {breakeven !== null && breakeven > 0 && breakeven <= maxQ && (
          <g>
            <line
              x1={nx(breakeven)} y1={PAD.top} x2={nx(breakeven)} y2={PAD.top + PH}
              stroke="var(--gc-text-3, #54585c)" strokeWidth="1" strokeDasharray="4,4"
            />
            <text
              x={nx(breakeven)} y={PAD.top - 2}
              textAnchor="middle"
              fontFamily="var(--gc-font-mono)"
              fontSize="11.5"
              fill="var(--gc-text-2, #3c3f42)"
            >
              breakeven ~{formatAxisQueries(breakeven)}/day
            </text>
          </g>
        )}

        {currentDailyQueries > 0 && (
          <line
            x1={nx(currentDailyQueries)} y1={PAD.top} x2={nx(currentDailyQueries)} y2={PAD.top + PH}
            stroke="var(--gc-link, #0066cc)" strokeWidth="1" strokeDasharray="2,3" opacity="0.5"
          />
        )}

        {yTickValues.map((v, i) => (
          <text
            key={`yl-${i}`}
            x={PAD.left - 6} y={ny(v) + 4}
            textAnchor="end"
            fontFamily="var(--gc-font-mono)"
            fontSize="11.5"
            fill="var(--gc-text-2, #3c3f42)"
          >
            {formatAxisCost(v)}
          </text>
        ))}

        {xTickValues.map((v, i) => (
          <text
            key={`xl-${i}`}
            x={nx(v)} y={H - 6}
            textAnchor="middle"
            fontFamily="var(--gc-font-mono)"
            fontSize="11.5"
            fill="var(--gc-text-2, #3c3f42)"
          >
            {formatAxisQueries(v)}/day
          </text>
        ))}
      </svg>

      <div className={styles.chartLegend}>
        <span><span className={styles.legendDot} style={{ background: 'var(--gc-c-orange, #b8390e)' }} /> Frontier API</span>
        <span><span className={styles.legendDot} style={{ background: 'var(--gc-success, #3e8635)' }} /> Self-hosted</span>
        {breakeven !== null && (
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--gc-font-mono)', fontSize: '12px' }}>
            Breakeven at ~{formatAxisQueries(breakeven)} queries/day
          </span>
        )}
      </div>
    </div>
  )
}
