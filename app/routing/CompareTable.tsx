'use client'

import { type TierCostResult, type TierState } from '@/lib/routing/calc'
import { type FrontierModel } from '@/lib/pricing/frontier-models'
import { type TierConfig } from '@/lib/routing/tier-defaults'
import { MODEL_CATALOG } from '@/lib/gpu-math/models'
import styles from './routing.module.css'

function fmtCost(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`
  return `$${v.toFixed(2)}`
}

function fmtPerQuery(annual: number, dailyQueries: number): string {
  if (dailyQueries <= 0) return '—'
  const perQuery = annual / (dailyQueries * 365)
  if (perQuery < 0.001) return `$${perQuery.toFixed(6)}`
  if (perQuery < 0.01) return `$${perQuery.toFixed(4)}`
  return `$${perQuery.toFixed(3)}`
}

interface Props {
  tierResults: TierCostResult[]
  tiers: TierState[]
  tierConfigs: TierConfig[]
  frontierModels: FrontierModel[]
  totalFrontier: number
  totalSelfHosted: number
}

export default function CompareTable({
  tierResults,
  tiers,
  tierConfigs,
  frontierModels,
  totalFrontier,
  totalSelfHosted,
}: Props) {
  return (
    <table className={styles.compareTable}>
      <thead>
        <tr>
          <th>Model</th>
          <th>Type</th>
          <th>Tier</th>
          <th>Annual cost</th>
          <th>Cost / query</th>
        </tr>
      </thead>
      <tbody>
        {tierResults.map((result, i) => {
          const tier = tiers[i]
          const config = tierConfigs[i]
          const frontier = frontierModels.find(m => m.id === tier.frontierModelId)
          const ossModel = MODEL_CATALOG.find(m => m.id === tier.ossModelId)

          return (
            <tr key={`${config.id}`}>
              <td colSpan={5} style={{ padding: 0, border: 'none' }}>
                <table className={styles.compareTable} style={{ marginBottom: 0 }}>
                  <tbody>
                    <tr>
                      <td>
                        <span className={styles.tierDot} style={{ background: config.color }} />
                        {frontier?.name ?? tier.frontierModelId}
                      </td>
                      <td>
                        <span className={`${styles.typeBadge} ${styles.typeFrontier}`}>API</span>
                      </td>
                      <td>{config.label}</td>
                      <td>{fmtCost(result.frontierAnnual)}</td>
                      <td>{fmtPerQuery(result.frontierAnnual, result.dailyQueries)}</td>
                    </tr>
                    <tr>
                      <td>
                        <span className={styles.tierDot} style={{ background: config.color }} />
                        {ossModel?.name ?? tier.ossModelId}
                      </td>
                      <td>
                        <span className={`${styles.typeBadge} ${styles.typeSelfHosted}`}>Self-hosted</span>
                      </td>
                      <td>{config.label}</td>
                      <td>{fmtCost(result.selfHostedAnnual)}</td>
                      <td>{fmtPerQuery(result.selfHostedAnnual, result.dailyQueries)}</td>
                    </tr>
                  </tbody>
                </table>
              </td>
            </tr>
          )
        })}
        <tr className={styles.totalRow}>
          <td colSpan={3}><strong>Total</strong></td>
          <td colSpan={2} style={{ padding: 0, border: 'none' }}>
            <table className={styles.compareTable} style={{ marginBottom: 0 }}>
              <tbody>
                <tr>
                  <td>
                    <span className={`${styles.typeBadge} ${styles.typeFrontier}`}>API</span>
                    &nbsp;{fmtCost(totalFrontier)}
                  </td>
                </tr>
                <tr>
                  <td>
                    <span className={`${styles.typeBadge} ${styles.typeSelfHosted}`}>Self-hosted</span>
                    &nbsp;{fmtCost(totalSelfHosted)}
                  </td>
                </tr>
              </tbody>
            </table>
          </td>
        </tr>
      </tbody>
    </table>
  )
}
