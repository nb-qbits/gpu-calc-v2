'use client';

import * as React from 'react';
import type { SavedEstimate } from '@/lib/saved-estimates';
import styles from './compare.module.css';

interface ComparisonRowProps {
  label: string;
  estimates: SavedEstimate[];
  getValue: (est: SavedEstimate) => number | string;
  format?: (val: number) => string;
  showBar?: boolean;
  barColor?: string;
  lowerIsBetter?: boolean;
}

export function ComparisonRow({
  label,
  estimates,
  getValue,
  format = (v) => v.toString(),
  showBar = false,
  barColor = '#0066cc',
  lowerIsBetter = false,
}: ComparisonRowProps) {
  const values = estimates.map(est => {
    const val = getValue(est);
    return typeof val === 'number' ? val : parseFloat(val) || 0;
  });

  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);

  const getBadge = (value: number, index: number) => {
    if (values.length < 2) return null;

    // Only show ONE badge per row for the outlier
    if (lowerIsBetter && value === minValue && values.filter(v => v === minValue).length === 1) {
      return <span className={styles.badgeBest}>best</span>;
    }
    if (!lowerIsBetter && value === maxValue && values.filter(v => v === maxValue).length === 1) {
      return <span className={styles.badgeHighest}>highest</span>;
    }
    // For GPU count, also mark "most"
    if (label === 'GPUs needed' && value === maxValue && values.filter(v => v === maxValue).length === 1) {
      return <span className={styles.badgeMost}>most</span>;
    }
    return null;
  };

  return (
    <tr>
      <td className={styles.metricCol}>{label}</td>
      {estimates.map((est, idx) => {
        const rawValue = getValue(est);
        const numValue = typeof rawValue === 'number' ? rawValue : parseFloat(rawValue) || 0;
        const displayValue = typeof rawValue === 'number' ? format(rawValue) : rawValue;
        const barWidth = maxValue > 0 ? (numValue / maxValue) * 100 : 0;

        return (
          <td key={est.id} className={styles.valueCell}>
            <div className={styles.cellContent}>
              <span className={styles.cellValue}>{displayValue}</span>
              {getBadge(numValue, idx)}
            </div>
            {showBar && (
              <div className={styles.barTrack}>
                <div
                  className={styles.bar}
                  style={{
                    width: `${barWidth}%`,
                    background: barColor,
                  }}
                />
              </div>
            )}
          </td>
        );
      })}
    </tr>
  );
}
