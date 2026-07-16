/**
 * General formatting utilities.
 */

/** Format a number to a fixed number of decimal places, stripping trailing zeros */
export function formatNumber(value: number, decimals = 2): string {
  return parseFloat(value.toFixed(decimals)).toString();
}

/** Format a GB value with appropriate unit */
export function formatMemory(gb: number): string {
  if (gb >= 1000) return `${formatNumber(gb / 1000, 1)} TB`;
  return `${formatNumber(gb, 1)} GB`;
}

/** Format a USD cost */
export function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

/** Format tokens per second */
export function formatThroughput(tps: number): string {
  if (tps >= 1000) return `${formatNumber(tps / 1000, 1)}k tok/s`;
  return `${formatNumber(tps, 0)} tok/s`;
}

/** Format a byte count using binary units (GiB, MiB, KiB) */
export function formatBytes(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${formatNumber(bytes / 1_073_741_824, 2)} GiB`;
  if (bytes >= 1_048_576) return `${formatNumber(bytes / 1_048_576, 2)} MiB`;
  if (bytes >= 1024) return `${formatNumber(bytes / 1024, 2)} KiB`;
  return `${bytes} B`;
}
