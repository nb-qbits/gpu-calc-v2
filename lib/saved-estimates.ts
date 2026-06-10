/**
 * Saved Estimates - localStorage utilities
 */

export interface SavedEstimate {
  id: string;
  name: string;
  tags: string;
  notes: string;
  savedAt: string;
  model: string;
  gpu: string;
  inputs: {
    isl: number;
    osl: number;
    concurrentUsers: number;
    workloadType: string;
    slaPriority: string;
    weightPrecision: string;
    kvCachePrecision: string;
  };
  results: {
    gpusRequired: number;
    tpSize: number;
    replicas: number;
    weightMemoryGB: number;
    kvCachePerUserGB: number;
    kvCacheTotalGB: number;
    kvCacheMBPerToken: number;
    kvCategory: string;
    kvCategoryLabel: string;
    cloudCostMonthly: number;
    cloudCost5Year: number;
    selfHostedCostMonthly: number;
    selfHostedCost5Year: number;
  };
}

const STORAGE_KEY = 'gc-saved-estimates';

export function getSavedEstimates(): SavedEstimate[] {
  if (typeof window === 'undefined') return [];
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Failed to load saved estimates:', error);
    return [];
  }
}

export function saveEstimate(estimate: Omit<SavedEstimate, 'id' | 'savedAt'>): SavedEstimate {
  const newEstimate: SavedEstimate = {
    ...estimate,
    id: Date.now().toString(36) + Math.random().toString(36).substring(2),
    savedAt: new Date().toISOString(),
  };

  const estimates = getSavedEstimates();
  estimates.unshift(newEstimate); // Add to beginning
  localStorage.setItem(STORAGE_KEY, JSON.stringify(estimates));
  return newEstimate;
}

export function deleteEstimate(id: string): void {
  const estimates = getSavedEstimates().filter(e => e.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(estimates));
}

export function clearAllEstimates(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function getSavedEstimateCount(): number {
  return getSavedEstimates().length;
}
