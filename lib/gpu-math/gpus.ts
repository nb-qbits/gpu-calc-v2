export interface GpuSpec {
  id: string;
  name: string;
  vramGb: number;
  bandwidthTbps: number;
  tflops: number;          // BF16 TFLOPS
  pricePerHour: number;    // cloud rental USD/hr
  hardwareCostPerGpu: number; // on-prem purchase price
  powerWatts: number;      // TDP
  cloudAvailabilityPct: number;
  tpuAvailabilityPct: number;
}

export const GPU_CATALOG: GpuSpec[] = [
  {
    id: 'h100',
    name: 'NVIDIA H100',
    vramGb: 80,
    bandwidthTbps: 3.35,
    tflops: 989,
    pricePerHour: 6.00,
    hardwareCostPerGpu: 30_000,
    powerWatts: 700,
    cloudAvailabilityPct: 7,
    tpuAvailabilityPct: 92,
  },
  {
    id: 'a100',
    name: 'NVIDIA A100',
    vramGb: 80,
    bandwidthTbps: 2.0,
    tflops: 312,
    pricePerHour: 3.20,
    hardwareCostPerGpu: 15_000,
    powerWatts: 400,
    cloudAvailabilityPct: 35,
    tpuAvailabilityPct: 78,
  },
  {
    id: 'l40s',
    name: 'NVIDIA L40S',
    vramGb: 48,
    bandwidthTbps: 0.864,
    tflops: 362,
    pricePerHour: 2.50,
    hardwareCostPerGpu: 12_000,
    powerWatts: 350,
    cloudAvailabilityPct: 52,
    tpuAvailabilityPct: 70,
  },
];

export const DEFAULT_GPU = GPU_CATALOG[0];
