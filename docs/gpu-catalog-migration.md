# GPU Catalog Migration

## Summary

Successfully migrated GPU catalog from hardcoded TypeScript to JSON file, based on `docs/gpu-data`.

## Before vs After

### Before
- **11 GPUs** hardcoded in [lib/gpu-math/gpus.ts](../lib/gpu-math/gpus.ts)
- Missing: H20, B100, B200, B300, B30A, MI250X, MI300A, MI350X
- Missing specs: FP8 TFLOPS, MFU metrics, architecture family

### After
- **15 GPUs** loaded from [lib/gpu-math/gpu-catalog.json](../lib/gpu-math/gpu-catalog.json)
- Added: 4 NVIDIA GPUs (H20, B100, B200, B300, B30A)
- Added: 3 AMD GPUs (MI250X, MI300A, MI350X)
- Enhanced specs: FP8 TFLOPS, MFU prefill/decode, architecture, color coding

## New GPU Catalog (15 GPUs)

### NVIDIA (10 GPUs)
| GPU | VRAM | Architecture | BF16 TFLOPS | FP8 TFLOPS | Hardware Cost | MFU Prefill | MFU Decode |
|-----|------|--------------|-------------|------------|---------------|-------------|------------|
| **Blackwell Series** |
| B300 | 288 GB | blackwell | 3000 | 6000 | $60,000 | 0.44 | 0.83 |
| B200 | 192 GB | blackwell | 2250 | 4500 | $45,000 | 0.44 | 0.85 |
| B100 | 192 GB | blackwell | 1750 | 3500 | $35,000 | 0.42 | 0.85 |
| B30A | 120 GB | blackwell | 1500 | 3000 | $20,000 | 0.40 | 0.83 |
| **Hopper Series** |
| H200 | 141 GB | hopper | 989 | 1979 | $32,000 | 0.38 | 0.87 |
| H100 | 80 GB | hopper | 989 | 1979 | $25,000 | 0.38 | 0.87 |
| H20 | 96 GB | hopper | 148 | 296 | $12,000 | 0.35 | 0.85 |
| **Ampere Series** |
| A100 80GB | 80 GB | ampere | 312 | - | $15,000 | 0.45 | 0.87 |
| A100 | 40 GB | ampere | 312 | - | $10,000 | 0.45 | 0.87 |
| **Ada Series** |
| L40S | 48 GB | ada | 733 | 1457 | $14,000 | 0.35 | 0.82 |

### AMD (5 GPUs)
| GPU | VRAM | Architecture | BF16 TFLOPS | FP8 TFLOPS | Hardware Cost | MFU Prefill | MFU Decode |
|-----|------|--------------|-------------|------------|---------------|-------------|------------|
| **CDNA4** |
| MI350X | 288 GB | cdna4 | 2600 | 4600 | $40,000 | 0.25 | 0.80 |
| **CDNA3** |
| MI325X | 256 GB | cdna3 | 1307 | 2615 | $30,000 | 0.20 | 0.78 |
| MI300X | 192 GB | cdna3 | 1307 | 2615 | $25,000 | 0.20 | 0.78 |
| MI300A | 128 GB | cdna3 | 1307 | 2615 | $20,000 | 0.20 | 0.78 |
| **CDNA2** |
| MI250X | 128 GB | cdna2 | 383 | - | $15,000 | 0.22 | 0.78 |

## New Fields in GPU Specs

### Performance Metrics
- **`tflops_bf16`**: Dense BF16 TFLOPS (official spec, non-sparse)
- **`tflops_fp8`**: Dense FP8 TFLOPS (2x BF16 for most GPUs)
- **`mfu_prefill`**: Model FLOPs Utilization for compute-bound prefill (from benchmarks)
  - Higher is better for long prompt workloads
  - NVIDIA Ampere: 0.45, Hopper: 0.38, Blackwell: 0.44
  - AMD CDNA3: 0.20 (lower due to software maturity)
- **`mfu_decode`**: Memory bandwidth utilization for decode
  - Critical for long-form generation
  - Most GPUs: 0.85-0.87 (excellent)

### Hardware Details
- **`architecture`**: Microarchitecture family (ampere, hopper, blackwell, cdna2/3/4)
- **`color`**: Hex color code for visualization in UI charts
- **`tokens_per_dollar`**: Cost efficiency metric (higher is better)
- **`nvlink_bandwidth_gbps`**: NVLink bandwidth (if available)

## API Response Changes

### New Response Format

```json
{
  "id": "h100-80gb",
  "name": "H100",
  "memory_gb": 80,
  "price_per_hour": 0,
  "hardware_cost": 25000,
  "memory_bandwidth_gbps": 3350,
  "tflops": 989,
  "power_watts": 700,
  "cloud_availability_pct": 0,
  
  // NEW FIELDS (from gpu-catalog.json)
  "vendor": "nvidia",
  "architecture": "hopper",
  "tflops_bf16": 989,
  "tflops_fp8": 1979,
  "mfu_prefill": 0.38,
  "mfu_decode": 0.87,
  "tokens_per_dollar": 18000,
  "color": "#5b9bd5"
}
```

## Backward Compatibility

All existing code continues to work:
- **Legacy field names**: `vramGb`, `memoryBandwidthGbps`, `tflops`, `pricePerHour`, etc.
- **New field names**: `vram_gb`, `memory_bandwidth_tbps`, `tflops_bf16`, etc.

Both naming conventions are available on every GPU object.

## Usage Examples

### Filter by Architecture
```typescript
import { getGpusByArchitecture } from '@/lib/gpu-math/gpus'

const hopperGpus = getGpusByArchitecture('hopper')
// Returns: H100, H200, H20
```

### Filter by Vendor
```typescript
import { getGpusByVendor } from '@/lib/gpu-math/gpus'

const amdGpus = getGpusByVendor('amd')
// Returns: MI250X, MI300X, MI325X, MI300A, MI350X
```

### Get GPU by ID
```typescript
import { getGpuById } from '@/lib/gpu-math/gpus'

const h100 = getGpuById('h100-80gb')
console.log(h100?.tflops_fp8) // 1979
console.log(h100?.mfu_prefill) // 0.38
```

### Filter by Minimum VRAM
```typescript
import { getGpusByMinVram } from '@/lib/gpu-math/gpus'

const largeMemoryGpus = getGpusByMinVram(192)
// Returns: B300 (288GB), MI350X (288GB), MI325X (256GB), B100/B200/MI300X (192GB)
```

## Data Source

Converted from `docs/gpu-data` JavaScript array to structured JSON.

**Original format:**
```javascript
{name:'H100', vendor:'nvidia', vram:80, price:25000, mem_bw:3.35, ...}
```

**New format:**
```json
{
  "id": "h100-80gb",
  "name": "H100",
  "vendor": "nvidia",
  "vram_gb": 80,
  "hardware_cost_usd": 25000,
  "memory_bandwidth_tbps": 3.35,
  ...
}
```

## Benefits

1. **Easy to update**: Edit JSON file instead of TypeScript code
2. **More GPUs**: 15 GPUs (was 11) with latest Blackwell/CDNA4 series
3. **Better specs**: FP8 TFLOPS, MFU metrics, architecture family
4. **Ready for database**: JSON structure maps directly to Cloudflare D1 if needed
5. **Visualization ready**: Color codes for charting

## Next Steps

### Option 1: Keep JSON file (current approach)
- ✅ Simple, no database required
- ✅ Fast (no API calls)
- ❌ Manual updates needed

### Option 2: Move to Cloudflare D1 database
- ✅ Centralized data (GPU specs + pricing in same DB)
- ✅ Easy to update via admin API
- ✅ Single source of truth
- ❌ Requires database migration

### Option 3: Hybrid (recommended for now)
- GPU **specs**: Keep in JSON (hardware doesn't change often)
- GPU **pricing**: Fetch from Cloudflare Worker (changes daily)
- VM/network **pricing**: Add to Cloudflare Worker (future)

## Testing

```bash
# Start dev server
npm run dev

# Test GPU catalog
curl http://localhost:3000/api/v1/gpus | jq '.data.gpus | length'
# Output: 15

# Test new fields
curl http://localhost:3000/api/v1/gpus | jq '.data.gpus[0] | {name, architecture, tflops_bf16, mfu_prefill}'

# Test live pricing integration
curl 'http://localhost:3000/api/v1/gpus?live_pricing=true' | jq '.data.gpus[] | select(.name == "H100")'
```

## Migration Checklist

- [x] Convert `docs/gpu-data` to `lib/gpu-math/gpu-catalog.json`
- [x] Update `lib/gpu-math/gpus.ts` to load from JSON
- [x] Add backward compatibility for legacy field names
- [x] Export helper functions (getGpuById, getGpusByVendor, etc.)
- [x] Export DEFAULT_GPU constant
- [x] Verify TypeScript compilation passes
- [x] Test API endpoint `/api/v1/gpus`
- [x] Test live pricing integration
- [ ] Update UI components to use new fields (architecture, MFU metrics)
- [ ] Add GPU architecture filter to UI
- [ ] Add MFU metrics to GPU comparison charts
- [ ] Document VM/network pricing requirements
