# Cloudflare Worker Integration

## Overview

Successfully integrated live GPU pricing from Cloudflare Worker (`gpu-pricing-worker.vikasgrover2004.workers.dev`) into Next.js inference config API.

## Architecture

```
┌─────────────────────────────────────────┐
│ Next.js App (Vercel)                   │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ Inference Config Engine          │  │
│  │ lib/gpu-math/inference-config/   │  │
│  │ (Static calculations)            │  │
│  └──────────────────────────────────┘  │
│                ↓                        │
│  ┌──────────────────────────────────┐  │
│  │ API Routes                       │  │
│  │ /api/v1/config  → Static data    │  │
│  │ /api/v1/gpus    → Live pricing ──┼──┐
│  │ /api/v1/models  → Static data    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
                                          │
                                          ↓ HTTPS
┌─────────────────────────────────────────────────────┐
│ Cloudflare Worker                                    │
│ https://gpu-pricing-worker.vikasgrover2004.workers.dev │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ D1 Database (SQL)                            │  │
│  │ ├─ prices        (current GPU/API pricing)   │  │
│  │ ├─ price_history (historical trends)         │  │
│  │ ├─ alerts        (price changes >20%)        │  │
│  │ └─ runs          (scraping metadata)         │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ KV Store (PRICE_CACHE, 6-hour TTL)          │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ Collectors (Multi-source scraping)           │  │
│  │ ├─ AWS      (hardcoded instances)            │  │
│  │ ├─ GCP      (hardcoded instances)            │  │
│  │ ├─ Azure    (live API)                       │  │
│  │ ├─ Vast.ai  (live API)                       │  │
│  │ ├─ RunPod   (GraphQL API)                    │  │
│  │ ├─ CoreWeave (HTML scrape)                   │  │
│  │ └─ API Tokens (Anthropic, OpenAI, Google)    │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  Cron: Bi-weekly automated scraping                 │
└─────────────────────────────────────────────────────┘
```

## Data Flow

1. **Static baseline**: GPU_CATALOG has hardcoded specs (VRAM, TFLOPS, bandwidth)
2. **Live pricing**: `/api/v1/gpus?live_pricing=true` fetches from Cloudflare Worker
3. **Enrichment**: Each GPU gets `livePricing` object with min/median/max prices
4. **Caching**: Next.js cache (6 hours) + Cloudflare KV cache (6 hours)

## Key Features

### Multi-cloud Price Comparison

H100 80GB example pricing:
- **On-demand**: $0.40/hr (GCP) to $18.16/hr (Azure West Europe)
- **Spot**: $0.21/hr (Azure) to $15.82/hr (Azure)
- **Median on-demand**: $1.14/hr (13 providers)
- **Median spot**: $2.04/hr (21 providers)

### API Response Structure

```json
{
  "success": true,
  "data": {
    "gpus": [
      {
        "id": "h100-sxm-80gb",
        "name": "H100 SXM 80 GB",
        "memory_gb": 80,
        "price_per_hour": 6.00,  // Static baseline
        "live_pricing": {
          "onDemand": {
            "min": 0.4,
            "median": 1.135,
            "max": 18.16,
            "count": 13,
            "providers": [
              { "provider": "GCP", "price_per_gpu": 0.4, "region": "us-central1" },
              { "provider": "Azure", "price_per_gpu": 0.8725, "region": "eastus" },
              ...
            ]
          },
          "spot": {
            "min": 0.21,
            "median": 2.04,
            "max": 15.82,
            "count": 21,
            "providers": [...]
          },
          "lastUpdated": "2026-06-03T22:55:24.645Z"
        }
      }
    ]
  }
}
```

## Implementation

### New Files Created

1. **[lib/api/cloudflare.ts](../lib/api/cloudflare.ts)**
   - `fetchGPUPricing()` - Fetch from Cloudflare Worker
   - `aggregateGPUPricing()` - Calculate min/median/max pricing
   - `fetchAPITokenPricing()` - API token pricing (future)

2. **[lib/gpu-math/gpus.ts](../lib/gpu-math/gpus.ts)** (enhanced)
   - Added `livePricing` optional field to `GpuSpec` interface

3. **[app/api/v1/gpus/route.ts](../app/api/v1/gpus/route.ts)** (enhanced)
   - Added `?live_pricing=true` query parameter
   - Enriches static GPU catalog with live multi-cloud pricing
   - Filters/sorts using live median pricing when available

4. **[lib/api/responses.ts](../lib/api/responses.ts)** (enhanced)
   - Updated `formatGpuCatalogResponse()` to include `live_pricing` field

## Usage

### Endpoint

```bash
# Get GPUs with static pricing (fast, cached)
GET /api/v1/gpus

# Get GPUs with live multi-cloud pricing
GET /api/v1/gpus?live_pricing=true

# Filter by VRAM and sort by live pricing
GET /api/v1/gpus?live_pricing=true&min_memory=80&sort=price
```

### Response Time

- **Static pricing**: ~50ms (Next.js cache hit)
- **Live pricing (first request)**: ~250-350ms (Cloudflare fetch + enrichment)
- **Live pricing (cached)**: ~50ms (Next.js 6-hour cache)

### Cloudflare Worker Endpoints

```bash
# Get all GPU cloud pricing
GET https://gpu-pricing-worker.vikasgrover2004.workers.dev/prices?category=gpu_cloud

# Filter by GPU type
GET https://gpu-pricing-worker.vikasgrover2004.workers.dev/prices?category=gpu_cloud&gpu=H100

# Price change diff (last 7 days)
GET https://gpu-pricing-worker.vikasgrover2004.workers.dev/diff?since=7d

# System status
GET https://gpu-pricing-worker.vikasgrover2004.workers.dev/status
```

## Pricing Coverage

### GPU Cloud (133 records)
- **H100 80GB**: 34 prices (13 on-demand, 21 spot)
- **H200 141GB**: 6 prices (all on-demand)
- **A100 80GB**: 7 prices
- **L40S 48GB**: 4 prices
- **Providers**: AWS, GCP, Azure, RunPod, Vast.ai

### API Tokens
- Anthropic (Claude 4.x family)
- OpenAI (GPT-4.x, o3/o4)
- Google (Gemini 2.x)
- Groq (Llama 3.x)

## Future Enhancements

1. **Real-time spot price tracking** - Add WebSocket for live spot price updates
2. **Cost optimizer** - Recommend cheapest provider for given workload
3. **Price alerts** - Notify when spot prices drop below threshold
4. **Historical trends** - Show 30/60/90-day price charts
5. **API token cost estimation** - Link model pricing to inference config

## Testing

```bash
# Start dev server
npm run dev

# Test static pricing
curl http://localhost:3000/api/v1/gpus

# Test live pricing
curl 'http://localhost:3000/api/v1/gpus?live_pricing=true' | jq '.data.gpus[0].live_pricing'

# Test filtering + live pricing
curl 'http://localhost:3000/api/v1/gpus?live_pricing=true&min_memory=80&sort=price'
```

## Performance Considerations

- **Cloudflare Worker**: Edge-deployed, <50ms latency globally
- **D1 Database**: SQLite at the edge, cold query ~30ms
- **KV Cache**: 6-hour TTL, <1ms read latency
- **Next.js Cache**: ISR with 6-hour revalidation
- **Total**: ~250ms end-to-end for live pricing (first request)

## Error Handling

If Cloudflare Worker is unavailable:
1. Route handler catches error
2. Falls back to static pricing from GPU_CATALOG
3. Logs error to console
4. Returns 200 OK with static data (graceful degradation)

## Monitoring

Watch dev server logs for:
```
[GPUs API] Fetching live pricing from Cloudflare Worker...
[GPUs API] Received 133 prices from Cloudflare
[GPUs API] H100 SXM 80 GB: Found 34 matching prices
[GPUs API] H100 SXM 80 GB: on-demand count=13, spot count=21
```

## Cost

- **Cloudflare Workers**: Free tier (100k requests/day)
- **D1 Database**: Free tier (5GB storage, 5M reads/month)
- **KV Store**: Free tier (100k reads/day)
- **Next.js on Vercel**: Existing deployment, no additional cost

Total: **$0/month** within free tiers
