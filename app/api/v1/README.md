# GPU Calc API v1

Production-ready REST API for LLM inference configuration, GPU sizing, and vLLM/llm-d optimization.

## Base URL

```
http://localhost:3000/api/v1  (development)
https://gpu-calc-three.vercel.app/api/v1  (production)
```

## Endpoints

### POST /api/v1/config

Generate complete inference configuration with vLLM settings, GPU sizing, and optimization recommendations.

**Request:**
```json
{
  "model_name": "meta-llama/Llama-3.1-70B-Instruct",
  "precision": "FP16",
  "gpu_type": "h100-sxm-80gb",
  "concurrent_users": 100,
  "isl": 2000,
  "osl": 500,
  "workload_type": "chat",
  "sla_priority": "ttft",
  "gpu_count": 8,  // Optional - engine recommends if not provided
  "network_topology": "nvlink",  // Optional: nvlink | infiniband | ethernet
  "enable_llmd": false  // Optional: enable disaggregated config
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "memory_analysis": {
      "weight_gb": 140,
      "weight_gb_per_gpu": 35,
      "usable_hbm_per_gpu": 68,
      "tp_size": 4,
      "replicas": 1,
      "kv_cache_budget_gb": 33,
      "max_sequences_from_memory": 66000
    },
    "vllm_config": {
      "tensor_parallel_size": 4,
      "max_model_len": 2500,
      "max_num_seqs": 120,
      "gpu_memory_utilization": 0.85,
      "max_num_batched_tokens": 512,
      "enable_chunked_prefill": true,
      "enable_prefix_caching": false,
      "quantization": "fp16"
    },
    "parallelism_strategy": {
      "strategy": "TP_ONLY",
      "pp_size": 1,
      "topology_note": "NVLink within node — optimal"
    },
    "bottleneck_analysis": {
      "primary": "MIXED",
      "risk": "Balanced ISL/OSL with high concurrency stresses both prefill and decode",
      "fix_suggestions": [...]
    },
    "diagnostics": {
      "nvidia_smi_watch": "nvidia-smi dmon -s pucvmet -c 10",
      "dcgm_metrics": [...],
      "vllm_metrics": [...]
    },
    "warnings": [...]
  },
  "metadata": {
    "generated_at": "2026-06-03T22:31:40.219Z",
    "version": "v1"
  }
}
```

**Error Responses:**
- `400` - Validation error (invalid parameters)
- `404` - Model or GPU not found in catalog
- `500` - Internal server error

---

### GET /api/v1/gpus

Get GPU catalog with optional filtering and sorting. Supports live pricing from Cloudflare Worker.

**Query Parameters:**
- `min_memory` (number, optional) - Minimum VRAM in GB
- `max_price` (number, optional) - Maximum price per hour
- `vendor` (string, optional) - Filter by vendor name (NVIDIA, AMD)
- `sort` (enum, optional) - Sort by: `memory`, `price`, `performance` (default: `memory`)
- `live_pricing` (boolean, optional) - Include live multi-cloud pricing (default: `false`)

**Examples:**
```bash
# Get all GPUs sorted by memory (default)
GET /api/v1/gpus

# Get GPUs with at least 80GB VRAM, sorted by price
GET /api/v1/gpus?min_memory=80&sort=price

# Get NVIDIA GPUs under $5/hour
GET /api/v1/gpus?vendor=nvidia&max_price=5

# Get GPUs with live pricing from AWS, GCP, Azure, RunPod, etc.
GET /api/v1/gpus?live_pricing=true
```

**Response:**
```json
{
  "success": true,
  "data": {
    "gpus": [
      {
        "id": "h100-sxm-80gb",
        "name": "H100 SXM 80 GB",
        "memory_gb": 80,
        "price_per_hour": 6.00,
        "hardware_cost": 30000,
        "memory_bandwidth_gbps": 3350,
        "tflops": 989,
        "power_watts": 700,
        "cloud_availability_pct": 7,
        "live_pricing": {
          "on_demand": {
            "min": 2.19,
            "median": 12.29,
            "max": 98.32,
            "providers": [
              { "provider": "RunPod", "price": 2.19, "region": "global" },
              { "provider": "AWS", "price": 12.29, "region": "us-east-1" }
            ]
          },
          "spot": {
            "min": 1.40,
            "median": 1.82,
            "max": 2.74,
            "providers": [...]
          },
          "last_updated": "2026-05-01T02:01:09.357Z"
        }
      },
      ...
    ],
    "count": 10
  }
}
```
*Note: `live_pricing` field only included when `?live_pricing=true` is set*

---

### GET /api/v1/models

Search and filter model catalog.

**Query Parameters:**
- `q` (string, optional) - Search query (matches name or HuggingFace ID)
- `vendor` (string, optional) - Filter by vendor (Meta, Google, Mistral, etc.)
- `min_params` (number, optional) - Minimum parameter count (in billions)
- `max_params` (number, optional) - Maximum parameter count (in billions)
- `limit` (number, optional) - Results limit (default: 50, max: 100)

**Examples:**
```bash
# Search for Llama models
GET /api/v1/models?q=llama

# Get Meta models between 10B and 100B parameters
GET /api/v1/models?vendor=Meta&min_params=10&max_params=100

# Get first 20 models
GET /api/v1/models?limit=20
```

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "llama-3.1-70b",
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "name": "Llama 3.1 70B",
        "vendor": "Meta",
        "param_label": "70B",
        "tags": [],
        "is_new": false
      },
      ...
    ],
    "count": 15
  }
}
```

---

## Validation

All endpoints use [Zod](https://zod.dev/) for request validation. Invalid requests return:

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": [
    {
      "path": ["precision"],
      "message": "Invalid enum value. Expected 'FP16' | 'FP8' | 'INT8' | 'INT4', received 'FP32'"
    }
  ]
}
```

## Rate Limiting

Currently no rate limiting (development). Production deployment will enforce:
- 100 requests/minute for `/config` (compute-heavy)
- 300 requests/minute for `/gpus` and `/models` (catalog lookups)

## Caching

- `/config` - 5 minutes cache, stale-while-revalidate for 10 minutes
- `/gpus` - 1 hour cache
- `/models` - 1 hour cache

## CORS

All endpoints support CORS with `Access-Control-Allow-Origin: *` for development.

Production deployment should restrict origins.

## Examples

### cURL

```bash
# Get inference config
curl -X POST http://localhost:3000/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "precision": "FP16",
    "gpu_type": "h100-sxm-80gb",
    "concurrent_users": 50,
    "isl": 1000,
    "osl": 200,
    "workload_type": "chat",
    "sla_priority": "ttft"
  }'

# Get GPUs sorted by price
curl "http://localhost:3000/api/v1/gpus?sort=price"

# Search models
curl "http://localhost:3000/api/v1/models?q=mistral"
```

### JavaScript/TypeScript

```typescript
// Get inference config
const response = await fetch('http://localhost:3000/api/v1/config', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_name: 'meta-llama/Llama-3.1-70B-Instruct',
    precision: 'FP16',
    gpu_type: 'h100-sxm-80gb',
    concurrent_users: 100,
    isl: 2000,
    osl: 500,
    workload_type: 'chat',
    sla_priority: 'ttft'
  })
})

const config = await response.json()
console.log(`TP size: ${config.data.vllm_config.tensor_parallel_size}`)
console.log(`Replicas: ${config.data.memory_analysis.replicas}`)
```

### Python

```python
import requests

# Get inference config
response = requests.post('http://localhost:3000/api/v1/config', json={
    'model_name': 'meta-llama/Llama-3.1-70B-Instruct',
    'precision': 'FP16',
    'gpu_type': 'h100-sxm-80gb',
    'concurrent_users': 100,
    'isl': 2000,
    'osl': 500,
    'workload_type': 'chat',
    'sla_priority': 'ttft'
})

config = response.json()
print(f"TP size: {config['data']['vllm_config']['tensor_parallel_size']}")
print(f"Recommended GPUs: {config['data']['memory_analysis']['tp_size'] * config['data']['memory_analysis']['replicas']}")
```

---

## Next Steps

Ready to use the API! For UI integration, see Phase 3: UI Redesign documentation.
