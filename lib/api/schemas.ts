// API Request/Response Schemas
// Zod schemas for validation

import { z } from 'zod'

// ═══ INFERENCE CONFIG REQUEST SCHEMA ═══

export const InferenceConfigRequestSchema = z.object({
  model_name: z.string().min(1, 'model_name is required'),
  precision: z.enum(['FP16', 'FP8', 'INT8', 'INT4']),
  gpu_type: z.string().min(1, 'gpu_type is required'),
  gpu_count: z.number().int().positive().optional(),
  concurrent_users: z.number().int().positive('concurrent_users must be positive'),
  isl: z.number().int().positive('isl must be positive'),
  osl: z.number().int().positive('osl must be positive'),
  workload_type: z.enum(['chat', 'web_search', 'rag', 'batch', 'coding']),
  sla_priority: z.enum(['ttft', 'tpot', 'throughput']),
  network_topology: z.enum(['nvlink', 'infiniband', 'ethernet']).optional(),
  enable_llmd: z.boolean().optional(),
  hf_token: z.string().optional(),
  kv_cache_precision: z.enum(['FP16', 'FP8']).optional()
})

export type InferenceConfigRequest = z.infer<typeof InferenceConfigRequestSchema>

// ═══ GPU CATALOG QUERY SCHEMA ═══

export const GpuCatalogQuerySchema = z.object({
  min_memory: z.string().transform(val => val ? parseFloat(val) : undefined).optional(),
  max_price: z.string().transform(val => val ? parseFloat(val) : undefined).optional(),
  vendor: z.string().optional(),
  sort: z.enum(['memory', 'price', 'performance']).default('memory')
})

export type GpuCatalogQuery = z.infer<typeof GpuCatalogQuerySchema>

// ═══ MODEL CATALOG QUERY SCHEMA ═══

export const ModelCatalogQuerySchema = z.object({
  q: z.string().optional(),  // Search query
  vendor: z.string().optional(),
  min_params: z.string().transform(val => val ? parseInt(val, 10) : undefined).optional(),
  max_params: z.string().transform(val => val ? parseInt(val, 10) : undefined).optional(),
  limit: z.string().optional().default('50').transform(val => parseInt(val, 10))
})

export type ModelCatalogQuery = z.infer<typeof ModelCatalogQuerySchema>
