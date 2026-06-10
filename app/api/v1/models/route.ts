// GET /api/v1/models
// Model catalog endpoint with search and filtering

import { NextRequest, NextResponse } from 'next/server'
import { ZodError } from 'zod'
import { ModelCatalogQuerySchema } from '@/lib/api/schemas'
import { ApiErrors } from '@/lib/api/errors'
import { formatModelCatalogResponse } from '@/lib/api/responses'
import { MODEL_CATALOG } from '@/lib/gpu-math/models'

export async function GET(req: NextRequest) {
  try {
    // Parse and validate query parameters
    const { searchParams } = new URL(req.url)
    const query = {
      q: searchParams.get('q'),
      vendor: searchParams.get('vendor'),
      min_params: searchParams.get('min_params'),
      max_params: searchParams.get('max_params'),
      limit: searchParams.get('limit')
    }

    const validatedQuery = ModelCatalogQuerySchema.parse(query)

    // Filter models
    let filteredModels = [...MODEL_CATALOG]

    // Search by query (matches name or hfId)
    if (validatedQuery.q) {
      const queryLower = validatedQuery.q.toLowerCase()
      filteredModels = filteredModels.filter(model =>
        model.name.toLowerCase().includes(queryLower) ||
        model.hfId.toLowerCase().includes(queryLower)
      )
    }

    // Filter by vendor
    if (validatedQuery.vendor) {
      filteredModels = filteredModels.filter(model =>
        model.vendor.toLowerCase() === validatedQuery.vendor!.toLowerCase()
      )
    }

    // Filter by parameter count (parse from paramLabel, e.g., "70B" → 70)
    if (validatedQuery.min_params || validatedQuery.max_params) {
      filteredModels = filteredModels.filter(model => {
        const match = model.paramLabel.match(/(\d+)B/)
        if (!match) return false
        const params = parseInt(match[1], 10)

        if (validatedQuery.min_params && params < validatedQuery.min_params) return false
        if (validatedQuery.max_params && params > validatedQuery.max_params) return false
        return true
      })
    }

    // Apply limit
    filteredModels = filteredModels.slice(0, validatedQuery.limit)

    // Return formatted response
    return NextResponse.json(formatModelCatalogResponse(filteredModels), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400'
      }
    })

  } catch (error) {
    if (error instanceof ZodError) {
      return NextResponse.json(
        ApiErrors.VALIDATION_ERROR(error.issues),
        { status: 400 }
      )
    }

    return NextResponse.json(
      ApiErrors.INTERNAL_ERROR('Failed to fetch model catalog'),
      { status: 500 }
    )
  }
}

// CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  })
}
