// POST /api/v1/config
// Main inference configuration endpoint

import { NextRequest, NextResponse } from 'next/server'
import { ZodError } from 'zod'
import { InferenceConfigRequestSchema } from '@/lib/api/schemas'
import { ApiErrors } from '@/lib/api/errors'
import { formatInferenceConfigResponse } from '@/lib/api/responses'
import { computeInferenceConfig } from '@/lib/gpu-math/inference-config'

export async function POST(req: NextRequest) {
  try {
    // Parse and validate request body
    const body = await req.json()
    const validatedData = InferenceConfigRequestSchema.parse(body)

    // Compute inference configuration using the engine
    const result = computeInferenceConfig(validatedData)

    // Return formatted response
    return NextResponse.json(formatInferenceConfigResponse(result), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600'
      }
    })

  } catch (error) {
    // Handle Zod validation errors
    if (error instanceof ZodError) {
      return NextResponse.json(
        ApiErrors.VALIDATION_ERROR(error.issues),
        { status: 400 }
      )
    }

    // Handle inference engine errors (validation, model/GPU not found, etc.)
    if (error instanceof Error) {
      // Check if it's a model/GPU not found error
      if (error.message.includes('not found in catalog')) {
        if (error.message.includes('Model')) {
          const match = error.message.match(/Model "([^"]+)"/)
          const modelName = match ? match[1] : 'unknown'
          return NextResponse.json(
            ApiErrors.MODEL_NOT_FOUND(modelName),
            { status: 404 }
          )
        }
        if (error.message.includes('GPU')) {
          const match = error.message.match(/GPU "([^"]+)"/)
          const gpuType = match ? match[1] : 'unknown'
          return NextResponse.json(
            ApiErrors.GPU_NOT_FOUND(gpuType),
            { status: 404 }
          )
        }
      }

      // Check if it's a validation error from the engine
      if (error.message.startsWith('Invalid inference request:')) {
        return NextResponse.json(
          ApiErrors.INVALID_REQUEST(error.message),
          { status: 400 }
        )
      }

      // Other errors - internal server error
      return NextResponse.json(
        ApiErrors.INTERNAL_ERROR(error.message),
        { status: 500 }
      )
    }

    // Unknown error
    return NextResponse.json(
      ApiErrors.INTERNAL_ERROR('An unexpected error occurred'),
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
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  })
}
