import { NextRequest, NextResponse } from 'next/server'
import { GpuSizerRequestSchema } from '@/lib/api/schemas'
import { callGpuSizer, generateRequestId } from '@/lib/api/gpu-sizer'

const ERROR_STATUS_MAP: Record<string, number> = {
  INVALID_REQUEST: 400,
  GPU_SIZER_NOT_CONFIGURED: 503,
  GPU_SIZER_AUTH_FAILED: 502,
  GPU_SIZER_UNAVAILABLE: 502,
  GPU_SIZER_TIMEOUT: 504,
  GPU_SIZER_INVALID_RESPONSE: 502,
  GPU_SIZER_NO_CONFIGURATION: 422,
  INTERNAL_ERROR: 500,
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const validated = GpuSizerRequestSchema.parse(body)
    const result = await callGpuSizer(validated)

    if (result.status === 'failed') {
      const httpStatus = ERROR_STATUS_MAP[result.error.code] ?? 500
      return NextResponse.json(result, { status: httpStatus })
    }

    return NextResponse.json(result, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (err: unknown) {
    const requestId = generateRequestId()

    if (err instanceof Error && err.constructor.name === 'ZodError') {
      const zodErr = err as Error & { issues: unknown[] }
      return NextResponse.json(
        {
          requestId,
          status: 'failed',
          error: {
            code: 'INVALID_REQUEST',
            message: 'Request validation failed',
            details: zodErr.issues,
          },
        },
        { status: 400 }
      )
    }

    return NextResponse.json(
      {
        requestId,
        status: 'failed',
        error: {
          code: 'INTERNAL_ERROR',
          message: err instanceof Error ? err.message : 'An unexpected error occurred',
        },
      },
      { status: 500 }
    )
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
