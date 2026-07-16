import { NextRequest, NextResponse } from 'next/server'
import { KvCacheCalcRequestSchema } from '@/lib/api/schemas'
import { callKvCacheCalc, generateKvRequestId } from '@/lib/api/kv-cache-calc'

const ERROR_STATUS_MAP: Record<string, number> = {
  INVALID_REQUEST: 400,
  KV_CACHE_NOT_CONFIGURED: 503,
  KV_CACHE_AUTH_FAILED: 502,
  KV_CACHE_UNAVAILABLE: 502,
  KV_CACHE_TIMEOUT: 504,
  KV_CACHE_INVALID_RESPONSE: 502,
  INTERNAL_ERROR: 500,
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const validated = KvCacheCalcRequestSchema.parse(body)
    const result = await callKvCacheCalc(validated)

    if (result.status === 'failed') {
      const httpStatus = ERROR_STATUS_MAP[result.error.code] ?? 500
      return NextResponse.json(result, { status: httpStatus })
    }

    return NextResponse.json(result, {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    })
  } catch (err: unknown) {
    const requestId = generateKvRequestId()

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
