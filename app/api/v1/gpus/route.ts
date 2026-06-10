// GET /api/v1/gpus
// GPU catalog endpoint with filtering + live pricing from Cloudflare Worker

import { NextRequest, NextResponse } from 'next/server'
import { ZodError } from 'zod'
import { GpuCatalogQuerySchema } from '@/lib/api/schemas'
import { ApiErrors } from '@/lib/api/errors'
import { formatGpuCatalogResponse } from '@/lib/api/responses'
import { GPU_CATALOG, type GpuSpec } from '@/lib/gpu-math/gpus'
import { fetchGPUPricing, aggregateGPUPricing } from '@/lib/api/cloudflare'

export async function GET(req: NextRequest) {
  try {
    // Parse and validate query parameters
    const { searchParams } = new URL(req.url)
    const query: Record<string, string> = {}

    const minMem = searchParams.get('min_memory')
    const maxPrice = searchParams.get('max_price')
    const vendor = searchParams.get('vendor')
    const sort = searchParams.get('sort')
    const includeLivePricing = searchParams.get('live_pricing') === 'true'

    if (minMem) query.min_memory = minMem
    if (maxPrice) query.max_price = maxPrice
    if (vendor) query.vendor = vendor
    if (sort) query.sort = sort

    const validatedQuery = GpuCatalogQuerySchema.parse(query)

    // Start with static GPU catalog
    let filteredGpus: GpuSpec[] = [...GPU_CATALOG]

    // Optionally enrich with live pricing from Cloudflare Worker
    if (includeLivePricing) {
      try {
        console.log('[GPUs API] Fetching live pricing from Cloudflare Worker...')
        const cloudflareData = await fetchGPUPricing()
        console.log(`[GPUs API] Received ${cloudflareData.prices.length} prices from Cloudflare`)

        // Enrich each GPU with live pricing
        filteredGpus = filteredGpus.map(gpu => {
          // Extract GPU model from name (e.g., "H100 SXM 80 GB" → "H100")
          const gpuModel = gpu.name.split(' ')[0]

          // Find matching prices from Cloudflare
          const matchingPrices = cloudflareData.prices.filter(p =>
            p.gpu === gpuModel &&
            p.vram_gb === gpu.vramGb
          )

          console.log(`[GPUs API] ${gpu.name}: Found ${matchingPrices.length} matching prices`)

          if (matchingPrices.length === 0) {
            return gpu
          }

          // Aggregate on-demand and spot pricing
          const onDemandPricing = aggregateGPUPricing(matchingPrices, 'on_demand')
          const spotPricing = aggregateGPUPricing(matchingPrices, 'spot')

          console.log(`[GPUs API] ${gpu.name}: on-demand count=${onDemandPricing.count}, spot count=${spotPricing.count}`)

          return {
            ...gpu,
            livePricing: {
              onDemand: onDemandPricing.count > 0 ? onDemandPricing : undefined,
              spot: spotPricing.count > 0 ? spotPricing : undefined,
              lastUpdated: cloudflareData.timestamp
            }
          }
        })
      } catch (cloudflareError) {
        // Log error but continue with static pricing
        console.error('[GPUs API] Failed to fetch live pricing from Cloudflare:', cloudflareError)
      }
    }

    // Filter by min_memory
    if (validatedQuery.min_memory) {
      filteredGpus = filteredGpus.filter(gpu => gpu.vramGb >= validatedQuery.min_memory!)
    }

    // Filter by max_price (use live pricing median if available, fallback to static)
    if (validatedQuery.max_price) {
      filteredGpus = filteredGpus.filter(gpu => {
        const effectivePrice = gpu.livePricing?.onDemand?.median ?? gpu.pricePerHour
        return effectivePrice <= validatedQuery.max_price!
      })
    }

    // Filter by vendor
    if (validatedQuery.vendor) {
      const vendorLower = validatedQuery.vendor.toLowerCase()
      filteredGpus = filteredGpus.filter(gpu =>
        gpu.name.toLowerCase().includes(vendorLower)
      )
    }

    // Sort GPUs
    switch (validatedQuery.sort) {
      case 'memory':
        filteredGpus.sort((a, b) => b.vramGb - a.vramGb)
        break
      case 'price':
        filteredGpus.sort((a, b) => {
          const priceA = a.livePricing?.onDemand?.median ?? a.pricePerHour
          const priceB = b.livePricing?.onDemand?.median ?? b.pricePerHour
          return priceA - priceB
        })
        break
      case 'performance':
        filteredGpus.sort((a, b) => b.tflops - a.tflops)
        break
    }

    // Return formatted response
    return NextResponse.json(formatGpuCatalogResponse(filteredGpus), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, s-maxage=21600, stale-while-revalidate=86400' // 6 hours
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
      ApiErrors.INTERNAL_ERROR('Failed to fetch GPU catalog'),
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
