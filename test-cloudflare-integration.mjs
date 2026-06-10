// Test Cloudflare integration directly
import { fetchGPUPricing, aggregateGPUPricing } from './lib/api/cloudflare.ts'

async function test() {
  console.log('Fetching GPU pricing from Cloudflare Worker...')

  try {
    const data = await fetchGPUPricing({ gpu: 'H100' })
    console.log(`\nTotal H100 prices: ${data.prices.length}`)
    console.log(`Source: ${data.source}`)
    console.log(`Timestamp: ${data.timestamp}`)

    if (data.prices.length > 0) {
      console.log('\nSample price record:')
      console.log(JSON.stringify(data.prices[0], null, 2))

      // Test aggregation
      const h100_80gb_prices = data.prices.filter(p => p.vram_gb === 80)
      console.log(`\nH100 80GB prices: ${h100_80gb_prices.length}`)

      const onDemand = aggregateGPUPricing(h100_80gb_prices, 'on_demand')
      const spot = aggregateGPUPricing(h100_80gb_prices, 'spot')

      console.log('\nOn-demand pricing:')
      console.log(JSON.stringify(onDemand, null, 2))

      console.log('\nSpot pricing:')
      console.log(JSON.stringify(spot, null, 2))
    }
  } catch (error) {
    console.error('Error:', error.message)
    console.error(error.stack)
  }
}

test()
