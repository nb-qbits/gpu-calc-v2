// Quick integration test for inference config engine
import { computeInferenceConfig } from './lib/gpu-math/inference-config'

console.log('Testing Inference Configuration Engine...\n')

// Test 1: Llama 3.1 70B on H100 - Chat workload
console.log('═══ Test 1: Llama 3.1 70B - Chat ═══')
try {
  const result1 = computeInferenceConfig({
    model_name: 'meta-llama/Llama-3.1-70B-Instruct',
    precision: 'FP16',
    gpu_type: 'h100-sxm-80gb',
    concurrent_users: 100,
    isl: 2000,
    osl: 500,
    workload_type: 'chat',
    sla_priority: 'ttft'
  })

  console.log('✅ Config generated successfully!')
  console.log(`GPU count: ${result1.memory_analysis.tp_size * result1.memory_analysis.replicas}`)
  console.log(`TP size: ${result1.vllm_config.tensor_parallel_size}`)
  console.log(`Replicas: ${result1.memory_analysis.replicas}`)
  console.log(`Weight memory: ${result1.memory_analysis.weight_gb.toFixed(1)} GB`)
  console.log(`Max sequences: ${result1.vllm_config.max_num_seqs}`)
  console.log(`Chunked prefill: ${result1.vllm_config.enable_chunked_prefill}`)
  console.log(`Prefix caching: ${result1.vllm_config.enable_prefix_caching}`)
  console.log(`Bottleneck: ${result1.bottleneck_analysis.primary}`)
  console.log(`Strategy: ${result1.parallelism_strategy.strategy}`)
  if (result1.warnings.length > 0) {
    console.log(`\nWarnings:`)
    result1.warnings.forEach(w => console.log(`  - ${w}`))
  }
} catch (error) {
  console.error('❌ Test 1 failed:', error instanceof Error ? error.message : error)
}

console.log('\n═══ Test 2: Llama 3.1 8B - RAG workload ═══')
try {
  const result2 = computeInferenceConfig({
    model_name: 'meta-llama/Llama-3.1-8B-Instruct',
    precision: 'FP8',
    gpu_type: 'l40s-48gb',
    gpu_count: 4,
    concurrent_users: 50,
    isl: 8000,  // Long RAG context
    osl: 500,
    workload_type: 'rag',
    sla_priority: 'throughput'
  })

  console.log('✅ Config generated successfully!')
  console.log(`GPU count: ${result2.memory_analysis.tp_size * result2.memory_analysis.replicas}`)
  console.log(`TP size: ${result2.vllm_config.tensor_parallel_size}`)
  console.log(`Replicas: ${result2.memory_analysis.replicas}`)
  console.log(`Weight memory: ${result2.memory_analysis.weight_gb.toFixed(1)} GB`)
  console.log(`Chunked prefill: ${result2.vllm_config.enable_chunked_prefill}`)
  console.log(`Prefix caching: ${result2.vllm_config.enable_prefix_caching}`)
  console.log(`Bottleneck: ${result2.bottleneck_analysis.primary}`)
  if (result2.warnings.length > 0) {
    console.log(`\nWarnings:`)
    result2.warnings.forEach(w => console.log(`  - ${w}`))
  }
} catch (error) {
  console.error('❌ Test 2 failed:', error instanceof Error ? error.message : error)
}

console.log('\n═══ Test 3: Validation - Invalid precision ═══')
try {
  const result3 = computeInferenceConfig({
    model_name: 'meta-llama/Llama-3.1-8B-Instruct',
    precision: 'FP32' as any,  // Invalid!
    gpu_type: 'h100-sxm-80gb',
    concurrent_users: 10,
    isl: 1000,
    osl: 200,
    workload_type: 'chat',
    sla_priority: 'ttft'
  })
  console.error('❌ Should have thrown validation error!')
} catch (error) {
  console.log('✅ Validation caught invalid input:', error instanceof Error ? error.message.split('\n')[0] : error)
}

console.log('\n✨ All tests complete!')
