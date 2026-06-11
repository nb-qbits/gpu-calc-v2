#!/bin/bash

# Test API KV category detection with HF config fetch
# Usage: HF_TOKEN=<token> ./test-api-kv-detection.sh

if [ -z "$HF_TOKEN" ]; then
  echo "❌ Error: HF_TOKEN environment variable is required"
  echo "Usage: HF_TOKEN=<your-token> ./test-api-kv-detection.sh"
  exit 1
fi

API_URL="http://localhost:3005/api/v1/config"

echo "Testing API KV category detection..."
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 1: Public model (DeepSeek V2 Lite) - no token needed, should detect KV-2
echo "Test 1: deepseek-ai/DeepSeek-V2-Lite (public, KV-2)"
result1=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deepseek-ai/DeepSeek-V2-Lite",
    "precision": "FP16",
    "gpu_type": "h200-141gb",
    "concurrent_users": 97,
    "isl": 1000,
    "osl": 150,
    "workload_type": "chat",
    "sla_priority": "ttft"
  }')

category1=$(echo "$result1" | jq -r '.data.memory_analysis.kv_category // "ERROR"')
weight1=$(echo "$result1" | jq -r '.data.memory_analysis.weight_gb // 0')

if [ "$category1" = "KV-2" ]; then
  echo "  ✅ PASS - Category: $category1, Weight: ${weight1}GB"
else
  echo "  ❌ FAIL - Got: $category1 (expected KV-2)"
fi
echo ""

# Test 2: Gated model (Gemma 3 4B) - requires token, should detect KV-3b
echo "Test 2: google/gemma-3-4b-it (gated, KV-3b, requires token)"
result2=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_name\": \"google/gemma-3-4b-it\",
    \"precision\": \"FP16\",
    \"gpu_type\": \"h200-141gb\",
    \"concurrent_users\": 97,
    \"isl\": 1000,
    \"osl\": 150,
    \"workload_type\": \"chat\",
    \"sla_priority\": \"ttft\",
    \"hf_token\": \"$HF_TOKEN\"
  }")

category2=$(echo "$result2" | jq -r '.data.memory_analysis.kv_category // "ERROR"')
weight2=$(echo "$result2" | jq -r '.data.memory_analysis.weight_gb // 0')

if [ "$category2" = "KV-3b" ]; then
  echo "  ✅ PASS - Category: $category2, Weight: ${weight2}GB"
else
  echo "  ❌ FAIL - Got: $category2 (expected KV-3b)"
fi
echo ""

# Test 3: SSM model (Mamba) - no token needed, should detect KV-5a
echo "Test 3: state-spaces/mamba-2.8b (public, KV-5a)"
result3=$(curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "state-spaces/mamba-2.8b",
    "precision": "FP16",
    "gpu_type": "h200-141gb",
    "concurrent_users": 97,
    "isl": 1000,
    "osl": 150,
    "workload_type": "chat",
    "sla_priority": "ttft"
  }')

category3=$(echo "$result3" | jq -r '.data.memory_analysis.kv_category // "ERROR"')
weight3=$(echo "$result3" | jq -r '.data.memory_analysis.weight_gb // 0')

if [ "$category3" = "KV-5a" ]; then
  echo "  ✅ PASS - Category: $category3, Weight: ${weight3}GB"
else
  echo "  ❌ FAIL - Got: $category3 (expected KV-5a)"
fi
echo ""

echo "════════════════════════════════════════════════════════════"

# Summary
total=3
pass=0
[ "$category1" = "KV-2" ] && pass=$((pass + 1))
[ "$category2" = "KV-3b" ] && pass=$((pass + 1))
[ "$category3" = "KV-5a" ] && pass=$((pass + 1))

echo "Results: $pass/$total tests passed"

if [ $pass -eq $total ]; then
  echo "✅ All tests passed!"
  exit 0
else
  echo "❌ Some tests failed"
  exit 1
fi
