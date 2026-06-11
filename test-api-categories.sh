#!/bin/bash

# Test API with models from each KV category
# Usage: HF_TOKEN=<your-token> ./test-api-categories.sh [local|prod]

MODE=${1:-local}

# Validate HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "❌ Error: HF_TOKEN environment variable is required"
  echo ""
  echo "Usage:"
  echo "  HF_TOKEN=<your-token> ./test-api-categories.sh [local|prod]"
  echo ""
  echo "Get your token at: https://huggingface.co/settings/tokens"
  exit 1
fi

if [ "$MODE" = "prod" ]; then
  API_URL="https://gpu-calc-v2.vercel.app/api/v1/config"
  echo "🌍 Testing PRODUCTION API: $API_URL"
else
  API_URL="http://localhost:3005/api/v1/config"
  echo "🏠 Testing LOCAL API: $API_URL"
fi

echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo ""

# Test models: "model|expected_category|expect_kv_zero"
TESTS=(
  "nvidia/Nemotron-Mini-4B-Instruct|KV-1|false"
  "deepseek-ai/DeepSeek-V2-Lite|KV-2|false"
  "microsoft/phi-3-mini-4k-instruct|KV-3a|false"
  "google/gemma-3-4b-it|KV-3b|false"
  "google/gemma-3-27b-it|KV-3b|false"
  "tiiuae/falcon-mamba-7b|KV-5a|true"
  "state-spaces/mamba-2.8b|KV-5a|true"
  "ai21labs/AI21-Jamba-Mini-1.5|KV-5b|false"
  "nvidia/Nemotron-H-4B-Base-8K|KV-5b|false"
  "deepseek-ai/DeepSeek-R1|KV-2|false"
  "zai-org/GLM-5.1-FP8|KV-2|false"
)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

printf "%-6s %-8s %-45s %-12s %-14s %-12s %s\n" "Status" "Category" "Model" "Weight(GB)" "KV_cache(GB)" "KV/req(MB)" "Notes"
echo "────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"

for test in "${TESTS[@]}"; do
  IFS='|' read -r model expected_category expect_kv_zero <<< "$test"

  # Call API with timeout
  response=$(curl -s --max-time 30 -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "'"$model"'",
      "precision": "FP16",
      "gpu_type": "h200-141gb",
      "concurrent_users": 97,
      "isl": 1000,
      "osl": 150,
      "workload_type": "chat",
      "sla_priority": "ttft",
      "hf_token": "'"$HF_TOKEN"'"
    }' 2>&1)

  # Check if request succeeded
  if ! echo "$response" | jq -e '.success' > /dev/null 2>&1; then
    # Request failed
    error_msg=$(echo "$response" | jq -r '.message // .error // "Unknown error"' 2>/dev/null || echo "API timeout or network error")
    printf "${RED}%-6s${NC} %-8s %-45s %-12s %-14s %-12s %s\n" "✗" "$expected_category" "$model" "ERROR" "-" "-" "$error_msg"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    sleep 2
    continue
  fi

  # Parse response fields
  weight_gb=$(echo "$response" | jq -r '.data.memory_analysis.weight_gb // 0')
  kv_cache_gb=$(echo "$response" | jq -r '.data.memory_analysis.kv_cache_used_gb // 0')
  actual_category=$(echo "$response" | jq -r '.data.memory_analysis.kv_category // "UNKNOWN"')
  warnings=$(echo "$response" | jq -r '.data.warnings // [] | join("; ")')

  # Calculate KV per request in MB
  kv_per_req_mb=$(echo "$kv_cache_gb * 1024 / 97" | bc -l | xargs printf "%.1f")

  # Check if using estimation
  is_estimated=""
  if echo "$warnings" | grep -qi "estimated"; then
    is_estimated="⚠️ estimated"
  fi

  # Validate category
  category_pass=false
  if [ "$actual_category" = "$expected_category" ]; then
    category_pass=true
  fi

  # Validate KV cache (zero vs non-zero)
  kv_zero_pass=false
  if [ "$expect_kv_zero" = "true" ]; then
    # Should be zero
    if [ "$(echo "$kv_cache_gb == 0" | bc -l)" -eq 1 ]; then
      kv_zero_pass=true
    fi
  else
    # Should be non-zero
    if [ "$(echo "$kv_cache_gb > 0" | bc -l)" -eq 1 ]; then
      kv_zero_pass=true
    fi
  fi

  # Overall pass/fail
  overall_pass=false
  if $category_pass && $kv_zero_pass; then
    overall_pass=true
  fi

  # Build notes
  notes=""
  if ! $category_pass; then
    notes="${notes}got $actual_category; "
  fi
  if ! $kv_zero_pass; then
    if [ "$expect_kv_zero" = "true" ]; then
      notes="${notes}expected KV=0; "
    else
      notes="${notes}expected KV>0; "
    fi
  fi
  if [ -n "$is_estimated" ]; then
    notes="${notes}${is_estimated}; "
  fi
  notes=$(echo "$notes" | sed 's/; $//')

  # Format output
  weight_str=$(printf "%.1f" "$weight_gb")
  kv_cache_str=$(printf "%.1f" "$kv_cache_gb")

  # Determine status symbol and color
  if $overall_pass; then
    if [ -n "$is_estimated" ]; then
      status="${YELLOW}⚠${NC}"
      notes="$is_estimated"
      WARN_COUNT=$((WARN_COUNT + 1))
    else
      status="${GREEN}✓${NC}"
      PASS_COUNT=$((PASS_COUNT + 1))
    fi
  else
    status="${RED}✗${NC}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  printf "%-6s %-8s %-45s %-12s %-14s %-12s %s\n" \
    "$status" \
    "$expected_category" \
    "$model" \
    "${weight_str}GB" \
    "${kv_cache_str}GB" \
    "${kv_per_req_mb}MB" \
    "$notes"

  # Rate limit
  sleep 2
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "SUMMARY:"
echo "  ${GREEN}✓ PASS:${NC}  $PASS_COUNT"
echo "  ${YELLOW}⚠ WARN:${NC}  $WARN_COUNT (estimated architecture)"
echo "  ${RED}✗ FAIL:${NC}  $FAIL_COUNT"
echo ""
echo "Total tests: ${#TESTS[@]}"

if [ $FAIL_COUNT -eq 0 ]; then
  echo ""
  echo "${GREEN}All tests passed!${NC}"
  exit 0
else
  echo ""
  echo "${RED}Some tests failed${NC}"
  exit 1
fi
