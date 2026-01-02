#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for the backend (expects backend reachable on localhost:8000).
# Run on your HOST machine (not in Docker):
#   ./scripts/e2e_smoke_test.sh

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "Checking $BASE_URL/health"
curl -sS "$BASE_URL/health" | cat
echo ""

echo "Checking $BASE_URL/feature_names"
curl -sS "$BASE_URL/feature_names" >/dev/null
echo "OK"

echo "Running /predict (AAPL, horizon=7)"
curl -sS -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","lookback_days":365,"horizon_days":7,"investment_usd":1000}' | head -c 400
echo ""

echo "Smoke test passed."


