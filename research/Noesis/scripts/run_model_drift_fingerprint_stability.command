#!/bin/zsh

set -e

cd /Users/javinahuja/Documents/ChatGPT/Startup/noesis-qos-phase0
export PYTHONPATH=.:/Users/javinahuja/dev/gpsaggese.github.io/helpers_root

# Collection is deliberately sequential: provider latency, not local CPU or
# memory, is the bottleneck. This keeps the run gentle on a 16 GB laptop.
python3 research/Noesis/scripts/run_model_drift_fingerprint_stability.py \
  --execute \
  --approved_cap_usd 0.50 \
  --prompt_for_key

echo ""
echo "Stability collection completed. Keep this window open for the result."
