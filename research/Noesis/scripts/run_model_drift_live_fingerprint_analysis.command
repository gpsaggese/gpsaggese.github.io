#!/bin/zsh

set -e

cd /Users/javinahuja/Documents/ChatGPT/Startup/noesis-qos-phase0
export NOESIS_MODEL_EQUALITY_REPO=/Users/javinahuja/Documents/ChatGPT/Startup/model-equality-testing
export MPLCONFIGDIR=/tmp/noesis-matplotlib
export PYTHONPATH=.:/Users/javinahuja/dev/gpsaggese.github.io/helpers_root

# One worker is intentional on this 16 GB Mac. The adapter clears the pinned
# reference package's per-permutation tensor cache, and the durable checkpoint
# resumes only the unfinished trials.
python3 research/Noesis/scripts/analyze_model_drift_live_fingerprint.py \
  --observations_path research/Noesis/docs/findings/model_drift_live_fingerprint/observations.jsonl \
  --observations_path research/Noesis/docs/findings/model_drift_live_fingerprint/topup-01/observations.jsonl \
  --dst_dir research/Noesis/docs/findings/model_drift_live_fingerprint \
  --partial_trials 100 \
  --workers 1

echo ""
echo "Lane 2 statistical analysis completed. You may close this window."
