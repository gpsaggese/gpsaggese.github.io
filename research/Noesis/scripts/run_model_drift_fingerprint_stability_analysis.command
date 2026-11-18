#!/bin/zsh

set -e

cd /Users/javinahuja/Documents/ChatGPT/Startup/noesis-qos-phase0
export NOESIS_MODEL_EQUALITY_REPO=/Users/javinahuja/Documents/ChatGPT/Startup/model-equality-testing
export MPLCONFIGDIR=/tmp/noesis-matplotlib
export PYTHONPATH=.:/Users/javinahuja/dev/gpsaggese.github.io/helpers_root

# Ten tests run strictly serially. A parent watchdog terminates the active
# worker if its resident memory exceeds 2 GiB, and each completed test is
# durably checkpointed before the next one starts.
python3 research/Noesis/scripts/analyze_model_drift_fingerprint_stability.py \
  --observations_path research/Noesis/docs/findings/model_drift_live_fingerprint/observations.jsonl \
  --observations_path research/Noesis/docs/findings/model_drift_live_fingerprint/topup-01/observations.jsonl \
  --observations_path research/Noesis/docs/findings/model_drift_fingerprint_stability_live/observations.jsonl \
  --dst_dir research/Noesis/docs/findings/model_drift_fingerprint_stability_live \
  --memory_limit_mb 2048

echo ""
echo "Stability analysis completed. You may close this window."
