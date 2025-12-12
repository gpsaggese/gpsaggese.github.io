#!/usr/bin/env bash
set -euo pipefail

# Phase 5: run training with W&B ONLINE logging.
# Run this inside the professor Docker container (with /venv activated).

export WANDB_MODE="${WANDB_MODE:-online}"

# These defaults match config/wandb.yaml. You can override by exporting before running.
export WANDB_ENTITY="${WANDB_ENTITY:-othakur-university-of-maryland-org}"
export WANDB_PROJECT="${WANDB_PROJECT:-time_seires_forecasting}"

echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"

python -m src.pipeline.training_pipeline


