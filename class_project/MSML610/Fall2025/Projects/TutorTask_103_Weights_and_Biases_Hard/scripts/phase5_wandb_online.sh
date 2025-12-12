#!/usr/bin/env bash
set -euo pipefail

# Phase 5: run training with W&B ONLINE logging.
# Run this inside the professor Docker container (with /venv activated).

export WANDB_MODE="${WANDB_MODE:-online}"

# Default to personal entity to avoid org/team permission issues.
# If your project lives under a TEAM, set WANDB_ENTITY to that team entity before running.
export WANDB_ENTITY="${WANDB_ENTITY:-othakur}"
export WANDB_PROJECT="${WANDB_PROJECT:-time_seires_forecasting}"

echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"

python -m src.pipeline.training_pipeline


