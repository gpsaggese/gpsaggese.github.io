#!/usr/bin/env bash
set -euo pipefail

# Phase 5: run training with W&B ONLINE logging.
# Run this inside the professor Docker container (with /venv activated).

export WANDB_MODE="${WANDB_MODE:-online}"

# IMPORTANT (W&B entity):
# Your W&B org is "University of Maryland", but you must log runs to a TEAM entity.
# Your team entity (per your dashboard) is:
#   othakur-university-of-maryland
#
# You can still override this by exporting WANDB_ENTITY before running.
export WANDB_ENTITY="${WANDB_ENTITY:-othakur-university-of-maryland}"
export WANDB_PROJECT="${WANDB_PROJECT:-time_seires_forecasting}"

echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"

python -m src.pipeline.training_pipeline


