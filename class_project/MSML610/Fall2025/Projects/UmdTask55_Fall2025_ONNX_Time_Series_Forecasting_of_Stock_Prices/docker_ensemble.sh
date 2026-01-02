#!/bin/bash
#
# Run the complete ensemble pipeline in the container.
# This script executes all ensemble steps in sequence:
#   1. Data preprocessing
#   2. LSTM training
#   3. TCN training
#   4. XGBoost training
#   5. Ensemble analysis
#
# Usage:
# > ./docker_ensemble.sh
#

set -e

# Parse params.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR"

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/docker_common/utils.sh

REPO_NAME=umd_msml610
IMAGE_NAME=onnx_timeseries_forecasting
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

print_docker_vars

# Run the script.
DOCKER_RUN_OPTS=""
if [[ $TARGET_DIR != "" ]]; then
    DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS -v $TARGET_DIR:/data"
fi

# Create command to run all ensemble scripts in sequence
CMD="cd /data && \
export PYTHONPATH=/data:\$PYTHONPATH && \
echo '=== Step 1/5: Preprocessing data ===' && \
python3 ensemble/01_preprocess_data.py && \
echo '=== Step 2/5: Training LSTM model ===' && \
python3 ensemble/02_train_lstm.py && \
echo '=== Step 3/5: Training TCN model ===' && \
python3 ensemble/03_train_tcn.py && \
echo '=== Step 4/5: Training XGBoost model ===' && \
python3 ensemble/04_train_xgboost.py && \
echo '=== Step 5/5: Running ensemble analysis ===' && \
python3 ensemble/05_ensemble_analysis.py && \
echo '=== Ensemble pipeline completed successfully ==="

# Run container
echo "Starting ensemble pipeline..."
echo "This will run all 5 steps in sequence:"
echo "  1. Data preprocessing"
echo "  2. LSTM training"
echo "  3. TCN training"
echo "  4. XGBoost training"
echo "  5. Ensemble analysis"
echo ""

docker run \
    --rm -ti \
    $DOCKER_RUN_OPTS \
    $FULL_IMAGE_NAME \
    bash -c "$CMD"
