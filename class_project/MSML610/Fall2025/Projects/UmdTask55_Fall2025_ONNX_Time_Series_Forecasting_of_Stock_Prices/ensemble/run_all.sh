#!/bin/bash

# MAG 7 Ensemble Training Pipeline
# This script runs all ensemble scripts sequentially

set -e  # Exit on any error

echo "=========================================="
echo "MAG 7 Ensemble Training Pipeline"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a script and measure time
run_script() {
    local script=$1
    local description=$2

    echo -e "${BLUE}=========================================="
    echo "Step $script: $description"
    echo -e "==========================================${NC}"

    start_time=$(date +%s)

    if uv run "ensemble/$script"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo -e "${GREEN} $script completed successfully in ${elapsed}s${NC}"
        echo ""
    else
        echo -e "${RED} $script failed!${NC}"
        exit 1
    fi
}

# Step 1: Data Preprocessing
run_script "01_preprocess_data.py" "Data Preprocessing"

# Step 2: LSTM Training
run_script "02_train_lstm.py" "LSTM Training & ONNX Conversion"

# Step 3: TCN Training
run_script "03_train_tcn.py" "TCN Training & ONNX Conversion"

# Step 4: XGBoost Training
run_script "04_train_xgboost.py" "XGBoost Training & ONNX Conversion"

# Step 5: Ensemble Analysis
run_script "05_ensemble_analysis.py" "Ensemble Creation & Analysis"

echo -e "${GREEN}=========================================="
echo " PIPELINE COMPLETE!"
echo "==========================================${NC}"
echo ""
echo "All models trained, evaluated, and analyzed!"
echo "Results saved in: ./models/"
echo ""
