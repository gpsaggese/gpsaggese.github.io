"""
End-to-End Example Script

This script demonstrates the complete house price prediction workflow
from data loading to model evaluation and predictions.

Usage:
    python scripts/example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils import config
from utils import data_utils

# Import other modules only if needed (they have heavy dependencies)
# from utils import feature_engineering
# from utils import model_utils
# from utils import evaluation_utils


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run the complete example workflow."""

    print_section("House Price Prediction - End-to-End Example")

    # ========================================================================
    # PHASE 1: Project Setup ✓
    # ========================================================================
    print_section("Phase 1: Project Foundation")
    print("[OK] Project structure created")
    print("[OK] Configuration loaded")
    print(f"[OK] Pipeline: {config.PIPELINE_NAME}")
    print(f"[OK] Data directory: {config.DATA_DIR}")
    print(f"[OK] Target column: {config.TARGET_COLUMN}")

    # ========================================================================
    # PHASE 2: Data Ingestion & Validation
    # ========================================================================
    print_section("Phase 2: Data Ingestion & Validation")
    print("TODO: Implement in Phase 2")
    print("- Load train.csv and test.csv")
    print("- Explore data statistics")
    print("- Implement CsvExampleGen")
    print("- Generate and validate schema")

    # Test data loading (simple version)
    try:
        train_df = data_utils.load_data("train")
        test_df = data_utils.load_data("test")
        print(f"\n[OK] Train data loaded: {train_df.shape}")
        print(f"[OK] Test data loaded: {test_df.shape}")

        # Basic exploration
        exploration = data_utils.explore_data(train_df)
        print(f"\n[OK] Dataset has {exploration['shape'][0]} rows and {exploration['shape'][1]} columns")

        missing_cols = [k for k, v in exploration['missing_values'].items() if v > 0]
        print(f"[OK] Found {len(missing_cols)} columns with missing values")

    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")

    # ========================================================================
    # PHASE 3: Feature Engineering
    # ========================================================================
    print_section("Phase 3: Feature Engineering & Transformation")
    print("TODO: Implement in Phase 3")
    print("- Handle missing values")
    print("- Scale numerical features")
    print("- Encode categorical features")
    print("- Create interaction terms")
    print("- Implement Transform component")

    # ========================================================================
    # PHASE 4: Model Training
    # ========================================================================
    print_section("Phase 4: Model Training")
    print("TODO: Implement in Phase 4")
    print("- Train XGBoost model")
    print("- Train TensorFlow DNN model")
    print("- Compare training metrics")

    # ========================================================================
    # PHASE 5: Model Evaluation
    # ========================================================================
    print_section("Phase 5: Model Evaluation")
    print("TODO: Implement in Phase 5")
    print("- Calculate RMSE and R² scores")
    print("- Perform cross-validation")
    print("- Compare XGBoost vs TensorFlow DNN")
    print("- Select best model")

    # ========================================================================
    # PHASE 6: Model Deployment & Predictions
    # ========================================================================
    print_section("Phase 6: Model Deployment & Predictions")
    print("TODO: Implement in Phase 6")
    print("- Deploy best model")
    print("- Generate predictions on test.csv")
    print("- Save predictions")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")
    print("Phase 1: [COMPLETE] Project foundation established")
    print("Phase 2: [PENDING] Data ingestion & validation")
    print("Phase 3: [PENDING] Feature engineering")
    print("Phase 4: [PENDING] Model training")
    print("Phase 5: [PENDING] Model evaluation")
    print("Phase 6: [PENDING] Model deployment")
    print("\nNext step: Proceed to Phase 2!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
