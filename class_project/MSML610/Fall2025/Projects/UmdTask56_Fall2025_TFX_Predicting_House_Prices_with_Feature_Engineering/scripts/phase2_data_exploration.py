"""
Phase 2: Data Exploration Script

This script performs comprehensive data exploration and generates reports
for the house price dataset without running the full TFX pipeline.

Usage:
    python scripts/phase2_data_exploration.py
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from utils import config
from utils import data_utils


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run Phase 2 data exploration."""

    print_section("Phase 2: Data Ingestion & Validation - Data Exploration")

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print_section("1. Loading Data")

    train_df = data_utils.load_data("train")
    test_df = data_utils.load_data("test")

    print(f"[OK] Train data loaded: {train_df.shape}")
    print(f"[OK] Test data loaded: {test_df.shape}")

    # ========================================================================
    # 2. Basic Information
    # ========================================================================
    print_section("2. Basic Dataset Information")

    print(f"Training Set:")
    print(f"  - Rows: {len(train_df):,}")
    print(f"  - Columns: {len(train_df.columns)}")
    print(f"  - Memory Usage: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\nTest Set:")
    print(f"  - Rows: {len(test_df):,}")
    print(f"  - Columns: {len(test_df.columns)}")

    # ========================================================================
    # 3. Feature Types
    # ========================================================================
    print_section("3. Feature Type Analysis")

    numerical, categorical = data_utils.get_feature_types(train_df)
    ordinal = data_utils.identify_ordinal_features(train_df)

    print(f"Numerical features: {len(numerical)}")
    print(f"Categorical features: {len(categorical)}")
    print(f"Ordinal features (from config): {len(ordinal)}")
    print(f"\nOrdinal features: {', '.join(ordinal[:10])}" +
          (f" ... (+{len(ordinal)-10} more)" if len(ordinal) > 10 else ""))

    # ========================================================================
    # 4. Missing Values Analysis
    # ========================================================================
    print_section("4. Missing Values Analysis")

    missing_df = data_utils.analyze_missing_values(train_df)

    print(f"Total features with missing values: {len(missing_df)}")
    print(f"\nTop 10 features with highest missing percentage:")
    print(missing_df.head(10).to_string(index=False))

    # Identify features with >50% missing
    high_missing = missing_df[missing_df['missing_pct'] > 50]
    if len(high_missing) > 0:
        print(f"\n[WARNING] {len(high_missing)} features have >50% missing values:")
        print(high_missing[['column', 'missing_pct']].to_string(index=False))

    # ========================================================================
    # 5. Target Variable Analysis
    # ========================================================================
    print_section("5. Target Variable Analysis (SalePrice)")

    target_stats = data_utils.analyze_target_variable(train_df)

    print(f"Count: {target_stats['count']:,}")
    print(f"Mean: ${target_stats['mean']:,.2f}")
    print(f"Median: ${target_stats['median']:,.2f}")
    print(f"Std Dev: ${target_stats['std']:,.2f}")
    print(f"Min: ${target_stats['min']:,.2f}")
    print(f"Max: ${target_stats['max']:,.2f}")
    print(f"25th Percentile: ${target_stats['q25']:,.2f}")
    print(f"75th Percentile: ${target_stats['q75']:,.2f}")
    print(f"\nSkewness: {target_stats['skewness']:.4f}")
    print(f"Kurtosis: {target_stats['kurtosis']:.4f}")

    if target_stats['skewness'] > 1:
        print("\n[INSIGHT] SalePrice is highly right-skewed (>1).")
        print("           Recommendation: Apply log transformation in Phase 3")

    # ========================================================================
    # 6. Data Validation
    # ========================================================================
    print_section("6. Data Schema Validation")

    train_validation = data_utils.validate_data_schema(train_df, "train")
    test_validation = data_utils.validate_data_schema(test_df, "test")

    print(f"Train data validation: {'[PASSED]' if train_validation['is_valid'] else '[FAILED]'}")
    if not train_validation['is_valid']:
        print(f"  Issues:")
        for issue in train_validation['issues']:
            print(f"    - {issue}")

    print(f"\nTest data validation: {'[PASSED]' if test_validation['is_valid'] else '[FAILED]'}")
    if not test_validation['is_valid']:
        print(f"  Issues:")
        for issue in test_validation['issues']:
            print(f"    - {issue}")

    # ========================================================================
    # 7. Numerical Features Summary
    # ========================================================================
    print_section("7. Numerical Features Summary (Top 10)")

    numerical_summary = data_utils.get_numerical_summary(train_df, numerical)
    print(numerical_summary.head(10).to_string(index=False))

    # ========================================================================
    # 8. Categorical Features Summary
    # ========================================================================
    print_section("8. Categorical Features Summary (Sample)")

    categorical_summary = data_utils.get_categorical_summary(train_df, categorical[:5])

    for feature, stats in list(categorical_summary.items())[:5]:
        print(f"\n{feature}:")
        print(f"  Unique values: {stats['unique_count']}")
        print(f"  Missing: {stats['missing_count']} ({stats['missing_pct']:.1f}%)")
        print(f"  Top values:")
        for value, count in list(stats['top_values'].items())[:3]:
            print(f"    - {value}: {count}")

    # ========================================================================
    # 9. Generate Comprehensive Reports
    # ========================================================================
    print_section("9. Generating Comprehensive Data Reports")

    print("Generating train data report...")
    train_report = data_utils.generate_data_report("train")

    print("Generating test data report...")
    test_report = data_utils.generate_data_report("test")

    # Save reports to JSON
    reports_dir = Path(__file__).parent.parent / "pipeline_outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_report_path = reports_dir / "train_data_report.json"
    test_report_path = reports_dir / "test_data_report.json"

    with open(train_report_path, 'w') as f:
        json.dump(train_report, f, indent=2, default=str)

    with open(test_report_path, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)

    print(f"\n[OK] Train report saved to: {train_report_path}")
    print(f"[OK] Test report saved to: {test_report_path}")

    # ========================================================================
    # 10. Key Insights Summary
    # ========================================================================
    print_section("Key Insights from Data Exploration")

    print("Dataset Characteristics:")
    print(f"  - {len(train_df):,} training samples, {len(test_df):,} test samples")
    print(f"  - {len(numerical)} numerical features, {len(categorical)} categorical features")
    print(f"  - {len(missing_df)} features have missing values")
    print(f"  - Target (SalePrice) is right-skewed (skewness={target_stats['skewness']:.2f})")

    print("\nData Quality:")
    if train_validation['is_valid'] and test_validation['is_valid']:
        print("  - [OK] Both train and test data passed validation")
    else:
        print("  - [WARNING] Some validation issues detected")

    print("\nNext Steps for Phase 3:")
    print("  1. Handle missing values (especially PoolQC, Fence, Alley with >50% missing)")
    print("  2. Apply log transformation to SalePrice (highly skewed)")
    print("  3. Encode ordinal features with proper ordering")
    print("  4. One-hot encode nominal categorical features")
    print("  5. Scale numerical features")
    print("  6. Create derived features (TotalSF, Age, etc.)")

    print_section("Phase 2 Data Exploration Complete")

    print("\nTo run the full TFX pipeline (requires Docker):")
    print("  python scripts/api.py")


if __name__ == "__main__":
    main()
