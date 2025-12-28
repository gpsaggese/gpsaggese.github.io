"""
Model Comparison Script

This script compares multiple regression models (including ensemble methods)
using cross-validation on the house price prediction dataset.

Usage:
    python scripts/compare_models.py [--use-transformed] [--cv-folds 5] [--models MODEL1 MODEL2 ...]

Examples:
    # Compare all models with 5-fold CV
    python scripts/compare_models.py

    # Compare specific models with 10-fold CV
    python scripts/compare_models.py --cv-folds 10 --models XGBoost RandomForest VotingEnsemble

    # Use transformed data from TFX pipeline
    python scripts/compare_models.py --use-transformed
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import config
from utils.model_comparison import compare_all_models, save_model, ModelRegistry
from utils.data_utils import load_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Multiple Regression Models for House Price Prediction"
    )

    parser.add_argument(
        "--use-transformed",
        action="store_true",
        help="Use transformed data from TFX pipeline instead of raw CSV"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to compare (default: all models)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.MODELS_DIR / "comparison"),
        help="Directory to save comparison results"
    )

    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save the best performing model"
    )

    return parser.parse_args()


def load_and_prepare_data(use_transformed=False):
    """
    Load and prepare data for model comparison.

    Args:
        use_transformed: Whether to use transformed data from TFX pipeline

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    print("=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)

    if use_transformed:
        print("\nLoading transformed data from TFX pipeline...")
        # TODO: Implement loading from TFX artifacts
        # For now, fall back to raw data
        print("WARNING: Transformed data loading not yet implemented, using raw data")
        use_transformed = False

    if not use_transformed:
        print(f"\nLoading raw data from {config.TRAIN_DATA_PATH}...")
        data = pd.read_csv(config.TRAIN_DATA_PATH)

        print(f"Data shape: {data.shape}")
        print(f"Target column: {config.TARGET_COLUMN}")

        # Separate features and target
        if config.TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in data")

        y = data[config.TARGET_COLUMN].values
        X = data.drop(columns=[config.TARGET_COLUMN])

        # Apply log transformation to target if configured
        if config.LOG_TRANSFORM_TARGET:
            print("Applying log transformation to target variable...")
            y = np.log1p(y)  # log(1 + y) to handle zeros

        # Basic preprocessing for raw data
        print("\nApplying basic preprocessing...")

        # Handle missing values in features
        # For numerical columns: fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        # For categorical columns: fill with 'Missing'
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col].fillna('Missing', inplace=True)

        # Convert categorical to numerical using one-hot encoding
        if len(categorical_cols) > 0:
            print(f"One-hot encoding {len(categorical_cols)} categorical features...")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        print(f"\nFinal feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Convert X to numpy array (y is already numpy array from .values above)
        X = X.values

    return X, y


def main():
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("HOUSE PRICE PREDICTION - MODEL COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  CV Folds: {args.cv_folds}")
    print(f"  Models: {args.models if args.models else 'All available models'}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Save Best Model: {args.save_best_model}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare data
    try:
        X, y = load_and_prepare_data(use_transformed=args.use_transformed)
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return 1

    # Compare models
    print("\n" + "=" * 80)
    print("STARTING MODEL COMPARISON")
    print("=" * 80)

    try:
        comparison_results = compare_all_models(
            X_train=X,
            y_train=y,
            use_cv=True,
            cv_folds=args.cv_folds,
            models_to_compare=args.models
        )

        # Save results to JSON
        results_file = os.path.join(args.output_dir, "comparison_results.json")

        # Convert results to JSON-serializable format
        json_results = {
            'comparison': comparison_results['comparison'],
            'models': {}
        }

        for model_name, model_result in comparison_results['results'].items():
            # Remove non-serializable items
            json_result = {k: v for k, v in model_result.items() if k != 'model'}
            json_results['models'][model_name] = json_result

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Save best model if requested
        if args.save_best_model:
            best_model_name = comparison_results['comparison']['best_model']
            print(f"\nRetraining best model ({best_model_name}) on full dataset...")

            # Get and train best model
            all_models = ModelRegistry.get_all_models()
            best_model = all_models[best_model_name]
            best_model.fit(X, y)

            # Save the model
            model_file = os.path.join(args.output_dir, f"best_model_{best_model_name}.pkl")
            save_model(best_model, model_file)

            # Save model metadata
            metadata = {
                'model_name': best_model_name,
                'metrics': comparison_results['results'][best_model_name],
                'cv_folds': args.cv_folds,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }

            metadata_file = os.path.join(args.output_dir, f"best_model_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Best model saved to: {model_file}")
            print(f"Model metadata saved to: {metadata_file}")

        print("\n" + "=" * 80)
        print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Print final summary
        print(f"\nBest Model: {comparison_results['comparison']['best_model']}")
        best_metrics = comparison_results['results'][comparison_results['comparison']['best_model']]

        if 'cv_mean_rmse' in best_metrics:
            print(f"CV RMSE: {best_metrics['cv_mean_rmse']:.4f} (+/- {best_metrics['cv_std_rmse']:.4f})")
        else:
            print(f"Test RMSE: {best_metrics['test_rmse']:.4f}")

        print(f"R² Score: {best_metrics['r2']:.4f}")

        return 0

    except Exception as e:
        print(f"\nError during model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
