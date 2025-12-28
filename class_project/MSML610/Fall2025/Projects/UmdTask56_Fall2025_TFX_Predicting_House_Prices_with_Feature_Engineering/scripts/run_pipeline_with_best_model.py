"""
Run TFX Pipeline with Best Model from Comparison

This script:
1. Runs model comparison to identify the best model
2. Updates the TFX pipeline to use that model
3. Runs the full pipeline with deployment

Usage:
    python scripts/run_pipeline_with_best_model.py [--cv-folds 5] [--force-comparison]

Examples:
    # Run comparison and pipeline with best model
    python scripts/run_pipeline_with_best_model.py

    # Force new comparison even if results exist
    python scripts/run_pipeline_with_best_model.py --force-comparison

    # Use specific CV folds
    python scripts/run_pipeline_with_best_model.py --cv-folds 10
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import trainer_pb2

from utils import config
from pipelines.house_price_pipeline import create_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TFX Pipeline with Best Model from Comparison"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for model comparison (default: 5)"
    )

    parser.add_argument(
        "--force-comparison",
        action="store_true",
        help="Force new model comparison even if results exist"
    )

    parser.add_argument(
        "--use-tensorflow-dnn",
        action="store_true",
        help="Use TensorFlow DNN instead of best sklearn model"
    )

    parser.add_argument(
        "--pipeline-name",
        type=str,
        default=config.PIPELINE_NAME,
        help="Name of the pipeline"
    )

    parser.add_argument(
        "--pipeline-root",
        type=str,
        default=config.PIPELINE_ROOT_STR,
        help="Root directory for pipeline outputs"
    )

    return parser.parse_args()


def run_model_comparison(cv_folds=5, force=False):
    """
    Run model comparison to identify best model.

    Args:
        cv_folds: Number of cross-validation folds
        force: Force new comparison even if results exist

    Returns:
        Dictionary with comparison results
    """
    comparison_dir = config.MODELS_DIR / "comparison"
    results_file = comparison_dir / "comparison_results.json"

    # Check if results already exist
    if results_file.exists() and not force:
        print(f"\nUsing existing comparison results from {results_file}")
        print("(Use --force-comparison to run a new comparison)")

        with open(results_file, 'r') as f:
            results = json.load(f)

        return results

    # Run new comparison
    print("\n" + "=" * 80)
    print("RUNNING MODEL COMPARISON")
    print("=" * 80)

    from utils.model_comparison import compare_all_models
    from utils.data_utils import load_data
    import pandas as pd
    import numpy as np

    # Load data
    print(f"\nLoading data from {config.TRAIN_DATA_PATH}...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    y = data[config.TARGET_COLUMN].values
    X = data.drop(columns=[config.TARGET_COLUMN])

    # Apply log transformation
    if config.LOG_TRANSFORM_TARGET:
        y = np.log1p(y)

    # Basic preprocessing
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col].fillna('Missing', inplace=True)

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = X.values

    # Compare models
    comparison_results = compare_all_models(
        X_train=X,
        y_train=y,
        use_cv=True,
        cv_folds=cv_folds
    )

    # Save results
    os.makedirs(comparison_dir, exist_ok=True)

    json_results = {
        'comparison': comparison_results['comparison'],
        'models': {}
    }

    for model_name, model_result in comparison_results['results'].items():
        json_result = {k: v for k, v in model_result.items() if k != 'model'}
        json_results['models'][model_name] = json_result

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return json_results


def get_trainer_module_file(model_name, use_tensorflow_dnn=False):
    """
    Get the appropriate trainer module file based on model type.

    Args:
        model_name: Name of the best model
        use_tensorflow_dnn: Force use of TensorFlow DNN

    Returns:
        Path to trainer module file
    """
    if use_tensorflow_dnn or model_name == "TF_DNN":
        # Use TensorFlow DNN trainer
        return str(Path(__file__).parent.parent / "utils" / "model_utils.py")
    else:
        # Use sklearn trainer
        return str(Path(__file__).parent.parent / "utils" / "sklearn_trainer.py")


def main():
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("TFX PIPELINE WITH BEST MODEL")
    print("=" * 80)

    # Step 1: Run model comparison
    if not args.use_tensorflow_dnn:
        comparison_results = run_model_comparison(
            cv_folds=args.cv_folds,
            force=args.force_comparison
        )

        best_model_name = comparison_results['comparison']['best_model']
        best_rmse = comparison_results['comparison']['summary']['best_rmse']

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(f"\nBest Model: {best_model_name}")
        print(f"CV RMSE (log scale): {best_rmse:.4f}")

        # Show rankings
        print("\nModel Rankings (by RMSE):")
        for i, model_name in enumerate(comparison_results['comparison']['rankings']['by_rmse'], 1):
            model_rmse = comparison_results['models'][model_name].get('rmse', float('inf'))
            print(f"  {i}. {model_name:20s} - RMSE: {model_rmse:.4f}")
    else:
        best_model_name = "TF_DNN"
        print("\nUsing TensorFlow DNN (--use-tensorflow-dnn flag set)")

    # Step 2: Set up TFX pipeline with best model
    print("\n" + "=" * 80)
    print("SETTING UP TFX PIPELINE")
    print("=" * 80)

    # Create necessary directories
    os.makedirs(args.pipeline_root, exist_ok=True)
    metadata_path = config.METADATA_PATH
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    os.makedirs(config.SERVING_MODEL_DIR, exist_ok=True)

    # Prepare data directory for CsvExampleGen
    train_data_dir = os.path.join(args.pipeline_root, "csv_data")
    os.makedirs(train_data_dir, exist_ok=True)

    import shutil
    train_csv_source = config.TRAIN_DATA_PATH
    train_csv_dest = os.path.join(train_data_dir, "train.csv")
    if os.path.exists(train_csv_source):
        shutil.copy(train_csv_source, train_csv_dest)
        print(f"Copied train.csv to {train_data_dir}")

    # Module file paths
    transform_module_file = str(Path(__file__).parent.parent / "utils" / "feature_engineering.py")
    trainer_module_file = get_trainer_module_file(best_model_name, args.use_tensorflow_dnn)

    print(f"\nPipeline Configuration:")
    print(f"  Model: {best_model_name}")
    print(f"  Trainer Module: {Path(trainer_module_file).name}")
    print(f"  Transform Module: {Path(transform_module_file).name}")
    print(f"  Pipeline Root: {args.pipeline_root}")
    print(f"  Metadata Path: {metadata_path}")
    print(f"  Serving Model Dir: {config.SERVING_MODEL_DIR}")

    # Step 3: Create and run pipeline
    print("\n" + "=" * 80)
    print("CREATING TFX PIPELINE")
    print("=" * 80)

    tfx_pipeline = create_pipeline(
        pipeline_name=args.pipeline_name,
        pipeline_root=args.pipeline_root,
        data_path=train_data_dir,
        transform_module_file=transform_module_file,
        trainer_module_file=trainer_module_file,
        metadata_path=metadata_path,
        serving_model_dir=str(config.SERVING_MODEL_DIR),
        trainer_custom_config={'model_name': best_model_name} if not args.use_tensorflow_dnn else None
    )

    print("\n" + "=" * 80)
    print("RUNNING PIPELINE")
    print("=" * 80)
    print(f"\nModel: {best_model_name}")
    print("This may take several minutes...\n")

    # Run the pipeline
    LocalDagRunner().run(tfx_pipeline)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest Model Deployed: {best_model_name}")
    print(f"Model Location: {config.SERVING_MODEL_DIR}")
    print(f"Pipeline Outputs: {args.pipeline_root}")
    print(f"Metadata: {metadata_path}")

    print("\nNext Steps:")
    print("  1. Check serving model in models/serving/")
    print("  2. Test predictions with the deployed model")
    print("  3. Review evaluation metrics in pipeline_outputs/")
    print(f"  4. Compare with baseline TensorFlow DNN model")

    return 0


if __name__ == "__main__":
    sys.exit(main())
