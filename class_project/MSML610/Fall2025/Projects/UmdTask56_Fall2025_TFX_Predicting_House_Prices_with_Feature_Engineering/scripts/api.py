"""
TFX Pipeline Runner Script

This script creates and runs the house price prediction TFX pipeline.

Usage:
    python scripts/api.py [--pipeline-name NAME] [--pipeline-root PATH]

Example:
    python scripts/api.py --pipeline-name house_price_pipeline
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from utils import config
from pipelines.house_price_pipeline import create_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the House Price Prediction TFX Pipeline"
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

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(config.DATA_DIR),
        help="Path to the data directory"
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default=config.METADATA_PATH,
        help="Path to metadata database"
    )

    parser.add_argument(
        "--serving-model-dir",
        type=str,
        default=config.SERVING_MODEL_DIR_STR,
        help="Directory for serving models"
    )

    return parser.parse_args()


def run_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    metadata_path: str,
    serving_model_dir: str
):
    """
    Create and run the TFX pipeline.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_root: Root directory for pipeline outputs
        data_path: Path to data directory
        metadata_path: Path to metadata database
        serving_model_dir: Directory for serving models
    """

    print("=" * 80)
    print("House Price Prediction TFX Pipeline")
    print("=" * 80)
    print(f"Pipeline Name: {pipeline_name}")
    print(f"Pipeline Root: {pipeline_root}")
    print(f"Data Path: {data_path}")
    print(f"Metadata Path: {metadata_path}")
    print(f"Serving Model Dir: {serving_model_dir}")
    print("=" * 80)

    # Create necessary directories
    os.makedirs(pipeline_root, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    os.makedirs(serving_model_dir, exist_ok=True)

    # Create a directory for CsvExampleGen with only train.csv
    # CsvExampleGen expects a directory with CSV files, all with the same schema
    train_data_dir = os.path.join(pipeline_root, "csv_data")
    os.makedirs(train_data_dir, exist_ok=True)

    # Copy train.csv to the CsvExampleGen input directory
    import shutil
    train_csv_source = os.path.join(data_path, "train.csv")
    train_csv_dest = os.path.join(train_data_dir, "train.csv")
    if os.path.exists(train_csv_source):
        shutil.copy(train_csv_source, train_csv_dest)
        print(f"Copied train.csv to {train_data_dir}")

    # Update data_path to point to the train-only directory
    data_path = train_data_dir

    # Create module file paths (for future phases)
    transform_module_file = str(Path(__file__).parent.parent / "utils" / "feature_engineering.py")
    trainer_module_file = str(Path(__file__).parent.parent / "utils" / "model_utils.py")

    print("\nCreating TFX pipeline...")
    print(f"Components: CsvExampleGen, StatisticsGen, SchemaGen, Transform, Trainer, Evaluator, Pusher (Phase 6)")

    # Create the pipeline
    tfx_pipeline = create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_path=data_path,
        transform_module_file=transform_module_file,
        trainer_module_file=trainer_module_file,
        metadata_path=metadata_path,
        serving_model_dir=serving_model_dir
    )

    print("\nRunning pipeline with LocalDagRunner...")
    print("This may take a few minutes...\n")

    # Run the pipeline
    LocalDagRunner().run(tfx_pipeline)

    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("=" * 80)
    print(f"\nOutputs saved to: {pipeline_root}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nNext steps:")
    print("  - Check pipeline_outputs/ for generated artifacts")
    print("  - Review trained model in pipeline_outputs/.../Trainer/...")
    print("  - Review evaluation metrics in pipeline_outputs/.../Evaluator/...")
    print("  - Check models/serving/ for the deployed model")
    print("  - Test model serving with sample predictions")
    print("  - All phases complete!")


def main():
    """Main function."""
    args = parse_args()

    run_pipeline(
        pipeline_name=args.pipeline_name,
        pipeline_root=args.pipeline_root,
        data_path=args.data_path,
        metadata_path=args.metadata_path,
        serving_model_dir=args.serving_model_dir
    )


if __name__ == "__main__":
    main()
