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

    # TODO: Implement pipeline creation in phases 2-6
    # For now, this is a placeholder

    print("\nPipeline creation placeholder - will be implemented in Phase 2")
    print("Once implemented, the pipeline will run with LocalDagRunner")

    # When ready, uncomment and implement:
    # transform_module_file = str(Path(__file__).parent.parent / "utils" / "feature_engineering.py")
    # trainer_module_file = str(Path(__file__).parent.parent / "utils" / "model_utils.py")
    #
    # pipeline = create_pipeline(
    #     pipeline_name=pipeline_name,
    #     pipeline_root=pipeline_root,
    #     data_path=data_path,
    #     transform_module_file=transform_module_file,
    #     trainer_module_file=trainer_module_file,
    #     metadata_path=metadata_path,
    #     serving_model_dir=serving_model_dir
    # )
    #
    # LocalDagRunner().run(pipeline)
    #
    # print("\n" + "=" * 80)
    # print("Pipeline execution completed successfully!")
    # print("=" * 80)


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
