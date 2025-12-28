"""
TFX Pipeline Utilities - Wrapper Layer

This module provides a simplified, high-level interface for working with
TensorFlow Extended (TFX) pipelines for house price prediction.

The wrapper layer builds on top of existing scripts:
- scripts/api.py - TFX pipeline runner
- scripts/compare_models.py - Model comparison
- scripts/run_pipeline_with_best_model.py - Automated best model deployment
- scripts/visualize_results.py - Results visualization

This provides a clean API for common operations without duplicating code.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
try:
    from utils import config
    from utils.model_comparison import ModelRegistry
    from utils.evaluation_utils import evaluate_model, cross_validate_model
except ImportError:
    import config
    from model_comparison import ModelRegistry
    from evaluation_utils import evaluate_model, cross_validate_model


class TFXPipelineWrapper:
    """
    High-level wrapper for TFX pipeline operations.

    This class provides a simplified interface by wrapping existing scripts:
    - api.py for pipeline execution
    - run_pipeline_with_best_model.py for automated deployment

    Example:
        >>> wrapper = TFXPipelineWrapper()
        >>> wrapper.run_pipeline()
        >>> model_path = wrapper.get_latest_model_path()
    """

    def __init__(
        self,
        pipeline_name: str = 'house_price_prediction_pipeline',
        pipeline_root: str = './pipeline_outputs',
        data_root: str = './data',
        model_dir: str = './models'
    ):
        """Initialize the TFX pipeline wrapper."""
        self.pipeline_name = pipeline_name
        # Convert to absolute paths to avoid path resolution issues
        self.pipeline_root = os.path.abspath(pipeline_root)
        self.data_root = os.path.abspath(data_root)
        self.model_dir = os.path.abspath(model_dir)

    def run_pipeline(self, trainer_module: str = 'utils.sklearn_trainer') -> None:
        """
        Run TFX pipeline using api.py script.

        Args:
            trainer_module: Python module path for trainer
        """
        from scripts import api
        from utils import config

        api.run_pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            data_path=self.data_root,
            metadata_path=config.METADATA_PATH,
            serving_model_dir=str(config.SERVING_MODEL_DIR)
        )

    def get_latest_model_path(self) -> Optional[str]:
        """Get path to latest deployed model."""
        serving_dir = os.path.join(self.model_dir, 'serving')
        if not os.path.exists(serving_dir):
            return None

        versions = [d for d in os.listdir(serving_dir) if d.isdigit()]
        if not versions:
            return None

        latest_version = max(versions, key=int)
        return os.path.join(serving_dir, latest_version)

    def load_model(self, model_path: Optional[str] = None) -> Any:
        """Load trained model (sklearn or TensorFlow)."""
        if model_path is None:
            model_path = self.get_latest_model_path()

        if model_path is None:
            raise ValueError("No model found")

        # Try sklearn pickle first
        sklearn_path = os.path.join(model_path, 'sklearn_model.pkl')
        if os.path.exists(sklearn_path):
            try:
                with open(sklearn_path, 'rb') as f:
                    return pickle.load(f)
            except (TypeError, AttributeError) as e:
                print(f"Warning: Could not load sklearn pickle due to version mismatch: {e}")
                print("This often happens with numpy version differences.")
                print("Please re-run model comparison to generate a fresh model.")
                raise ValueError(
                    "Model pickle incompatible with current numpy version. "
                    "Please delete models/comparison/*.pkl and re-run model comparison."
                ) from e

        # Otherwise TensorFlow SavedModel
        import tensorflow as tf
        return tf.saved_model.load(model_path)


class ModelComparisonWrapper:
    """
    Wrapper for model comparison using existing compare_models.py script.

    Example:
        >>> comparator = ModelComparisonWrapper()
        >>> results = comparator.compare_all_models(cv_folds=5)
        >>> best_model = comparator.get_best_model_name()
    """

    def __init__(self, output_dir: str = './models/comparison'):
        """Initialize model comparison wrapper."""
        self.output_dir = output_dir

    def compare_all_models(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Compare all models using compare_models.py script.

        Args:
            cv_folds: Number of cross-validation folds

        Returns:
            Comparison results dictionary
        """
        import subprocess
        import sys

        # Get project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Run the comparison script as a subprocess with proper arguments
        result = subprocess.run(
            [
                sys.executable,
                'scripts/compare_models.py',
                '--cv-folds', str(cv_folds),
                '--output-dir', 'models/comparison',
                '--save-best-model'
            ],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError("Model comparison failed")

        # Load and return results
        return self.load_results()

    def load_results(self) -> Dict[str, Any]:
        """Load comparison results from JSON."""
        results_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(results_path, 'r') as f:
            return json.load(f)

    def get_best_model_name(self) -> str:
        """Get name of best performing model."""
        results = self.load_results()
        return results['comparison']['best_model']


class DataPipelineWrapper:
    """
    Wrapper for data loading operations.

    Example:
        >>> data = DataPipelineWrapper()
        >>> train_df = data.load_training_data()
    """

    def __init__(self, data_root: str = './data'):
        """Initialize data pipeline wrapper."""
        # Convert to absolute path to work from any directory
        self.data_root = os.path.abspath(data_root)

    def load_training_data(self) -> pd.DataFrame:
        """Load raw training data."""
        return pd.read_csv(os.path.join(self.data_root, 'train.csv'))

    def load_test_data(self) -> pd.DataFrame:
        """Load raw test data."""
        return pd.read_csv(os.path.join(self.data_root, 'test.csv'))


def run_complete_pipeline(cv_folds: int = 5) -> Tuple[Dict, str]:
    """
    Run complete pipeline: compare models, select best, deploy.

    This uses run_pipeline_with_best_model.py script.

    Args:
        cv_folds: Number of CV folds

    Returns:
        (comparison_results, model_path)

    Example:
        >>> results, path = run_complete_pipeline(cv_folds=5)
        >>> print(f"Best: {results['comparison']['best_model']}")
    """
    from scripts.run_pipeline_with_best_model import main as pipeline_main
    import argparse

    args = argparse.Namespace(cv_folds=cv_folds)
    pipeline_main(args)

    # Get results and model path
    wrapper = TFXPipelineWrapper()
    comparison = ModelComparisonWrapper()

    return comparison.load_results(), wrapper.get_latest_model_path()


def visualize_results(output_dir: str = './docs/visualizations') -> None:
    """
    Generate visualizations using visualize_results.py script.

    Args:
        output_dir: Output directory for plots
    """
    # Check if comparison results exist
    results_path = os.path.abspath('models/comparison/comparison_results.json')
    if not os.path.exists(results_path):
        print(f"Warning: Comparison results not found at {results_path}")
        print("Please run model comparison first using ModelComparisonWrapper.compare_all_models()")
        return

    # The script uses hardcoded paths, so we need to run it from project root
    import subprocess
    import sys

    # Get project root (parent of utils directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Run the visualization script
    result = subprocess.run(
        [sys.executable, 'scripts/visualize_results.py'],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error generating visualizations: {result.stderr}")
