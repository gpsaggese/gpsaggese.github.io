"""
Utils Module for House Price Prediction TFX Pipeline

This module contains helper functions and utilities for:
- Data processing and validation
- Feature engineering and transformation
- Model training and evaluation
- Pipeline configuration and orchestration
"""

__version__ = "1.0.0"
__author__ = "MSML610 Fall 2025 Project"

# Import submodules for easier access
from . import config
from . import data_utils
# Commenting out feature_engineering and model_utils to avoid tensorflow_transform import at module load time
# from . import feature_engineering
# from . import model_utils
from . import evaluation_utils

# Import optional modules only when explicitly needed
# (avoid circular imports in TFX wheel packaging)
try:
    from . import model_comparison
    from . import sklearn_trainer
    __all__ = [
        "config",
        "data_utils",
        # "feature_engineering",  # Available but not auto-imported (requires tensorflow_transform)
        # "model_utils",  # Available but not auto-imported (requires tensorflow_transform)
        "evaluation_utils",
        "model_comparison",
        "sklearn_trainer",
    ]
except ImportError:
    # sklearn/model_comparison not available in TFX wheel
    __all__ = [
        "config",
        "data_utils",
        # "feature_engineering",  # Available but not auto-imported (requires tensorflow_transform)
        # "model_utils",  # Available but not auto-imported (requires tensorflow_transform)
        "evaluation_utils",
    ]
