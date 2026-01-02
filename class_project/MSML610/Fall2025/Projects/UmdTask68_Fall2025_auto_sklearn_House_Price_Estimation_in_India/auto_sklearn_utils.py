"""
auto sklearn utils wrapper - main module

provides a unified interface to all utility functions
import from this module for backwards compatibility
"""

# NOTE:
# This file is the notebook-facing API surface. Keep notebooks lightweight by
# placing reusable helpers here and re-exporting functionality from utils_*.

# exporting all public functions and classes
from utils_transformers import AmenitiesEncoder
from utils_data_io import load_housing_data
from utils_feature_engineering import get_column_groups, encode_binary_columns
from utils_preprocessing import create_preprocessor, prepare_data

def to_float32(X):
    """
    Convert feature matrices to float32 for auto-sklearn compatibility.

    Supports both NumPy arrays and SciPy sparse matrices.
    """
    import numpy as np
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:  # pragma: no cover
        sp = None

    if sp is not None and sp.issparse(X):
        return X.astype(np.float32)
    return np.asarray(X, dtype=np.float32)

__all__ = [
    "AmenitiesEncoder",
    "load_housing_data",
    "get_column_groups",
    "encode_binary_columns",
    "create_preprocessor",
    "prepare_data",
    "to_float32",
]