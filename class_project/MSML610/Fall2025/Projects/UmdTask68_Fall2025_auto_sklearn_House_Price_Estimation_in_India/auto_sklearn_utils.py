"""
auto sklearn utils wrapper - main module

provides a unified interface to all utility functions
import from this module for backwards compatibility
"""

# exporting all public functions and classes
from utils_transformers import AmenitiesEncoder
from utils_data_io import load_housing_data
from utils_feature_engineering import get_column_groups, encode_binary_columns
from utils_preprocessing import create_preprocessor, prepare_data

__all__ = [
    "AmenitiesEncoder",
    "load_housing_data",
    "get_column_groups",
    "encode_binary_columns",
    "create_preprocessor",
    "prepare_data",
]