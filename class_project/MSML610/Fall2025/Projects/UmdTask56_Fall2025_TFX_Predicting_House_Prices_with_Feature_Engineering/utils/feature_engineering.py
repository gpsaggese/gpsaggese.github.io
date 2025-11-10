"""
Feature engineering utilities for the house price prediction pipeline

This module contains:
- Feature transformation functions
- TFX Transform preprocessing_fn
- Feature creation and interaction terms
"""

import tensorflow as tf
import tensorflow_transform as tft

from . import config


def preprocessing_fn(inputs):
    """
    TFX Transform preprocessing function.

    This function will be called by the Transform component to preprocess
    both training and serving data consistently.

    Args:
        inputs: Dictionary of input features

    Returns:
        Dictionary of transformed features

    TODO: Implement in Phase 3
    - Handle missing values
    - Scale numerical features
    - Encode categorical features
    - Create interaction terms
    - Create derived features
    """
    outputs = {}

    # Placeholder - will be implemented in Phase 3
    # For now, just pass through inputs
    for key, value in inputs.items():
        outputs[key] = value

    return outputs


def create_derived_features(df):
    """
    Create derived features from raw data.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with additional derived features

    TODO: Implement in Phase 3
    - Age features (YrSold - YearBuilt, etc.)
    - Total square footage combinations
    - Boolean indicators (HasGarage, HasPool, etc.)
    - Quality interaction terms
    """
    # Placeholder - will be implemented in Phase 3
    return df


def test_feature_engineering():
    """Test function to verify feature engineering works."""
    print("Feature engineering placeholder - will be implemented in Phase 3")


if __name__ == "__main__":
    test_feature_engineering()
