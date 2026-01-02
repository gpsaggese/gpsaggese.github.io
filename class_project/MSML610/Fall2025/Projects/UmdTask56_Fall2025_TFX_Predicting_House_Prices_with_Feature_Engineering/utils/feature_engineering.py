"""
Feature engineering utilities for the house price prediction pipeline

This module contains:
- Feature transformation functions
- TFX Transform preprocessing_fn
- Feature creation and interaction terms
"""

import tensorflow as tf
import tensorflow_transform as tft


# Define feature name constants
# Note: These are defined directly here to avoid import issues when TFX packages this module
TARGET = 'SalePrice'  # Target column in the dataset
LABEL_KEY = 'SalePrice_log'  # Log-transformed target


def preprocessing_fn(inputs):
    """
    TFX Transform preprocessing function.

    This function will be called by the Transform component to preprocess
    both training and serving data consistently.

    Args:
        inputs: Dictionary of input features (tensors)

    Returns:
        Dictionary of transformed features
    """
    outputs = {}

    # ========================================================================
    # 1. TARGET VARIABLE TRANSFORMATION
    # ========================================================================
    if TARGET in inputs:
        # Apply log transformation to reduce skewness (skewness was 1.88)
        # Add small constant to avoid log(0)
        target_float = tf.cast(inputs[TARGET], tf.float32)
        target_log = tf.math.log(target_float + 1.0)
        outputs[LABEL_KEY] = target_log

    # ========================================================================
    # 2. NUMERICAL FEATURES - Imputation and Scaling
    # ========================================================================
    numerical_features = [
        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
        'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
        'MiscVal', 'MoSold', 'YrSold'
    ]

    for feature in numerical_features:
        if feature in inputs:
            # Only scale if the feature is numeric (int or float)
            # Skip string/categorical features
            if inputs[feature].dtype in (tf.int64, tf.int32, tf.float32, tf.float64):
                # Standardize (z-score normalization)
                # tft.scale_to_z_score handles missing values automatically
                outputs[f'{feature}_scaled'] = tft.scale_to_z_score(inputs[feature])

    # ========================================================================
    # 3. DERIVED NUMERICAL FEATURES
    # ========================================================================
    # Total Square Footage
    if all(f in inputs for f in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        total_sf = inputs['TotalBsmtSF'] + inputs['1stFlrSF'] + inputs['2ndFlrSF']
        outputs['TotalSF_scaled'] = tft.scale_to_z_score(total_sf)

    # Total Bathrooms
    if all(f in inputs for f in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        # Cast to float for arithmetic operations
        total_bath = (tf.cast(inputs['FullBath'], tf.float32) +
                      0.5 * tf.cast(inputs['HalfBath'], tf.float32) +
                      tf.cast(inputs['BsmtFullBath'], tf.float32) +
                      0.5 * tf.cast(inputs['BsmtHalfBath'], tf.float32))
        outputs['TotalBath_scaled'] = tft.scale_to_z_score(total_bath)

    # House Age
    if all(f in inputs for f in ['YrSold', 'YearBuilt']):
        age = inputs['YrSold'] - inputs['YearBuilt']
        outputs['HouseAge_scaled'] = tft.scale_to_z_score(age)

    # Years Since Remodel
    if all(f in inputs for f in ['YrSold', 'YearRemodAdd']):
        years_since_remod = inputs['YrSold'] - inputs['YearRemodAdd']
        outputs['YearsSinceRemodel_scaled'] = tft.scale_to_z_score(years_since_remod)

    # ========================================================================
    # 4. ORDINAL FEATURES - Encode with proper ordering
    # ========================================================================
    ordinal_mappings = {
        'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtExposure': ['NA', 'No', 'Mn', 'Av', 'Gd'],
        'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'FireplaceQu': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageQual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageCond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    }

    for feature, ordered_values in ordinal_mappings.items():
        if feature in inputs:
            # Use vocabulary to map to integers (maintains order)
            outputs[f'{feature}_ordinal'] = tft.compute_and_apply_vocabulary(
                inputs[feature],
                vocab_filename=f'{feature}_vocab',
                default_value=0
            )

    # ========================================================================
    # 5. CATEGORICAL FEATURES - One-hot or vocabulary encoding
    # ========================================================================
    categorical_features = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
        'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive',
        'SaleType', 'SaleCondition'
    ]

    for feature in categorical_features:
        if feature in inputs:
            # Use vocabulary-based encoding (more efficient than one-hot for high cardinality)
            outputs[f'{feature}_encoded'] = tft.compute_and_apply_vocabulary(
                inputs[feature],
                vocab_filename=f'{feature}_vocab',
                default_value=0,
                top_k=50  # Limit to top 50 categories
            )

    # ========================================================================
    # 6. BOOLEAN FEATURES
    # ========================================================================
    # Has Pool
    if 'PoolArea' in inputs:
        outputs['HasPool'] = tf.cast(inputs['PoolArea'] > 0, tf.int64)

    # Has Garage
    if 'GarageArea' in inputs:
        outputs['HasGarage'] = tf.cast(inputs['GarageArea'] > 0, tf.int64)

    # Has Fireplace
    if 'Fireplaces' in inputs:
        outputs['HasFireplace'] = tf.cast(inputs['Fireplaces'] > 0, tf.int64)

    # Is Remodeled
    if all(f in inputs for f in ['YearRemodAdd', 'YearBuilt']):
        outputs['IsRemodeled'] = tf.cast(
            inputs['YearRemodAdd'] > inputs['YearBuilt'],
            tf.int64
        )

    return outputs


def create_derived_features(df):
    """
    Create derived features from raw data (for exploratory analysis).

    This is a pandas-based version for notebooks/exploration.
    The actual pipeline uses preprocessing_fn above.

    Args:
        df: Input pandas DataFrame

    Returns:
        DataFrame with additional derived features
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Total Square Footage
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Total Bathrooms
    if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df['TotalBath'] = (df['FullBath'] +
                           0.5 * df['HalfBath'] +
                           df['BsmtFullBath'] +
                           0.5 * df['BsmtHalfBath'])

    # House Age
    if all(col in df.columns for col in ['YrSold', 'YearBuilt']):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    # Years Since Remodel
    if all(col in df.columns for col in ['YrSold', 'YearRemodAdd']):
        df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    # Boolean features
    if 'PoolArea' in df.columns:
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)

    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    if all(col in df.columns for col in ['YearRemodAdd', 'YearBuilt']):
        df['IsRemodeled'] = (df['YearRemodAdd'] > df['YearBuilt']).astype(int)

    # Log transform SalePrice if present
    if 'SalePrice' in df.columns:
        df['SalePrice_log'] = np.log(df['SalePrice'] + 1)

    return df


def test_feature_engineering():
    """Test function to verify feature engineering works."""
    print("Testing feature engineering...")

    # Test with dummy data
    import pandas as pd
    import numpy as np

    # Create sample data
    sample_data = pd.DataFrame({
        'TotalBsmtSF': [1000, 800, 1200],
        '1stFlrSF': [1000, 900, 1100],
        '2ndFlrSF': [800, 600, 900],
        'FullBath': [2, 1, 2],
        'HalfBath': [1, 0, 1],
        'BsmtFullBath': [1, 0, 1],
        'BsmtHalfBath': [0, 0, 1],
        'YrSold': [2008, 2007, 2009],
        'YearBuilt': [2000, 1990, 2005],
        'YearRemodAdd': [2005, 1990, 2008],
        'PoolArea': [0, 100, 0],
        'GarageArea': [400, 0, 500],
        'Fireplaces': [1, 0, 2],
        'SalePrice': [200000, 150000, 250000]
    })

    # Test derived features
    result = create_derived_features(sample_data)

    print(f"\nOriginal features: {len(sample_data.columns)}")
    print(f"Features after engineering: {len(result.columns)}")
    print(f"New features created: {len(result.columns) - len(sample_data.columns)}")

    print("\nNew features:")
    new_cols = set(result.columns) - set(sample_data.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")

    print("\nFeature engineering test passed!")


if __name__ == "__main__":
    test_feature_engineering()
