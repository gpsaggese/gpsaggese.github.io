"""
Configuration module for TFX pipeline

This module contains all configuration settings for the house price prediction pipeline.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
DATA_DESCRIPTION_PATH = DATA_DIR / "data_description.txt"

# Pipeline directories
PIPELINE_ROOT = PROJECT_ROOT / "pipeline_outputs"
MODELS_DIR = PROJECT_ROOT / "models"
SERVING_MODEL_DIR = MODELS_DIR / "serving"

# Ensure directories exist
PIPELINE_ROOT.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SERVING_MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

PIPELINE_NAME = "house_price_prediction_pipeline"
PIPELINE_ROOT_STR = str(PIPELINE_ROOT / PIPELINE_NAME)

# Metadata configuration
METADATA_PATH = str(PIPELINE_ROOT / "metadata" / PIPELINE_NAME / "metadata.db")

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Target column
TARGET_COLUMN = "SalePrice"

# Feature columns (will be auto-detected from schema)
# Numerical features (examples)
NUMERICAL_FEATURES = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1",
    "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
    "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",
    "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal", "MoSold", "YrSold"
]

# Categorical features (examples)
CATEGORICAL_FEATURES = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
    "LandContour", "Utilities", "LotConfig", "LandSlope",
    "Neighborhood", "Condition1", "Condition2", "BldgType",
    "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "Foundation", "Heating",
    "CentralAir", "Electrical", "Functional", "GarageType",
    "GarageFinish", "PavedDrive", "SaleType", "SaleCondition"
]

# Ordinal features (quality/condition ratings)
ORDINAL_FEATURES = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "random_state": 42,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "early_stopping_rounds": 50
}

# TensorFlow DNN hyperparameters
TF_DNN_PARAMS = {
    "hidden_units": [128, 64, 32],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping_patience": 10
}

# Training configuration
TRAIN_STEPS = 5000
EVAL_STEPS = 500

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Evaluation metrics
EVAL_METRICS = ["rmse", "mae", "r2"]

# Model approval thresholds
RMSE_THRESHOLD = 30000  # Max acceptable RMSE
R2_THRESHOLD = 0.85     # Min acceptable R²

# Cross-validation configuration
CV_FOLDS = 5

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Missing value imputation strategies
NUMERICAL_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_VALUE = "Missing"

# Scaling method
SCALING_METHOD = "standard"  # Options: "standard", "minmax"

# Create interaction features
CREATE_INTERACTIONS = True

# Create polynomial features
CREATE_POLYNOMIALS = False
POLYNOMIAL_DEGREE = 2

# Apply log transformation to target
LOG_TRANSFORM_TARGET = True

# ============================================================================
# SERVING CONFIGURATION
# ============================================================================

# Model serving directory
SERVING_MODEL_DIR_STR = str(SERVING_MODEL_DIR)

# TensorFlow Serving configuration
TF_SERVING_PORT = 8501
TF_SERVING_REST_PORT = 8501
TF_SERVING_GRPC_PORT = 8500

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_path(split="train"):
    """Get the path to train or test data."""
    if split == "train":
        return str(TRAIN_DATA_PATH)
    elif split == "test":
        return str(TEST_DATA_PATH)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

def get_pipeline_root():
    """Get the pipeline root directory."""
    return PIPELINE_ROOT_STR

def get_metadata_path():
    """Get the metadata database path."""
    return METADATA_PATH

def get_serving_model_dir():
    """Get the serving model directory."""
    return SERVING_MODEL_DIR_STR

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
