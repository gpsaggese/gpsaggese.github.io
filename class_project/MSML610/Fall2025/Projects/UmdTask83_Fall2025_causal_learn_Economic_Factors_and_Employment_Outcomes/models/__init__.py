"""
Machine Learning models for economic forecasting and comparison with causal analysis.

This module provides:
- RandomForestModel: Tree-based ensemble for wage/employment prediction
- LSTMModel: Deep learning model for temporal dependencies
- CausalModel: Wrapper for causal-learn algorithms
"""

from models.random_forest_model import RandomForestModel
from models.lstm_model import LSTMModel
from models.causal_model import CausalModel

__all__ = [
    'RandomForestModel',
    'LSTMModel',
    'CausalModel'
]
