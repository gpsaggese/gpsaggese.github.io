"""
Base model class defining the interface for all models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models in the project.
    
    All models should implement:
    - fit(): Train the model
    - predict(): Make predictions
    - evaluate(): Calculate performance metrics
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize base model.
        
        Args:
            name: Model name for logging and identification
        """
        self.name = name
        self.is_fitted = False
        self.metrics = {}
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
            
        Returns:
            self: Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True targets
            metrics: List of metrics to compute (default: ['rmse', 'mae', 'r2'])
            
        Returns:
            Dictionary of metric names and values
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted. Call fit() first.")
        
        y_pred = self.predict(X)
        
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = float(np.sqrt(mean_squared_error(y, y_pred)))
        if 'mae' in metrics:
            results['mae'] = float(mean_absolute_error(y, y_pred))
        if 'r2' in metrics:
            results['r2'] = float(r2_score(y, y_pred))
        if 'mse' in metrics:
            results['mse'] = float(mean_squared_error(y, y_pred))
            
        self.metrics = results
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        return self
    
    def summary(self) -> str:
        """Return model summary string."""
        summary = f"Model: {self.name}\n"
        summary += f"Fitted: {self.is_fitted}\n"
        if self.metrics:
            summary += "Metrics:\n"
            for k, v in self.metrics.items():
                summary += f"  {k}: {v:.4f}\n"
        return summary
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
