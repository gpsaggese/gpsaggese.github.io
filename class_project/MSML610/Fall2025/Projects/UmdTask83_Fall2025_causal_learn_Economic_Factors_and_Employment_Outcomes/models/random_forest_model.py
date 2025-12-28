"""
Random Forest model for economic prediction.

This model provides:
- Non-linear relationship modeling
- Feature importance analysis
- Comparison baseline for causal analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Regressor for predicting economic outcomes.
    
    Example:
        >>> model = RandomForestModel(n_estimators=100, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        name: str = "RandomForest"
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            name: Model name
        """
        super().__init__(name=name)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.feature_names = None
        self.feature_importances_ = None
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            feature_names: Optional list of feature names
            **kwargs: Additional parameters passed to sklearn fit()
            
        Returns:
            self: Fitted model
        """
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
            
        # Initialize and fit model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"{self.name} training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def get_feature_importance(
        self, 
        sort: bool = True
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            sort: Sort by importance (descending)
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        })
        
        if sort:
            importance_df = importance_df.sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
        
        return importance_df
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'RandomForestModel':
        """Set model hyperparameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of folds
            
        Returns:
            Dictionary with mean and std of scores
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return {
            'cv_mean': float(np.mean(scores)),
            'cv_std': float(np.std(scores)),
            'cv_scores': scores.tolist()
        }
