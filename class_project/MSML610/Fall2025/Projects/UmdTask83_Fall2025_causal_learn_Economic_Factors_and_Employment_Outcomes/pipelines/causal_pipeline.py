"""
Main causal inference pipeline definition.

This module defines the complete causal inference pipeline including:
- Data preprocessing
- Causal discovery
- Causal effect estimation
- Temporal analysis
"""

import logging
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from utils.utils_data_io import (
    load_economic_data,
    prepare_features_for_causal_discovery,
    create_derived_features
)
from utils.utils_post_processing import (
    discover_causal_structure,
    estimate_causal_effects,
    visualize_causal_graph,
    rolling_window_causal_discovery,
    temporal_effect_estimation,
    prepare_lstm_data
)
from models import RandomForestModel, LSTMModel, CausalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalInferencePipeline:
    """
    Main pipeline class for causal inference analysis on economic data.
    """
    
    # Default variables for economic causal analysis
    DEFAULT_VARIABLES = [
        'unemployment_rate',
        'inflation_rate', 
        'wage_growth',
        'gdp_growth',
        'federal_funds_rate'
    ]
    
    def __init__(
        self,
        data_path: str = None,
        algorithm: str = "PC",
        alpha: float = 0.05,
        window_size: int = 36
    ):
        """
        Initialize the causal inference pipeline.
        
        Args:
            data_path: Path to economic data CSV (default: data/economic_data.csv)
            algorithm: Causal discovery algorithm ('PC', 'GES', 'FCI')
            alpha: Significance level for independence tests
            window_size: Size of rolling window for temporal analysis (months)
        """
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, 'data', 'economic_data.csv')
        
        self.data_path = data_path
        self.algorithm = algorithm
        self.alpha = alpha
        self.window_size = window_size
        
        self.raw_data = None
        self.causal_data = None
        self.causal_graph = None
        self.edges = None
        self.effects = {}
        
        # ML models for comparison
        self.rf_model = None
        self.lstm_model = None
        self.ml_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the economic data.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        self.raw_data = load_economic_data(self.data_path)
        self.raw_data = create_derived_features(self.raw_data)
        
        logger.info(f"Loaded {len(self.raw_data)} observations")
        return self.raw_data
    
    def prepare_causal_data(
        self,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare data for causal discovery.
        
        Args:
            variables: List of variables to include (None for defaults)
            
        Returns:
            DataFrame ready for causal discovery
        """
        if self.raw_data is None:
            self.load_data()
        
        if variables is None:
            variables = self.DEFAULT_VARIABLES
        
        # Filter to available variables
        available = [v for v in variables if v in self.raw_data.columns]
        
        self.causal_data = prepare_features_for_causal_discovery(
            self.raw_data, available
        )
        
        logger.info(f"Prepared {len(self.causal_data)} observations for causal discovery")
        return self.causal_data
    
    def discover_structure(
        self,
        variables: Optional[List[str]] = None
    ) -> Any:
        """
        Discover causal structure from data.
        
        Args:
            variables: List of variables to include (None for defaults)
            
        Returns:
            CausalGraph object
        """
        if self.causal_data is None:
            self.prepare_causal_data(variables)
        
        logger.info(f"Running {self.algorithm} algorithm (alpha={self.alpha})")
        
        self.causal_graph, self.edges = discover_causal_structure(
            data=self.causal_data,
            algorithm=self.algorithm,
            alpha=self.alpha
        )
        
        logger.info(f"Discovered {len(self.edges)} causal relationships")
        return self.causal_graph
    
    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        method: str = 'regression'
    ) -> Dict[str, float]:
        """
        Estimate causal effect of treatment on outcome.
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            method: Estimation method ('regression', 'SEM')
            
        Returns:
            Dictionary with effect estimates
        """
        if self.causal_graph is None:
            self.discover_structure()
        
        effect = estimate_causal_effects(
            data=self.causal_data,
            causal_graph=self.causal_graph,
            treatment=treatment,
            outcome=outcome,
            method=method
        )
        
        key = f"{treatment} -> {outcome}"
        self.effects[key] = effect
        
        return effect
    
    def run_temporal_analysis(
        self,
        treatment: str,
        outcome: str
    ) -> Dict[str, np.ndarray]:
        """
        Perform temporal analysis over rolling windows.
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            
        Returns:
            Dictionary with temporal effect estimates
        """
        if self.raw_data is None:
            self.load_data()
        
        temporal = temporal_effect_estimation(
            data=self.raw_data,
            window_size=self.window_size,
            treatment=treatment,
            outcome=outcome,
            time_column='date'
        )
        
        return temporal
    
    def visualize(
        self,
        output_path: Optional[str] = None,
        title: str = 'Economic Causal Structure'
    ) -> None:
        """
        Visualize the causal graph.
        
        Args:
            output_path: Path to save visualization (None to display)
            title: Title for the plot
        """
        if self.causal_graph is None:
            self.discover_structure()
        
        visualize_causal_graph(
            graph=self.causal_graph,
            output_path=output_path,
            title=title
        )
    
    def train_random_forest(
        self,
        features: Optional[List[str]] = None,
        target: str = 'wage_growth',
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train Random Forest model for comparison with causal analysis.
        
        Args:
            features: Feature columns (None for defaults)
            target: Target variable
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with model metrics and feature importance
        """
        from sklearn.model_selection import train_test_split
        
        if self.causal_data is None:
            self.prepare_causal_data()
        
        if features is None:
            features = [c for c in ['unemployment_rate', 'inflation_rate', 'federal_funds_rate'] 
                       if c in self.causal_data.columns]
        
        if target not in self.causal_data.columns:
            raise ValueError(f"Target '{target}' not in data")
        
        # Prepare data
        X = self.causal_data[features].dropna()
        y = self.causal_data.loc[X.index, target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train using modular RandomForestModel
        self.rf_model = RandomForestModel(n_estimators=100, max_depth=10)
        self.rf_model.fit(X_train, y_train, feature_names=features)
        
        # Evaluate
        metrics = self.rf_model.evaluate(X_test, y_test)
        importance = self.rf_model.get_feature_importance()
        
        self.ml_results['random_forest'] = {
            'metrics': metrics,
            'feature_importance': importance.to_dict('records')
        }
        
        logger.info(f"Random Forest R²: {metrics['r2']:.4f}")
        return self.ml_results['random_forest']
    
    def train_lstm(
        self,
        features: Optional[List[str]] = None,
        target: str = 'wage_growth',
        sequence_length: int = 12,
        epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Train LSTM model for temporal comparison.
        
        Args:
            features: Feature columns
            target: Target variable
            sequence_length: Number of time steps
            epochs: Training epochs
            
        Returns:
            Dictionary with model metrics
        """
        if self.raw_data is None:
            self.load_data()
        
        if features is None:
            features = [c for c in ['unemployment_rate', 'inflation_rate', 'federal_funds_rate'] 
                       if c in self.raw_data.columns]
        
        # Prepare sequences
        X_seq, y_seq = prepare_lstm_data(
            self.raw_data, features, target, sequence_length
        )
        
        # Split
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Train using modular LSTMModel
        self.lstm_model = LSTMModel(
            sequence_length=sequence_length,
            lstm_units=64
        )
        self.lstm_model.fit(X_train, y_train, epochs=epochs, verbose=0)
        
        # Evaluate
        metrics = self.lstm_model.evaluate(X_test, y_test)
        
        self.ml_results['lstm'] = {
            'metrics': metrics,
            'training_history': self.lstm_model.get_training_history()
        }
        
        logger.info(f"LSTM R²: {metrics['r2']:.4f}")
        return self.ml_results['lstm']
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare causal effects with ML model predictions.
        
        Returns:
            DataFrame comparing all models
        """
        comparison = []
        
        # Causal effects
        for key, effect in self.effects.items():
            comparison.append({
                'model': 'Causal (SEM)',
                'relationship': key,
                'coefficient': effect['coefficient'],
                'p_value': effect['p_value'],
                'r2': None,
                'rmse': None
            })
        
        # Random Forest
        if 'random_forest' in self.ml_results:
            rf = self.ml_results['random_forest']
            comparison.append({
                'model': 'Random Forest',
                'relationship': 'Prediction',
                'coefficient': None,
                'p_value': None,
                'r2': rf['metrics']['r2'],
                'rmse': rf['metrics']['rmse']
            })
        
        # LSTM
        if 'lstm' in self.ml_results:
            lstm = self.ml_results['lstm']
            comparison.append({
                'model': 'LSTM',
                'relationship': 'Prediction',
                'coefficient': None,
                'p_value': None,
                'r2': lstm['metrics']['r2'],
                'rmse': lstm['metrics']['rmse']
            })
        
        return pd.DataFrame(comparison)
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete causal analysis pipeline.
        
        Returns:
            Dictionary with all results
        """
        # Load and prepare data
        self.load_data()
        self.prepare_causal_data()
        
        # Discover causal structure
        self.discover_structure()
        
        # Estimate key effects
        causal_pairs = [
            ('unemployment_rate', 'wage_growth'),
            ('inflation_rate', 'wage_growth'),
            ('federal_funds_rate', 'inflation_rate')
        ]
        
        for treatment, outcome in causal_pairs:
            if treatment in self.causal_data.columns and outcome in self.causal_data.columns:
                self.estimate_effect(treatment, outcome)
        
        # Temporal analysis
        temporal = None
        if 'unemployment_rate' in self.causal_data.columns and 'wage_growth' in self.causal_data.columns:
            temporal = self.run_temporal_analysis('unemployment_rate', 'wage_growth')
        
        # Train ML models for comparison
        try:
            self.train_random_forest()
        except Exception as e:
            logger.warning(f"Random Forest training failed: {e}")
        
        return {
            'n_observations': len(self.causal_data),
            'n_edges': len(self.edges),
            'edges': self.edges,
            'effects': self.effects,
            'temporal': temporal,
            'ml_results': self.ml_results
        }


def run_pipeline(
    data_path: str = None,
    algorithm: str = "PC",
    alpha: float = 0.05
) -> CausalInferencePipeline:
    """
    Run the complete causal inference pipeline.
    
    Args:
        data_path: Path to data file
        algorithm: Causal discovery algorithm
        alpha: Significance level
        
    Returns:
        CausalInferencePipeline object with results
    """
    pipeline = CausalInferencePipeline(
        data_path=data_path,
        algorithm=algorithm,
        alpha=alpha
    )
    
    pipeline.run_full_analysis()
    
    return pipeline


if __name__ == "__main__":
    # Run pipeline when executed directly
    pipeline = run_pipeline()
    
    print("\nResults Summary:")
    print(f"  Edges: {len(pipeline.edges)}")
    for src, tgt in pipeline.edges:
        print(f"    {src} -> {tgt}")
    
    print(f"\n  Effects:")
    for key, effect in pipeline.effects.items():
        sig = "***" if effect['p_value'] < 0.001 else "**" if effect['p_value'] < 0.01 else "*" if effect['p_value'] < 0.05 else ""
        print(f"    {key}: {effect['coefficient']:.4f} {sig}")
