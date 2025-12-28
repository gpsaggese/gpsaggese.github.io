"""
Causal model wrapper for causal-learn algorithms.

This model provides:
- Unified interface for causal discovery (PC, GES, FCI)
- Causal effect estimation
- Integration with the base model interface
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging

from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalModel(BaseModel):
    """
    Causal model wrapper for causal-learn algorithms.
    
    Provides a unified interface for:
    - Causal structure discovery (PC, GES, FCI algorithms)
    - Causal effect estimation using regression adjustment
    - Comparison with predictive ML models
    
    Example:
        >>> model = CausalModel(algorithm='PC', alpha=0.05)
        >>> model.fit(data, variables=['inflation', 'unemployment', 'wages'])
        >>> effects = model.estimate_effect('inflation', 'wages')
    """
    
    def __init__(
        self,
        algorithm: str = 'PC',
        alpha: float = 0.05,
        indep_test: str = 'fisherz',
        name: str = "CausalModel"
    ):
        """
        Initialize causal model.
        
        Args:
            algorithm: Causal discovery algorithm ('PC', 'GES', 'FCI')
            alpha: Significance level for independence tests
            indep_test: Independence test method ('fisherz', 'chisq', 'gsq')
            name: Model name
        """
        super().__init__(name=name)
        
        self.algorithm = algorithm.upper()
        self.alpha = alpha
        self.indep_test = indep_test
        
        self.causal_graph = None
        self.edges = []
        self.variables = None
        self.data = None
        self.adjacency_matrix = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> 'CausalModel':
        """
        Discover causal structure from data.
        
        Args:
            X: Data array or DataFrame
            y: Ignored (for API compatibility)
            variables: List of variable names
            **kwargs: Additional algorithm parameters
            
        Returns:
            self: Fitted model with discovered structure
        """
        logger.info(f"Running {self.algorithm} causal discovery (alpha={self.alpha})")
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            self.variables = list(X.columns)
            self.data = X.copy()
            data_array = X.dropna().values.astype(np.float64)
        else:
            data_array = np.asarray(X, dtype=np.float64)
            if variables is not None:
                self.variables = variables
            else:
                self.variables = [f"X{i}" for i in range(data_array.shape[1])]
            self.data = pd.DataFrame(data_array, columns=self.variables)
        
        n_vars = len(self.variables)
        
        # Remove rows with NaN
        mask = ~np.isnan(data_array).any(axis=1)
        data_array = data_array[mask]
        
        logger.info(f"Data: {len(data_array)} samples, {n_vars} variables")
        
        # Run causal discovery
        self.edges = []
        
        try:
            self._run_causal_discovery(data_array)
        except ImportError:
            logger.warning("causal-learn not installed. Using correlation fallback.")
            self._fallback_discovery(data_array)
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            self._fallback_discovery(data_array)
        
        self.is_fitted = True
        logger.info(f"Discovered {len(self.edges)} causal relationships")
        
        return self
    
    def _run_causal_discovery(self, data: np.ndarray) -> None:
        """Run causal-learn algorithms."""
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ScoreBased.GES import ges
        from causallearn.search.ConstraintBased.FCI import fci
        
        n_vars = len(self.variables)
        
        if self.algorithm == 'PC':
            cg = pc(
                data,
                alpha=self.alpha,
                indep_test=self.indep_test,
                stable=True,
                show_progress=False
            )
            adj_matrix = cg.G.graph
            self.causal_graph = cg
            
        elif self.algorithm == 'GES':
            record = ges(data, score_func='local_score_BIC')
            adj_matrix = record['G'].graph
            self.causal_graph = record
            
        elif self.algorithm == 'FCI':
            g, _ = fci(data, self.indep_test, self.alpha, verbose=False)
            adj_matrix = g.graph
            self.causal_graph = g
            
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.adjacency_matrix = adj_matrix
        
        # Extract edges from adjacency matrix
        for i in range(n_vars):
            for j in range(n_vars):
                # Directed edge i -> j
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    self.edges.append((self.variables[i], self.variables[j]))
                # Undirected edge
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                    self.edges.append((self.variables[i], self.variables[j]))
    
    def _fallback_discovery(self, data: np.ndarray) -> None:
        """Fallback using correlations when causal-learn unavailable."""
        from scipy import stats
        
        n_vars = len(self.variables)
        df = pd.DataFrame(data, columns=self.variables)
        corr = df.corr()
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                r = corr.iloc[i, j]
                n = len(data)
                t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                
                if p_value < self.alpha and abs(r) > 0.1:
                    self.edges.append((self.variables[i], self.variables[j]))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Causal models don't make predictions in the traditional sense.
        This method returns causal effect estimates for each sample.
        """
        raise NotImplementedError(
            "Causal models estimate effects, not predictions. "
            "Use estimate_effect() instead."
        )
    
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
            Dictionary with effect estimates and statistics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if treatment not in self.variables:
            raise ValueError(f"Treatment '{treatment}' not in variables")
        if outcome not in self.variables:
            raise ValueError(f"Outcome '{outcome}' not in variables")
        
        # Identify confounders
        confounders = self._identify_confounders(treatment, outcome)
        
        # Prepare data
        cols = [treatment, outcome] + confounders
        analysis_data = self.data[cols].dropna()
        
        # Regression adjustment
        return self._regression_estimate(
            analysis_data, treatment, outcome, confounders
        )
    
    def _identify_confounders(
        self, 
        treatment: str, 
        outcome: str
    ) -> List[str]:
        """Identify confounders from causal graph."""
        # Find variables that cause both treatment and outcome
        causes_treatment = {src for src, tgt in self.edges if tgt == treatment}
        causes_outcome = {src for src, tgt in self.edges if tgt == outcome}
        confounders = list(causes_treatment & causes_outcome)
        return [c for c in confounders if c != treatment and c != outcome]
    
    def _regression_estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> Dict[str, float]:
        """Estimate effect using OLS regression."""
        from scipy import stats
        
        # Build regression
        feature_cols = [treatment] + confounders
        X = data[feature_cols].values
        y = data[outcome].values
        
        # Add intercept
        X_int = np.column_stack([np.ones(len(X)), X])
        
        # OLS
        beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
        y_pred = X_int @ beta
        residuals = y - y_pred
        
        n, p = X_int.shape
        mse = np.sum(residuals**2) / (n - p)
        var_beta = mse * np.linalg.inv(X_int.T @ X_int)
        se = np.sqrt(np.diag(var_beta))
        
        # Treatment effect (first coefficient after intercept)
        coef = beta[1]
        se_coef = se[1]
        t_stat = coef / se_coef
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))
        
        return {
            'coefficient': float(coef),
            'se': float(se_coef),
            'ci_lower': float(coef - 1.96 * se_coef),
            'ci_upper': float(coef + 1.96 * se_coef),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_observations': int(n),
            'confounders': confounders,
            'method': 'OLS Regression Adjustment'
        }
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Get discovered causal edges."""
        return self.edges.copy()
    
    def get_adjacency_matrix(self) -> Optional[np.ndarray]:
        """Get adjacency matrix of causal graph."""
        return self.adjacency_matrix
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'algorithm': self.algorithm,
            'alpha': self.alpha,
            'indep_test': self.indep_test
        }
    
    def summary(self) -> str:
        """Return model summary."""
        summary = f"Causal Model: {self.algorithm}\n"
        summary += f"Alpha: {self.alpha}\n"
        summary += f"Fitted: {self.is_fitted}\n"
        
        if self.is_fitted:
            summary += f"Variables: {len(self.variables)}\n"
            summary += f"Discovered edges: {len(self.edges)}\n"
            if self.edges:
                summary += "Causal relationships:\n"
                for src, tgt in self.edges:
                    summary += f"  {src} → {tgt}\n"
        
        return summary
