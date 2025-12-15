"""
FairnessPP: Fairness in Predictive Policing

A high-level Python API for building fair machine learning models in high-stakes 
public sector applications. Provides multiple mitigation strategies and comprehensive
evaluation utilities.

Course: MSML610 Fall 2025
Project: Fairness in Predictive Policing
"""

import pandas as pd
import numpy as np
import os
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

# Native API imports
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fairlearn imports
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    MetricFrame, 
    equalized_odds_difference,
    demographic_parity_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION OBJECTS
# =============================================================================

@dataclass
class ModelConfig:
    """
    Stable configuration for model training.
    
    Attributes:
        n_estimators: Number of boosting rounds for the base classifier
        max_depth: Maximum tree depth for base classifier
        learning_rate: Boosting learning rate
        random_state: Seed for reproducibility
        max_iter_mitigation: Maximum iterations for fairness mitigation algorithm
        constraint_type: Fairness constraint ("equalized_odds" or "demographic_parity")
    """
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    random_state: int = 42
    max_iter_mitigation: int = 50
    constraint_type: str = "equalized_odds"
    

@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation metrics returned by FairnessPredictor.evaluate().
    
    Attributes:
        accuracy: Overall classification accuracy
        balanced_accuracy: Mean recall per class (handles imbalance)
        precision: Precision score
        recall: Recall score (sensitivity)
        f1_score: Harmonic mean of precision and recall
        auc_roc: Area under ROC curve (0 if unavailable)
        fairness_disparity: Equalized odds difference (alias)
        demographic_parity_diff: Difference in selection rates across groups
        equalized_odds_diff: Max difference in TPR/FPR across groups
        group_metrics: DataFrame with per-group performance metrics
        confusion_matrix: 2x2 confusion matrix
        selection_rates: Series of selection rates per demographic group
    """
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    fairness_disparity: float
    demographic_parity_diff: float
    equalized_odds_diff: float
    group_metrics: pd.DataFrame
    confusion_matrix: np.ndarray
    selection_rates: pd.Series


# =============================================================================
# CORE WRAPPER CLASS
# =============================================================================

class FairnessPredictor:
    """
    Enhanced wrapper around Scikit-Learn and Fairlearn for fair ML.
    
    Provides a unified interface for training models with or without fairness
    constraints, using either in-processing (ExponentiatedGradient) or 
    post-processing (ThresholdOptimizer) mitigation strategies.
    
    Example:
        >>> config = ModelConfig(n_estimators=100, constraint_type="equalized_odds")
        >>> predictor = FairnessPredictor(config)
        >>> predictor.train(X_train, y_train, A=A_train, mitigate=True)
        >>> result = predictor.evaluate(X_test, y_test, A_test)
        >>> print(f"Fairness: {result.equalized_odds_diff:.3f}")
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize the FairnessPredictor.
        
        Args:
            config: ModelConfig object with hyperparameters (uses defaults if None)
        """
        self.config = config if config is not None else ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_mitigated = False
        self.mitigation_type = None
        self.feature_names = None
        self._base_model_for_postprocessing = None

    def train(self, X, y, A=None, mitigate: bool = False, mitigation_strategy: str = "inprocessing", class_weight: str = "balanced"):
        """
        Train the model with optional fairness mitigation.
        
        Args:
            X: Feature matrix (DataFrame or ndarray)
            y: Target variable (Series or ndarray)
            A: Sensitive attributes (required if mitigate=True)
            mitigate: Whether to apply fairness constraints
            mitigation_strategy: "inprocessing" (ExponentiatedGradient) or 
                                 "postprocessing" (ThresholdOptimizer)
            class_weight: "balanced" to handle class imbalance, None otherwise
        
        Raises:
            ValueError: If mitigate=True but A is not provided
        """
        self.is_mitigated = mitigate
        self.mitigation_type = mitigation_strategy if mitigate else None
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        if self.feature_names:
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Calculate sample weights for class balancing
        sample_weights = None
        if class_weight == "balanced":
            class_counts = np.bincount(y.astype(int))
            total = len(y)
            weights_per_class = total / (2.0 * class_counts)
            sample_weights = np.array([weights_per_class[int(yi)] for yi in y])
            print(f"  Class weights: No Arrest={weights_per_class[0]:.2f}, Arrest={weights_per_class[1]:.2f}")
        
        # Training logic based on mitigation strategy
        if not mitigate:
            self._train_baseline(X_scaled, y, sample_weights)
            
        elif mitigation_strategy == "inprocessing":
            if A is None:
                raise ValueError("Sensitive attribute 'A' required for mitigation.")
            self._train_inprocessing(X_scaled, y, A, sample_weights)
        
        elif mitigation_strategy == "postprocessing":
            if A is None:
                raise ValueError("Sensitive attribute 'A' required for mitigation.")
            self._train_postprocessing(X_scaled, y, A, sample_weights)
        
        else:
            raise ValueError(f"Unknown mitigation_strategy: {mitigation_strategy}")

    def _train_baseline(self, X_scaled, y, sample_weights):
        """Train a standard Gradient Boosting model without fairness constraints."""
        print("Training Baseline Model (GradientBoosting)...")
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            min_samples_split=20,
            min_samples_leaf=10
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Report training predictions distribution
        train_preds = self.model.predict(X_scaled)
        print(f"  Training predictions: {np.sum(train_preds==0)} negative, {np.sum(train_preds==1)} positive")

    def _train_inprocessing(self, X_scaled, y, A, sample_weights):
        """
        Train using Fairlearn's ExponentiatedGradient (in-processing).
        
        This method applies fairness constraints during training by iteratively
        reweighting samples to satisfy the chosen constraint.
        """
        print(f"Training Fair Model (In-Processing: ExponentiatedGradient)...")
        print(f"  Constraint: {self.config.constraint_type}")
        
        # Use LogisticRegression as base estimator - more stable with ExponentiatedGradient
        # GradientBoosting can be unstable with the reweighting scheme
        base_estimator = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=self.config.random_state,
            class_weight='balanced'  # Handle imbalance in base estimator
        )
        
        # Select constraint
        if self.config.constraint_type == "demographic_parity":
            constraint = DemographicParity()
        else:
            constraint = EqualizedOdds()
        
        # Configure ExponentiatedGradient with tuned parameters
        # nu: trade-off parameter (smaller = stricter fairness, larger = more utility)
        # eps: constraint tolerance (smaller = stricter enforcement)
        self.model = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            max_iter=self.config.max_iter_mitigation,
            nu=1e-4,      # Balance between fairness and utility
            eta0=2.0,     # Learning rate for the reduction
            eps=0.005     # Stricter constraint enforcement
        )
        
        # Fit the model - ExponentiatedGradient handles sample weighting internally
        # through its constraint satisfaction mechanism
        self.model.fit(X_scaled, y, sensitive_features=A)
        
        n_predictors = len(self.model.predictors_) if hasattr(self.model, 'predictors_') else 1
        print(f"  Trained ensemble of {n_predictors} predictors")
        
        # Report training predictions distribution
        train_preds = self.model.predict(X_scaled)
        print(f"  Training predictions: {np.sum(train_preds==0)} negative, {np.sum(train_preds==1)} positive")

    def _train_postprocessing(self, X_scaled, y, A, sample_weights):
        """
        Train using Fairlearn's ThresholdOptimizer (post-processing).
        
        This method first trains a standard model, then adjusts decision thresholds
        per demographic group to satisfy fairness constraints.
        """
        print("Training Fair Model (Post-Processing: ThresholdOptimizer)...")
        print(f"  Constraint: {self.config.constraint_type}")
        
        # Step 1: Train a well-calibrated base model
        # Key insight: ThresholdOptimizer needs good probability estimates,
        # not aggressive weighting that distorts probabilities
        print("  Step 1: Training base model with balanced class weights...")
        
        base_model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            min_samples_split=20,
            min_samples_leaf=10
        )
        
        # Use moderate sample weights - not too aggressive
        base_model.fit(X_scaled, y, sample_weight=sample_weights)
        self._base_model_for_postprocessing = base_model
        
        # Verify base model makes meaningful predictions
        base_preds = base_model.predict(X_scaled)
        base_proba = base_model.predict_proba(X_scaled)[:, 1]
        print(f"    Base model train preds: {np.sum(base_preds==0)} neg, {np.sum(base_preds==1)} pos")
        print(f"    Probability range: [{base_proba.min():.3f}, {base_proba.max():.3f}]")
        
        # Step 2: Apply ThresholdOptimizer
        print("  Step 2: Optimizing decision thresholds per group...")
        
        # Note: ThresholdOptimizer with equalized_odds can be very conservative
        # Use demographic_parity for more stable results, or equalized_odds as specified
        constraint_str = "demographic_parity" if self.config.constraint_type == "demographic_parity" else "equalized_odds"
        
        self.model = ThresholdOptimizer(
            estimator=base_model,
            constraints=constraint_str,
            predict_method="predict_proba",
            prefit=True,
            objective="balanced_accuracy_score"  # Optimize for balanced accuracy
        )
        
        self.model.fit(X_scaled, y, sensitive_features=A)
        
        # Report post-processed predictions
        train_preds_fair = self.model.predict(X_scaled, sensitive_features=A)
        print(f"    Post-processed train preds: {np.sum(train_preds_fair==0)} neg, {np.sum(train_preds_fair==1)} pos")

    def predict(self, X, A=None):
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            A: Sensitive attributes (required for post-processing models)
        
        Returns:
            ndarray: Binary predictions (0 or 1)
        """
        X_scaled = self.scaler.transform(X)
        if self.feature_names:
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if self.mitigation_type == "postprocessing" and A is not None:
            return self.model.predict(X_scaled, sensitive_features=A)
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X, A=None):
        """
        Generate probability predictions if available.
        
        Args:
            X: Feature matrix
            A: Sensitive attributes (unused, for API consistency)
        
        Returns:
            ndarray: Probability estimates [n_samples, 2] or None if unavailable
        """
        X_scaled = self.scaler.transform(X)
        if self.feature_names:
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # For postprocessing, use the base model's probabilities
        if self.mitigation_type == "postprocessing" and self._base_model_for_postprocessing is not None:
            return self._base_model_for_postprocessing.predict_proba(X_scaled)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        
        # ExponentiatedGradient doesn't have standard predict_proba
        return None

    def evaluate(self, X, y, A) -> EvaluationResult:
        """
        Comprehensive evaluation with performance and fairness metrics.
        
        Args:
            X: Feature matrix
            y: True labels
            A: Sensitive attributes
        
        Returns:
            EvaluationResult: Dataclass with all computed metrics
        """
        y_pred = self.predict(X, A=A)
        
        # Performance metrics
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # AUC-ROC (requires probabilities)
        auc = 0.0
        try:
            y_proba = self.predict_proba(X, A=A)
            if y_proba is not None and len(y_proba.shape) == 2:
                auc = roc_auc_score(y, y_proba[:, 1])
        except Exception:
            pass
        
        # Fairness metrics
        eq_odds_diff = equalized_odds_difference(y, y_pred, sensitive_features=A)
        dem_parity_diff = demographic_parity_difference(y, y_pred, sensitive_features=A)
        
        # Group-level metrics using Fairlearn's MetricFrame
        mf = MetricFrame(
            metrics={
                "Selection Rate": selection_rate,
                "Accuracy": accuracy_score,
                "Precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
                "Recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
                "FPR": false_positive_rate,
                "FNR": false_negative_rate,
                "TPR": true_positive_rate
            },
            y_true=y,
            y_pred=y_pred,
            sensitive_features=A
        )
        
        cm = confusion_matrix(y, y_pred)
        
        return EvaluationResult(
            accuracy=acc,
            balanced_accuracy=bal_acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc_roc=auc,
            fairness_disparity=eq_odds_diff,
            demographic_parity_diff=dem_parity_diff,
            equalized_odds_diff=eq_odds_diff,
            group_metrics=mf.by_group,
            confusion_matrix=cm,
            selection_rates=mf.by_group["Selection Rate"]
        )


# =============================================================================
# DATA LOADING
# =============================================================================

API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

def load_chicago_data(local_cache_path="data/chicago_crime_2020_2023.csv", use_enhanced_features=True):
    """
    Load and preprocess Chicago Crime Data with feature engineering.
    
    Args:
        local_cache_path: Path to cached CSV file (created if API succeeds)
        use_enhanced_features: Include rich temporal/spatial features
    
    Returns:
        Tuple of (X, y, A, dates):
            X: Feature DataFrame
            y: Binary arrest indicator Series
            A: Intersectional demographic groups Series
            dates: Datetime Series for temporal splits
    """
    if os.path.exists(local_cache_path):
        print(f"Loading cached data from {local_cache_path}...")
        df = pd.read_csv(local_cache_path)
    else:
        print("Fetching data from Chicago Open Data API...")
        years = [2020, 2021, 2022, 2023] 
        all_data = []
        try:
            for year in years:
                url = f"{API_URL}?$limit=20000&year={year}&$order=date%20DESC"
                print(f"  Fetching {year}...")
                df_year = pd.read_json(url)
                all_data.append(df_year)
            df = pd.concat(all_data, ignore_index=True)
            os.makedirs(os.path.dirname(local_cache_path), exist_ok=True) if os.path.dirname(local_cache_path) else None
            df.to_csv(local_cache_path, index=False)
        except Exception as e:
            print(f"API Error: {e}. Generating mock data for demonstration.")
            df = _generate_mock_data(80000)

    # Standardize column names
    df.rename(columns={
        'latitude': 'Latitude', 
        'longitude': 'Longitude', 
        'arrest': 'Arrest', 
        'domestic': 'Domestic', 
        'date': 'Date',
        'primary_type': 'Primary_Type'
    }, inplace=True)
    
    # Clean data
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Feature engineering
    df = _engineer_features(df, use_enhanced_features)
    df = _engineer_demographics(df)
    
    # Select features
    if use_enhanced_features:
        feature_cols = [
            'Latitude', 'Longitude', 'Domestic',
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
            'Crime_Density', 'Arrest_Rate_Historic',
            'Distance_Downtown', 'Hour_Sin', 'Hour_Cos',
            'Month_Sin', 'Month_Cos'
        ]
    else:
        feature_cols = ['Latitude', 'Longitude', 'Domestic']
    
    X = df[feature_cols].copy()
    y = df['Arrest'].astype(int)
    A = df['Intersectional_Group']
    dates = df['Date']
    
    print(f"Loaded {len(df)} records with {len(feature_cols)} features")
    print(f"Arrest rate: {y.mean():.2%}")
    print(f"Demographic groups: {A.nunique()}")
    
    return X, y, A, dates


def _engineer_features(df, use_enhanced=True):
    """Add temporal and spatial features to the dataset."""
    # Basic temporal features
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Grid-based location binning
    df['lat_bin'] = (df['Latitude'] / 0.01).astype(int)
    df['lon_bin'] = (df['Longitude'] / 0.01).astype(int)
    df['Grid_ID'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)
    
    if use_enhanced:
        # Cyclic encoding for temporal features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Distance from downtown Chicago
        chicago_downtown = (41.8781, -87.6298)
        df['Distance_Downtown'] = np.sqrt(
            (df['Latitude'] - chicago_downtown[0])**2 + 
            (df['Longitude'] - chicago_downtown[1])**2
        )
        
        # Historical crime density per grid cell
        grid_crime_counts = df.groupby('Grid_ID').size()
        df['Crime_Density'] = df['Grid_ID'].map(grid_crime_counts).fillna(0)
        
        # Historical arrest rate per grid cell
        grid_arrest_rates = df.groupby('Grid_ID')['Arrest'].mean()
        df['Arrest_Rate_Historic'] = df['Grid_ID'].map(grid_arrest_rates).fillna(0.5)
    
    return df


def _engineer_demographics(df):
    """
    Create simulated demographic attributes for fairness analysis.
    
    Note: Real demographic data would come from census integration.
    This simulation creates realistic neighborhood-level distributions.
    """
    np.random.seed(42)
    
    unique_grids = df['Grid_ID'].unique()
    demographic_profiles = {}
    
    for grid in unique_grids:
        # Simulate different neighborhood types
        cluster_type = np.random.choice(
            ['diverse', 'majority_minority', 'majority_white'], 
            p=[0.3, 0.4, 0.3]
        )
        
        if cluster_type == 'diverse':
            race_dist = [0.25, 0.25, 0.25, 0.15, 0.10]
        elif cluster_type == 'majority_minority':
            race_dist = [0.50, 0.10, 0.30, 0.05, 0.05]
        else:
            race_dist = [0.10, 0.60, 0.15, 0.10, 0.05]
        
        income_dist = [0.35, 0.30, 0.25, 0.10]
        
        demographic_profiles[grid] = {
            'Majority_Race': np.random.choice(
                ['Black', 'White', 'Hispanic', 'Asian', 'Other'], 
                p=race_dist
            ),
            'Income_Level': np.random.choice(
                ['Low', 'Medium-Low', 'Medium-High', 'High'], 
                p=income_dist
            )
        }
    
    df['Majority_Race'] = df['Grid_ID'].map(lambda x: demographic_profiles[x]['Majority_Race'])
    df['Income_Level'] = df['Grid_ID'].map(lambda x: demographic_profiles[x]['Income_Level'])
    df['Intersectional_Group'] = df['Majority_Race'] + "_" + df['Income_Level']
    
    return df


def _generate_mock_data(n):
    """Generate realistic mock data when API is unavailable."""
    np.random.seed(42)
    
    # Chicago lat/lon bounds
    lat_range = (41.6, 42.0)
    lon_range = (-87.9, -87.5)
    
    # Crime hotspot centers
    hotspot_centers = [(41.85, -87.65), (41.75, -87.70), (41.95, -87.70)]
    
    lats, lons = [], []
    for _ in range(n):
        if np.random.random() < 0.6:  # 60% near hotspots
            center = hotspot_centers[np.random.randint(0, 3)]
            lats.append(np.random.normal(center[0], 0.05))
            lons.append(np.random.normal(center[1], 0.05))
        else:
            lats.append(np.random.uniform(*lat_range))
            lons.append(np.random.uniform(*lon_range))
    
    # ~15% arrest rate (realistic for Chicago data)
    arrests = np.random.random(n) < 0.15
    
    # Date range: 2020-2023
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-12-31')
    date_range_days = (end_date - start_date).days
    random_days = np.random.randint(0, date_range_days, n)
    dates = [start_date + pd.Timedelta(days=int(d)) for d in random_days]
    
    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'arrest': arrests,
        'domestic': np.random.choice([True, False], n, p=[0.2, 0.8]),
        'date': dates
    })


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_fairness_tradeoff(results_dict, save_path=None):
    """
    Plot the Pareto frontier of accuracy vs fairness.
    
    Args:
        results_dict: Dict mapping model names to EvaluationResult objects
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(results_dict.keys())
    accuracies = [results_dict[m].accuracy for m in models]
    balanced_accs = [results_dict[m].balanced_accuracy for m in models]
    fairness = [results_dict[m].equalized_odds_diff for m in models]
    
    # Color map for different model types
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Left plot: Accuracy vs Fairness
    for i, model in enumerate(models):
        axes[0].scatter(fairness[i], accuracies[i], s=200, alpha=0.7, 
                       c=[colors[i]], label=model, edgecolors='black', linewidth=1)
    axes[0].set_xlabel('Equalized Odds Difference (Lower = Fairer)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Fairness Trade-off', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(alpha=0.3)
    
    # Right plot: Balanced Accuracy vs Fairness
    for i, model in enumerate(models):
        axes[1].scatter(fairness[i], balanced_accs[i], s=200, alpha=0.7,
                       c=[colors[i]], label=model, edgecolors='black', linewidth=1)
    axes[1].set_xlabel('Equalized Odds Difference (Lower = Fairer)', fontsize=12)
    axes[1].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[1].set_title('Balanced Accuracy vs Fairness Trade-off', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_group_metrics(eval_result, title="Group Fairness Metrics", save_path=None):
    """
    Visualize per-group performance and fairness metrics.
    
    Args:
        eval_result: EvaluationResult from FairnessPredictor.evaluate()
        title: Plot title
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    
    metrics = eval_result.group_metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Selection Rate
    metrics['Selection Rate'].plot(kind='bar', ax=axes[0,0], color='steelblue', edgecolor='black')
    axes[0,0].set_title('Selection Rate by Group', fontweight='bold')
    axes[0,0].set_ylabel('Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=metrics['Selection Rate'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3, axis='y')
    
    # Accuracy
    metrics['Accuracy'].plot(kind='bar', ax=axes[0,1], color='green', edgecolor='black')
    axes[0,1].set_title('Accuracy by Group', fontweight='bold')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].axhline(y=metrics['Accuracy'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3, axis='y')
    
    # True Positive Rate
    if 'TPR' in metrics.columns:
        metrics['TPR'].plot(kind='bar', ax=axes[1,0], color='orange', edgecolor='black')
        axes[1,0].set_title('True Positive Rate by Group', fontweight='bold')
        axes[1,0].set_ylabel('TPR')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].axhline(y=metrics['TPR'].mean(), color='red', linestyle='--', label='Mean')
        axes[1,0].legend()
        axes[1,0].grid(alpha=0.3, axis='y')
    
    # False Positive Rate
    if 'FPR' in metrics.columns:
        metrics['FPR'].plot(kind='bar', ax=axes[1,1], color='red', edgecolor='black')
        axes[1,1].set_title('False Positive Rate by Group', fontweight='bold')
        axes[1,1].set_ylabel('FPR')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].axhline(y=metrics['FPR'].mean(), color='blue', linestyle='--', label='Mean')
        axes[1,1].legend()
        axes[1,1].grid(alpha=0.3, axis='y')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_table(results_dict):
    """
    Create a formatted comparison table of all models.
    
    Args:
        results_dict: Dict mapping model names to EvaluationResult objects
    
    Returns:
        pandas DataFrame with formatted metrics
    """
    comparison_data = []
    for model_name, result in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{result.accuracy:.3f}",
            'Balanced Acc': f"{result.balanced_accuracy:.3f}",
            'Precision': f"{result.precision:.3f}",
            'Recall': f"{result.recall:.3f}",
            'F1-Score': f"{result.f1_score:.3f}",
            'AUC-ROC': f"{result.auc_roc:.3f}",
            'EO Disparity': f"{result.equalized_odds_diff:.3f}",
            'DP Disparity': f"{result.demographic_parity_diff:.3f}"
        })
    
    return pd.DataFrame(comparison_data)


def plot_selection_rate_comparison(results_dict, save_path=None):
    """
    Create a grouped bar chart comparing selection rates across models and groups.
    
    Args:
        results_dict: Dict mapping model names to EvaluationResult objects
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    
    selection_df = pd.DataFrame({
        name: result.selection_rates 
        for name, result in results_dict.items()
    })
    
    fig, ax = plt.subplots(figsize=(14, 6))
    selection_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
    
    ax.set_title('Selection Rate Comparison Across Models and Groups', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Demographic Group', fontsize=12)
    ax.set_ylabel('Selection Rate', fontsize=12)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig