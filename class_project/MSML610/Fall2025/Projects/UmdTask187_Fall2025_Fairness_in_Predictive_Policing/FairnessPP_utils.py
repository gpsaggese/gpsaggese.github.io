import pandas as pd
import numpy as np
import os
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# Native API imports
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import MetricFrame, equalized_odds_difference, selection_rate

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION OBJECTS ---
@dataclass
class ModelConfig:
    """Stable configuration for model training."""
    n_estimators: int = 50
    random_state: int = 42
    max_iter_mitigation: int = 50

@dataclass
class EvaluationResult:
    """Stable return type for evaluations."""
    accuracy: float
    balanced_accuracy: float
    fairness_disparity: float
    group_metrics: pd.DataFrame

# --- WRAPPER LAYER ---
class FairnessPredictor:
    """
    A lightweight wrapper around Scikit-Learn and Fairlearn.
    """
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.model = None
        self.is_mitigated = False

    def train(self, X, y, A=None, mitigate: bool = False):
        self.is_mitigated = mitigate
        
        # Native Estimator
        base_estimator = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )

        if mitigate:
            if A is None:
                raise ValueError("Sensitive attribute 'A' is required for mitigation.")
            print(f"Training Mitigated Model (Fairlearn ExponentiatedGradient)...")
            self.model = ExponentiatedGradient(
                estimator=base_estimator,
                constraints=EqualizedOdds(),
                max_iter=self.config.max_iter_mitigation
            )
            self.model.fit(X, y, sensitive_features=A)
        else:
            print("Training Baseline Model (Native GradientBoosting)...")
            self.model = base_estimator
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, A) -> EvaluationResult:
        y_pred = self.predict(X)
        
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        disp = equalized_odds_difference(y, y_pred, sensitive_features=A)
        
        mf = MetricFrame(
            metrics={"Selection Rate": selection_rate, "Accuracy": accuracy_score},
            y_true=y,
            y_pred=y_pred,
            sensitive_features=A
        )
        
        return EvaluationResult(
            accuracy=acc,
            balanced_accuracy=bal_acc,
            fairness_disparity=disp,
            group_metrics=mf.by_group
        )

# --- UTILITY FUNCTIONS ---
API_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

def load_chicago_data(local_cache_path="data/chicago_crime_2020_2023.csv") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Smart data loader that handles API fetching, caching, and feature engineering.
    """
    if os.path.exists(local_cache_path):
        print(f"Loading cached data from {local_cache_path}...")
        df = pd.read_csv(local_cache_path)
    else:
        print("Fetching multi-year data from Chicago API...")
        years = [2020, 2021, 2022, 2023] 
        all_data = []
        try:
            for year in years:
                url = f"{API_URL}?$limit=20000&year={year}&$order=date DESC"
                print(f" - Fetching {year}...")
                df_year = pd.read_json(url)
                all_data.append(df_year)
            df = pd.concat(all_data, ignore_index=True)
            os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
            df.to_csv(local_cache_path, index=False)
        except Exception as e:
            print(f"API Error: {e}. Generating Mock Data.")
            df = _generate_mock_data(2000)

    # Preprocessing
    df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 
                       'arrest': 'Arrest', 'domestic': 'Domestic', 'date': 'Date'}, inplace=True)
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature Engineering
    df['lat_bin'] = (df['Latitude'] / 0.005).astype(int)
    df['lon_bin'] = (df['Longitude'] / 0.005).astype(int)
    df['Grid_ID'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)

    # Simulated Demographics
    np.random.seed(42)
    regions = df['Grid_ID'].unique()
    region_demographics = {rid: {'Majority_Race': np.random.choice(['Black', 'White', 'Hispanic'], p=[0.4, 0.3, 0.3]),
                                 'Income_Level': np.random.choice(['Low', 'High'], p=[0.6, 0.4])} for rid in regions}
    
    df['Majority_Race'] = df['Grid_ID'].map(lambda x: region_demographics[x]['Majority_Race'])
    df['Income_Level'] = df['Grid_ID'].map(lambda x: region_demographics[x]['Income_Level'])
    df['Intersectional_Group'] = df['Majority_Race'] + "_" + df['Income_Level']

    X = df[['Latitude', 'Longitude', 'Domestic']] 
    y = df['Arrest'].astype(int) 
    A = df['Intersectional_Group']
    dates = df['Date']

    return X, y, A, dates

def _generate_mock_data(n):
    return pd.DataFrame({
        'latitude': np.random.uniform(41.6, 42.0, n),
        'longitude': np.random.uniform(-87.9, -87.5, n),
        'arrest': np.random.choice([True, False], n),
        'domestic': np.random.choice([True, False], n),
        'date': pd.date_range(start='1/1/2020', periods=n)
    })