"""
causal_impact_utils.py

A utility module for estimating causal effects using CausalML.
This module provides wrappers around Meta-Learners (T-Learner, X-Learner)
and handles data preprocessing, propensity checks, and heterogeneity analysis.

Classes:
    CausalNavigator: Main wrapper for the CausalML workflow.

Functions:
    load_cdc_data: Specific helper to load and clean the CDC Diabetes dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union

# CausalML Imports
from causalml.inference.meta import BaseXRegressor, BaseTRegressor, BaseSRegressor
from causalml.match import NearestNeighborMatch

# Machine Learning Imports (Base Learners)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set generic style for plots
sns.set_theme(style="whitegrid")


# #############################################################################
# Data Loading & Specific Preprocessing 
# #############################################################################
def load_cdc_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CDC Diabetes Health Indicators dataset and performs 
    initial cleaning.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {filepath}. Please check the path.")
    # Drop duplicates if any
    df = df.drop_duplicates()
    # ensure float types for consistency with XGBoost
    df = df.astype(float)
    return df

def preprocess_for_causal(df: pd.DataFrame, 
                          treatment_col: str, 
                          outcome_col: str, 
                          covariate_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares arrays X, T, Y for CausalML.
    
    Args:
        df (pd.DataFrame): The full dataframe.
        treatment_col (str): Name of the binary treatment column.
        outcome_col (str): Name of the outcome column.
        covariate_cols (List[str]): List of covariate names.
        
    Returns:
        tuple: (df_filtered, X, T, Y)
    """
    # Basic Validation
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment col '{treatment_col}' not found.")
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome col '{outcome_col}' not found.")
    # Filter data to ensure columns exist and drop NAs
    keep_cols = covariate_cols + [treatment_col, outcome_col]
    df_clean = df[keep_cols].dropna().copy()
    # Extract Vectors
    X = df_clean[covariate_cols]
    T = df_clean[treatment_col]
    Y = df_clean[outcome_col]
    return df_clean, X, T, Y

# #############################################################################
#  API Wrapper Class
# #############################################################################
class CausalNavigator:
    """
    A unified interface for CausalML meta-learners.
    
    This class wraps the complexity of:
    1. Propensity Score Estimation (for overlap checks)
    2. Meta-Learner Initialization (S-Learner, T-Learner, X-Learner)
    3. CATE Estimation
    4. Visualization of Heterogeneity
    """
    def __init__(self, 
                 learner_type: str = 'X', 
                 control_name: str = 'Control', 
                 treatment_name: str = 'Treatment'):
        """
        Initialize the CausalNavigator.

        Args:
            learner_type (str): 'S', 'T', or 'X'. Defaults to 'X' (X-Learner).
            control_name (str): Label for T=0.
            treatment_name (str): Label for T=1.
        """
        self.learner_type = learner_type.upper()
        self.control_name = control_name
        self.treatment_name = treatment_name
        # Define Base Learners (Using XGBoost for speed and performance)
        self.model_t = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss')
        self.model_y = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        
        # Initialize the CausalML Meta-Learner
        if self.learner_type == 'X':
            self.learner = BaseXRegressor(learner=self.model_y, control_name=control_name)
        elif self.learner_type == 'T':
            self.learner = BaseTRegressor(learner=self.model_y, control_name=control_name)
        elif self.learner_type == 'S':
            self.learner = BaseSRegressor(learner=self.model_y, control_name=control_name)
        else:
            raise ValueError("learner_type must be 'S', 'T', or 'X'")
            
        self.cate_estimates = None
        self.feature_names = None
        
    def check_overlap(self, X: pd.DataFrame, T: pd.Series):
        """
        Diagnose the Common Support (Overlap) assumption.
        Calculates propensity scores and plots the distribution.
        
        Args:
            X (pd.DataFrame): Covariates.
            T (pd.Series): Treatment vector.
        """
        print("Calculating Propensity Scores for Overlap Check...")
        ps_model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42)
        ps_model.fit(X, T)
        p_scores = ps_model.predict_proba(X)[:, 1]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(p_scores[T == 0], shade=True, color='red', label=self.control_name)
        sns.kdeplot(p_scores[T == 1], shade=True, color='blue', label=self.treatment_name)
        plt.title('Propensity Score Overlap (Common Support Check)')
        plt.xlabel('Propensity Score (Probability of Treatment)')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        # Interpretation text
        print("\n--- Diagnostic Interpretation ---")
        print("Good Overlap: The red and blue distributions share the same x-axis range.")
        print("Bad Overlap: One group is clustered at 0 and the other at 1 (Positivity Violation).\n")

    def fit_estimate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """
        Fits the meta-learner and estimates the Conditional Average Treatment Effect (CATE).
        
        Args:
            X (pd.DataFrame): Covariates.
            T (pd.Series): Treatment assignment.
            Y (pd.Series): Outcome.
        """
        self.feature_names = X.columns.tolist()
        print(f"Training {self.learner_type}-Learner with XGBoost base models...")
        # fit_predict returns the CATE (difference in potential outcomes)
        # Note: causalml's API varies slightly by version; fit_predict is standard for BaseX/T/S.
        cate = self.learner.fit_predict(X=X, treatment=T, y=Y)
        # Handle shape differences (sometimes returns 2D array)
        if cate.ndim > 1 and cate.shape[1] == 1:
            cate = cate.flatten()  
        self.cate_estimates = cate
        ate = cate.mean()
        print(f"Done. Estimated Average Treatment Effect (ATE): {ate:.4f}")
        return cate

    def get_cate_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the original dataframe with an added 'CATE' column.
        """
        if self.cate_estimates is None:
            raise ValueError("Model not fitted. Run fit_estimate first.") 
        df_out = df_original.copy()
        df_out['cate'] = self.cate_estimates
        return df_out

    def plot_heterogeneity(self, df_with_cate: pd.DataFrame, col: str, bins: int = 5):
        """
        Plots the average treatment effect grouped by a specific feature.
        Useful for answering: "Who benefits most?"
        
        Args:
            df_with_cate (pd.DataFrame): DF containing 'cate' column.
            col (str): The column to group by (e.g., 'Age', 'Income').
            bins (int): Number of bins if the column is continuous.
        """
        if col not in df_with_cate.columns:
            raise ValueError(f"Column {col} not found.")            
        plt.figure(figsize=(10, 6))
        # Check if column is effectively continuous or categorical
        unique_vals = df_with_cate[col].nunique()
        is_categorical = unique_vals < 15
        if is_categorical:
            # Bar plot for categorical
            sns.barplot(x=col, y='cate', data=df_with_cate, ci=95, palette="viridis")
        else:
            # Binning for continuous
            df_with_cate[f'{col}_bin'] = pd.cut(df_with_cate[col], bins=bins)
            sns.barplot(x=f'{col}_bin', y='cate', data=df_with_cate, ci=95, palette="viridis")
            plt.xticks(rotation=45)            
        plt.title(f'Heterogeneous Treatment Effect by {col}')
        plt.ylabel('Estimated Treatment Effect (CATE)')
        plt.xlabel(col)        
        # Add a reference line at ATE
        ate = df_with_cate['cate'].mean()
        plt.axhline(ate, color='r', linestyle='--', label=f'Average Effect ({ate:.3f})')
        plt.legend()
        plt.tight_layout()
        plt.show()