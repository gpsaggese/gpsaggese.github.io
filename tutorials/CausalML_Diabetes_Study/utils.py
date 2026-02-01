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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Optional, Tuple, Union

# CausalML Imports
from causalml.inference.meta import BaseXRegressor, BaseTRegressor, BaseSRegressor, BaseRRegressor, BaseDRRegressor
from causalml.match import NearestNeighborMatch
from causalml.metrics import plot_gain, auuc_score 

# Machine Learning Imports (Base Learners)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set generic style for plots
sns.set_theme(style="whitegrid")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# #############################################################################
# Data Loading & Specific Preprocessing 
# #############################################################################
def load_cdc_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CDC Diabetes Health Indicators dataset from a local CSV.
    Performs basic cleaning (duplicate removal, type casting) for CausalML.
    
    Args:
        filepath (str): Relative path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for processing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Could not find file at: {filepath}")
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")
    # Ensure all data is float for XGBoost compatibility
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
        logger.info("Calculating Propensity Scores for Overlap Check...")
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
        logger.info("--- Diagnostic Interpretation ---")
        logger.info("Good Overlap: The red and blue distributions share the same x-axis range.")
        logger.info("Bad Overlap: One group is clustered at 0 and the other at 1 (Positivity Violation).\n")

    def fit_estimate(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """
        Fits the meta-learner and estimates the Conditional Average Treatment Effect (CATE).
        
        Args:
            X (pd.DataFrame): Covariates.
            T (pd.Series): Treatment assignment.
            Y (pd.Series): Outcome.
        """
        self.feature_names = X.columns.tolist()
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            logger.info(f"Mapping labels: 0.0 -> {self.control_name}, 1.0 -> {self.treatment_name}")
            map_dict = {0: self.control_name, 1: self.treatment_name, 
                        0.0: self.control_name, 1.0: self.treatment_name}
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        logger.info(f"Training {self.learner_type}-Learner with XGBoost base models...")
        # fit_predict returns the CATE (difference in potential outcomes)
        # Note: causalml's API varies slightly by version; fit_predict is standard for BaseX/T/S.
        cate = self.learner.fit_predict(X=X, treatment=T_proc, y=Y)
        # Handle shape differences (sometimes returns 2D array)
        if cate.ndim > 1 and cate.shape[1] == 1:
            cate = cate.flatten()  
        self.cate_estimates = cate
        ate = cate.mean()
        logger.info(f"Done. Estimated Average Treatment Effect (ATE): {ate:.4f}")
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

    def run_placebo_test(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series, n_simulations: int = 10):
        """
        Validates the model by randomizing the Treatment vector.
        
        Hypothesis: If T is random, the estimated ATE should be ~0.
        If the model finds a strong effect on random data, the original result is suspect.
        
        Args:
            n_simulations (int): How many times to shuffle and retrain. 
                                 Keep low (e.g. 5-10) for speed on large data.
        """
        logger.info(f"Running Placebo Test ({n_simulations} permutations)...")    
        if self.cate_estimates is None:
            raise ValueError("Run fit_estimate() first to establish a baseline.")    
        original_ate = self.cate_estimates.mean()
        placebo_ates = []
        # Handle label mapping once for consistency
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            map_dict = {0: self.control_name, 1: self.treatment_name, 
                        0.0: self.control_name, 1.0: self.treatment_name}
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        for i in range(n_simulations):
            # Shuffle Treatment (break the causal link)
            T_shuffled = T_proc.sample(frac=1, random_state=i).reset_index(drop=True)
            T_shuffled.index = X.index # Align indices
            # We create a fresh instance to avoid side effects
            if self.learner_type == 'X':
                temp_learner = BaseXRegressor(learner=self.model_y, control_name=self.control_name)
            elif self.learner_type == 'T':
                temp_learner = BaseTRegressor(learner=self.model_y, control_name=self.control_name)
            else:
                temp_learner = BaseSRegressor(learner=self.model_y, control_name=self.control_name)    
            # Estimate Pseudo-Effect
            cate_placebo = temp_learner.fit_predict(X=X, treatment=T_shuffled, y=Y)
            if cate_placebo.ndim > 1: cate_placebo = cate_placebo.flatten()
            placebo_ates.append(cate_placebo.mean())
            logger.info(f"   Sim {i+1}/{n_simulations}: Placebo ATE = {cate_placebo.mean():.5f}")
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(placebo_ates, color='grey', kde=True, label='Placebo Estimates (Random T)')
        plt.axvline(original_ate, color='red', linestyle='--', linewidth=3, label=f'Actual Estimate ({original_ate:.4f})')
        plt.axvline(0, color='black', linestyle='-', linewidth=1)
        plt.title('Placebo Test: Actual Effect vs. Random Noise')
        plt.xlabel('Estimated ATE')
        plt.legend()
        plt.show()

    def run_sensitivity_analysis(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """
        Performs a Sensitivity Analysis by iteratively removing covariates.
        
        Purpose: Tests model stability. If removing a single confounder (e.g., 'Age') 
        drastically changes the ATE, the result is highly sensitive to that variable.
        If the ATE remains stable across removals, the causal finding is robust.
        """
        logger.info("Running Sensitivity Analysis (Covariate Removal)...")
        # Establish Baseline
        if self.cate_estimates is None:
            self.fit_estimate(X, T, Y)
        baseline_ate = self.cate_estimates.mean()
        logger.info(f"   Baseline ATE: {baseline_ate:.5f}")
        sensitivity_results = {}
        # Iterate through covariates
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            map_dict = {0: self.control_name, 1: self.treatment_name, 
                        0.0: self.control_name, 1.0: self.treatment_name}
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        # We create a new learner instance for each iteration to ensure a clean state
        for feature in X.columns:
            logger.info(f"   ... testing robustness without '{feature}'")
            X_drop = X.drop(columns=[feature])        
            # Re-initialize learner (same type as original)
            if self.learner_type == 'X':
                temp_learner = BaseXRegressor(learner=self.model_y, control_name=self.control_name)
            elif self.learner_type == 'T':
                temp_learner = BaseTRegressor(learner=self.model_y, control_name=self.control_name)
            else:
                temp_learner = BaseSRegressor(learner=self.model_y, control_name=self.control_name)
            # Estimate
            cate_drop = temp_learner.fit_predict(X=X_drop, treatment=T_proc, y=Y)
            if cate_drop.ndim > 1: cate_drop = cate_drop.flatten()            
            sensitivity_results[feature] = cate_drop.mean()
        # Visualization
        # Convert to DF for plotting
        sens_df = pd.DataFrame.from_dict(sensitivity_results, orient='index', columns=['ATE'])
        sens_df = sens_df.sort_values(by='ATE')
        plt.figure(figsize=(10, 8))
        # Plot bars
        sns.barplot(x=sens_df['ATE'], y=sens_df.index, palette='viridis')
        # Add baseline reference line
        plt.axvline(baseline_ate, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_ate:.4f})')
        plt.axvline(0, color='black', linestyle='-', linewidth=1)
        plt.title('Sensitivity Analysis: ATE Stability upon Covariate Removal')
        plt.xlabel('Estimated ATE')
        plt.ylabel('Removed Covariate')
        plt.legend()
        plt.tight_layout()
        plt.show()
        logger.info("Interpretation: Variables causing large shifts from the red line are critical confounders.")

    def compare_estimators(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        """
        Runs a 'Horse Race' tournament between S, T, X, R, and DR Learners.
        
        Since we lack ground truth CATE, we use the 'Gain' (Uplift) Curve on a held-out test set.
        The model with the highest area under the curve (AUUC) is best at ranking 
        patients from 'highest benefit' to 'lowest benefit'.
        """
        logger.info("Starting Estimator Tournament...")
        # Split Data 
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
            X, T, Y, test_size=0.3, random_state=42
        )
        # Define Candidates (Using XGBoost for consistency)
        # Note: R and DR learners are more sensitive to hyperparams, but we use defaults.
        learners = {
            'S-Learner': BaseSRegressor(learner=self.model_y),
            'T-Learner': BaseTRegressor(learner=self.model_y),
            'X-Learner': BaseXRegressor(learner=self.model_y),
            'R-Learner': BaseRRegressor(learner=self.model_y),
            'DR-Learner': BaseDRRegressor(learner=self.model_y)
        }
        pred_results = pd.DataFrame()
        # Train and Predict loop
        for name, model in learners.items():
            logger.info(f"Training {name}...")
            # We assume numeric T (0/1) which we ensured in preprocessing
            try:
                # Fit on Train
                model.fit(X=X_train, treatment=T_train, y=y_train) 
                # Predict on Test (CATE)
                cate_test = model.predict(X=X_test)
                # Standardize dimensions (some return 2D arrays)
                if cate_test.ndim > 1: cate_test = cate_test.flatten()
                pred_results[name] = cate_test
            except Exception as e:
                logger.error(f"{name} failed: {e}")
        # Evaluate using Cumulative Gain (Qini Curve)
        logger.info("Generating Uplift Curves (Metrics on Test Set)...")
        # plot_gain expects a DataFrame containing the predictions, outcome, and treatment
        df_preds = pred_results.copy()
        df_preds['y'] = y_test.values
        df_preds['t'] = T_test.values
        # Calculate AUUC Score
        auuc = auuc_score(
            df_preds,
            outcome_col='y',
            treatment_col='t',
            normalize=True
        )
        # Display Table
        logger.info("--- Qini / AUUC Scores (Higher is Better) ---")
        logger.info("\n" + str(auuc.sort_values(ascending=False)))
        plot_gain(
            df_preds,
            outcome_col='y',
            treatment_col='t',
            normalize=True,
            random_seed=42,
            figsize=(10, 6)
        )
        plt.title("Estimator Comparison: Cumulative Gain (Uplift)")
        plt.show()
        logger.info("Interpretation: The line that stays highest on the Y-axis sorts patients best.")