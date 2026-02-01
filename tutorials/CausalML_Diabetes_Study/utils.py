"""
Import as:

import tutorials.CausalML_Diabetes_Study.utils as tcdistut
"""

import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
)
from causalml.metrics import auuc_score, plot_gain
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# Set generic style for plots.
sns.set_theme(style="whitegrid")


# #############################################################################
# Data Loading & Specific Preprocessing
# #############################################################################


def load_cdc_data(filepath: str) -> pd.DataFrame:
    """
    Load the CDC Diabetes Health Indicators dataset from a local CSV.

    Perform basic cleaning (duplicate removal, type casting) for CausalML.

    :param filepath: relative path to the CSV file
    :return: cleaned dataframe ready for processing
    """
    hdbg.dassert(os.path.exists(filepath), "File not found:", filepath)
    _LOG.info("Loading data from: %s", filepath)
    df = pd.read_csv(filepath)
    # Drop duplicates.
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        _LOG.info("Dropped %s duplicate rows", initial_len - len(df))
    # Ensure all data is float for XGBoost compatibility.
    df = df.astype(float)
    return df


def preprocess_for_causal(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare arrays X, T, Y for CausalML.

    :param df: the full dataframe
    :param treatment_col: name of the binary treatment column
    :param outcome_col: name of the outcome column
    :param covariate_cols: list of covariate names
    :return: tuple of (df_filtered, X, T, Y)
    """
    # Basic validation.
    hdbg.dassert_in(
        treatment_col, df.columns, "Treatment column not found:", treatment_col
    )
    hdbg.dassert_in(
        outcome_col, df.columns, "Outcome column not found:", outcome_col
    )
    # Filter data to ensure columns exist and drop NAs.
    keep_cols = covariate_cols + [treatment_col, outcome_col]
    df_clean = df[keep_cols].dropna().copy()
    # Extract vectors.
    X = df_clean[covariate_cols]
    T = df_clean[treatment_col]
    Y = df_clean[outcome_col]
    return df_clean, X, T, Y


# #############################################################################
# CausalNavigator
# #############################################################################


class CausalNavigator:
    """
    A unified interface for CausalML meta-learners.
    """

    def __init__(
        self,
        *,
        learner_type: str = "X",
        control_name: str = "Control",
        treatment_name: str = "Treatment",
    ) -> None:
        """
        Initialize the CausalNavigator.

        :param learner_type: 'S', 'T', or 'X', defaults to 'X' (X-Learner)
        :param control_name: label for T=0
        :param treatment_name: label for T=1
        """
        self.learner_type = learner_type.upper()
        hdbg.dassert_in(
            self.learner_type,
            ["S", "T", "X"],
            "learner_type must be 'S', 'T', or 'X'",
        )
        self.control_name = control_name
        self.treatment_name = treatment_name
        # Define base learners (using XGBoost for speed and performance).
        self.model_t = XGBClassifier(
            n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss"
        )
        self.model_y = XGBRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )
        # Initialize the CausalML meta-learner.
        if self.learner_type == "X":
            self.learner = BaseXRegressor(
                learner=self.model_y, control_name=control_name
            )
        elif self.learner_type == "T":
            self.learner = BaseTRegressor(
                learner=self.model_y, control_name=control_name
            )
        elif self.learner_type == "S":
            self.learner = BaseSRegressor(
                learner=self.model_y, control_name=control_name
            )
        self.cate_estimates = None
        self.feature_names = None

    def check_overlap(self, X: pd.DataFrame, T: pd.Series) -> None:
        """
        Diagnose the Common Support (Overlap) assumption.

        Calculate propensity scores and plot the distribution.

        :param X: covariates
        :param T: treatment vector
        """
        _LOG.info("Calculating Propensity Scores for Overlap Check")
        ps_model = XGBClassifier(
            n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42
        )
        ps_model.fit(X, T)
        p_scores = ps_model.predict_proba(X)[:, 1]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            p_scores[T == 0], shade=True, color="red", label=self.control_name
        )
        sns.kdeplot(
            p_scores[T == 1], shade=True, color="blue", label=self.treatment_name
        )
        plt.title("Propensity Score Overlap (Common Support Check)")
        plt.xlabel("Propensity Score (Probability of Treatment)")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        # Interpretation text.
        _LOG.info("--- Diagnostic Interpretation ---")
        _LOG.info(
            "Good Overlap: The red and blue distributions share the same x-axis range"
        )
        _LOG.info(
            "Bad Overlap: One group is clustered at 0 and the other at 1 (Positivity Violation)"
        )

    def fit_estimate(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> np.ndarray:
        """
        Fit the meta-learner and estimate the Conditional Average Treatment Effect.

        :param X: covariates
        :param T: treatment assignment
        :param Y: outcome
        :return: CATE estimates
        """
        self.feature_names = X.columns.tolist()
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            _LOG.info(
                "Mapping labels: 0.0 -> %s, 1.0 -> %s",
                self.control_name,
                self.treatment_name,
            )
            map_dict = {
                0.0: self.control_name,
                1.0: self.treatment_name,
            }
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        _LOG.info(
            "Training %s-Learner with XGBoost base models", self.learner_type
        )
        # fit_predict returns the CATE (difference in potential outcomes).
        # Note: causalml's API varies slightly by version; fit_predict is standard for BaseX/T/S.
        cate = self.learner.fit_predict(X=X, treatment=T_proc, y=Y)
        # Handle shape differences (sometimes returns 2D array).
        if cate.ndim > 1 and cate.shape[1] == 1:
            cate = cate.flatten()
        self.cate_estimates = cate
        ate = cate.mean()
        _LOG.info("Done. Estimated Average Treatment Effect (ATE): %.4f", ate)
        return cate

    def get_cate_df(self, df_original: pd.DataFrame) -> pd.DataFrame:
        """
        Return the original dataframe with an added 'CATE' column.

        :param df_original: original dataframe
        :return: dataframe with CATE column added
        """
        hdbg.dassert_is_not(
            self.cate_estimates, None, "Model not fitted. Run fit_estimate first"
        )
        df_out = df_original.copy()
        df_out["cate"] = self.cate_estimates
        return df_out

    def plot_heterogeneity(
        self, df_with_cate: pd.DataFrame, col: str, *, bins: int = 5
    ) -> None:
        """
        Plot the average treatment effect grouped by a specific feature.

        Useful for answering: "Who benefits most?"

        :param df_with_cate: dataframe containing 'cate' column
        :param col: the column to group by (e.g., 'Age', 'Income')
        :param bins: number of bins if the column is continuous
        """
        hdbg.dassert_in(col, df_with_cate.columns, "Column not found:", col)
        plt.figure(figsize=(10, 6))
        # Check if column is effectively continuous or categorical.
        unique_vals = df_with_cate[col].nunique()
        is_categorical = unique_vals < 15
        if is_categorical:
            # Bar plot for categorical.
            sns.barplot(
                x=col, y="cate", data=df_with_cate, ci=95, palette="viridis"
            )
        else:
            # Binning for continuous.
            df_with_cate[f"{col}_bin"] = pd.cut(df_with_cate[col], bins=bins)
            sns.barplot(
                x=f"{col}_bin",
                y="cate",
                data=df_with_cate,
                ci=95,
                palette="viridis",
            )
            plt.xticks(rotation=45)
        plt.title(f"Heterogeneous Treatment Effect by {col}")
        plt.ylabel("Estimated Treatment Effect (CATE)")
        plt.xlabel(col)
        # Add a reference line at ATE.
        ate = df_with_cate["cate"].mean()
        plt.axhline(
            ate, color="r", linestyle="--", label=f"Average Effect ({ate:.3f})"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_placebo_test(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
        *,
        n_simulations: int = 10,
    ) -> None:
        """
        Validate the model by randomizing the Treatment vector.

        Hypothesis: If T is random, the estimated ATE should be ~0.
        If the model finds a strong effect on random data, the original result is suspect.

        :param X: covariates
        :param T: treatment vector
        :param Y: outcome
        :param n_simulations: how many times to shuffle and retrain (keep low, e.g. 5-10)
        """
        _LOG.info("Running Placebo Test (%s permutations)", n_simulations)
        hdbg.dassert_is_not(
            self.cate_estimates,
            None,
            "Run fit_estimate() first to establish a baseline",
        )
        original_ate = self.cate_estimates.mean()
        placebo_ates = []
        # Handle label mapping once for consistency.
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            map_dict = {
                0.0: self.control_name,
                1.0: self.treatment_name,
            }
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        for i in range(n_simulations):
            # Shuffle treatment (break the causal link).
            T_shuffled = T_proc.sample(frac=1, random_state=i).reset_index(
                drop=True
            )
            T_shuffled.index = X.index  # Align indices.
            # We create a fresh instance to avoid side effects.
            if self.learner_type == "X":
                temp_learner = BaseXRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            elif self.learner_type == "T":
                temp_learner = BaseTRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            else:
                temp_learner = BaseSRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            # Estimate pseudo-effect.
            cate_placebo = temp_learner.fit_predict(
                X=X, treatment=T_shuffled, y=Y
            )
            if cate_placebo.ndim > 1:
                cate_placebo = cate_placebo.flatten()
            placebo_ates.append(cate_placebo.mean())
            _LOG.info(
                "   Sim %s/%s: Placebo ATE = %.5f",
                i + 1,
                n_simulations,
                cate_placebo.mean(),
            )
        # Visualization.
        plt.figure(figsize=(10, 6))
        sns.histplot(
            placebo_ates,
            color="grey",
            kde=True,
            label="Placebo Estimates (Random T)",
        )
        plt.axvline(
            original_ate,
            color="red",
            linestyle="--",
            linewidth=3,
            label=f"Actual Estimate ({original_ate:.4f})",
        )
        plt.axvline(0, color="black", linestyle="-", linewidth=1)
        plt.title("Placebo Test: Actual Effect vs. Random Noise")
        plt.xlabel("Estimated ATE")
        plt.legend()
        plt.show()

    def run_sensitivity_analysis(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> None:
        """
        Perform a Sensitivity Analysis by iteratively removing covariates.

        Purpose: Test model stability. If removing a single confounder (e.g., 'Age')
        drastically changes the ATE, the result is highly sensitive to that variable.
        If the ATE remains stable across removals, the causal finding is robust.

        :param X: covariates
        :param T: treatment vector
        :param Y: outcome
        """
        _LOG.info("Running Sensitivity Analysis (Covariate Removal)")
        # Establish baseline.
        if self.cate_estimates is None:
            self.fit_estimate(X, T, Y)
        baseline_ate = self.cate_estimates.mean()
        _LOG.info("   Baseline ATE: %.5f", baseline_ate)
        sensitivity_results = {}
        # Iterate through covariates.
        T_proc = T.copy()
        unique_vals = sorted(T_proc.unique())
        if self.control_name not in unique_vals:
            map_dict = {
                0.0: self.control_name,
                1.0: self.treatment_name,
            }
            T_proc = T_proc.map(lambda x: map_dict.get(x, x))
        # We create a new learner instance for each iteration to ensure a clean state.
        for feature in X.columns:
            _LOG.info("   ... testing robustness without '%s'", feature)
            X_drop = X.drop(columns=[feature])
            # Re-initialize learner (same type as original).
            if self.learner_type == "X":
                temp_learner = BaseXRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            elif self.learner_type == "T":
                temp_learner = BaseTRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            else:
                temp_learner = BaseSRegressor(
                    learner=self.model_y, control_name=self.control_name
                )
            # Estimate.
            cate_drop = temp_learner.fit_predict(X=X_drop, treatment=T_proc, y=Y)
            if cate_drop.ndim > 1:
                cate_drop = cate_drop.flatten()
            sensitivity_results[feature] = cate_drop.mean()
        # Visualization.
        # Convert to DF for plotting.
        sens_df = pd.DataFrame.from_dict(
            sensitivity_results, orient="index", columns=["ATE"]
        )
        sens_df = sens_df.sort_values(by="ATE")
        plt.figure(figsize=(10, 8))
        # Plot bars.
        sns.barplot(x=sens_df["ATE"], y=sens_df.index, palette="viridis")
        # Add baseline reference line.
        plt.axvline(
            baseline_ate,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Baseline ({baseline_ate:.4f})",
        )
        plt.axvline(0, color="black", linestyle="-", linewidth=1)
        plt.title("Sensitivity Analysis: ATE Stability upon Covariate Removal")
        plt.xlabel("Estimated ATE")
        plt.ylabel("Removed Covariate")
        plt.legend()
        plt.tight_layout()
        plt.show()
        _LOG.info(
            "Interpretation: Variables causing large shifts from the red line are critical confounders"
        )

    def compare_estimators(
        self, X: pd.DataFrame, T: pd.Series, Y: pd.Series
    ) -> None:
        """
        Run a 'Horse Race' tournament between S, T, X, R, and DR Learners.

        Since we lack ground truth CATE, we use the 'Gain' (Uplift) Curve on a held-out test set.
        The model with the highest area under the curve (AUUC) is best at ranking
        patients from 'highest benefit' to 'lowest benefit'.

        :param X: covariates
        :param T: treatment vector
        :param Y: outcome
        """
        _LOG.info("Starting Estimator Tournament")
        # Split data.
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
            X, T, Y, test_size=0.3, random_state=42
        )
        # Define candidates (using XGBoost for consistency).
        # Note: R and DR learners are more sensitive to hyperparams, but we use defaults.
        learners = {
            "S-Learner": BaseSRegressor(learner=self.model_y),
            "T-Learner": BaseTRegressor(learner=self.model_y),
            "X-Learner": BaseXRegressor(learner=self.model_y),
            "R-Learner": BaseRRegressor(learner=self.model_y),
            "DR-Learner": BaseDRRegressor(learner=self.model_y),
        }
        pred_results = pd.DataFrame()
        # Train and predict loop.
        for name, model in learners.items():
            _LOG.info("Training %s", name)
            # We assume numeric T (0/1) which we ensured in preprocessing.
            try:
                # Fit on train.
                model.fit(X=X_train, treatment=T_train, y=y_train)
                # Predict on test (CATE).
                cate_test = model.predict(X=X_test)
                # Standardize dimensions (some return 2D arrays).
                if cate_test.ndim > 1:
                    cate_test = cate_test.flatten()
                pred_results[name] = cate_test
            except Exception as e:
                _LOG.error("%s failed: %s", name, str(e))
        # Evaluate using Cumulative Gain (Qini Curve).
        _LOG.info("Generating Uplift Curves (Metrics on Test Set)")
        # plot_gain expects a DataFrame containing the predictions, outcome, and treatment.
        df_preds = pred_results.copy()
        df_preds["y"] = y_test.values
        df_preds["t"] = T_test.values
        # Calculate AUUC score.
        auuc = auuc_score(
            df_preds, outcome_col="y", treatment_col="t", normalize=True
        )
        # Display table.
        _LOG.info("--- Qini / AUUC Scores (Higher is Better) ---")
        _LOG.info("\n%s", str(auuc.sort_values(ascending=False)))
        plot_gain(
            df_preds,
            outcome_col="y",
            treatment_col="t",
            normalize=True,
            random_seed=42,
            figsize=(10, 6),
        )
        plt.title("Estimator Comparison: Cumulative Gain (Uplift)")
        plt.show()
        _LOG.info(
            "Interpretation: The line that stays highest on the Y-axis sorts patients best"
        )
