# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import os
import seaborn as sns
import warnings

from utils import CausalNavigator, load_cdc_data, preprocess_for_causal

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# %%
filename = "diabetes_binary_health_indicators_BRFSS2015.csv"
DATA_PATH = os.path.join("data", "unprocessed", filename)
try:
    print(f"Attempting to load: {DATA_PATH}")
    df_raw = load_cdc_data(DATA_PATH)
    print(f"Data Loaded Successfully. Rows: {df_raw.shape[0]}")
except Exception as e:
    print(f"Error: {e}")
    print("Please ensure the full dataset is in the data/unprocessed/ folder.")

# %%
# Define Causal Targets
treatment_col = "PhysActivity"
outcome_col = "Diabetes_binary"
# Define Covariates (Confounders)
# We use a rich set of features to satisfy the "Unconfoundedness" assumption.
# We include 'BMI' to see the direct effect, acknowledging it might be a mediator.
covariate_cols = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
    "BMI",
]
df_clean, X, T, Y = preprocess_for_causal(
    df_raw, treatment_col, outcome_col, covariate_cols
)
print("--- Analysis Population ---")
print(f"Total N: {len(df_clean)}")
print(f"Treatment Rate (Active): {T.mean():.2%}")
print(f"Outcome Rate (Diabetes): {Y.mean():.2%}")

# %%
# Initialize our CausalNavigator with X-Learner
navigator = CausalNavigator(
    learner_type="X", control_name="Sedentary", treatment_name="Active"
)
# Check Common Support
print("Calculating Propensity Scores and checking overlap...")
navigator.check_overlap(X, T)
# Diagnostic Interpretation
# Good Overlap: The red and blue distributions share the same x-axis range.
# Bad Overlap: One group is clustered at 0 and the other at 1 (Positivity Violation).

# %%
print("Training X-Learner on full dataset...")
# Fit the model
cate_estimates = navigator.fit_estimate(X, T, Y)
df_results = navigator.get_cate_df(df_clean)
# Calculate Overall ATE
ate = cate_estimates.mean()
print("\nTraining Complete.")
print(f"Average Treatment Effect (ATE): {ate:.4f}")
print(
    f"Interpretation: Physical Activity reduces diabetes risk by {abs(ate) * 100:.2f}% on average."
)

# %%
# Question: Does the benefit of exercise increase as we age?
# Age 1 = 18-24, Age 13 = 80+
print("Analyzing Heterogeneity by Age...")
navigator.plot_heterogeneity(df_results, col="Age")

# %%
# Question: Is the effect consistent across Income levels?
# Income 1 = <$10k, Income 8 = >$75k
navigator.plot_heterogeneity(df_results, col="Income")

# %%
# Question: Who benefits more? People who already feel healthy (1-2) or those with poor health (4-5)?
# GenHlth: 1=Excellent, 5=Poor
navigator.plot_heterogeneity(df_results, col="GenHlth")

# %%
# Robustness Check using Placebo Test
# We challenge our model: "Could this result just be random noise?"
# We randomize who exercises (shuffling T) and see if the model still finds an effect.
# - If the Placebo bars are near 0, and our Red Line is far away => GOOD RESULT.
# - If the Red Line is inside the Grey bars => BAD RESULT (Model is overfitting).
# Note: We use n=5 to keep execution time reasonable
navigator.run_placebo_test(X, T, Y, n_simulations=5)

# %% [markdown]
# The actual estimate (-0.002) falls outside the placebo distribution, 
# indicating the result is not attributable to random chance. However, 
# the placebo distribution's slight positive bias (~+0.001) suggests 
# residual confounding or model instability, consistent with our 
# cross-sectional design limitations.

# %%
# Sensitivity Analysis
# We test how dependent our result is on specific variables.
# If removing "GenHlth" makes the effect disappear, our finding depends entirely on that variable.
# If the effect remains negative (protective) regardless of which variable is dropped, the result is robust.
navigator.run_sensitivity_analysis(X, T, Y)

# %% [markdown]
# ### Sensitivity Analysis: Covariate Stability and Causal Pathways
#
# To assess model robustness, we iteratively removed individual covariates and re-estimated the Average Treatment Effect (ATE). Stable estimates across permutations imply that the result is not driven by a single spurious correlation.
#
# **1. Robustness of Direction**
# The estimated treatment effect remained consistently negative (protective) across all covariate removal scenarios. No single variable removal caused the effect to disappear or flip direction, indicating that the association between physical activity and reduced diabetes risk is robust to variable selection.
#
# **2. Identification of Critical Variables**
# While most variables (such as Education or Sex) showed minimal impact when removed, two variables exhibited strong suppression effects, substantially increasing the estimated protective effect when omitted:
# *   **BMI Removal:** Removing BMI resulted in the largest shift, increasing the magnitude of the protective effect by approximately 5-fold compared to the baseline.
# *   **GenHlth Removal:** Removing General Health caused the second-largest shift, roughly quadrupling the estimated effect size.
#
# **3. Structural Interpretation (Mediation vs. Confounding)**
# The dramatic increase in the protective effect upon removing BMI strongly suggests that BMI functions largely as a **mediator**.
# *   *Mechanism:* Physical activity reduces Diabetes risk *through* the mechanism of weight loss ($T \rightarrow BMI \rightarrow Y$).
# *   *Direct vs. Total Effect:* By controlling for BMI in our baseline model, we blocked this indirect pathway, isolating the conservative **Direct Effect** (metabolic benefits independent of weight). Removing BMI reveals the larger **Total Effect**.
#
# **4. Methodological Limitation**
# The high sensitivity to BMI and General Health underscores the "structural identifiability" challenge inherent in cross-sectional data. Without longitudinal temporal ordering, these variables occupy an ambiguous causal position:
# *   **As Confounders:** Poor baseline health leads to inactivity ($X \rightarrow T$).
# *   **As Mediators:** Inactivity leads to poor health ($T \rightarrow M$).
# *   **As Colliders:** Diabetes status and activity levels both influence reported health ($T \rightarrow C \leftarrow Y$).
#
# Our baseline model follows the conservative approach of controlling for these variables. This likely blocks some true causal pathways (mediation), meaning our reported ATE of -0.2% serves as a lower bound for the true causal effect.

# %%
# The "Horse Race"
# We chose X-Learner earlier. Was that the right choice?
# We now train S, T, X, R, and DR Learners and compare them using Uplift Curves.
# Since we don't know the "True" effect for any person, we test how well each model ranks users by responsiveness.
navigator.compare_estimators(X, T, Y)

# %% [markdown]
# The comparative evaluation of meta-learners reveals three distinct 
# performance tiers based on Qini coefficients and uplift curves.
#
# **Top-performing estimators (Qini: -0.468 to -0.481):**
# The T-Learner, X-Learner, and R-Learner exhibit statistically similar 
# performance, with Qini scores varying by less than 3%. The T-Learner 
# achieves the highest score (-0.481), followed closely by X-Learner 
# (-0.469) and R-Learner (-0.468). These methods share a common 
# theoretical foundation: separate modeling of treatment and control 
# response surfaces, which enables flexible estimation of heterogeneous 
# treatment effects.
#
# **Moderate performance (Qini: -0.449):**
# The DR-Learner demonstrates approximately 7% lower performance relative 
# to top-tier methods. While doubly robust estimators offer theoretical 
# protection against misspecification of either the propensity or outcome 
# model, this advantage requires at least one model to be correctly 
# specified. In observational settings with violated identification 
# assumptions, the dual-model dependency may amplify rather than mitigate 
# estimation error.
#
# **Poor performance (Qini: -0.362):**
# The S-Learner substantially underperforms, achieving 25% lower scores 
# than top methods. This result is consistent with known limitations of 
# single-model approaches: when treatment effects are small relative to 
# main effects of confounders, regularization techniques prioritize 
# fitting dominant predictors at the expense of subtle treatment 
# heterogeneity.
#
# **Model selection justification:**
# The X-Learner is retained as the primary estimator for this analysis. 
# While the T-Learner achieves marginally superior performance (ΔQini = 
# 0.012), the X-Learner's explicit propensity score weighting provides 
# more stable estimates under treatment imbalance, a key characteristic 
# of our data (74% treatment prevalence). The negligible performance 
# difference validates this methodological choice.
