# Explainable Credit Scoring with XGBoost and SHAP: A Walkthrough of SHAP_Credit.example.ipynb

## Overview

The SHAP_Credit.example.ipynb notebook is a complete demonstration of credit scoring with explainability. It's a comprehensive ML project that shows you how to build an XGBoost credit risk model using the German Credit dataset. The notebook covers everything from handling class imbalance to tuning hyperparameters, calibrating probabilities, and explaining every prediction with SHAP (SHapley Additive exPlanations).

This notebook is designed for both beginners and intermediate ML practitioners. It combines gradient boosting, proper model evaluation, and modern explainability techniques. The goal is to show how banks can make lending decisions that are both accurate and transparent. It builds on the ideas from shap_credit_API.ipynb and serves as the main resource for understanding production-ready credit scoring systems.

## Purpose

The SHAP_Credit.example.ipynb notebook aims to showcase a complete, production-quality credit scoring pipeline. Here are the key objectives:

- Build baseline models (Logistic Regression and XGBoost) to establish performance benchmarks.
- Tune XGBoost hyperparameters using GridSearchCV with cross-validation for near-optimal performance.
- Handle the roughly 70-30 class imbalance using scale_pos_weight so the model pays more attention to risky borrowers.
- Find decision thresholds based on business costs (approving bad loans is more expensive than rejecting good applicants).
- Calibrate probability predictions so they are trustworthy for business decisions, not just good for ranking.
- Explain model decisions globally (which features matter most) and locally (why this specific loan was approved or denied) using SHAP.
- Perform sensitivity analysis to show borrowers and risk teams how changing key features affects approval chances.

This document includes detailed explanations, beginner-friendly insights, troubleshooting tips, and best practices for building explainable ML systems in regulated industries.

## Key conventions used in this notebook

### Labels and the positive class

This notebook uses a binary target with these meanings:

- y = 0 means Good (repaid).
- y = 1 means Bad (defaulted).

When you see `predict_proba(X)[:, 1]`, it means probability of the positive class, which here is P(y=1), the probability a loan is Bad (default risk).

### Decision rule we use

We use a simple decision rule:

- If P(Bad) is greater than or equal to threshold, predict Bad.
- Else, predict Good.

If you map "predict Good" to "approve", then:

- False positive (FP): predicted Bad but actually Good. This means we rejected a good applicant.
- False negative (FN): predicted Good but actually Bad. This means we approved a bad loan.

This mapping is crucial in the cost-based threshold step.

## Notebook Structure

The notebook is organized into seventeen steps, grouped into five major sections.

## Quick API reference used in this notebook

### XGBoost (xgboost.XGBClassifier)

We use `xgboost.XGBClassifier` for binary classification.

Common calls in this notebook:

- `xgb_model = xgb.XGBClassifier(**params)` creates a gradient boosted tree model.

- `xgb_model.fit(X_train, y_train)` trains the model on labeled data.

- `y_proba = xgb_model.predict_proba(X_test)[:, 1]` returns P(y=1) for each row. In this project, y=1 is Bad (default risk).

- `y_pred = (y_proba >= threshold).astype(int)` turns probabilities into class predictions using a decision threshold.

Class imbalance call used later:

- `scale_pos_weight = neg_count / pos_count` changes how much the model cares about class y=1 during training. In this project, y=1 is Bad, so `scale_pos_weight` greater than 1 upweights the Bad class and makes the model pay more attention to catching defaults.

### SHAP (shap.TreeExplainer)

We use SHAP to explain a trained tree model.

Common calls in this notebook:

- `explainer = shap.TreeExplainer(model)` builds an explainer for a tree-based model.

- `shap_values = explainer.shap_values(X)` computes per-feature contributions for each row in X.

Binary classification note: Sometimes `shap_values` is returned as a list with one array per class. If that happens, the array at index 1 corresponds to class y=1 (Bad).

## Section 1: Setup and Baseline Models (Steps 1 to 5)

### Step 1: Imports, Configuration, and Data Loading

Purpose: Set up the environment and load the German Credit dataset.

Process: Imports necessary libraries (pandas, scikit-learn, xgboost, shap, matplotlib), loads configuration from the credit_scoring_shap package, and loads both raw and preprocessed data.

Output:

- Raw data shape: (1000, 21)
- Encoded training data shape: (800, 61)
- 200 rows are held out for testing.
- The positive class is explicitly set to Bad (1).
- Train and test Bad rates are both roughly 0.30, so roughly 70% Good versus 30% Bad.

Insights:

- The jump from 21 to 61 features is due to one-hot encoding of categorical variables (checking account status, savings, housing, purpose, etc.).
- The 70-30 imbalance will challenge naive models that tend to predict the majority Good class.

### Step 2: Logistic Regression Baseline Training

Purpose: Train a simple logistic regression model as a strong baseline. Any complex model must at least match this.

Process: Fits `LogisticRegression(max_iter=1000, solver="liblinear")` on the training set, predicts probabilities on the test set, and evaluates using ROC AUC, confusion matrix, and classification report.

Output:

- Test ROC AUC: roughly 0.804
- Confusion matrix at threshold 0.5:
  - 124 true Good correctly predicted Good (TN)
  - 16 Good predicted Bad (FP)
  - 28 Bad predicted Good (FN)
  - 32 Bad predicted Bad (TP)

Insights:

- Accuracy is roughly 0.78.
- Recall (Good) is roughly 0.89. The model approves most Good borrowers.
- Recall (Bad) is roughly 0.53. It only catches about half of the risky borrowers.
- This tells us the problem is solvable, but there is room to improve the detection of Bad loans.

### Step 3: Logistic Regression Visualizations

Purpose: Generate publication-quality plots for the baseline model to visualize performance.

Process: Uses helper functions from the evaluation module to create confusion matrix heatmap, ROC curve, and Precision-Recall (PR) curve. It also saves metrics to a text file.

Output:

- Heatmap showing the confusion matrix above.
- ROC curve with AUC roughly 0.80 and corresponding PR curve.

Insights:

- The heatmap makes it obvious where errors occur (especially the 28 false negatives, which are Bad loans approved as Good).
- ROC shows decent separation from random guessing. The PR curve is especially informative given the class imbalance.

### Step 4: Baseline XGBoost Training

Purpose: Upgrade to XGBoost, a state-of-the-art gradient boosting algorithm, but without tuning yet.

Process:

- Builds an XGBoost classifier with default hyperparameters from the config (e.g. learning_rate, max_depth, n_estimators).
- Trains on `X_train, y_train`.
- Evaluates on `X_test`.

Output:

- Test ROC AUC: roughly 0.80.
- Confusion matrix at threshold 0.5:
  - 127 true Good correctly predicted Good (TN)
  - 13 Good predicted Bad (FP)
  - 33 Bad predicted Good (FN)
  - 27 Bad predicted Bad (TP)

Insights:

- Untuned XGBoost reaches a similar AUC to logistic regression but shifts the error mix:
  - It correctly identifies slightly more Good borrowers (higher Good recall),
  - But misses more Bad borrowers (lower Bad recall).
- This is common: a complex model can behave differently at a given threshold even when overall ranking power (AUC) is similar.

### Step 5: Baseline XGBoost Visualizations

Purpose: Visualize baseline XGBoost performance for comparison.

Process: Generates confusion matrix, ROC, and PR plots for the baseline XGBoost model.

Output: Plots look broadly similar to those from logistic regression, but with the different confusion matrix above.

Insights: Side-by-side comparison confirms that before tuning, XGBoost does not provide a clear advantage over the linear model in discrimination, but it does change how errors are distributed across Good and Bad borrowers.

## Section 2: Hyperparameter Tuning (Steps 6 to 9)

### Step 6: GridSearchCV for Hyperparameter Tuning

Purpose: Systematically search for better XGBoost hyperparameters using cross-validation.

Process:

- Defines a parameter grid with 72 combinations, for example:
  - max_depth: [3, 4, 5]
  - learning_rate: [0.01, 0.05, 0.1]
  - n_estimators: [200, 400]
  - subsample: [0.8, 1.0]
  - colsample_bytree: [0.8, 1.0]
- Uses 5-fold stratified CV with ROC AUC as the scoring metric.
- Total fits: 72 times 5 equals 360 models.

Output:

- Log message `"Fitting 5 folds for each of 72 candidates, totalling 360 fits"`.

Insights:

- Stratified folds keep the roughly 70-30 Good/Bad split in each fold.
- `n_jobs = -1` parallelizes the search across CPU cores.
- The grid is small enough to be feasible on a laptop, but rich enough to reveal useful patterns.

### Step 7: Best Parameters from Cross-Validation

Purpose: Identify which hyperparameter combination performed best.

Process: Reads `grid_cv.best_score_` and `grid_cv.best_params_`.

Output:

- Best CV AUC roughly 0.8009.
- Best parameters, for example:
  - colsample_bytree: 0.8
  - learning_rate: 0.01
  - max_depth: 5
  - n_estimators: 400
  - subsample: 0.8
  - plus fixed params like objective, eval_metric, etc.

Insights:

- A relatively deep but heavily regularized model (depth=5 with subsampling and low learning rate) gives the best CV performance.
- CV AUC is similar to the baseline XGBoost AUC, but we now have a principled, data-driven set of hyperparameters.

### Step 8: Evaluate Tuned Model on Test Set

Purpose: Check tuned model performance on truly unseen data.

Process: Uses `grid_cv.best_estimator_` to predict on `X_test` and computes AUC and confusion matrix.

Output:

- Test ROC AUC: roughly 0.79.
- Confusion matrix (threshold 0.5):
  - 122 true Good correctly predicted Good (TN)
  - 18 Good predicted Bad (FP)
  - 31 Bad predicted Good (FN)
  - 29 Bad predicted Bad (TP)

Insights:

- AUC is in the same ballpark as the baselines.
- Compared with the baseline XGBoost [[127,13],[33,27]]:
  - False positives increase (13 → 18), so more Good borrowers are rejected.
  - False negatives decrease slightly (33 → 31), so a few more Bad borrowers are caught.
- Hyperparameter tuning mainly shifts the error trade-off at threshold 0.5 instead of dramatically boosting ranking quality.

### Step 9: Tuned Model Visualizations

Purpose: Visualize changes introduced by tuning.

Process: Recreates confusion matrix, ROC, and PR plots for the tuned model.

Output: Updated diagnostic plots.

Insights:

- Compared with the baseline XGBoost plots, you can see fewer Bad borrowers being missed but more Good borrowers being flagged as Bad at the default 0.5 threshold.
- This motivates explicit handling of class imbalance to better control the trade-off.

## Section 3: Handling Class Imbalance (Steps 10 to 12)

### Step 10: Applying scale_pos_weight for Class Balance

Purpose: Counteract bias toward the majority Good class.

Process:

- Counts training labels:
  - 560 negatives (Good)
  - 240 positives (Bad)
- Computes `scale_pos_weight = 560 / 240` which is roughly 2.333.
- Copies the best parameters from GridSearchCV, adds `scale_pos_weight = 2.333`, and trains a balanced tuned XGBoost model.

Output:

- Test ROC AUC: roughly 0.80.
- Confusion matrix at threshold 0.5:
  - 113 true Good correctly predicted Good (TN)
  - 27 Good predicted Bad (FP)
  - 20 Bad predicted Good (FN)
  - 40 Bad predicted Bad (TP)

Insights:

- Compared with the tuned model without class weighting [[122,18],[31,29]]:
  - False negatives (Bad predicted as Good) drop from 31 to 20. The model catches more risky borrowers.
  - False positives (Good predicted as Bad) rise from 18 to 27. More safe borrowers are rejected.
- Overall AUC remains high (around 0.80). The main change is in who the model flags as risky.
- This is often a good trade-off in credit risk: avoiding approvals of clearly bad loans is worth reviewing more borderline good ones.

### Step 11: Balanced Model Visualizations

Purpose: Visualize how balancing changes the confusion matrix and curves.

Process: Plots confusion matrix, ROC, and PR curves for the balanced tuned model.

Output: Heatmap highlights more Bad loans caught (TP = 40) and more Good loans rejected (FP = 27).

Insights:

- The ROC curve is similar to the tuned model's, but the operating point at threshold 0.5 clearly moves toward fewer FN and more FP.
- This is a model that prioritizes catching Bad loans more aggressively.

### Step 12: Threshold Tuning with Business Costs

Purpose: Replace the arbitrary threshold 0.5 with a cost-aware choice.

Process:

- Defines costs:
  - `fn_cost = 5.0` (approving a Bad loan is 5 times as costly)
  - `fp_cost = 1.0` (rejecting a Good applicant)
- Sweeps thresholds from 0.1 to 0.9. For each threshold:
  - Predicts Bad if P(Bad) is greater than or equal to the threshold, else Good.
  - Computes total cost = `fn_cost * FN + fp_cost * FP`.
- Selects the threshold with minimum total cost for the **balanced tuned** model.

Output:

- Best threshold by cost: **0.35**.
- Confusion matrix at this threshold:
  - 94 true Good correctly predicted Good (TN)
  - 46 Good predicted Bad (FP)
  - 12 Bad predicted Good (FN)
  - 48 Bad predicted Bad (TP)

Insights:

- The cost-optimal threshold is lower than 0.5, making the model more conservative than default:
  - Bad recall rises to about 80% (most Bad loans are flagged).
  - Good recall falls to about 67% (more Good applicants are rejected).
- Accuracy is lower than at 0.5, but the **cost** of mistakes is better aligned with the business preference to avoid bad loans.
- This illustrates that threshold selection is a business decision, not a technical afterthought. Real deployments should use realistic monetary costs and target approval rates.

### Role of SHAP in this project

The central problem this project addresses is that tree-based credit models like XGBoost are often accurate but opaque. SHAP is the main tool that resolves this tension: it keeps the predictive power of XGBoost while providing model-wide and borrower-level explanations. Globally, SHAP identifies the true drivers of portfolio risk and informs policy and pricing design; locally, it generates reason codes and what-if analyses for individual applicants. In other words, SHAP is how we turn a high-performing black-box model into an explainable credit scoring system that can actually be deployed in a regulated environment.

## Section 4: Calibration and Explainability (Steps 13 to 17)

### Step 13: Probability Calibration

Purpose: Ensure predicted probabilities are numerically meaningful, not just well ranked.

Process:

- Wraps the balanced tuned model in `CalibratedClassifierCV` with:
  - `method="isotonic"`
  - 5-fold CV
- Compares uncalibrated versus calibrated predictions on the test set using Brier score and a calibration curve.

Output:

- Brier score (uncalibrated) roughly **0.1693**
- Brier score (calibrated) roughly **0.1568**
- Calibration plot shows the isotonic curve closer to the 45-degree diagonal.

Insights:

- XGBoost's raw probabilities are decent but slightly miscalibrated.
- Isotonic calibration improves them, yielding more reliable probability-of-default (PD) estimates.
- This matters for downstream tasks like pricing, capital allocation, and stress testing where the numeric PD value is used, not just the ranking.

### Step 14: SHAP Global Explanations

Purpose: Understand which features drive the model's decisions overall.

Process:

- Builds a `shap.TreeExplainer` on the balanced tuned model.
- Computes SHAP values on the training set.
- Produces:
  1. Bar plot of mean |SHAP| (global importance)
  2. Beeswarm plot (distribution plus direction)
  3. Dependence plot for key features (e.g. `status_checking_account_A14`)

Output: Top features by average SHAP magnitude include:

- `status_checking_account_A14`
- `duration_months`
- `credit_amount`
- `credit_history_A34`
- `savings_account_A61`
- `status_checking_account_A11`
- `age_years`
- `purpose_A41`
- `other_installment_plans_A143`, etc.

Insights:

- These are very plausible credit risk drivers: account status, loan duration and amount, credit history, savings, age, and purpose.
- In the beeswarm plot, SHAP values are defined for the Bad (1) class:
  - Points to the right (positive SHAP) push the prediction toward Bad (higher default risk).
  - Points to the left (negative SHAP) push toward Good (lower risk).
  - For example, higher `credit_amount` and longer `duration_months` tend to have positive SHAP values, increasing default risk.
  - Favourable statuses (certain savings/checking categories) tend to have negative SHAP values, reducing risk.

### Step 15: SHAP Local Explanations (Decision Plots)

Purpose: Explain why specific borrowers were scored as Good or Bad.

Process:

- Selects one Bad borrower (y = 1) and one Good borrower (y = 0) from the test set.
- Plots SHAP decision plots for each:
  - Start at the expected log-odds of Bad (global baseline)
  - Add feature contributions one by one
  - End at that borrower's final log-odds and predicted probability of Bad

Output: Two decision plots, one for each borrower, showing cumulative contributions.

Insights:

- Features that move the curve to the right increase the predicted probability of Bad. Leftward moves decrease it.
- For the Bad borrower, unfavourable account status, long duration, and high credit amount dominate and push the score into high-risk territory.
- For the Good borrower, favourable account/savings features and moderate amounts pull the score into a low-risk region.
- These plots provide regulator-friendly, borrower-specific explanations and can be mapped to reason codes ("high loan amount", "no savings account", etc.).

### Step 16: Sensitivity Analysis

Purpose: Turn the model into a "what-if" tool.

Process:

- For selected Good and Bad borrowers:
  1. Identify top features by |SHAP|.
  2. Sweep each feature across a plausible range (or {0,1} for binaries), keeping others fixed.
  3. Recompute P(Good) (or P(Bad)) at each point.
  4. Plot probability versus feature value.

Output: Sensitivity plots for continuous features like `age_years`, `credit_amount`, `duration_months`, and for binary features such as `status_checking_account_A14`, `savings_account_A61`, `savings_account_A64`, `savings_account_A65`, `employment_since_A74`, etc.

Insights:

- Continuous features often show non-linear patterns:
  - Moderate credit amounts may be safest. Very small or very large amounts can look riskier.
  - The impact of duration is borrower-specific. For some borrowers longer terms look safer, for others medium terms are riskiest.
- Binary indicators frequently act like on/off switches:
  - Flipping certain savings or checking account categories can move P(Good) by many percentage points, in either direction depending on the rest of the profile.
- This analysis reveals which levers matter most for each borrower and how much "room for improvement" exists.

### Step 17: Display Sensitivity Plots

Purpose: Show all saved sensitivity plots inline for inspection.

Process: Scans the `reports/` directory for files starting with `"sensitivity_"` and displays each PNG.

Output: A gallery of sensitivity curves for the selected borrowers and features.

Insights: Steeper curves indicate features where small changes produce large shifts in predicted risk. These are ideal candidates for manual review or customer advice.

## Educational Value

This notebook is an excellent learning resource because it:

- Covers the full pipeline: from data loading through tuned, calibrated, cost-aware, and explained predictions.
- Tackles real-world challenges: class imbalance, hyperparameter tuning, cost-sensitive thresholds, calibration, and explainability.
- Goes beyond "the model predicted X" to "the model predicted X because of Y, and here's how X would change if Y changed."
- Uses production-style code (config-driven, modular helpers, saved reports), not one-off script hacks.
- Is beginner-friendly: concepts are motivated and illustrated, not just dropped as formulas.
- Teaches transferable techniques that apply to any high-stakes classification problem (fraud, medical risk, churn, etc.).

## How to Use

To run this notebook, follow the setup instructions in the project's main README to install dependencies and configure paths. Then:

1. Open the notebook in Jupyter: `jupyter notebook SHAP_Credit.example.ipynb`
2. Run cells sequentially to execute the full pipeline.
3. Experiment: change hyperparameters, cost ratios, or SHAP test cases.
4. Review the conclusions section for key takeaways and ideas for future work.

## Common Pitfalls

If your confusion matrix looks "flipped", check which class is treated as y=1. In this project, y=1 is Bad (default risk), and `predict_proba(X)[:, 1]` returns P(Bad).

If the cost-based threshold picks a surprising threshold, confirm your cost mapping:

- Does FP really mean "rejected Good"?
- Does FN really mean "approved Bad"?
- Are the relative costs realistic?

If SHAP returns a list for `shap_values`, remember to use index 1 for class y=1 (Bad).

## Requirements

- Python 3.8 or higher
- pandas, numpy, scikit-learn, xgboost, shap, matplotlib
- Jupyter Notebook

## Key Takeaways

- **Baseline first:** Logistic regression with AUC roughly 0.80 is a strong reference point. Untuned XGBoost reaches similar AUC and slightly different errors. Complexity alone isn't enough.

- **Hyperparameters shape behaviour:** Tuning keeps AUC in the roughly 0.79 to 0.80 range but shifts where errors occur, especially when combined with class weighting.

- **Class imbalance is critical:** Using `scale_pos_weight ≈ 2.33` raises Bad recall from about **48%** (29 out of 60 Bad loans caught by the tuned model) to about **67%** (40 out of 60 caught by the balanced tuned model), at the cost of more rejected Good borrowers.

- **Thresholds are business decisions:** A cost-based sweep chose a threshold of about **0.35** rather than 0.5, trading some accuracy and Good approvals for substantially better Bad recall. Real deployments must set thresholds based on money, risk appetite, and regulation.

- **Calibration enables trust:** Improving Brier score from about **0.1693** to **0.1568** makes probabilities reliable enough to use as PD estimates in pricing, reserves, and stress tests.

- **SHAP provides transparency:** Global and local SHAP plots show which features drive risk (account status, duration, amount, credit history, savings, age, purpose) and explain each borrower's score.

- **Sensitivity analysis is actionable:** Feature-wise "what-if" curves turn the model into an advisor, showing how changes in loan amount, duration, and account/savings status would shift the predicted risk.

## Risk Assessment and Implications for Credit Scoring

### 1. Model discrimination and portfolio risk

With ROC AUC consistently around 0.80, the models have good but not perfect discriminatory power. In practice this means:

- The model can meaningfully separate higher-risk from lower-risk borrowers, but
- There will still be misclassifications, especially near the decision boundary.

For risk management, this implies:

- The model is suitable as a front-line decision engine for standard retail cases, but
- High-exposure or borderline cases should still be subject to manual review or additional checks (champion-challenger models, policy rules, etc.).

### 2. Error trade-offs and risk appetite

The move from tuned model without class weighting [[122,18],[31,29]], to balanced tuned model at threshold 0.5 [[113,27],[20,40]], to the cost-optimised threshold at 0.35 [[94,46],[12,48]] shows a controlled progression from approval-focused to risk-averse strategies:

- At 0.5 without class weighting, a fair number of Bad loans slip through (31 FN) and relatively few Good applicants are rejected (18 FP).
- With `scale_pos_weight`, Bad recall improves (FN = 20) at the cost of more rejected Good applicants (FP = 27).
- At threshold 0.35, the bank catches most Bad borrowers (FN = 12) but rejects more Good ones (FP = 46).

Implication: Credit risk teams can dial the operating point along this spectrum depending on:

- Macroeconomic conditions (tighten thresholds during downturns),
- Capital constraints and target default rates,
- Regulatory expectations (e.g. stress-test scenarios).

The notebook makes this trade-off explicit rather than hiding it in an arbitrary 0.5 threshold.

### 3. Calibrated PDs and capital planning

Calibrated probabilities (Brier roughly 0.1693 to 0.1568) are crucial for:

- Expected loss calculations (EL = PD × LGD × EAD),
- Provisioning and capital under IFRS 9 / Basel-style frameworks,
- Portfolio monitoring and stress testing (e.g. "what happens if PDs double?").

If PDs are systematically under-estimated or over-estimated, banks may:

- Under-reserve and be vulnerable in downturns, or
- Over-reserve and tie up unnecessary capital.

By improving calibration, the notebook demonstrates how to move from "scores" to risk-quantified PDs that can feed directly into enterprise risk systems.

### 4. Feature drivers, policy design, and fairness

Global SHAP analysis highlights:

- Checking account and savings account status,
- Loan amount and duration,
- Credit history,
- Age and purpose,

as primary drivers of risk. This has several implications:

- **Policy formulation:** risk teams can design or refine cut-off rules and pricing tiers that align with the model's main levers (e.g. higher rates for large, long-tenor loans with weak account history).
- **Governance and fairness:** if certain features proxy sensitive attributes (e.g. age, certain account types), SHAP can be used to:
  - Detect disproportionate impact on specific groups,
  - Support fairness testing and model-risk reviews,
  - Provide transparent reason codes to comply with regulations.

### 5. Explainability for regulatory compliance

Local SHAP decision plots and sensitivity curves provide:

- A clear audit trail of why each decision was made,
- A basis for adverse action notices ("application declined because of: high credit amount, no savings account, short employment history"),
- Evidence for model risk management and internal validation teams.

Explainability reduces the "black-box" concerns often associated with tree ensembles and helps satisfy regulatory expectations around transparency and customer communication.

### 6. Operational usage and limitations

The findings suggest the model is well-suited for:

- Automating decisions on standard, lower-ticket loans, with
- Escalation rules for high-exposure or borderline cases,
- Continuous monitoring of:
  - Default rates by score band,
  - Calibration drift (Brier score, calibration plots),
  - Impact of threshold and policy changes on approval rates.

Limitations to keep in mind:

- The German Credit dataset is relatively small (1000 rows) and dated. Real banks use richer, larger datasets with behavioural and bureau data.
- The model is static. In production, models need retraining and backtesting to handle population drift.
- This notebook does not address macro stress scenarios or LGD/EAD modelling, which are essential for full risk-capital frameworks.

Overall, the project demonstrates how a modern ML model can support sound credit risk management when combined with:

- Proper calibration and threshold setting,
- Clear documentation of trade-offs,
- Strong explainability for both global behaviour and individual decisions.

### 7. Business Impact of Threshold Selection and Revenue Trade-offs

The threshold analysis reveals critical business trade-offs that executive leadership must understand.

Using the **balanced tuned model**:

- At threshold **0.50**, the model:
  - Approves roughly two-thirds of applicants (about 133 out of 200 in the test set),
  - With a default rate among approved loans of around **15%** (about 20 Bad loans among those approvals).

- At the cost-optimal threshold **0.35**, the model:
  - Approves roughly half of applicants (about 106 out of 200),
  - With a lower default rate among approved loans of around **11%** (about 12 Bad loans among approvals).

This has direct business implications:

- **Revenue impact:** A higher threshold (more conservative policy) reduces the number of loans and therefore interest income.
- **Risk impact:** The same conservative policy reduces expected losses because fewer Bad borrowers are approved.
- **Competitive positioning:** If competitors operate with looser thresholds (approving a higher share of applicants), a very conservative policy may protect credit quality but sacrifice market share.

The optimal threshold depends on:

- Funding costs and capital constraints,
- Competitive landscape and market position,
- Risk appetite set by the board,
- Economic outlook and expected default rates,
- Regulatory requirements on portfolio quality.

Threshold setting therefore requires joint input from data science, risk, finance, and business leadership.

### 8. Model Monitoring and Performance Degradation Over Time

Credit models degrade over time due to:

- **Population drift:** The characteristics of applicants change. For example, during economic booms more marginal borrowers apply; during recessions, only the most confident borrowers apply.
- **Behavioural changes:** Consumer behaviour shifts (e.g. rise of BNPL, gig work). Features that predicted default well years ago may be less predictive now.
- **Economic regime changes:** Models trained during low interest-rate environments may not perform well when rates rise sharply.

The notebook shows performance on a static test set, but production deployment requires:

- **Monthly monitoring:** Track actual default rates by score band and compare to predictions.
- **Recalibration:** Even if rank-ordering power (AUC) stays stable, probabilities may need recalibration every 6–12 months using recent default data.
- **Champion-challenger testing:** Keep the current model as champion, train a new model on recent data as challenger, and compare them on new applications before switching.
- **Early warning indicators:** Monitor feature distributions; sudden shifts may signal population drift or data quality issues.

### 9. Fairness, Bias, and Disparate Impact Considerations

While this notebook does not explicitly test for fairness, the SHAP framework enables it. Organisations must assess whether the model treats protected groups fairly:

- **Age** is a key driver of risk in the model. This may be legally permissible but can still create disparate impact for very young or very old borrowers.
- **Account status** features may proxy for socio-economic status. If the model heavily penalises "no savings" or similar indicators, it may indirectly disadvantage lower-income applicants.

Testing for fairness requires:

- Splitting test data by protected attributes (age bands, gender, geography as proxy for race),
- Comparing approval rates, default rates, and model performance across groups,
- Using SHAP to understand if features affect different groups differently,
- Considering threshold adjustments or policy overlays if disparate impact is found (subject to legal constraints).

Regulatory frameworks such as ECOA (US), the Consumer Credit Act (UK), and EU non-discrimination directives all require lenders to demonstrate that models do not unfairly discriminate. SHAP-based analysis can be part of this evidence.

### 10. Integration with Broader Credit Risk Ecosystem

This model does not exist in isolation. Real credit decisioning involves:

- **Credit bureau scores:** The model can supplement or refine bureau scores or act as a challenger when bureau data is thin.
- **Fraud detection:** A separate fraud model should run before credit scoring to flag suspicious applications.
- **Affordability assessment:** Regulations often require lenders to verify that borrowers can afford the loan (debt-to-income, expenditure analysis).
- **Pricing and collections:** The predicted PD should feed into pricing, limit setting, and collections strategies.

A typical decisioning flow might be:

1. Application received,
2. Fraud model runs; if flagged, reject or route to investigation,
3. Credit bureau data pulled,
4. This XGBoost model runs using bureau plus application features,
5. Affordability check,
6. Final decision combining fraud score, credit score, and affordability,
7. If approved, price is set using the PD,
8. If borderline, route to human underwriter with SHAP explanations.

### 11. Stress Testing and Scenario Analysis for Portfolio Risk

Regulators require banks to stress test portfolios under adverse scenarios (recession, housing crash, etc.). This model can support stress testing by:

- **Scenario probability adjustments:** Multiply PDs under assumed shocks and recompute expected losses.
- **Feature-based scenarios:** Use sensitivity analysis to simulate macro shocks (e.g. higher unemployment, income drops, interest-rate hikes).
- **Portfolio-level aggregation:** Apply the model to the entire loan book to estimate losses under stress.

Combined with LGD and EAD models, this forms the foundation of credit risk capital calculations.

The sensitivity analysis in this notebook demonstrates how individual feature changes affect risk. When scaled to portfolio level, this becomes:

- "What if unemployment rises 5 percentage points across our customer base?"
- "What if interest rates force a large share of our borrowers to refinance at higher rates?"
- "What if housing prices drop 20% and affect collateral values?"

These are exactly the questions risk committees and regulators ask. A well-calibrated, explainable model like this one provides the analytical foundation to answer them credibly.

## References

### Core libraries

XGBoost documentation: https://xgboost.readthedocs.io/

SHAP documentation: https://shap.readthedocs.io/

scikit-learn documentation: https://scikit-learn.org/stable/

### scikit-learn APIs used in this notebook

GridSearchCV:  
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

ROC AUC score:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

Confusion matrix:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

Classification report:  
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

Calibration with CalibratedClassifierCV:  
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

Calibration guide:  
https://scikit-learn.org/stable/modules/calibration.html

### Dataset

Statlog German Credit Data (UCI):  
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

### Background papers

XGBoost paper: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016):  
https://arxiv.org/abs/1603.02754

SHAP paper: Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (2017):  
https://arxiv.org/abs/1705.07874
