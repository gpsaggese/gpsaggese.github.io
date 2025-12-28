# Model Comparison Visualizations Guide

This document explains the visualizations generated from our comprehensive model comparison for house price prediction.

## Overview

We compared **8 different regression models** using **5-fold cross-validation** to predict house prices on the Ames Housing Dataset. The visualizations showcase performance metrics, training efficiency, and model stability.

---

## Visualization Files

All visualizations are located in: `docs/visualizations/`

### 1. Summary Dashboard (`summary_dashboard.png`)

**Purpose:** Comprehensive overview of all key metrics in one view.

**What it shows:**
- CV RMSE comparison (top-left)
- Best model information box (top-right)
- Training time for each model (middle-left)
- R² scores comparison (middle-center)
- CV standard deviation for stability (middle-right)
- Top 5 models ranking (bottom, full width)

**Key Insights:**
- **StackingEnsemble** is highlighted in green as the best model
- Best CV RMSE: **0.1271** (log scale)
- Improvement over worst model: **0.0226** (15.1% better)
- Trade-off between performance and training time is visible

**Use this for:** Executive summary, presentations, quick overview

---

### 2. CV RMSE Comparison (`cv_rmse_comparison.png`)

**Purpose:** Direct comparison of model performance using the primary metric.

**What it shows:**
- Bar chart of mean cross-validation RMSE for each model
- Error bars showing ± standard deviation
- Best model highlighted in green
- Exact RMSE values labeled on each bar

**Key Findings:**
| Rank | Model | CV RMSE | Interpretation |
|------|-------|---------|----------------|
| 1 | StackingEnsemble | 0.1271 | Best - combines multiple models |
| 2 | GradientBoosting | 0.1273 | Very close second |
| 3 | XGBoost | 0.1300 | Strong individual model |
| 4 | VotingEnsemble | 0.1301 | Good ensemble performance |
| 8 | RandomForest | 0.1497 | Worst - simpler approach |

**Interpretation:**
- Lower is better (RMSE measures prediction error)
- Target is log-transformed, so RMSE of 0.1271 means average error of ~12.7% in log space
- Ensemble methods (Stacking, Voting) dominate top rankings

**Use this for:** Model selection justification, performance reports

---

### 3. CV Score Distributions (`cv_score_distributions.png`)

**Purpose:** Show performance consistency across different data folds.

**What it shows:**
- Box plots for each model's 5 CV fold scores
- Median (red line), quartiles (box), and outliers
- Notched boxes indicate confidence intervals

**Key Insights:**
- **StackingEnsemble** has relatively tight distribution → consistent performance
- **Ridge** has wide spread → less stable, sensitive to data splits
- Models with smaller boxes are more reliable for production

**Interpretation:**
- Tight distribution = model generalizes well
- Wide distribution = performance varies significantly with data
- Outliers indicate folds where model struggled

**Use this for:** Assessing model reliability, understanding variance

---

### 4. Training Time Comparison (`training_time_comparison.png`)

**Purpose:** Compare computational efficiency of different models.

**What it shows:**
- Horizontal bar chart of training time in seconds
- Sorted from fastest to slowest
- Time includes full 5-fold cross-validation

**Key Findings:**
| Model | Training Time | Speed Category |
|-------|---------------|----------------|
| Ridge | 0.2s | Ultra-fast |
| ElasticNet | 0.2s | Ultra-fast |
| Lasso | 0.2s | Ultra-fast |
| RandomForest | 2.5s | Fast |
| XGBoost | 9.7s | Moderate |
| GradientBoosting | 24.6s | Moderate |
| VotingEnsemble | 30.6s | Slow |
| StackingEnsemble | **136.0s** | Very slow |

**Trade-off Analysis:**
- **StackingEnsemble** takes 680x longer than Ridge but has 10% better RMSE
- **XGBoost** is 14x faster than Stacking with only 2% worse RMSE → good balance
- Linear models (Ridge, Lasso, ElasticNet) are fastest but 10-15% worse

**Use this for:** Production deployment decisions, cost-benefit analysis

---

### 5. Multi-Metric Comparison (`multi_metric_comparison.png`)

**Purpose:** Compare models across multiple performance dimensions.

**What it shows:**
Three subplots:
1. **CV RMSE** (left) - Primary metric, lower is better
2. **Train MAE** (center) - Mean Absolute Error, lower is better
3. **Train R²** (right) - Goodness of fit, higher is better (0-1 scale)

**Key Insights:**
- **RMSE ranking:** StackingEnsemble > GradientBoosting > XGBoost
- **MAE ranking:** XGBoost best (0.0195) - excels at average case predictions
- **R² ranking:** XGBoost best (0.9953) - explains 99.5% of variance in training

**Interpretation:**
- Different metrics favor different models
- XGBoost has lowest training error but slightly higher CV error → mild overfitting
- StackingEnsemble balances all metrics well → better generalization

**Use this for:** Understanding strengths/weaknesses of each model

---

### 6. CV Variability (`cv_variability.png`)

**Purpose:** Measure model stability and reliability.

**What it shows:**
- Standard deviation of CV scores for each model
- Lower = more stable and predictable
- Percentage shown is coefficient of variation (std/mean × 100)

**Key Findings:**
| Model | CV Std Dev | CV % | Stability |
|-------|------------|------|-----------|
| RandomForest | 0.0100 | 6.7% | Most stable |
| GradientBoosting | 0.0101 | 7.9% | Very stable |
| VotingEnsemble | 0.0106 | 8.2% | Stable |
| XGBoost | 0.0117 | 9.0% | Stable |
| StackingEnsemble | 0.0135 | **10.6%** | Less stable |
| Ridge | 0.0254 | 18.0% | Least stable |

**Interpretation:**
- **StackingEnsemble** is powerful but has 10.6% variability
- **RandomForest** is most consistent across folds
- **Ridge** high variability suggests sensitivity to data distribution

**Use this for:** Risk assessment, production reliability planning

---

### 7. Performance vs Time Trade-off (`performance_time_tradeoff.png`)

**Purpose:** Visualize the efficiency-accuracy balance.

**What it shows:**
- Scatter plot with training time (x-axis, log scale) vs CV RMSE (y-axis)
- Best model highlighted in larger green circle
- Bottom-left corner is ideal: fast + accurate

**Key Strategic Insights:**

**Pareto Optimal Models:**
1. **Ridge** (0.2s, 0.1410 RMSE) - Ultra-fast baseline
2. **XGBoost** (9.7s, 0.1300 RMSE) - Best speed/accuracy balance
3. **GradientBoosting** (24.6s, 0.1273 RMSE) - Moderate time, excellent accuracy
4. **StackingEnsemble** (136s, 0.1271 RMSE) - Best accuracy, high cost

**Decision Guide:**
- **Real-time serving (< 1s latency):** Use Ridge or ElasticNet
- **Batch predictions (daily/hourly):** Use XGBoost (best trade-off)
- **Maximum accuracy needed:** Use StackingEnsemble (accept 2min training)
- **Budget-constrained:** Use GradientBoosting (middle ground)

**Use this for:** Architecture decisions, resource planning

---

## Summary of Key Findings

### Best Overall Model: **StackingEnsemble** ✅

**Performance:**
- CV RMSE: 0.1271 ± 0.0135
- Train R²: 0.9808
- Train MAE: 0.0398

**Why it's best:**
- Combines strengths of XGBoost, RandomForest, GradientBoosting, and Ridge
- Ridge meta-learner prevents overfitting
- Best generalization to unseen data

**Trade-offs:**
- Slowest training: 136 seconds (2.3 minutes)
- Higher complexity: 4 base models + meta-learner
- 10.6% CV variability (not the most stable)

### Runner-Up: **GradientBoosting**

- Only 0.0002 worse RMSE (0.02% difference)
- 5.5x faster than StackingEnsemble (24.6s vs 136s)
- More stable (7.9% CV variability)
- Simpler to explain and deploy

### Best Speed/Accuracy: **XGBoost**

- 2.3% worse than best (still excellent)
- 14x faster than StackingEnsemble
- Industry-standard for tabular data
- Best training metrics (R² = 0.9953)

---

## Recommendations by Use Case

### For Production Deployment:
**Primary:** StackingEnsemble (deployed model)
**Backup:** XGBoost (faster alternative with 98% of performance)

### For Real-Time API:
**Use:** XGBoost or GradientBoosting
**Reason:** Sub-second inference, excellent accuracy

### For Experimentation:
**Use:** Ridge or ElasticNet
**Reason:** Instant training, good baseline

### For Maximum Accuracy:
**Use:** StackingEnsemble (current deployment)
**Reason:** Best CV RMSE, accepts longer training time

---

## How to Regenerate Visualizations

```bash
# Inside Docker container or with dependencies installed
python scripts/visualize_results.py
```

**Requirements:**
- matplotlib
- seaborn
- pandas
- numpy

**Output:**
- All visualizations saved to `docs/visualizations/`
- High resolution PNG (300 DPI)
- Publication-ready quality

---

## Interpreting RMSE Values

Since our target (SalePrice) is **log-transformed**:

```
Actual prediction error = exp(predicted) - exp(actual)
```

**Example:** RMSE of 0.1271 means:
- Average log-space error: 0.1271
- For a $200,000 house: typically within ±$27,000 (13.5%)
- For a $400,000 house: typically within ±$54,000 (13.5%)

**Why log-transform?**
- House prices are right-skewed (few very expensive houses)
- Log-transform normalizes distribution
- Makes percentage errors consistent across price ranges

---

## Next Steps

1. **Deploy StackingEnsemble** ✅ (Already done - see `models/serving/`)
2. **Monitor performance** on new data
3. **A/B test** StackingEnsemble vs XGBoost in production
4. **Retrain quarterly** with new housing data
5. **Consider ensemble pruning** if inference speed becomes critical

---

## Files Reference

- **Comparison Results:** `models/comparison/comparison_results.json`
- **Deployed Model:** `models/serving/<timestamp>/sklearn_model.pkl`
- **Visualizations:** `docs/visualizations/*.png`
- **This Guide:** `docs/VISUALIZATION_GUIDE.md`

---

*Generated from model comparison on Ames Housing Dataset with 1,460 training samples and 80 features.*
