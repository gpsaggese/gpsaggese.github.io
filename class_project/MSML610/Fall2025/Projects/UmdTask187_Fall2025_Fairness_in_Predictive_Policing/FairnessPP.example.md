# FairnessPP Example: Bias Mitigation in Crime Prediction

## 1. Problem Statement

Predictive policing algorithms face critical challenges that can perpetuate systemic injustices:

- **Historical Bias**: Training data reflects decades of over-policing in certain communities
- **Feedback Loops**: Biased predictions lead to biased enforcement, generating more biased data
- **High Stakes**: Algorithmic errors can result in civil rights violations and eroded public trust
- **Class Imbalance**: Arrests are rare events (~15%), making balanced prediction difficult

This example demonstrates how **FairnessPP** addresses these challenges through:

1. Rich feature engineering (temporal, spatial, historical)
2. Multiple mitigation strategies (in-processing and post-processing)
3. Comprehensive fairness evaluation across demographic groups
4. Transparent trade-off analysis between accuracy and fairness

---

## 2. Methodology

### 2.1 Data Engineering

We enhance the Chicago Crime dataset with engineered features:

**Temporal Features:**
- Hour of day with cyclic encoding (sin/cos transformations)
- Day of week, month, weekend indicator
- Captures crime patterns (night vs day, weekday vs weekend)

**Spatial Features:**
- Distance from downtown Chicago
- Grid-based crime density (historical counts per area)
- Historical arrest rates per grid cell

**Demographic Groups:**
- 5 racial categories: Black, White, Hispanic, Asian, Other
- 4 income levels: Low, Medium-Low, Medium-High, High
- 20 intersectional groups for comprehensive fairness analysis

### 2.2 Model Training Strategy

We train **4 models** to demonstrate the progression from biased to fair:

| Model | Description | Purpose |
|-------|-------------|---------|
| **Baseline (Unmitigated)** | No class balancing, no fairness | Show majority class collapse |
| **Balanced Baseline** | Class weights, no fairness | Show bias with functional model |
| **Fair (In-Processing)** | ExponentiatedGradient + Equalized Odds | Demonstrate in-processing mitigation |
| **Fair (Post-Processing)** | ThresholdOptimizer + Equalized Odds | Demonstrate post-processing mitigation |

### 2.3 Evaluation Metrics

**Performance Metrics:**
- Accuracy: Overall correctness
- Balanced Accuracy: Mean recall per class (critical for imbalanced data)
- Precision, Recall, F1-Score: Trade-offs between error types
- AUC-ROC: Discrimination ability

**Fairness Metrics:**
- Equalized Odds Difference: Maximum difference in TPR and FPR across groups
- Demographic Parity Difference: Difference in selection rates across groups
- Per-group metrics: Selection rate, TPR, FPR for each demographic group

---

## 3. Results

### 3.1 Model Comparison

| Model | Accuracy | Balanced Acc | Recall | EO Disparity |
|-------|----------|--------------|--------|--------------|
| Baseline (Unmitigated) | 0.850 | 0.501 | 0.003 | 0.045 |
| Balanced Baseline | 0.563 | 0.565 | 0.568 | **0.413** |
| Fair (In-Processing) | 0.545 | 0.516 | 0.474 | **0.187** |
| Fair (Post-Processing) | 0.557 | 0.554 | 0.551 | **0.154** |

### 3.2 Key Findings

**Finding 1: Unmitigated Baseline Shows Model Collapse**

Without class balancing, the model predicts almost entirely the majority class (no arrest):
- Only 27 positive predictions out of 19,858 test samples
- Recall of 0.3% means it catches almost no actual arrests
- High accuracy (85%) is misleading - it simply predicts "no arrest" for everyone
- Low EO disparity (0.045) is an artifact of making no real predictions

**Finding 2: Class Balancing Reveals the Bias Problem**

With balanced class weights, the model makes meaningful predictions:
- Recall increases from 0.3% to 56.8%
- Balanced accuracy jumps to 56.5%
- **EO disparity of 0.413 indicates 41% difference in error rates across groups**
- This reveals the bias hidden in the collapsed baseline

**Finding 3: In-Processing Mitigation Reduces Disparity**

ExponentiatedGradient with Equalized Odds constraint:
- Reduces EO disparity from 0.413 to 0.187 (**55% improvement**)
- Maintains reasonable recall (47.4%)
- Uses ensemble of 27 predictors to satisfy fairness constraints
- Slight reduction in balanced accuracy (56.5% to 51.6%)

**Finding 4: Post-Processing Achieves Best Fairness**

ThresholdOptimizer with group-specific thresholds:
- Reduces EO disparity from 0.413 to 0.154 (**63% improvement**)
- Maintains higher balanced accuracy (55.4%) than in-processing
- Preserves recall (55.1%)
- Adjusts decision boundaries per demographic group

**Finding 5: Trade-offs Are Manageable**

The accuracy-fairness trade-off is reasonable:
- Balanced accuracy drops 2-9% for 55-63% fairness improvement
- All fair models maintain recall above 47%
- Post-processing offers the best balance

---

## 4. Visualization Analysis

### 4.1 Pareto Frontier

The accuracy-fairness trade-off plot shows:
- X-axis: Equalized Odds Difference (lower = fairer)
- Y-axis: Balanced Accuracy (higher = better)
- Fair models move toward the ideal top-left corner
- Post-processing achieves the best position on the frontier

### 4.2 Group Metrics Comparison

Four-panel visualization reveals:
- **Selection Rate**: Fair models show more uniform rates across groups
- **Accuracy**: Maintained across groups in fair models
- **True Positive Rate**: More balanced in fair models
- **False Positive Rate**: Reduced variance in fair models

### 4.3 Selection Rate Distribution

Bar chart comparison across all 20 demographic groups:
- Baseline shows wide variance (some groups 3-5x higher than others)
- Fair models compress this variance significantly
- Visual evidence of successful bias mitigation

---

## 5. Impact Analysis

### 5.1 Policing Implications

**Without Fairness Constraints:**
- Some neighborhoods receive disproportionate police attention
- Creates feedback loop: more policing leads to more arrests leads to higher predicted crime
- Erodes community trust and perpetuates historical injustices

**With Fairness Constraints:**
- Police resources distributed more equitably across demographics
- Predictions based on actual risk factors, not historical over-policing
- Breaks the feedback loop of biased enforcement

### 5.2 Technical Lessons

1. **Class Imbalance Must Be Addressed First**
   - Without balanced weights, model collapses to majority class
   - This masks the underlying bias problem

2. **Multiple Metrics Are Essential**
   - Accuracy alone hides bias (85% accuracy with 0.3% recall)
   - Balanced accuracy and recall reveal true performance
   - Fairness metrics (EO, DP) quantify group disparities

3. **Both Mitigation Strategies Work**
   - In-processing: Stronger guarantees, more complex training
   - Post-processing: Faster, can be applied to existing models
   - Choice depends on deployment constraints

4. **Trade-offs Are Quantifiable**
   - 55-63% fairness improvement for 2-9% accuracy cost
   - Enables informed policy decisions

---

## 6. Ethical Considerations

### 6.1 What This Tool Provides

- Technical mechanism to measure algorithmic bias
- Methods to reduce disparity across demographic groups
- Quantitative basis for policy discussions
- Transparency in accuracy-fairness trade-offs

### 6.2 What This Tool Cannot Do

- Eliminate human bias in data collection or labeling
- Address root causes of crime (poverty, inequality, lack of opportunity)
- Define what "fairness" means (contested and context-dependent)
- Replace human oversight and community input

### 6.3 Recommendations for Deployment

1. **Transparency**: Publish fairness metrics alongside accuracy
2. **Community Input**: Include affected communities in defining fairness goals
3. **Regular Audits**: Re-evaluate quarterly as demographics and patterns shift
4. **Minimum Utility Bounds**: Ensure recall exceeds threshold for practical use
5. **Policy Integration**: Algorithm is one tool; must combine with systemic reforms

---

## 7. Conclusion

This example demonstrates that:

1. **Bias is measurable**: 41% equalized odds disparity detected in balanced baseline

2. **Mitigation works**: Both strategies reduced disparity by 55-63%

3. **Trade-offs are reasonable**: Small accuracy cost for large fairness gain

4. **Multiple strategies available**: In-processing and post-processing offer different trade-offs

5. **Transparency is achievable**: Comprehensive metrics enable informed decisions

The **FairnessPP** library provides accessible tools for implementing these techniques, but technology alone is insufficient. Fair algorithms must be deployed alongside policy reforms, community engagement, and ongoing audits to achieve equitable outcomes in high-stakes applications like predictive policing.

---

## 8. Reproducibility

To reproduce these results:

```python
from FairnessPP_utils import (
    load_chicago_data,
    FairnessPredictor,
    ModelConfig,
    create_comparison_table,
    plot_fairness_tradeoff
)

# Load data
X, y, A, dates = load_chicago_data(use_enhanced_features=True)

# Temporal split
train_mask = dates.dt.year < 2023
X_train, y_train, A_train = X[train_mask], y[train_mask], A[train_mask]
X_test, y_test, A_test = X[~train_mask], y[~train_mask], A[~train_mask]

# Train models
config = ModelConfig(n_estimators=100, max_depth=5, constraint_type="equalized_odds")

# Baseline
baseline = FairnessPredictor(config)
baseline.train(X_train, y_train, mitigate=False, class_weight="balanced")

# Fair (In-Processing)
fair_inproc = FairnessPredictor(config)
fair_inproc.train(X_train, y_train, A=A_train, mitigate=True, 
                  mitigation_strategy="inprocessing", class_weight="balanced")

# Fair (Post-Processing)
fair_postproc = FairnessPredictor(config)
fair_postproc.train(X_train, y_train, A=A_train, mitigate=True,
                    mitigation_strategy="postprocessing", class_weight="balanced")

# Evaluate and compare
results = {
    'Baseline': baseline.evaluate(X_test, y_test, A_test),
    'Fair (In-Proc)': fair_inproc.evaluate(X_test, y_test, A_test),
    'Fair (Post-Proc)': fair_postproc.evaluate(X_test, y_test, A_test)
}

print(create_comparison_table(results))
plot_fairness_tradeoff(results)
```

All experiments use `random_state=42` for reproducibility.