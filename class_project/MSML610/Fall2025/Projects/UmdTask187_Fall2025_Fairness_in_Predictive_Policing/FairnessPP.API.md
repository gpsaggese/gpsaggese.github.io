# FairnessPP API Documentation

## Overview

**FairnessPP** (Fairness in Predictive Policing) is a Python API designed to simplify the development of fair machine learning models for high-stakes public sector applications. It provides a clean abstraction layer over scikit-learn and Fairlearn, enabling practitioners to build, evaluate, and compare models with different fairness constraints.

### Design Philosophy

1. **Simplicity**: Single `FairnessPredictor` class handles all use cases
2. **Flexibility**: Multiple mitigation strategies through simple configuration
3. **Transparency**: Rich evaluation metrics and visualization utilities
4. **Reproducibility**: Dataclass-based configuration ensures consistent results

---

## 1. Configuration Objects

### ModelConfig

A dataclass for stable model configuration.

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_estimators: int = 100           # Number of boosting rounds
    max_depth: int = 5                # Maximum tree depth
    learning_rate: float = 0.1        # Boosting learning rate
    random_state: int = 42            # Reproducibility seed
    max_iter_mitigation: int = 50     # Iterations for fairness algorithm
    constraint_type: str = "equalized_odds"  # Fairness constraint
```

**Constraint Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| `equalized_odds` | Equalizes TPR and FPR across groups | Criminal justice (recommended) |
| `demographic_parity` | Equalizes selection rates | Hiring, lending |

**Usage:**

```python
from FairnessPP_utils import ModelConfig

# Default configuration
config = ModelConfig()

# Custom configuration
config = ModelConfig(
    n_estimators=150,
    max_depth=4,
    constraint_type="demographic_parity",
    max_iter_mitigation=75
)
```

---

### EvaluationResult

A dataclass containing comprehensive evaluation metrics.

```python
@dataclass
class EvaluationResult:
    accuracy: float                    # Overall accuracy
    balanced_accuracy: float           # Mean recall per class
    precision: float                   # Precision score
    recall: float                      # Recall (sensitivity)
    f1_score: float                    # Harmonic mean of precision/recall
    auc_roc: float                     # Area under ROC curve
    fairness_disparity: float          # Equalized odds difference
    demographic_parity_diff: float     # Selection rate difference
    equalized_odds_diff: float         # Max TPR/FPR difference across groups
    group_metrics: pd.DataFrame        # Per-group performance metrics
    confusion_matrix: np.ndarray       # 2x2 confusion matrix
    selection_rates: pd.Series         # Selection rate per group
```

**Accessing Results:**

```python
result = predictor.evaluate(X_test, y_test, A_test)

# Performance metrics
print(f"Accuracy: {result.accuracy:.3f}")
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"Recall: {result.recall:.3f}")

# Fairness metrics
print(f"EO Disparity: {result.equalized_odds_diff:.3f}")
print(f"DP Disparity: {result.demographic_parity_diff:.3f}")

# Group-level analysis
print(result.group_metrics)
print(result.selection_rates)
```

---

## 2. Core Wrapper Class

### FairnessPredictor

The main entry point for training and evaluating fair models.

#### Constructor

```python
def __init__(self, config: ModelConfig = None)
```

**Parameters:**
- `config`: ModelConfig object (uses defaults if None)

**Example:**

```python
from FairnessPP_utils import FairnessPredictor, ModelConfig

# With default config
predictor = FairnessPredictor()

# With custom config
config = ModelConfig(n_estimators=150, constraint_type="equalized_odds")
predictor = FairnessPredictor(config=config)
```

---

#### Method: train()

```python
def train(
    self, 
    X, 
    y, 
    A=None, 
    mitigate: bool = False,
    mitigation_strategy: str = "inprocessing",
    class_weight: str = "balanced"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | DataFrame/ndarray | Feature matrix |
| `y` | Series/ndarray | Target variable (binary) |
| `A` | Series/ndarray | Sensitive attributes (required if mitigate=True) |
| `mitigate` | bool | Whether to apply fairness constraints |
| `mitigation_strategy` | str | `"inprocessing"` or `"postprocessing"` |
| `class_weight` | str | `"balanced"` or `None` |

**Mitigation Strategies:**

| Strategy | Algorithm | Description |
|----------|-----------|-------------|
| `inprocessing` | ExponentiatedGradient | Applies constraints during training via iterative reweighting |
| `postprocessing` | ThresholdOptimizer | Adjusts decision thresholds per group after training |

**Examples:**

```python
# Baseline (no fairness, no class balancing)
predictor.train(X_train, y_train, mitigate=False, class_weight=None)

# Class-balanced baseline
predictor.train(X_train, y_train, mitigate=False, class_weight="balanced")

# Fair model with in-processing
predictor.train(
    X_train, y_train, A=A_train,
    mitigate=True,
    mitigation_strategy="inprocessing",
    class_weight="balanced"
)

# Fair model with post-processing
predictor.train(
    X_train, y_train, A=A_train,
    mitigate=True,
    mitigation_strategy="postprocessing",
    class_weight="balanced"
)
```

---

#### Method: predict()

```python
def predict(self, X, A=None) -> np.ndarray
```

**Parameters:**
- `X`: Feature matrix
- `A`: Sensitive attributes (required for post-processing models)

**Returns:** Binary predictions (0 or 1)

**Example:**

```python
predictions = predictor.predict(X_test)

# For post-processing models
predictions = predictor.predict(X_test, A=A_test)
```

---

#### Method: predict_proba()

```python
def predict_proba(self, X, A=None) -> np.ndarray
```

**Parameters:**
- `X`: Feature matrix
- `A`: Sensitive attributes (unused, for API consistency)

**Returns:** Probability estimates [n_samples, 2] or None if unavailable

**Example:**

```python
probabilities = predictor.predict_proba(X_test)
if probabilities is not None:
    positive_proba = probabilities[:, 1]
```

---

#### Method: evaluate()

```python
def evaluate(self, X, y, A) -> EvaluationResult
```

**Parameters:**
- `X`: Feature matrix
- `y`: True labels
- `A`: Sensitive attributes

**Returns:** EvaluationResult dataclass with all metrics

**Computed Metrics:**
- Overall: accuracy, balanced_accuracy, precision, recall, F1, AUC-ROC
- Fairness: equalized_odds_diff, demographic_parity_diff
- Group-level: Selection rates, TPR, FPR, FNR, accuracy per group

**Example:**

```python
result = predictor.evaluate(X_test, y_test, A_test)

print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"EO Disparity: {result.equalized_odds_diff:.3f}")
print("\nGroup Metrics:")
print(result.group_metrics)
```

---

## 3. Data Loading

### load_chicago_data()

```python
def load_chicago_data(
    local_cache_path: str = "data/chicago_crime_2020_2023.csv",
    use_enhanced_features: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
```

**Parameters:**
- `local_cache_path`: Path to cached CSV file
- `use_enhanced_features`: Include rich temporal/spatial features

**Returns:** Tuple of (X, y, A, dates)
- `X`: Feature DataFrame
- `y`: Binary arrest indicator
- `A`: Intersectional demographic groups
- `dates`: Datetime Series for temporal splits

**Features:**

| Category | Features | Description |
|----------|----------|-------------|
| Basic | Latitude, Longitude, Domestic | Core spatial and categorical |
| Temporal | Hour, DayOfWeek, Month, IsWeekend | Time-based patterns |
| Cyclic | Hour_Sin, Hour_Cos, Month_Sin, Month_Cos | Cyclic encoding |
| Spatial | Distance_Downtown, Crime_Density, Arrest_Rate_Historic | Location context |

**Example:**

```python
from FairnessPP_utils import load_chicago_data

X, y, A, dates = load_chicago_data(use_enhanced_features=True)

# Temporal split
train_mask = dates.dt.year < 2023
X_train, y_train, A_train = X[train_mask], y[train_mask], A[train_mask]
X_test, y_test, A_test = X[~train_mask], y[~train_mask], A[~train_mask]
```

---

## 4. Visualization Utilities

### plot_fairness_tradeoff()

```python
def plot_fairness_tradeoff(results_dict: dict, save_path: str = None)
```

Visualizes the Pareto frontier of accuracy vs fairness.

**Parameters:**
- `results_dict`: Dict mapping model names to EvaluationResult objects
- `save_path`: Optional path to save figure

**Example:**

```python
results = {
    'Baseline': result_baseline,
    'Fair Model': result_fair
}
fig = plot_fairness_tradeoff(results, save_path='tradeoff.png')
plt.show()
```

---

### plot_group_metrics()

```python
def plot_group_metrics(eval_result, title: str = "Group Fairness Metrics", save_path: str = None)
```

Visualizes per-group performance metrics in a 4-panel plot.

**Parameters:**
- `eval_result`: EvaluationResult from evaluate()
- `title`: Plot title
- `save_path`: Optional path to save figure

**Panels:**
1. Selection Rate by Group
2. Accuracy by Group
3. True Positive Rate by Group
4. False Positive Rate by Group

---

### create_comparison_table()

```python
def create_comparison_table(results_dict: dict) -> pd.DataFrame
```

Creates a formatted comparison table of multiple models.

**Parameters:**
- `results_dict`: Dict mapping model names to EvaluationResult objects

**Returns:** DataFrame with all metrics for comparison

---

## 5. Complete Workflow Example

```python
from FairnessPP_utils import (
    FairnessPredictor, 
    ModelConfig,
    load_chicago_data,
    plot_fairness_tradeoff,
    create_comparison_table
)
import matplotlib.pyplot as plt

# Step 1: Load data
X, y, A, dates = load_chicago_data(use_enhanced_features=True)

# Step 2: Temporal split
train_mask = dates.dt.year < 2023
X_train, y_train, A_train = X[train_mask], y[train_mask], A[train_mask]
X_test, y_test, A_test = X[~train_mask], y[~train_mask], A[~train_mask]

# Step 3: Train baseline
config = ModelConfig(n_estimators=100, constraint_type="equalized_odds")
baseline = FairnessPredictor(config)
baseline.train(X_train, y_train, mitigate=False, class_weight="balanced")
result_baseline = baseline.evaluate(X_test, y_test, A_test)

# Step 4: Train fair model
fair_model = FairnessPredictor(config)
fair_model.train(
    X_train, y_train, A=A_train,
    mitigate=True,
    mitigation_strategy="inprocessing",
    class_weight="balanced"
)
result_fair = fair_model.evaluate(X_test, y_test, A_test)

# Step 5: Compare
results = {'Baseline': result_baseline, 'Fair Model': result_fair}
print(create_comparison_table(results))
plot_fairness_tradeoff(results)
plt.show()
```

---

## 6. Implementation Details

### Internal Architecture

| Component | Implementation |
|-----------|---------------|
| Feature Scaling | StandardScaler (automatic) |
| Base Classifier | GradientBoostingClassifier (baseline, post-processing) |
| In-Processing | LogisticRegression + ExponentiatedGradient |
| Post-Processing | ThresholdOptimizer |
| Evaluation | Fairlearn MetricFrame |

### Hyperparameter Defaults

**ExponentiatedGradient:**
- `nu=1e-4`: Balance between fairness and utility
- `eta0=2.0`: Learning rate for reduction
- `eps=0.005`: Constraint tolerance

**GradientBoostingClassifier:**
- `n_estimators=100`
- `max_depth=5`
- `learning_rate=0.1`
- `min_samples_split=20`
- `min_samples_leaf=10`

### Performance Characteristics

| Operation | Time (80K samples) |
|-----------|-------------------|
| Baseline training | 10-30 seconds |
| In-processing | 1-3 minutes |
| Post-processing | 30-60 seconds |
| Evaluation | 2-5 seconds |

---

## 7. Best Practices

### Always Use Temporal Validation

```python
# Correct: Temporal split
train_mask = dates.dt.year < 2023
X_train = X[train_mask]

# Incorrect: Random split (data leakage)
X_train, X_test = train_test_split(X, test_size=0.2)
```

### Address Class Imbalance

```python
# Correct: Use class weighting
predictor.train(X, y, A=A, mitigate=True, class_weight="balanced")

# Incorrect: Model may collapse to majority class
predictor.train(X, y, A=A, mitigate=True, class_weight=None)
```

### Evaluate Multiple Metrics

```python
# Correct: Comprehensive evaluation
result = predictor.evaluate(X_test, y_test, A_test)
print(f"Balanced Acc: {result.balanced_accuracy:.3f}")
print(f"Recall: {result.recall:.3f}")
print(f"EO Disparity: {result.equalized_odds_diff:.3f}")

# Incorrect: Only accuracy (hides bias)
print(f"Accuracy: {result.accuracy:.3f}")
```

---

## 8. Troubleshooting

### Model predicts all negative class

**Symptoms:** Selection rate ~0, recall ~0, balanced accuracy ~0.5

**Solutions:**
1. Use `class_weight="balanced"`
2. Reduce `max_iter_mitigation` (try 30-50)
3. Check class distribution in training data

### Fairness disparity still high

**Symptoms:** EO disparity >0.20 after mitigation

**Solutions:**
1. Increase `max_iter_mitigation` (try 75-100)
2. Try post-processing instead of in-processing
3. Check for proxy features correlated with sensitive attributes

### Training very slow

**Symptoms:** In-processing takes >5 minutes

**Solutions:**
1. Reduce `n_estimators` (try 50)
2. Reduce `max_iter_mitigation` (try 30)
3. Use post-processing (faster)
4. Subsample training data if >200K samples