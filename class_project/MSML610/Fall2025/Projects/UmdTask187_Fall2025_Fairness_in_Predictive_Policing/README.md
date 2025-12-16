# FairnessPP: Fairness in Predictive Policing

**Name:** Pranav Srinivasan K N  
**UID:** 121334572  
**Project Tag:** Fall2025_Fairness_in_Predictive_Policing  
**GitHub Issue:** #187  
**Course:** MSML610 Fall 2025  
**Difficulty Level:** 3 (Hard)

---

## Project Overview

**FairnessPP** is a Python library that demonstrates how to build fair machine learning models for high-stakes predictive policing applications. The project addresses a critical real-world problem: algorithmic bias in crime prediction systems that can perpetuate historical injustices.

### Key Features

- **Multiple Fairness Mitigation Strategies**: In-processing (ExponentiatedGradient) and post-processing (ThresholdOptimizer)
- **Rich Feature Engineering**: Temporal, spatial, and historical crime patterns
- **Comprehensive Evaluation**: Performance and fairness metrics across 20 demographic groups
- **Visualization Tools**: Pareto frontiers and group-level fairness dashboards
- **Clean API**: Configurable, type-safe interface with extensive documentation

### Problem Statement

Predictive policing algorithms trained on historical arrest data often learn to over-police specific demographic groups because these groups are historically over-represented in the training data. This creates a harmful feedback loop:

```
Biased Historical Data → Biased Model → Biased Policing → More Biased Data
```

**FairnessPP** breaks this cycle by:
1. Measuring bias across intersectional demographic groups
2. Applying fairness constraints during or after training
3. Enabling transparent trade-off analysis between accuracy and fairness

---

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (recommended) or local Python environment
- 8GB+ RAM recommended

### Installation

**Option 1: Docker (Recommended)**

```bash
# Build Docker image
docker build -t fairness-pp .

# Run Jupyter Lab
docker run -p 8888:8888 -v "${PWD}:/workspace" fairness-pp

# Open http://localhost:8888 in browser
```

**Option 2: Local Installation**

```bash
pip install -r requirements.txt
jupyter notebook
```

### Basic Usage

```python
from FairnessPP_utils import FairnessPredictor, ModelConfig, load_chicago_data

# Load data
X, y, A, dates = load_chicago_data(use_enhanced_features=True)

# Split temporally
train_mask = dates.dt.year < 2023
X_train, y_train, A_train = X[train_mask], y[train_mask], A[train_mask]
X_test, y_test, A_test = X[~train_mask], y[~train_mask], A[~train_mask]

# Train fair model
config = ModelConfig(constraint_type="equalized_odds")
predictor = FairnessPredictor(config)
predictor.train(
    X_train, y_train, A=A_train,
    mitigate=True,
    mitigation_strategy="inprocessing",
    class_weight="balanced"
)

# Evaluate
result = predictor.evaluate(X_test, y_test, A_test)
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"EO Disparity: {result.equalized_odds_diff:.3f}")
```

---

## Project Structure

```
FairnessPP/
│
├── FairnessPP_utils.py          # Core library implementation
│   ├── FairnessPredictor        # Main wrapper class
│   ├── ModelConfig              # Configuration dataclass
│   ├── EvaluationResult         # Results dataclass
│   ├── load_chicago_data()      # Data loader with feature engineering
│   └── Visualization utilities  # plot_fairness_tradeoff(), etc.
│
├── FairnessPP_API.ipynb         # API demonstration notebook
├── FairnessPP_API.md            # Complete API reference
│
├── FairnessPP_example.ipynb     # End-to-end workflow notebook
├── FairnessPP_example.md        # Results interpretation guide
│
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container setup
└── README.md                    # This file
```

---

## Key Results

Experiments on Chicago Crime Data (80,000 records, 2020-2023):


| Model | Balanced Acc | Recall | EO Disparity | Fairness Improvement |
|-------|--------------|--------|--------------|---------------------|
| Baseline (Unmitigated) | 0.501 | 0.003 | 0.045 | N/A (collapsed) |
| Balanced Baseline | 0.565 | 0.568 | 0.413 | Reference |
| **Fair (In-Processing)** | 0.516 | 0.474 | 0.187 | **55% reduction** |
| **Fair (Post-Processing)** | 0.554 | 0.551 | 0.154 | **63% reduction** |

*Note: Model results may vary slightly between runs due to threshold optimization. Improvements typically range from 35-65%.*

### Key Findings

1. **Without class balancing, models collapse** to predicting majority class
2. **Balanced baseline reveals 41% disparity** in error rates across groups
3. **In-processing reduces disparity by 55%** with slight accuracy trade-off
4. **Post-processing achieves 63% reduction** while maintaining higher accuracy
5. **Trade-offs are manageable**: 2-9% accuracy cost for 55-63% fairness gain

---

## Documentation

### For API Users
- **[FairnessPP_API.md](FairnessPP.API.md)**: Complete API reference with examples
- **[FairnessPP_API.ipynb](FairnessPP.API.ipynb)**: Interactive API demonstration

### For Understanding Results
- **[FairnessPP_example.md](FairnessPP.example.md)**: Interpretation guide and policy implications
- **[FairnessPP_example.ipynb](FairnessPP.example.ipynb)**: Full end-to-end workflow

### For Developers
- **[FairnessPP_utils.py](FairnessPP_utils.py)**: Well-documented source code

---

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Base ML algorithms |
| fairlearn | Fairness mitigation |
| matplotlib, seaborn | Visualization |
| jupyter | Interactive notebooks |

### Algorithms

**Base Classifier:** GradientBoostingClassifier (scikit-learn)

**Fairness Mitigation:**
- In-processing: Fairlearn ExponentiatedGradient with LogisticRegression
- Post-processing: Fairlearn ThresholdOptimizer

**Features Engineering:**
- Temporal: Hour, day, month with cyclic encoding
- Spatial: Grid-based density, distance to downtown
- Historical: Arrest rates per geographic cell

**Demographics:**
- 5 racial categories × 4 income levels = 20 intersectional groups

### Performance

| Operation | Time (80K samples) |
|-----------|-------------------|
| Data loading | 5-10 seconds |
| Baseline training | 10-30 seconds |
| In-processing | 1-3 minutes |
| Post-processing | 30-60 seconds |

---

## Running the Project

### Step 1: Build Docker Image

```bash
docker build -t fairness-pp .
```

Expected output:
```
Successfully built <image_id>
Successfully tagged fairness-pp:latest
```

### Step 2: Run Container

```bash
docker run -p 8888:8888 -v $(pwd):/workspace fairness-pp
```

### Step 3: Open Jupyter

Navigate to `http://localhost:8888` and run:
1. **FairnessPP_API.ipynb** - API overview
2. **FairnessPP_example.ipynb** - Full workflow

### Step 4: Verify Results

Execute notebooks with "Restart & Run All" to confirm end-to-end functionality.

---

## Evaluation Metrics

### Performance Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| Accuracy | Overall correctness | Can be misleading with imbalance |
| Balanced Accuracy | Mean recall per class | Better for imbalanced data |
| Recall | True positive rate | Critical for catching events |
| Precision | Positive predictive value | Reduces false alarms |

### Fairness Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Equalized Odds Diff | Max difference in TPR/FPR across groups | < 0.10 |
| Demographic Parity Diff | Difference in selection rates | < 0.10 |
| Selection Rate Variance | Spread of positive prediction rates | Minimize |

---

## Ethical Considerations

### What This Tool Provides

- Technical mechanism to measure and reduce algorithmic bias
- Transparent evaluation of fairness/accuracy trade-offs
- Quantitative basis for policy decisions

### What This Tool Cannot Do

- Eliminate human bias in data collection
- Address root causes of crime (poverty, inequality)
- Define "fairness" (context-dependent and contested)
- Replace human oversight and community input

### Deployment Recommendations

1. **Transparency**: Publish fairness metrics with accuracy
2. **Community Oversight**: Include affected communities in defining fairness
3. **Regular Audits**: Re-evaluate quarterly
4. **Utility Bounds**: Ensure minimum recall for practical use
5. **Policy Integration**: Combine with systemic reforms

---

## Troubleshooting

### Model predicts all negative class

**Cause:** Class imbalance not addressed

**Solution:** Use `class_weight="balanced"` in training

### Fairness disparity still high

**Cause:** Insufficient mitigation iterations

**Solution:** Increase `max_iter_mitigation` or try post-processing

### Training very slow

**Cause:** In-processing with many iterations

**Solution:** Reduce `n_estimators` or use post-processing

---

## Future Enhancements

- [ ] Individual fairness metrics
- [ ] Causal inference for proxy detection
- [ ] Multi-objective Pareto optimization
- [ ] Temporal drift monitoring
- [ ] Interactive policy dashboard

---

## Acknowledgments

**Data Source:** Chicago Crime Data via Chicago Open Data Portal

**Libraries:** Built on scikit-learn and Fairlearn

**Course:** MSML610 Advanced Machine Learning, Fall 2025

---

## License

This project is for educational purposes as part of MSML610 coursework.