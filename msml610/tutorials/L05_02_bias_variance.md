---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Imports

```python
%load_ext autoreload
%autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
```

```python
import msml610_utils as ut
import L05_02_bias_variance_utils as utils

ut.config_notebook()

# Initialize logger.
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)
```

## Cell 1: Approximation

This cell demonstrates the concept of approximation - how well different models
can fit a target function.

**Purpose**: Visualize how a constant model (horizontal line) and a linear
model (diagonal line) approximate a sinusoidal target function. This shows
the difference in approximation capability between simple and more complex
models.

**Target Function**:
- f(x) = sin(pi*x) for x in [-1, 1]

**Models**:
- **Constant Model (g_0)**: g_0(x) = b, where b is the mean of f(x)
- **Linear Model (g_1)**: g_1(x) = a*x + b, where a and b are fitted using
  least squares

**Three plots**:
1. **True Function vs Constant Model**: Shows the target function (blue) and
   the best constant approximation (green), with orange shading indicating
   approximation error
2. **True Function vs Linear Model**: Shows the target function (blue) and
   the best linear approximation (magenta), with orange shading indicating
   approximation error
3. **Comments**: Displays the approximation errors and key observations

**Key observations**:
- The constant model has high approximation error - it cannot capture any
  variation in the target function
- The linear model has lower approximation error - it can capture the general
  trend, though not the curvature
- Lower approximation error means better fit, but doesn't always mean better
  learning (as we'll see with bias-variance tradeoff)

```python
# Display approximation comparison between constant and linear models.
utils.cell1_approximation()
```

## Cell 2: Learning Once

This cell demonstrates the difference between learning and approximation.
Learning uses a limited training set, whereas approximation assumes full
knowledge of the target function.

**Purpose**: Show how models trained on a few random samples perform
differently than models that approximate the full function. This illustrates
the key difference between in-sample error (E_in) and out-of-sample error
(E_out).

**Parameters**:
- `seed`: Random seed controlling which training points are sampled
- `N_samples`: Number of random points in the training set (default: 2)

**Target Function**:
- f(x) = sin(pi*x) for x in [-1, 1]

**Models**:
- **Constant Model (g_0)**: Fitted to training data only
- **Linear Model (g_1)**: Fitted to training data only

**Three plots**:
1. **Constant Model**: Shows true function, fitted constant model, and
   training points (red dots). Title displays E_in and E_out.
2. **Linear Model**: Shows true function, fitted linear model, and training
   points (red dots). Title displays E_in and E_out.
3. **Comments**: Displays errors and key observations about learning vs
   approximation.

**Key observations**:
- E_in measures how well the model fits the training data
- E_out measures how well the model generalizes to the full function
- With very few samples (e.g., N=2), a linear model can achieve E_in=0
  (perfect fit on training data) but still have high E_out
- This demonstrates that learning from limited data is fundamentally
  different from approximation
- Try different seeds to see how training set selection affects performance

```python
# Display learning from N random samples with interactive controls.
utils.cell2_learning_once()
```

## Cell 3: Learning (Bias-Variance Decomposition)

This cell visualizes the bias-variance tradeoff by showing how models
trained on different random training sets vary around the true function.

**Purpose**: Demonstrate bias and variance decomposition by running multiple
learning experiments with different training sets. Shows how model complexity
affects both bias (systematic error) and variance (sensitivity to training
data).

**Parameters**:
- `seed`: Random seed for reproducibility
- `N_samples`: Number of training points per experiment (default: 2)
- `N_experiments`: Number of different training sets to generate (default: 20)

**Target Function**:
- f(x) = sin(pi*x) for x in [-1, 1]

**Visualization**:
- For each experiment, sample N_samples random points and fit both models
- Plot all fitted models with transparency (alpha=0.5) to show variation
- Plot average model across all experiments as a dashed line

**Three plots**:
1. **Constant Models**: Shows true function and all fitted constant models
   (green lines with transparency). Dashed line shows average model.
2. **Linear Models**: Shows true function and all fitted linear models
   (magenta lines with transparency). Dashed line shows average model.
3. **Comments**: Displays average errors and explanation of bias-variance
   tradeoff.

**Key observations**:
- **Constant model (g_0)**: Low variance (all lines very similar), high bias
  (far from true function). The model is too simple to capture the pattern.
- **Linear model (g_1)**: Higher variance (lines spread out more), lower bias
  (average model closer to true function). The model is more flexible but
  sensitive to training data.
- This illustrates the **BIAS-VARIANCE TRADEOFF**: simpler models have low
  variance but high bias; more complex models have lower bias but higher
  variance.
- The total out-of-sample error decomposes as: E_out = bias^2 + variance + noise
- Try increasing N_samples to see how more data reduces variance
- Try increasing N_experiments to get more stable estimates of bias and variance

```python
# Display bias-variance decomposition over multiple experiments.
utils.cell3_learning_bias_variance()
```
