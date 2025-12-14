# GluonTS API Documentation

**A friendly guide to using GluonTS for time series forecasting**

---

## Overview

This document explains how to use the three GluonTS models demonstrated in `GluonTS.API.ipynb`. Think of this as your reference guide while the notebook is your hands-on tutorial.

### What is GluonTS?

GluonTS is a Python library for probabilistic time series forecasting. Unlike traditional models that give you a single prediction, GluonTS models give you a **distribution of possible futures** - which means you get predictions with confidence intervals!

### Why Probabilistic Forecasting?

Imagine you're planning hospital resources during COVID-19. Instead of hearing "we'll have 50,000 cases tomorrow," wouldn't it be more useful to hear "we expect 50,000 cases, but there's a 90% chance it will be between 40,000 and 60,000"? That's what probabilistic forecasting gives you - uncertainty quantification that helps with planning.

---

## The Three Models

### 1. DeepAR - The Pattern Learner

**What it is:** DeepAR uses recurrent neural networks (RNNs) to learn temporal patterns. Think of it as a model with memory - it remembers what happened in the past to make smarter predictions about the future.

**When to use it:**
- You have complex, repeating patterns (like COVID waves)
- Your data has seasonality (weekly reporting cycles, holiday effects)
- You need to capture long-term dependencies
- Accuracy matters more than training speed

**How it works:** DeepAR learns patterns by looking at historical sequences. For COVID data, it learns that:
- Cases tend to follow a weekly pattern (lower on weekends)
- Waves have characteristic rise and fall shapes
- External factors (mobility, deaths) help predict case trends

**Key strengths:**
- Excellent at capturing complex temporal dynamics
- Handles seasonality naturally
- Produces well-calibrated uncertainty estimates

**Limitations:**
- Slower to train (1-5 minutes typically)
- More hyperparameters to tune
- Requires more data to learn effectively

---

### 2. SimpleFeedForward - The Speed Demon

**What it is:** A straightforward neural network that maps recent history directly to future predictions. No fancy memory, no recurrent connections - just simple, fast, effective.

**When to use it:**
- You need fast experiments (prototyping, parameter search)
- Your data has stable, smooth trends
- You want a baseline to beat
- You need quick retraining in production

**How it works:** SimpleFeedForward looks at the recent past (say, last 60 days) and directly predicts the next 14 days. It's like drawing a trend line, but with a neural network that can learn non-linear patterns.

**Key strengths:**
- Trains 10x faster than DeepAR
- Fewer hyperparameters = easier to tune
- Great baseline for comparison
- Works well for stable trends

**Limitations:**
- Doesn't capture complex temporal dependencies
- Less effective for highly seasonal data
- Can struggle with sudden regime changes

---

### 3. DeepNPTS - The Flexible Learner

**What it is:** DeepNPTS (Deep Non-Parametric Time Series) is special - it doesn't assume your data follows any particular distribution (like normal or Poisson). Instead, it learns the distribution directly from your data.

**When to use it:**
- Your data distribution keeps changing (like different COVID waves)
- You have unusual, non-standard distributions
- You see regime changes (patterns shift over time)
- Your data has heavy tails or rare extreme events

**How it works:** Most models say "I assume your data is normally distributed" or "I assume it follows a Poisson distribution." DeepNPTS says "Show me your data, and I'll figure out the distribution myself." This makes it incredibly flexible for COVID data where each wave behaves differently.

**Key strengths:**
- Adapts to changing data distributions
- No distribution assumptions needed
- Handles regime changes gracefully
- Good for unusual, non-standard data

**Limitations:**
- More complex than SimpleFeedForward
- Requires careful hyperparameter tuning
- Can be slower to train

---

## Model Parameters Explained

### Common Parameters (All Models)

#### `freq` (string)
**What it does:** Specifies your data's frequency.  
**Typical values:** `'D'` (daily), `'H'` (hourly), `'M'` (monthly)  
**Example:** `freq='D'` for daily COVID cases

#### `prediction_length` (int)
**What it does:** How many time steps ahead to forecast.  
**Typical values:** 7 (one week), 14 (two weeks), 30 (one month)  
**COVID example:** `prediction_length=14` to forecast 2 weeks of cases

#### `context_length` (int)
**What it does:** How much historical data to use for predictions.  
**Rule of thumb:** 2-4× prediction_length  
**Example:** `context_length=60` uses 2 months of history to predict 2 weeks

#### `num_feat_dynamic_real` (int)
**What it does:** Number of external features (covariates) you're providing.  
**COVID example:** If you have deaths + 3 mobility features = `num_feat_dynamic_real=4`  
**Important:** Must match the number of feature columns in your data!

#### `epochs` (int)
**What it does:** How many times to go through the training data.  
**Trade-off:** More epochs = better fit but longer training  
**Typical values:** 
- DeepAR: 20-50 epochs
- SimpleFeedForward: 50-100 epochs (it's fast!)
- DeepNPTS: 20-40 epochs

#### `lr` (float, learning rate)
**What it does:** How fast the model learns from data.  
**Typical values:** 0.001 (default, good starting point), 0.0001 (careful learning)  
**Rule:** Too high = unstable training, too low = very slow learning

---

### DeepAR-Specific Parameters

#### `num_layers` (int)
**What it does:** How many RNN layers to stack.  
**Impact:** More layers = can learn more complex patterns  
**Typical values:** 2-3 layers  
**COVID example:** `num_layers=2` captures weekly cycles + wave patterns

#### `hidden_size` (int)
**What it does:** Size of the hidden state in the RNN (network capacity).  
**Impact:** Bigger = more complex patterns, but slower training  
**Typical values:** 40-100  
**Rule of thumb:** Start with 40, increase if underfitting

#### `dropout_rate` (float)
**What it does:** Randomly drops connections during training to prevent overfitting.  
**Typical values:** 0.1-0.2 (10-20% dropout)  
**When to increase:** If your model memorizes training data

---

### SimpleFeedForward-Specific Parameters

#### `hidden_dims` (list of ints)
**What it does:** Sizes of hidden layers in the network.  
**Example:** `hidden_dims=[40, 40]` means two layers, 40 units each  
**Impact:** More layers or bigger layers = more capacity  
**COVID example:** `[40, 40]` is usually sufficient

---

### DeepNPTS-Specific Parameters

#### `hidden_size` (int)
**What it does:** Network capacity for learning the data distribution.  
**Typical values:** 40-100  
**Impact:** Controls how flexible the learned distribution can be

---

## Choosing the Right Model

Here's a decision tree to help you choose:

```
Start here!
│
├─ Do you need FAST training (< 1 min)?
│  │
│  ├─ YES → Use SimpleFeedForward
│  │        Great for: baselines, experiments, stable trends
│  │
│  └─ NO → Continue...
│
├─ Does your data have COMPLEX patterns?
│  (multiple cycles, strong seasonality, long-term dependencies)
│  │
│  ├─ YES → Use DeepAR
│  │        Great for: COVID waves, retail sales, web traffic
│  │
│  └─ NO → Continue...
│
└─ Does your data distribution CHANGE over time?
   (regime shifts, different behavior in different periods)
   │
   ├─ YES → Use DeepNPTS
   │        Great for: COVID (each wave is different), finance
   │
   └─ NO → Use SimpleFeedForward or DeepAR
           Test both and compare!
```

---

## Quick Start: Model Configuration

### DeepAR - Good Defaults for COVID

```python
from gluonts.torch.model.deepar import DeepAREstimator

estimator = DeepAREstimator(
    freq='D',                    # Daily data
    prediction_length=14,        # Forecast 2 weeks
    context_length=60,           # Use 2 months of history
    num_feat_dynamic_real=3,     # Example: 3 features (deaths + 2 mobility)
    
    # Network architecture
    num_layers=2,                # Good balance
    hidden_size=40,              # Sufficient for COVID patterns
    dropout_rate=0.1,            # Prevent overfitting
    
    # Training
    lr=0.001,                    # Standard learning rate
    epochs=20,                   # Good for demos (use 30-50 in production)
    batch_size=32,
    num_batches_per_epoch=50,
    trainer_kwargs={"max_epochs": 20}
)
```

### SimpleFeedForward - Fast Baseline

```python
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

estimator = SimpleFeedForwardEstimator(
    freq='D',
    prediction_length=14,
    context_length=60,
    num_feat_dynamic_real=3,
    
    # Network architecture
    hidden_dims=[40, 40],        # Two hidden layers
    
    # Training (can use more epochs since it's fast!)
    lr=0.001,
    epochs=50,                   # Still trains in under a minute
    batch_size=32,
    num_batches_per_epoch=50,
    trainer_kwargs={"max_epochs": 50}
)
```

### DeepNPTS - Flexible Distribution Learning

```python
from gluonts.torch.model.deep_npts import DeepNPTSEstimator

estimator = DeepNPTSEstimator(
    freq='D',
    prediction_length=14,
    context_length=60,
    num_feat_dynamic_real=3,
    
    # Network architecture
    hidden_size=40,
    
    # Training
    lr=0.001,
    epochs=30,                   # Moderate training time
    batch_size=32,
    num_batches_per_epoch=50,
    trainer_kwargs={"max_epochs": 30}
)
```

---

## Generating Forecasts

All three models follow the same prediction pattern:

```python
from gluonts.evaluation import make_evaluation_predictions

# Train the model
predictor = estimator.train(train_dataset)

# Generate probabilistic forecasts
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,
    predictor=predictor,
    num_samples=100  # Generate 100 sample paths for uncertainty
)

# Convert to lists
forecasts = list(forecast_it)
ground_truths = list(ts_it)

# Get your forecast
forecast = forecasts[0]  # For single time series

# Access predictions
mean_forecast = forecast.mean              # Mean prediction
median_forecast = forecast.quantile(0.5)   # Median (50th percentile)
lower_bound = forecast.quantile(0.1)       # Lower confidence (10th percentile)
upper_bound = forecast.quantile(0.9)       # Upper confidence (90th percentile)
```

---

## Understanding Probabilistic Forecasts

### What You Get

When you generate a forecast, you don't just get one number - you get a whole distribution!

```python
forecast.mean           # Average of all predictions
forecast.median         # Middle value (50th percentile)
forecast.quantile(0.1)  # 10% of predictions are below this
forecast.quantile(0.9)  # 90% of predictions are below this
```

### Confidence Intervals

You can create confidence intervals at any level:

```python
# 80% confidence: 80% of the time, true value is in this range
lower_80 = forecast.quantile(0.1)
upper_80 = forecast.quantile(0.9)

# 90% confidence: 90% of the time, true value is in this range
lower_90 = forecast.quantile(0.05)
upper_90 = forecast.quantile(0.95)

# 95% confidence: even wider
lower_95 = forecast.quantile(0.025)
upper_95 = forecast.quantile(0.975)
```

### Why This Matters

For COVID-19 planning:
- **Mean forecast:** "We expect 50,000 cases"
- **80% interval:** "We're 80% confident it'll be between 40,000-60,000"
- **Planning:** "Better prepare resources for up to 60,000 to be safe"

---

## Common Issues and Solutions

### Issue 1: "My model is overfitting!"

**Symptoms:** Great training performance, poor test performance

**Solutions:**
- Increase `dropout_rate` (try 0.2 or 0.3 for DeepAR)
- Reduce `hidden_size` or `hidden_dims`
- Use less `context_length`
- Get more training data if possible

### Issue 2: "Training is too slow!"

**Solutions:**
- Use SimpleFeedForward instead of DeepAR
- Reduce `epochs` (start with 10-20)
- Reduce `hidden_size`
- Reduce `num_batches_per_epoch`

### Issue 3: "Predictions are way off!"

**Debugging checklist:**
1. Check `num_feat_dynamic_real` matches your actual features
2. Verify your data has no NaN values
3. Try increasing `context_length` (use more history)
4. Try more `epochs`
5. Check if your features are properly normalized

### Issue 4: "Model gives weird uncertainty estimates"

**Solutions:**
- Increase `num_samples` in `make_evaluation_predictions` (try 200)
- Train for more `epochs` - models need time to learn uncertainty
- For DeepAR: tune `dropout_rate`
- For DeepNPTS: this model specifically learns uncertainty well!

---

## Performance Tips

### For Best Accuracy
1. Use DeepAR with:
   - `context_length` = 4× prediction_length
   - `epochs` = 30-50
   - Include relevant external features

2. Tune hyperparameters:
   - Try different `hidden_size` values (40, 60, 80, 100)
   - Experiment with `num_layers` (2, 3, 4 for DeepAR)

### For Fastest Training
1. Use SimpleFeedForward
2. Reduce `num_batches_per_epoch` to 25-30
3. Use `hidden_dims=[30, 30]` instead of `[40, 40]`

### For Best Uncertainty Quantification
1. Use DeepNPTS or DeepAR
2. Generate forecasts with `num_samples=200` or more
3. Train for more epochs
4. Use dropout (DeepAR)

---

## Evaluation Metrics

### MAE (Mean Absolute Error)
**What it is:** Average absolute difference between prediction and actual  
**Interpretation:** Lower is better, measured in same units as your data  
**COVID example:** MAE of 5,000 means predictions are off by 5,000 cases on average

### RMSE (Root Mean Squared Error)
**What it is:** Like MAE but penalizes large errors more  
**Interpretation:** Lower is better, more sensitive to outliers than MAE  
**When to use:** When large errors are especially bad

### MAPE (Mean Absolute Percentage Error)
**What it is:** Average error as a percentage of actual values  
**Interpretation:** Lower is better, gives scale-independent comparison  
**COVID example:** MAPE of 10% means predictions are off by 10% on average  
**Good performance:** < 10% excellent, < 20% good, < 30% acceptable

### CRPS (Continuous Ranked Probability Score)
**What it is:** Measures quality of probabilistic forecasts  
**Interpretation:** Lower is better, evaluates the entire distribution  
**Why it matters:** MAE/RMSE only look at point forecasts, CRPS evaluates uncertainty too

---

## Next Steps

### Learn by Doing
- Run `GluonTS.API.ipynb` - hands-on tutorial with COVID data
- Experiment with different parameters
- Try all three models and compare results

### See a Complete Application
- Check out `GluonTS.example.ipynb` - full COVID-19 forecasting pipeline
- Learn about model comparison, uncertainty quantification, scenario analysis

### Further Reading
- [GluonTS Documentation](https://ts.gluon.ai/)
- [DeepAR Paper](https://arxiv.org/abs/1704.04110)
- [Time Series Forecasting Best Practices](https://github.com/microsoft/forecasting)

---

