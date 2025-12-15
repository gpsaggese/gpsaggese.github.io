# GluonTS COVID-19 Forecasting: Complete Example

## Overview

**GluonTS.example.ipynb** is your complete, production-ready reference implementation for COVID-19 forecasting. While the API notebook teaches you *how* to use GluonTS models, this example shows you *what* you can build with them!

This notebook demonstrates a full end-to-end forecasting application that public health officials could actually use to guide decision-making during a pandemic.

---

## What Makes This Example Special?

### Complete Application Flow

This isn't just model training - it's a complete application:

1. ** Data Pipeline**: Load, explore, and understand real COVID-19 data
2. ** Feature Engineering**: Create advanced features that improve predictions
3. ** Multi-Model Training**: Train and compare three different approaches
4. ** Comprehensive Evaluation**: Use multiple metrics and visualizations
5. ** Scenario Analysis**: Simulate "what if" public health interventions
6. ** Actionable Insights**: Generate recommendations for decision-makers

### Real-World Problem Solving

We tackle an actual public health challenge:

**Problem**: Hospital systems need to predict COVID-19 case surges to:
- Allocate ICU beds and ventilators
- Schedule staff effectively
- Plan intervention strategies (lockdowns, vaccination campaigns)
- Communicate risk to the public

**Solution**: A forecasting system that provides:
- 14-day case predictions with uncertainty bounds
- Model comparison to find the best approach
- Scenario analysis to evaluate intervention strategies
- Clear visualizations for non-technical stakeholders

---

## Getting Started

### Prerequisites

Make sure you've completed the setup from the main README:
- Docker environment built and running
- All data files in the `data/` directory
- Utility modules working correctly

### Running the Example

1. **Start Jupyter** (if not already running):
   ```bash
   ./docker_jupyter.sh
   ```

2. **Open the notebook**:
   - Navigate to `GluonTS.example.ipynb`
   - You'll see a friendly introduction and clear structure

3. **Run it**:
   - Click "Restart & Run All" to execute the entire pipeline
   - Or run cells one by one to follow along step-by-step
   - **Expected runtime**: 1-2 minutes on CPU

4. **Explore the results**:
   - Beautiful visualizations of data and forecasts
   - Model performance comparisons
   - Scenario analysis insights

---

## Notebook Structure

The notebook is organized into 7 main sections:

### 1. Introduction and Setup (Cells 1-2)

**What it does**: Sets the stage and imports all necessary tools

**Key points**:
- Explains the real-world problem we're solving
- Lists data sources (JHU COVID-19, Google Mobility)
- Imports our custom utilities and GluonTS models
- Sets up nice plotting defaults

**What you'll learn**:
- How to structure a complete forecasting application
- What tools and data sources you need

---

### 2. Data Loading and Exploration (Cells 3-5)

**What it does**: Loads real COVID-19 data and helps you understand it

**Key features**:
- Uses our convenient `load_covid_data_for_gluonts()` function
- Shows data statistics and date ranges
- Visualizes time series patterns (cases, deaths, mobility)
- Identifies key patterns (waves, seasonality, behavioral changes)

**What you'll learn**:
- How to load and prepare COVID-19 data
- What patterns exist in the data
- Why visualization matters before modeling

**Tip**: The three time series plots show you:
1. **Cases**: Multiple distinct waves over time
2. **Deaths**: Lagged correlation with cases
3. **Mobility**: Behavioral responses to the pandemic

---

### 3. Feature Engineering (Cells 6-7)

**What it does**: Explains and visualizes the engineered features

**Features created**:
- `Daily_Cases_MA7`: Smoothed case counts (removes weekly noise)
- `Daily_Deaths_MA7`: Smoothed death counts
- `CFR`: Case Fatality Ratio (deaths/cases)
- 6 mobility metrics (retail, workplace, transit, etc.)

**Why these matter**:
- **Deaths**: Strong predictor of case severity
- **CFR**: Captures how deadly the virus is at different times
- **Mobility**: Shows behavioral changes (lockdowns work!)
- **Moving averages**: Remove reporting artifacts (weekend dips)

**What you'll learn**:
- How to engineer meaningful features for forecasting
- How to analyze feature correlations
- Which features are most predictive

**Tip**: The correlation plot shows which features move together with cases. Positive correlation = feature increases with cases. Negative = feature decreases when cases rise.

---

### 4. Model Training (Cells 8-13)

**What it does**: Trains all three models with detailed output

**Models trained**:

#### 4.1 DeepAR (Most Sophisticated)
- **What**: Autoregressive RNN with external features
- **Strengths**: Captures complex temporal patterns, uses all features
- **Training time**: ~3-4 minutes
- **Best for**: When you have rich feature data and need high accuracy

#### 4.2 SimpleFeedForward (Baseline)
- **What**: Simple neural network baseline
- **Strengths**: Very fast training, easy to understand
- **Training time**: ~30-60 seconds
- **Best for**: Quick experiments, benchmarking, stable trends
- **Limitation**: Doesn't use external features (deaths, mobility)

#### 4.3 DeepNPTS (Flexible)
- **What**: Non-parametric time series model
- **Strengths**: No distribution assumptions, handles regime changes well
- **Training time**: ~3-4 minutes
- **Best for**: Data with shifting patterns (perfect for COVID waves!)

**What you'll learn**:
- How to configure each model appropriately
- What makes each model unique
- How to use our convenient wrapper functions

**Tip**: All three models train simultaneously in the notebook. You can compare their outputs side-by-side!

---

### 5. Model Comparison (Cells 14-16)

**What it does**: Compares all models systematically

**Metrics used**:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAPE** (Mean Absolute Percentage Error): Error as a percentage
- **Training Time**: How fast each model trains

**Visualizations**:
- Side-by-side forecast plots for all three models
- Confidence intervals showing uncertainty
- Comparison against actual values

**What you'll learn**:
- How to evaluate forecasting models properly
- How to interpret multiple metrics
- When to use each model based on tradeoffs

**Tip**: Lower is better for MAE, RMSE, and MAPE. The "best" model depends on your priorities:
- **Need accuracy?** Choose the model with lowest MAPE
- **Need speed?** SimpleFeedForward wins
- **Need flexibility?** DeepNPTS handles regime changes best

---

### 6. Scenario Analysis (Cells 17-18)

**What it does**: Simulates public health interventions

**Scenarios explored**:

#### Baseline Scenario
- **Assumption**: No changes to current behavior
- **Result**: Shows expected case trajectory
- **Use**: Understand what happens if we do nothing

#### Intervention Scenario
- **Assumption**: Strong lockdown (30% mobility reduction)
- **Result**: Shows potential case reduction
- **Use**: Quantify intervention impact

**Key outputs**:
- Cases prevented by intervention
- Percentage reduction in transmission
- Visual comparison of scenarios

**What you'll learn**:
- How to use forecasts for policy decisions
- How to simulate "what if" scenarios
- How to quantify intervention tradeoffs

**Real-world application**: This helps public health officials answer:
- "Should we implement a lockdown?"
- "How many cases could we prevent?"
- "Is the intervention worth the economic cost?"

---

### 7. Conclusions and Next Steps (Cells 19-21)

**What it does**: Synthesizes insights and provides guidance

**Key findings summary**:
- Model performance comparison
- Feature importance insights
- Uncertainty quantification lessons
- Scenario analysis takeaways

**Recommendations**:
- When to use multiple models
- How to monitor uncertainty
- Why frequent retraining matters
- How to combine models with domain expertise

**Next steps**:
- Immediate improvements (state-level data, vaccination data)
- Advanced techniques (ensembles, hierarchical forecasting)
- Production deployment considerations

**What you'll learn**:
- How to translate model results into actionable insights
- What improvements to prioritize
- How to deploy models in production

---

## Key Concepts Explained

### Probabilistic Forecasting

Unlike simple point predictions, our models provide **probabilistic forecasts**:

```python
# Not just: "We predict 50,000 cases"
# But: "We predict 50,000 cases, with 80% confidence it will be between 40,000-60,000"
```

**Why this matters**:
- **Risk assessment**: Wide intervals = high uncertainty = higher risk
- **Resource planning**: Plan for the range, not just the average
- **Decision-making**: Know when predictions are reliable vs. uncertain

**In the notebook**: Look for the shaded regions around forecasts - these show confidence intervals!

---

### External Features (Covariates)

Most forecasting models only use historical values. Our advanced models (DeepAR, DeepNPTS) also use **external features**:

- **Deaths data**: Leading indicator of severity
- **Mobility patterns**: Behavioral changes affect transmission
- **CFR**: Captures how deadly the virus is

**Why this matters**:
- More information = better predictions
- Can capture causality (lockdowns → reduced mobility → fewer cases)
- Enables scenario analysis (change mobility, predict case impact)

**In the notebook**: Compare DeepAR (uses features) vs. SimpleFeedForward (doesn't) to see the impact!

---

### Model Comparison

We train three different models because:

1. **No single model is always best**: Different models excel in different situations
2. **Robustness**: If all models agree, you can be more confident
3. **Learning**: Understanding tradeoffs helps you pick the right tool

**How to choose**:
- **Need accuracy?** Test all models, pick the best performer
- **Need speed?** SimpleFeedForward trains in seconds
- **Regime changes?** DeepNPTS handles distribution shifts well
- **Rich features?** DeepAR leverages external data best

---

### Scenario Analysis

The most powerful application of forecasting is asking **"what if?"**

Example questions you can answer:
- "What if we implement a lockdown?"
- "What if vaccination rates increase?"
- "What if a new variant emerges?"

**How it works**:
1. Train model on historical data
2. Create scenarios with different future conditions
3. Generate forecasts for each scenario
4. Compare outcomes to guide decisions

**In the notebook**: We simulate a strong intervention and quantify its impact. In a real application, you'd create multiple scenarios with different intervention strengths!

---

## Learning Trajectory

### If You're New to Forecasting

Start here to build intuition:

1. **Run the whole notebook** ("Restart & Run All") to see the complete flow
2. **Focus on Section 2** (Data Exploration) - understand the problem first
3. **Study Section 5** (Model Comparison) - see how models differ
4. **Experiment** - change parameters, try different train/test splits

**Key sections**: 2, 5, 7

---

### If You Know Forecasting Basics

Dive deeper into the techniques:

1. **Section 3** - Study feature engineering strategies
2. **Section 4** - Compare model architectures and training approaches
3. **Section 6** - Learn scenario analysis techniques
4. **Modify the code** - try different features, model parameters, scenarios

**Key sections**: 3, 4, 6

---

### If You've worked with it before

Focus on production-readiness:

1. **Code organization** - See how utilities are structured
2. **Evaluation** - Study the comprehensive metrics and visualizations
3. **Section 7** - Production deployment considerations
4. **Adapt to your domain** - Replace COVID data with your own time series

**Key sections**: 4, 5, 7

---

## Customization Guide

### Using Your Own Data

Replace COVID-19 data with your own time series:

1. **Prepare your data**:
   ```python
   # Your time series (target variable)
   df = pd.DataFrame({
       'Date': [...],
       'target_value': [...]
   })
   
   # Optional: external features
   features_df = pd.DataFrame({
       'Date': [...],
       'feature1': [...],
       'feature2': [...]
   })
   ```

2. **Merge and format**:
   ```python
   from GluonTS_utils_gluonts import create_gluonts_dataset
   
   train_ds, test_ds = create_gluonts_dataset(
       df, 
       target_col='target_value',
       feature_cols=['feature1', 'feature2'],
       ...
   )
   ```

3. **Train models**: Use the same training code!

---

### Adjusting Forecast Horizon

Change from 14-day to different horizons:

```python
# 7-day forecast
prediction_length = 7
context_length = 28  # Usually 2-4x prediction length

# 30-day forecast  
prediction_length = 30
context_length = 90
```

**Tradeoff**: Longer horizons are harder to predict accurately but more useful for long-term planning.

---

### Tuning Model Parameters

Experiment with these key parameters:

**DeepAR**:
```python
epochs = 20          # More = better fit (but slower)
hidden_size = 60     # Bigger = more capacity
num_layers = 3       # Deeper = more complex patterns
dropout = 0.2        # Higher = more regularization
```

**SimpleFeedForward**:
```python
epochs = 30          # It's fast, so you can train longer!
hidden_dimensions = [60, 60, 30]  # Can add more layers
```

**DeepNPTS**:
```python
epochs = 20
num_hidden_nodes = [60]  # Can add more nodes
dropout_rate = 0.15
```

---

### Adding More Scenarios

Create additional "what if" scenarios:

```python
scenarios = {
    'Baseline': {
        'mobility_change': 0.0,
        'expected_reduction': 0.0
    },
    'Mild Intervention': {
        'mobility_change': -0.15,  # 15% reduction
        'expected_reduction': 0.08  # 8% case reduction
    },
    'Strong Intervention': {
        'mobility_change': -0.30,  # 30% reduction
        'expected_reduction': 0.15  # 15% case reduction
    },
    'Full Lockdown': {
        'mobility_change': -0.60,  # 60% reduction
        'expected_reduction': 0.35  # 35% case reduction
    }
}
```

Then visualize all scenarios side-by-side!

---

## Expected Outputs

### Visualizations

You'll see these key plots:

1. **Time Series Plots** (Section 2)
   - Cases, deaths, and mobility over time
   - Shows pandemic waves and behavioral responses

2. **Correlation Heatmap** (Section 3)
   - Which features predict cases best
   - Helps understand feature relationships

3. **Forecast Plots** (Section 5)
   - Three plots showing each model's forecast
   - Confidence intervals showing uncertainty
   - Actual values for comparison

4. **Scenario Comparison** (Section 6)
   - Bar chart comparing baseline vs. intervention
   - Shows quantified impact of policy changes

### Metrics

Performance metrics for all three models:

```
Model               MAE      RMSE     MAPE    Time
─────────────────────────────────────────────────
DeepAR             X,XXX    X,XXX    X.X%    XXXs
SimpleFeedForward  X,XXX    X,XXX    X.X%    XXs
DeepNPTS           X,XXX    X,XXX    X.X%    XXXs
```

**Interpretation**:
- **Lower MAE/RMSE/MAPE = better accuracy**
- Compare across models to find the best performer
- Consider training time for production use

### Insights

Actionable insights you'll discover:

- Which model performs best for COVID-19 forecasting
- How deaths and mobility data improve predictions
- Impact of interventions on case trajectories
- Recommendations for public health policy

---

## Common Questions

### "Which model should I use?"

**Answer**: It depends on your priorities!

| Priority | Best Model | Why |
|----------|-----------|-----|
| Highest accuracy | Test all, pick best | Performance varies by dataset |
| Fast training | SimpleFeedForward | Trains in <1 minute |
| Rich features | DeepAR | Best at using external data |
| Regime changes | DeepNPTS | Handles distribution shifts |

**Pro tip**: In production, many teams use **ensembles** - combine multiple models for better robustness!

---

### "How accurate are these forecasts?"

**Answer**: It varies!

- **Short-term** (1-7 days): Usually quite accurate (MAPE < 10%)
- **Medium-term** (7-14 days): Good for planning (MAPE 10-20%)
- **Long-term** (>14 days): Higher uncertainty (MAPE > 20%)

**Factors affecting accuracy**:
- **Data quality**: More frequent data = better predictions
- **Pattern stability**: Stable trends are easier to predict
- **External shocks**: New variants, policy changes create uncertainty

**Key insight**: Always look at confidence intervals, not just point predictions!

---

### "Can I use this for other diseases?"

**Absolutely!** The approach generalizes:

- **Flu forecasting**: Replace COVID data with flu data
- **Hospital admissions**: Use admission counts instead of cases
- **Disease outbreaks**: Apply to any infectious disease

**What to change**:
1. Load your disease data instead of COVID
2. Engineer relevant features (may differ by disease)
3. Adjust forecast horizons based on disease dynamics
4. Same models, same evaluation approach!

---

### "How often should I retrain models?"

**Answer**: It depends on your application!

**For COVID-19**:
- **Daily**: During active outbreaks (patterns change fast)
- **Weekly**: During stable periods (less critical)
- **After shocks**: Always retrain after major events (new variants, policy changes)

**General rule**: Retrain when:
- New data substantially changes patterns
- Forecast accuracy degrades
- External conditions shift dramatically

**In production**: Set up automated retraining (daily/weekly schedule) and monitor forecast quality metrics!

---

### "What if my forecasts are way off?"

**Debugging checklist**:

1. **Check data quality**:
   - Missing values?
   - Outliers or data errors?
   - Sufficient history (at least 2x forecast horizon)?

2. **Verify features**:
   - Are features available for forecast period?
   - Features properly aligned with target?
   - Correct number of features specified?

3. **Tune models**:
   - Try more training epochs
   - Adjust network size (hidden_size, hidden_dimensions)
   - Change context length

4. **Evaluate uncertainty**:
   - Wide confidence intervals = model knows it's uncertain!
   - This is actually a feature, not a bug

**Pro tip**: Sometimes poor forecasts reveal real changes in the underlying process (new variant, behavioral shift). Investigate why, don't just tune blindly!
