# COVID-19 Case Prediction with GluonTS

Complete example of using GluonTS for real-world pandemic forecasting.

## Project Overview

This notebook (`GluonTS.example.ipynb`) demonstrates COVID-19 forecasting using
three GluonTS models. It shows how to build an end-to-end application that
public health officials could use for resource planning and intervention
strategies.

**Problem Statement:** Hospital systems need to predict COVID-19 case surges 14
days ahead to:

- Allocate ICU beds and ventilators
- Schedule healthcare staff
- Plan intervention strategies
- Communicate risk to the public

**Solution:** A forecasting system that:

- Predicts daily cases with uncertainty bounds
- Compares three different models (DeepAR, SimpleFeedForward, DeepNPTS)
- Simulates intervention scenarios
- Provides actionable insights

## Data Sources

We use three COVID-19 datasets:

1. **Cases Data** (JHU CSSE)
   - Daily confirmed COVID-19 cases
   - County-level for entire US
   - Aggregated to national level

2. **Deaths Data** (JHU CSSE)
   - Daily COVID-19 deaths
   - Used as predictive feature
   - Case Fatality Rate (CFR) derived

3. **Mobility Data** (Google)
   - Changes in movement patterns
   - Six categories (retail, parks, transit, etc.)
   - Captures behavioral response to pandemic

**Data Period:** January 2020 - March 2023 (covers multiple COVID variant waves)

## Why These Features?

### Target Variable: Daily Cases (7-Day MA)

- Smooths weekend reporting artifacts
- Reduces noise for better model training
- Clinically relevant metric

### Feature 1: Deaths Data

- Deaths lag cases by 2-3 weeks
- Strong correlation with severe outcomes
- Helps predict case trends

### Feature 2: Case Fatality Rate (CFR)

- CFR = Deaths / Cases
- Indicates healthcare strain
- Changes with virus variants

### Feature 3: Mobility Patterns

- Decreased mobility → Fewer cases (with lag)
- Captures policy interventions (lockdowns)
- Real-time behavioral indicator

## Model Selection for COVID-19

We compare three approaches, each with strengths for pandemic data:

### DeepAR

**Why chosen:** COVID-19 has complex temporal patterns

- Multiple waves with different shapes
- Weekly seasonality (lower reporting on weekends)
- Long-term dependencies between waves

**Configuration:**

- `context_length=60`: 2 months of history
- `num_layers=2`: Deep enough for complex patterns
- `hidden_size=40`: Balanced capacity
- `epochs=10`: Fast enough for demos (use 20-30 for higher accuracy, trains
  slower though)

**Expected performance:** Best overall accuracy, captures wave dynamics

### SimpleFeedForward

**Why chosen:** Fast baseline for comparison

- Stable trends between major surges
- Quick retraining as new data arrives
- Computationally efficient

**Configuration:**

- `context_length=60`: Same lookback as DeepAR
- `hidden_dimensions=[40, 40]`: Two-layer network
- `epochs=10`: Very fast training

**Expected performance:** Good for stable periods, struggles with sudden changes

### DeepNPTS

**Why chosen:** COVID-19 distribution changes across waves

- Delta wave behaves differently than Omicron
- No single distribution fits all periods
- Adapts to regime changes

**Configuration:**

- `context_length=60`: Consistent with other models
- `num_hidden_nodes=[40, 40]`: Similar capacity
- `epochs=10`: Moderate training time

**Expected performance:** Best for transitional periods and new variants

## Notebook Structure

### Section 1: Data Loading and Exploration

**What it does:**

- Loads and merges three data sources
- Creates visualizations of pandemic timeline
- Explores correlations between features
- Shows data quality and coverage

**Key insights:**

- Multiple distinct waves visible
- Clear weekly seasonality pattern
- Mobility drops precede case declines
- Deaths lag cases consistently

### Section 2: Feature Engineering

**What it does:**

- Creates 7-day moving averages
- Calculates Case Fatality Rate
- Derives cumulative metrics
- Prepares features for GluonTS

**Features created:**

- `Daily_Cases_MA7`: Target variable (smoothed)
- `Daily_Deaths_MA7`: Death trends
- `Cumulative_Deaths`: Overall severity
- `CFR`: Healthcare strain indicator
- Mobility metrics: Behavioral indicators

### Section 3: Train/Test Split

**Strategy:**

- Training: ~3 years (Jan 2020 - Feb 2023)
- Testing: 14 days (late Feb - early Mar 2023)
- Prediction length: 14 days (2 weeks ahead)

**Why this split:**

- Representative test period (post-Omicron)
- 14-day horizon matches public health planning cycles
- Sufficient training data for all three models

### Section 4: Model Training

**DeepAR Training:**
```
Configuration:
  - Context: 60 days
  - Prediction: 14 days ahead
  - Features: 3 (deaths, CFR, mobility subset)
  - Architecture: 2 layers, 40 units
  - Training: 10 epochs (~2-3 minutes)

Results:
  - Learns wave patterns
  - Captures weekly seasonality
  - Produces calibrated uncertainty
```

**SimpleFeedForward Training:**
```
Configuration:
  - Context: 60 days
  - Prediction: 14 days ahead
  - No features (target only)
  - Architecture: 2 layers, 40 units each
  - Training: 10 epochs (~30 seconds)

Results:
  - Fast baseline
  - Smooth trend extrapolation
  - Wider uncertainty bounds
```

**DeepNPTS Training:**
```
Configuration:
  - Context: 60 days
  - Prediction: 14 days ahead
  - Features: 3 (deaths, CFR, mobility subset)
  - Architecture: 2 layers, 40 nodes each
  - Training: 10 epochs (~1-2 minutes)

Results:
  - Adapts to distribution shifts
  - Flexible uncertainty estimates
  - Good for transitional periods
```

### Section 5: Model Evaluation

**Metrics used:**

1. **MAE (Mean Absolute Error)**
   - Average prediction error in cases
   - Easy to interpret (in original units)
   - Robust to outliers

2. **RMSE (Root Mean Square Error)**
   - Penalizes large errors more
   - Common forecasting metric
   - In same units as target

3. **MAPE (Mean Absolute Percentage Error)**
   - Scale-independent (useful for comparison)
   - Interpretation: average % error
   - Shown as percentage

4. **CRPS (Continuous Ranked Probability Score)**
   - Evaluates full probabilistic forecast
   - Lower is better
   - Rewards calibrated uncertainty

**Typical results:**
```
Model              MAE      RMSE     MAPE    CRPS
--------------------------------------------------------
DeepAR             2,500    3,200    5.2%    1,800
SimpleFeedForward  3,100    4,000    6.5%    2,400
DeepNPTS           2,700    3,500    5.7%    2,000
```

(Actual values depend on test period and data quality)

**Visualizations:**

- Forecast vs actual plots
- Confidence intervals (10th-90th percentile)
- Quantile predictions
- Error distributions

### Section 6: Model Comparison

**Comparison criteria:**

1. **Accuracy:** Which model has lowest error?
2. **Uncertainty:** Are confidence intervals calibrated?
3. **Speed:** Training time matters for production
4. **Interpretability:** Can we explain predictions?

**Typical findings:**

**Best overall: DeepAR**

- Lowest MAE and MAPE
- Captures wave dynamics well
- Well-calibrated uncertainty
- Worth the extra training time

**Best for speed: SimpleFeedForward**

- 10x faster training
- Good enough for stable periods
- Great for quick experiments
- Use as baseline

**Best for regime changes: DeepNPTS**

- Handles distribution shifts
- Good during variant transitions
- Flexible uncertainty
- Consider during new waves

### Section 7: Scenario Analysis

**What it is:** Simulating "what if" public health interventions to guide policy
decisions.

**Scenarios tested:**

#### Scenario 1: Baseline (No Intervention)

- Current trends continue
- No policy changes
- Forecast: Expected trajectory

#### Scenario 2: Moderate Intervention

- 20% reduction in mobility
- Simulates mask mandates, capacity limits
- Forecast: Slower case growth

#### Scenario 3: Strong Intervention

- 40% reduction in mobility
- Simulates lockdowns, school closures
- Forecast: Significant case reduction

**How it works:**

1. Adjust mobility features by intervention strength
2. Re-run forecasts with modified features
3. Compare predicted case counts
4. Estimate intervention effectiveness

**Example results:**
```
Scenario              Predicted Cases (14 days)    Reduction
----------------------------------------------------------------
Baseline              65,000 (±8,000)              --
Moderate (-20%)       52,000 (±6,500)              20% fewer
Strong (-40%)         38,000 (±5,000)              42% fewer
```

**Insights for decision-makers:**

- Quantify intervention impact
- Balance health vs economic costs
- Plan resource allocation
- Communicate risk clearly

## Key Takeaways

### Technical Lessons

1. **Feature engineering matters:** Deaths and mobility data significantly
   improve forecasts

2. **Model choice depends on context:**
   - Stable periods → SimpleFeedForward
   - Complex patterns → DeepAR
   - Regime changes → DeepNPTS

3. **Uncertainty quantification is critical:** Point forecasts alone are
   insufficient for planning

4. **Computational efficiency:** SimpleFeedForward trains 10x faster with
   acceptable accuracy loss

### Domain Insights

1. **COVID-19 is highly seasonal:** Weekly patterns must be captured

2. **Multiple waves require flexible models:** One-size-fits-all doesn't work

3. **Behavioral data is predictive:** Mobility changes lead case changes

4. **Deaths data improves case forecasts:** Despite lag, strong correlation
   helps

### Practical Considerations

1. **14-day horizon is realistic:** Matches public health planning cycles

2. **Intervention scenarios are valuable:** Quantifying policy impact aids
   decision-making

3. **Real-time updates are essential:** Retrain as new data arrives

4. **Communication matters:** Visualizations bridge technical and non-technical
   audiences

## Running the Example

### Prerequisites

- Docker environment running (see main README)
- Data files in `data/` directory (auto-downloaded if missing)
- Estimated time: 10-15 minutes

### Steps

1. **Start Jupyter:**

   ```bash
   ./docker_jupyter.sh
   ```

2. **Open notebook:**
   - Navigate to `GluonTS.example.ipynb`

3. **Run all cells:**
   - "Restart & Run All" or step through manually

4. **Explore results:**
   - Visualizations of data and forecasts
   - Model comparison metrics
   - Scenario analysis insights

## Extending This Example

### Add More Features

- Vaccination rates (data in `data/vaccine.csv`)
- Weather data (temperature, humidity)
- Policy stringency indices
- Genomic surveillance (variant prevalence)

### Try Different Models

- Prophet (Facebook's forecasting tool)
- ARIMA (classical statistical model)
- LSTM (vanilla PyTorch implementation)

### Adjust Forecast Horizon

- Short-term: 7 days (operational planning)
- Medium-term: 14 days (current choice)
- Long-term: 28 days (strategic planning)

### Enhance Scenario Analysis

- Multiple intervention combinations
- Staged interventions (gradual relaxation)
- Regional variations (state-level)
- Cost-benefit analysis

## Reproducibility

All results are reproducible given:

- Same data files (in `data/`)
- Same random seeds (set in notebook)
- Same Docker environment
- Same hyperparameters

**Note:** Results may vary slightly due to:

- PyTorch non-determinism
- Training data order
- Hardware differences

## Further Reading

### GluonTS Resources

- [GluonTS Documentation](https://ts.gluon.ai/)
- [DeepAR Paper](https://arxiv.org/abs/1704.04110)

### COVID-19 Data

- [JHU COVID-19 Repository](https://github.com/CSSEGISandData/COVID-19)
- [Google Mobility Reports](https://www.google.com/covid19/mobility/)

### Time Series Forecasting

- "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

## See Also

- **GluonTS.API.md**: Reference guide for GluonTS models and parameters
- **README.md**: Project setup and quick start guide
