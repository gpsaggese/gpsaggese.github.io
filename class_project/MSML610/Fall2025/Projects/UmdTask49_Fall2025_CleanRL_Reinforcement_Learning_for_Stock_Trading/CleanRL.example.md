# CleanRL for Stock Trading Signal Generation

## Building an Ensemble RL Strategy

After extensive research into applying reinforcement learning to financial markets, I discovered that single RL agents (PPO, SAC) are not enough on their own. Through experimentation and failed attempts, I learned several critical lessons that shaped this pipeline.

## Why I Built This Architecture

### The Problem with Pure RL Approaches

When I first started this project, I tried the straightforward approach: feed OHLCV data and news directly into an RL agent and let it learn trading signals. This failed spectacularly for several reasons:

1. **RL agents are decision makers, not forecasters**: I realized that PPO and SAC are powerful at learning optimal policies given good state representations, but they're fundamentally not designed to forecast asset movements or detect market regimes from raw price data. They need features that already encode meaningful patterns.

2. **Raw TF-IDF vectors don't work**: My initial attempt involved dumping TF-IDF news vectors directly into the RL agent's observation space. The performance remained completely unchanged—the agent couldn't extract anything meaningful from the high-dimensional sparse text features. It was like giving someone a dictionary and expecting them to understand the story.

3. **Uncertainty matters in finance**: Markets are inherently uncertain, and point forecasts are insufficient. I needed a way to quantify uncertainty and give the RL agent confidence intervals rather than single predictions.

### The Solution: An Ensemble Pipeline

After reviewing recent research on hybrid approaches in algorithmic trading, I designed a three-model ensemble:

**Model A-L (LSTM Forecaster with MC Dropout)**

- Purpose: Forecasts uncertainty cones for price and volatility
- Why: LSTMs can capture temporal dependencies in price movements, and Monte Carlo Dropout provides uncertainty quantification
- Output: 10th, 50th, 90th percentiles for both price and volatility (6 paths × 20 days = 120 features)

**Model B-L (News Interpreter with LM Dictionary + LDA)**

- Purpose: Extracts structured context from news text
- Why: Instead of raw TF-IDF, I use LM (Loughran-McDonald) financial sentiment dictionary combined with LDA topic modeling to create interpretable features
- Output: 3 sentiment scores (positive/negative/uncertainty) + 15 topic probabilities = 18 features

**Model C (RL Agent - PPO/SAC)**

- Purpose: Synthesizes the enriched state into optimal trading signals
- Why: Now that the state encodes meaningful patterns (uncertainty + context), the RL agent can focus on decision-making
- Output: Trading signal vector [directional_bias, volatility_bias, duration]

This decomposition works because each model does what it's good at: the LSTM forecasts, the NLP model interprets, and the RL agent decides.

## Why Monte Carlo Dropout and Uncertainty Forecasting?

I chose Monte Carlo Dropout for uncertainty quantification for several quantitative reasons:

1. **Bayesian approximation at low cost**: MC Dropout provides an approximate Bayesian posterior without the computational overhead of full Bayesian inference. Each forward pass with dropout enabled samples from an implicit distribution over predictions.

2. **Epistemic uncertainty**: In financial forecasting, we care about model uncertainty (epistemic) not just data noise (aleatoric). MC Dropout captures epistemic uncertainty by showing us how confident the model is in its predictions.

3. **Practical benefits for RL**:

   - **Risk management**: Wide uncertainty cones → agent learns to reduce position size or duration
   - **Regime detection**: Uncertainty patterns change across market regimes (bull/bear/sideways)
   - **Calibration**: Percentile forecasts provide natural confidence intervals that the RL agent can condition on

The key insight: by giving the RL agent a distribution of possible futures rather than a single prediction, it learns more robust policies that account for uncertainty.

## The Full Pipeline: From Data to Signals

Let me walk through each step and where the code lives:

### Step 1: Data Preparation (`rl_utils/data.py`)

**What happens:**

- Fetch historical OHLCV data for the target asset
- Fetch corresponding news articles from Polygon API
- Calculate technical indicators (moving averages, volatility, momentum, etc.)
- Create rolling news windows (e.g., past 30 days of news for each trading day)

**Code location:** `rl_utils/data.py::prepare_data_features()`
**Note** all code within `rl_utils` folder is prebuilt and imported from my personal quantitative strategy project.

**Key design choice:** I use caching aggressively because API calls are slow and expensive. Each data/news fetch is cached locally with a hash-based key.

### Step 2: Model Training (`rl_env.py`)

**Model A-L (LSTM Forecaster):**

```python
# Location: rl_env.py::train_uncertainty_forecaster()
# Trains on: 60-day lookback windows → 20-day forward predictions
# Output: Saves trained model, used for inference in next step
```

The LSTM is trained to predict two targets:

- Log returns: `log(price_t+1 / price_t)`
- Realized volatility: Rolling 5-day standard deviation of returns

Why these targets? Log returns are more stationary than raw prices, and realized volatility is what we actually care about for risk management.

**Model B-L (News Interpreter):**

```python
# Location: rl_env.py::NewsInterpreter.fit()
# Fits on: Entire corpus of news documents
# Output: Fitted TF-IDF vectorizer + LDA model
```

The interpreter is fitted once on all available news and then used for fast inference. I use:

- TF-IDF with financial stop words removed
- LDA with 15 topics (emperically, more topics led to noise)
- LM Dictionary for sentiment (pre-defined lists of positive/negative/uncertain financial words)

### Step 3: Feature Pre-computation (`rl_env.py::prepare_forecasts_and_context()`)

**Critical optimization:** Instead of computing forecasts on-the-fly during RL training (which would be impossibly slow), I pre-compute everything once:

```python
for t in range(lookback_window, len(data) - forecast_horizon):
    # Generate MC Dropout samples (100 forward passes)
    uncertainty_cone = forecaster.predict_with_uncertainty(window_t)

    # Extract news context
    news_context = interpreter.get_full_context(news_t)

    # Cache both
    cache[t] = (uncertainty_cone, news_context)
```

This takes time upfront (a few minutes for 200 days) but makes training instant. The RL agent just loads precomputed features.

### Step 4: Environment Registration (`rl_env.py::register_cleanrl_env()`)

This is the glue that makes everything work with cleanRL:

```python
def register_cleanrl_env(env_id, data, news_documents, ...):
    # 1. Pre-compute all forecasts and contexts
    uncertainty_forecasts, news_contexts = prepare_forecasts_and_context(...)

    # 2. Register a gym id that closes over these precomputed features
    def _entry_point():
        return SignalTesterEnv(
            data=data,
            uncertainty_forecasts=uncertainty_forecasts,  # Shared reference
            news_contexts=news_contexts  # Shared reference
        )

    gym.register(id=env_id, entry_point=_entry_point)
```

Now `gym.make('SignalTester-v0')` creates environments that share the precomputed features (memory efficient).

### Step 5: RL Training

TODO

## Strategy Validation and Performance

TODO

### How I Test Signals

TODO

**Validation approach:**

1. **Out-of-sample testing**: Train on first 70% of data, test on last 30%
2. **Walk-forward validation**: Retrain periodically and test on next unseen window
3. **Signal quality metrics**: Measure signal consistency, turnover, and execution cost

**Code location:** `rl_env.py::SignalTesterEnv._calculate_reward()`

The environment calculates realized metrics during testing:

```python
# For each signal, measure actual outcomes:
actual_sharpe = (mean_return / std_return) * sqrt(252)  # Annualized
max_drawdown = max(cumulative_peak - cumulative_return)
win_rate = (profitable_signals / total_signals)
```

### Performance Metrics I Track

TODO

### Computational Cost

**Training time:** (on M1 Mac, CPU only)

TODO

## Conclusion

TODO
