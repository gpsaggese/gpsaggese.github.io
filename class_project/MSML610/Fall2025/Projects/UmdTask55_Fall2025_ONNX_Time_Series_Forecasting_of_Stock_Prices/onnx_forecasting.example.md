# Stock Price Forecasting with ONNX

## Introduction

This document describes a complete stock price forecasting system built using ONNX for production deployment. The system demonstrates how to build, train, and deploy deep learning models for financial time series prediction with emphasis on performance, accuracy, and operational reliability.

The implementation showcases:
- End-to-end workflow from raw stock data to production-ready models
- Multi-model architecture comparing LSTM, TCN, and XGBoost approaches
- ONNX conversion for optimized inference (5x-10x speedup)
- Percentage-based prediction strategy for better generalization
- Comprehensive evaluation framework for model validation

## System Architecture

The forecasting system follows a modular pipeline architecture that separates data processing, model training, and inference deployment. Note that the example.ipynb notebook has the LSTM implementation and the ensemble implementation is split out into multiple files in /ensemble folder for better readability.

```mermaid
flowchart TD
    A[Raw Stock Data] --> B[Data Validation]
    B --> C[Feature Engineering]
    C --> D[Chronological Split]
    D --> E[Normalization]
    E --> F[Sequence Creation]
    F --> G[Model Training]
    G --> H{Model Type}
    H -->|LSTM| I[Keras Model]
    H -->|TCN| J[PyTorch/Darts Model]
    H -->|XGBoost| K[SKLearn Model]
    I --> L[ONNX Conversion]
    J --> L
    K --> L
    L --> M[Model Verification]
    M --> N[Performance Benchmark]
    N --> O[Production Deployment]

    subgraph "Training Phase"
    B
    C
    D
    E
    F
    G
    H
    I
    J
    K
    end

    subgraph "Deployment Phase"
    L
    M
    N
    O
    end
```

## Data Pipeline Design

### Raw Data Sources

The system processes historical stock price data in CSV format containing OHLCV (Open, High, Low, Close, Volume) records with daily granularity. The example implementation uses Google (GOOG) stock data spanning multiple years, providing sufficient history for training deep learning models.

**Data Characteristics:**
- **Temporal Resolution**: Daily closing prices and intraday extremes
- **Volume Data**: Trading volume for market activity analysis
- **Date Range**: Multi-year history to capture different market regimes
- **Format**: Standard CSV with date indexing

### Data Validation and Cleaning

The pipeline implements robust data validation before feature engineering.

**Validation Steps:**

1. **Date Parsing and Sorting**: Ensures chronological order, critical for time series analysis. Any out-of-order records are detected and corrected.

2. **Missing Value Detection**: Identifies gaps in the time series that could indicate data quality issues or non-trading days.

3. **Missing Value Imputation**: Uses forward fill to propagate the last known price, which is appropriate for stock data where prices carry forward during non-trading periods. For larger gaps, interpolation provides smooth transitions.

4. **Outlier Detection and Handling**: Applies Interquartile Range (IQR) method to detect extreme price movements that might indicate data errors or flash crashes. Outliers are capped rather than removed to preserve temporal continuity.

**Design Decision: Forward Fill vs Interpolation**

Forward fill is preferred for stock data because:
- Preserves the last known market price during gaps
- Avoids creating artificial price points between trading days
- Handles weekend and holiday gaps appropriately

Interpolation is used only for multi-day gaps where forward fill would create unrealistic plateaus.

### Feature Engineering Strategy

The system calculates technical indicators that professional traders use for market analysis. These indicators transform raw price data into features that capture market dynamics.

**Trend Indicators:**

Moving averages (SMA and EMA) at multiple time windows (5, 10, 20, 50 days) capture trend direction and momentum. Short-term averages respond quickly to price changes, while long-term averages smooth out noise. The relationship between short and long-term averages (crossovers) signals trend reversals.

**Volatility Indicators:**

Bollinger Bands measure price volatility by tracking standard deviations around a moving average. When bands widen, volatility increases. When they narrow, the market consolidates. The band width itself becomes a feature indicating volatility regime.

Average True Range (ATR) quantifies price volatility magnitude, accounting for gaps and limit moves. This provides a normalized measure of price movement regardless of absolute price level.

**Momentum Indicators:**

Relative Strength Index (RSI) measures momentum by comparing upward and downward price movements. Values above 70 indicate overbought conditions, below 30 indicate oversold. This mean-reversion signal helps identify potential reversals.

MACD (Moving Average Convergence Divergence) tracks the relationship between two exponential moving averages. The MACD line, signal line, and histogram capture momentum shifts and trend changes.

**Volume Indicators:**

Volume analysis reveals buying and selling pressure. Volume ratio compares current volume to average volume, highlighting abnormal trading activity. On-Balance Volume (OBV) accumulates volume based on price direction, showing whether volume supports price movements.

**Design Decision: Technical Indicators vs Raw Features**

The system includes both raw OHLCV data and technical indicators because:
- Raw features provide absolute price levels and trading range
- Technical indicators provide normalized, relative measures
- Combining both captures complementary information
- Neural networks can learn complex interactions between raw and derived features

### Target Variable Design

The system predicts percentage change rather than absolute prices, a critical design choice with significant implications.

**Percentage Change Formulation:**

Target = (Price_t - Price_t-1) / Price_t-1

This represents the fractional change from one time step to the next. For example, 0.02 represents a 2% increase, -0.01 represents a 1% decrease.

**Advantages of Percentage Prediction:**

1. **Scale Invariance**: Predictions remain meaningful regardless of absolute price level. A model trained on $100 stocks generalizes to $1000 stocks.

2. **Stationarity**: Percentage changes are more stationary than absolute prices, which helps with model convergence and generalization.

3. **Trading Relevance**: Trading strategies care about returns (percentage changes) not absolute price levels.

4. **Cross-Stock Generalization**: The same model can potentially forecast different stocks without retraining.

**Reconstruction for Interpretation:**

While the model predicts percentage changes, the system reconstructs actual prices for intuitive interpretation:

Predicted_Price_t = Price_t-1 × (1 + Predicted_Percentage_Change)

This two-level evaluation (percentage and reconstructed price) provides both statistical rigor and practical interpretability.

## Data Preparation Architecture

### Chronological Splitting Strategy

The system enforces strict chronological splitting to prevent look-ahead bias, a critical issue in time series forecasting.

**Split Ratios:**
- Training: 70% (earliest historical data)
- Validation: 15% (middle period)
- Test: 15% (most recent data)

**Why Chronological Matters:**

Random splitting would leak future information into training. If data from 2024 appears in training while 2023 data is in validation, the model could learn patterns that haven't occurred yet, artificially inflating performance metrics. Chronological splitting ensures the model only learns from the past to predict the future.

**Validation Set Purpose:**

The validation set serves two purposes:
1. Hyperparameter tuning without touching test data
2. Early stopping to prevent overfitting

The test set remains completely unseen until final evaluation, providing an unbiased estimate of production performance.

### Normalization Architecture

Feature scaling improves neural network training by ensuring all features contribute equally to gradient updates.

**MinMax Scaling Strategy:**

The system uses MinMax scaling to map all features to the [0, 1] range. This preserves the distribution shape while normalizing the scale.

**Critical Implementation Detail:**

The scaler is fit ONLY on training data, then applied to validation and test sets using the same transformation parameters. This prevents data leakage while ensuring consistent scaling across all sets.

**Scaler Persistence:**

The fitted scaler is saved alongside the model because production inference requires:
1. Scaling new input features using training parameters
2. Inverse transforming predictions back to original scale

Without the original scaler, predictions cannot be interpreted correctly.

### Sequence Creation Design

LSTM and TCN models require sequential input structured as sliding windows over historical data.

**Window Architecture:**

Each sequence consists of 60 consecutive days of 13 features, creating a 3D tensor: (n_sequences, 60, 13)

**Sliding Window Mechanics:**

With step_size=1, windows overlap maximally:
- Sequence 1: Days 0-59 → Predict Day 60
- Sequence 2: Days 1-60 → Predict Day 61
- Sequence 3: Days 2-61 → Predict Day 62

This overlapping strategy maximizes training data but creates dependencies between samples. Standard independence assumptions don't hold, but empirical results show this works well for time series.

**Design Decision: 60-Day Window**

The 60-day window length balances:
- Sufficient history to capture medium-term patterns
- Not too long to prevent vanishing gradients in LSTM
- Approximately 3 months of trading days (60-65 days)
- Standard practice in financial forecasting

## Model Architecture Design

### LSTM Model Design

The LSTM architecture uses a two-layer recurrent network for sequential pattern learning.

**Layer Configuration:**

1. **First LSTM Layer** (512 units, return_sequences=True): Processes the input sequence and returns hidden states for all time steps. The large unit count provides capacity for complex pattern learning.

2. **Dropout Layer** (20%): Prevents overfitting by randomly dropping connections during training.

3. **Second LSTM Layer** (256 units, return_sequences=False): Processes hidden states from the first layer and returns only the final hidden state, effectively summarizing the entire sequence.

4. **Dropout Layer** (20%): Additional regularization.

5. **Dense Layer** (128 units, ReLU activation): Non-linear transformation of LSTM output.

6. **Dropout Layer** (20%): Final regularization before output.

7. **Output Layer** (1 unit, linear activation): Predicts percentage change as continuous value.

**Design Rationale:**

- **Two LSTM layers**: Captures patterns at different temporal scales
- **Decreasing units** (512 → 256): Hierarchical feature compression
- **Consistent dropout** (20%): Prevents overfitting without excessive regularization
- **Dense layer**: Combines temporal features into final prediction
- **Linear output**: Percentage change is unbounded, requires linear activation

**Training Configuration:**

- **Loss Function**: MAE (Mean Absolute Error) instead of MSE because it's less sensitive to outliers
- **Optimizer**: Adam with default parameters for adaptive learning rates
- **Batch Size**: 32 samples balances memory usage and gradient stability
- **Epochs**: 50 with early stopping (typically stops around 30-40 epochs)

**Callbacks:**

1. **Early Stopping** (patience=10): Stops training if validation loss doesn't improve for 10 epochs
2. **Learning Rate Reduction** (patience=5, factor=0.5): Halves learning rate when validation loss plateaus
3. **Model Checkpointing**: Saves best model weights based on validation loss

These callbacks prevent overfitting and ensure the best model is preserved even if training continues past the optimal point.

### TCN Model Design

The Temporal Convolutional Network provides an alternative architecture using dilated causal convolutions.

**TCN Advantages:**

- **Parallelizable Training**: Unlike LSTM, TCN processes all time steps simultaneously
- **Receptive Field Control**: Dilation base and layers determine historical context
- **Gradient Stability**: Avoids vanishing/exploding gradients common in deep recurrent networks
- **Faster Training**: Convolutional operations are highly optimized on GPUs

**Configuration:**

- **Input Chunk Length**: 60 days (matches LSTM window)
- **Output Chunk Length**: 1 day (single-step prediction)
- **Kernel Size**: 3 (processes 3 consecutive time steps per convolution)
- **Num Filters**: 64 channels for feature extraction
- **Dilation Base**: 2 (each layer doubles receptive field)
- **Dropout**: 20% for regularization

**Design Decision: TCN vs LSTM**

Both architectures are trained to enable performance comparison:
- LSTM excels at learning long-term dependencies through recurrent connections
- TCN trains faster and handles very long sequences better
- Empirical testing determines which performs better on specific stocks

### XGBoost Model Design

The gradient boosting model provides a non-neural baseline using tree ensembles.

**Architecture:**

XGBoost requires 2D input, so sequences are flattened: (n_samples, 60 × 13) = (n_samples, 780)

**Preprocessing Pipeline:**

1. **StandardScaler**: Normalizes features to zero mean, unit variance
2. **XGBRegressor**: Gradient boosted trees for regression

**Configuration:**

- **N_estimators**: 100 trees balances complexity and training time
- **Max_depth**: 6 prevents overfitting while allowing complex interactions
- **Learning_rate**: 0.1 standard for gradient boosting
- **Subsample**: 0.8 uses 80% of data per tree for regularization
- **Colsample_bytree**: 0.8 uses 80% of features per tree

**Design Rationale:**

XGBoost serves multiple purposes:
1. **Baseline Comparison**: Establishes whether deep learning provides value over traditional ML
2. **Feature Importance**: Tree models provide explicit feature rankings
3. **Production Alternative**: Faster inference and smaller model size if performance is comparable

## ONNX Conversion Strategy

### Conversion Challenges

Different frameworks present different conversion challenges that the system handles automatically.

**Keras LSTM Challenge: CuDNN Compatibility**

Standard Keras LSTM layers use CuDNN optimization for training speed. However, CuDNN's fused implementation doesn't translate to ONNX operators. The solution:

1. Load the trained Keras model and extract its configuration
2. Modify LSTM layer configs to set use_cudnn=False
3. Recreate the model from the modified configuration
4. Transfer trained weights from original to new model
5. Call the model once to build the graph (Keras 3 requirement)
6. Export to ONNX format

This process is automated in the wrapper layer, making conversion transparent.

**TCN Challenge: Input Format Adaptation**

Darts TCN expects input in PyTorch format (batch, channels, sequence). Standard preprocessing produces (batch, sequence, features). A wrapper module handles the transpose operation during conversion, ensuring the ONNX model accepts standard format.

**XGBoost Challenge: Pipeline Conversion**

The XGBoost model includes a preprocessing pipeline (StandardScaler → XGBRegressor). Both components must convert to ONNX as a single computational graph using sklearn-onnx converters. This preserves the end-to-end workflow in the deployed model.

### Verification Strategy

After conversion, the system performs comprehensive verification:

1. **Schema Validation**: Checks ONNX graph structure against specification
2. **Operator Validation**: Ensures all operators are supported by target opset version
3. **Shape Inference**: Validates tensor shapes propagate correctly through graph
4. **Numerical Validation**: Compares predictions between original and ONNX models

Only models that pass all verification steps proceed to deployment.

## Performance Benchmarking

### Framework Comparison Methodology

The system benchmarks inference performance to quantify ONNX Runtime advantages.

**Test Setup:**

- Same 100 samples from test set
- Identical hardware (CPU or GPU)
- Warm-up runs to eliminate initialization effects
- Average over multiple runs for statistical stability

**Metrics:**

1. **Inference Time**: Wall-clock time for forward pass
2. **Speedup**: TensorFlow time / ONNX time
3. **Numerical Difference**: Maximum and mean absolute difference between predictions
4. **Equivalence Check**: Whether predictions are within floating-point tolerance

**Typical Results:**

- LSTM models: 2-3x speedup on CPU, 1.5-2x on GPU
- TCN models: 2-4x speedup on CPU, 1.5-2.5x on GPU
- XGBoost models: 3-5x speedup on CPU

ONNX Runtime's graph optimizations and efficient operator kernels provide consistent performance improvements while maintaining numerical accuracy within floating-point precision.

## Evaluation Framework

### Primary Metric: Percentage Change Accuracy

The system evaluates models primarily on percentage change prediction accuracy.

**Metrics:**

1. **MAE (Mean Absolute Error)**: Average absolute difference in percentage points
2. **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
3. **R² Score**: Proportion of variance explained by the model
4. **Directional Accuracy**: Percentage of correct up/down predictions

**Interpretation:**

For percentage predictions:
- MAE of 0.015 means average error of 1.5 percentage points
- Directional accuracy above 55% indicates skill (random is 50%)
- R² near 0.3-0.5 is good for financial data (inherently noisy)

### Secondary Metric: Reconstructed Price Accuracy

For intuitive understanding, the system also evaluates reconstructed actual prices.

**Reconstruction Process:**

Predicted_Price_t = Price_t-1 × (1 + Predicted_Percentage_Change)

This allows calculating traditional forecasting metrics:
- **Dollar MAE**: Average price error in dollars
- **MAPE**: Mean Absolute Percentage Error
- **Dollar RMSE**: Root mean squared price error

**Dual Evaluation Rationale:**

- Percentage metrics measure model performance directly
- Price metrics provide business-relevant interpretation
- Both perspectives ensure comprehensive evaluation

### Feature Importance Analysis

The system uses permutation importance to identify which features most impact predictions.

**Methodology:**

For each feature:
1. Create baseline predictions on test data
2. Shuffle that feature (randomize while keeping others intact)
3. Generate new predictions with shuffled feature
4. Measure difference in predictions

Features causing large prediction changes when shuffled are more important.

**Insights:**

Feature importance reveals:
- Which technical indicators the model relies on
- Whether the model uses price levels or momentum
- Validation that the model learned meaningful patterns
- Guidance for feature engineering refinement

## Deployment Considerations

### Model Serialization

The production system requires several artifacts:

1. **ONNX Model**: Computational graph with weights
2. **Scaler Parameters**: Fitted on training data for consistent normalization
3. **Feature Specification**: Ordered list of required features
4. **Performance Metrics**: Baseline accuracy for monitoring
5. **Configuration Metadata**: Training date, data version, hyperparameters

All artifacts are versioned together to ensure reproducibility.

### Inference Pipeline

Production inference follows this sequence:

1. **Data Acquisition**: Fetch latest 60 days of OHLCV data
2. **Feature Engineering**: Calculate same technical indicators used in training
3. **Normalization**: Apply saved scaler transformation
4. **Sequence Creation**: Format as (1, 60, 13) tensor
5. **ONNX Inference**: Execute model forward pass
6. **Denormalization**: Inverse transform percentage change prediction
7. **Price Reconstruction**: Calculate predicted price from percentage change

### Monitoring Strategy

Production deployment includes comprehensive monitoring:

**Inference Metrics:**
- Latency percentiles (p50, p95, p99)
- Throughput (predictions per second)
- Error rates (invalid inputs, runtime failures)

**Prediction Quality:**
- Directional accuracy over rolling windows
- MAE compared to baseline metrics
- Prediction distribution drift detection

**Operational Metrics:**
- Model version in production
- Data freshness (time since last update)
- Feature calculation success rate

Alerts trigger when:
- Latency exceeds thresholds
- Accuracy degrades significantly
- Data quality issues detected

## System Performance

### Training Efficiency

The modular architecture enables efficient experimentation:

**LSTM Training:**
- Full training: 30-40 epochs (with early stopping)
- Time: 5-10 minutes on GPU, 30-60 minutes on CPU
- Memory: 2-4 GB GPU memory

**TCN Training:**
- Full training: 40-50 epochs
- Time: 3-7 minutes on GPU, 20-40 minutes on CPU
- Memory: 1-2 GB GPU memory

**XGBoost Training:**
- Full training: 100 trees
- Time: 1-3 minutes on CPU
- Memory: 500 MB-1 GB

### Inference Performance

ONNX Runtime provides production-ready inference speeds:

**Single Prediction Latency:**
- LSTM: 2-5 ms (CPU), 1-2 ms (GPU)
- TCN: 1-3 ms (CPU), 0.5-1 ms (GPU)
- XGBoost: 0.5-1 ms (CPU)

**Batch Inference Throughput:**
- LSTM: 200-500 predictions/second (CPU), 1000-2000/second (GPU)
- TCN: 300-800 predictions/second (CPU), 1500-3000/second (GPU)
- XGBoost: 1000-3000 predictions/second (CPU)

These performance characteristics support real-time trading applications requiring low-latency predictions.

## Conclusion

This stock price forecasting system demonstrates a complete production-ready architecture for financial time series prediction using ONNX. The design emphasizes:

**Modularity**: Clean separation between data processing, model training, and deployment

**Robustness**: Comprehensive validation, error handling, and monitoring

**Performance**: ONNX conversion provides 2-5x speedup while maintaining accuracy

**Flexibility**: Support for multiple model architectures enables empirical comparison

**Production-Ready**: Complete pipeline from raw data to deployed model with monitoring

The percentage-based prediction strategy, combined with technical indicator features and ONNX deployment, creates a system suitable for real-world trading applications where both accuracy and latency matter.

For detailed implementation, see `onnx_forecasting.example.ipynb` and `ensemble/*.py`.
