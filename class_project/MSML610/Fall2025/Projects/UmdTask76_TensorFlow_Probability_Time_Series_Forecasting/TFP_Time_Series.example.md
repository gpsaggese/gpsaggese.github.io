# Example: Air Quality Forecasting with TFP

## The Problem
Air quality monitoring is critical for public health. Sensors often capture data with missing chunks (due to maintenance or failure). We need a model that can:
1.  **Forecast** future Carbon Monoxide (CO) levels.
2.  **Quantify Uncertainty** (How confident are we?).
3.  **Handle Missing Data** gracefully.

## The Data
- **Source:** [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/air+quality).
- **Features:** Hourly averaged CO concentrations from a sensor in an Italian city.
- **Preprocessing:**
    - Parsed timestamps.
    - Resampled to strict hourly frequency.
    - Interpolated small gaps to ensure continuity.

## The Approach
We compare two methods:

### 1. Baseline: ARIMA
A standard statistical model (AutoRegressive Integrated Moving Average).
- **Configuration:** Order (24, 1, 0) to account for the strong 24-hour daily cycle.
- **Limitation:** Provides a point forecast but assumes constant variance (homoscedasticity).

### 2. Probabilistic: Gaussian Process (Our Tool)
A Bayesian non-parametric model using TensorFlow Probability.
- **Kernel:** `ExponentiatedQuadratic` (Trend) + `ExpSinSquared` (Seasonality).
- **Advantage:** Outputs a full probability distribution for every time step.

## Results
The Gaussian Process successfully captured the daily peaks and troughs of traffic pollution.

| Model | RMSE (Lower is Better) | Notes |
| :--- | :--- | :--- |
| **ARIMA** | 1.4053 | Good baseline, captured seasonality well. |
| **Gaussian Process** | **1.3922** | **Best Performance.** Also provided 95% confidence intervals. |

## Conclusion
The **Gaussian Process** outperformed the ARIMA baseline. More importantly, the GP provided "error bars" (uncertainty), which are crucial for decision-making in environmental safety.

## System Workflow

```mermaid
flowchart TD
    A[Raw Air Quality Data] --> B(Preprocessing)
    B -->|Resample & Interpolate| C{Model Selection}
    C -->|Baseline| D[ARIMA Model]
    C -->|Probabilistic| E[Gaussian Process]
    D --> F[Point Forecast]
    E --> G[Forecast + Uncertainty Bands]
    F --> H[RMSE Comparison]
    G --> H
    H --> I[Final Decision]