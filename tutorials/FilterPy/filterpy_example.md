# FilterPy for Financial Applications - Jupyter Notebook Script

A Jupyter notebook for college students explaining FilterPy Kalman filter
variants applied to financial time series problems.

---

## Cell 1: Introduction - Signal vs Noise in Financial Markets

- Purpose: Build intuition for why Bayesian filtering matters in finance by
  framing price data as a noisy observation of a hidden market state
- Display:
  - Two-panel side-by-side matplotlib figure:
    - Panel 1: Raw daily stock price time series (jagged orange line) overlaid
      with a smooth "true underlying value" curve (blue); title "What We See
      (Noisy Prices) vs What We Want (True Value)"
    - Panel 2: Static flow diagram using annotated arrows:
      True Market State -> [Process Noise Q: new information, macro shocks] ->
      Predicted State -> [Measurement Noise R: bid-ask spread, rounding,
      sentiment] -> Observed Price -> [Kalman Gain K] -> Updated State Estimate
  - Markdown cell below explaining the two questions a filter answers:
    - "Where is the market now?" (state estimate x)
    - "How confident are we?" (uncertainty P)
  - Table summarizing Kalman matrix roles in finance:
    - F: price dynamics model (e.g., random walk, mean reversion)
    - H: mapping from hidden state to observed price
    - Q: process noise (market volatility, model uncertainty)
    - R: measurement noise (observation error, microstructure noise)
    - P: current uncertainty about the hidden state
- Interactive widget: None
- Key insights: Every Kalman variant shares the predict-update cycle; they
  differ only in how they handle nonlinearity in F or H; finance provides
  natural examples of all four filter types

---

## Cell 2: Linear Kalman Filter - Extracting Price Trend from Noise

- Purpose: Show how KalmanFilter tracks the latent price trend from noisy
  daily observations using a constant-velocity (trend + drift) state model
- Display:
  - Single-panel time series plot with three elements:
    - True latent price trend (smooth blue line)
    - Noisy observed prices (orange scatter dots)
    - KF estimated trend (green solid line)
    - Shaded 1-sigma confidence band (light green) around KF estimate
  - Title: "Kalman Filter: Extracting Trend from Noisy Stock Prices"
  - x-axis: "Trading Day"; y-axis: "Price ($)"
  - ylim fixed to [80, 140] to avoid jumps when widgets change
  - Legend showing RMSE of KF estimate vs true trend
- Interactive widget:
  - Slider for measurement noise R (0.5 to 50, step 0.5, initial 5.0), label
    "Observation Noise R (microstructure noise)"
  - Slider for process noise Q (0.001 to 2.0, step 0.01, initial 0.1), label
    "Process Noise Q (market volatility)"
  - Output: redraws the full plot with updated R and Q; RMSE updates in legend
- Key insights: Larger R means we trust prices less and smooth more; larger Q
  lets the filter track rapid price changes; the ratio R/Q is the key tuning
  knob; this is equivalent to an exponential moving average when Q is small

---

## Cell 3: Linear Kalman Filter - Kalman Gain and Uncertainty Convergence

- Purpose: Show how the state covariance P and Kalman gain K evolve over
  time, giving intuition about when the filter trusts new prices vs the model
- Display:
  - Two-panel row:
    - Panel 1: Time series of position variance P[0,0] (blue) and velocity
      variance P[1,1] (orange) converging to steady state; x-axis "Trading
      Day"; y-axis "Variance"; title "Uncertainty P Converges Over Time";
      ylim fixed to [0, 500]
    - Panel 2: Time series of Kalman gain K[0,0] (the weight on new
      observations) starting near 1 and decreasing to steady state; title
      "Kalman Gain K: How Much to Trust New Prices"; ylim fixed to [0, 1.1];
      annotated horizontal dashed line at K=0.5 labeled "Equal trust:
      model vs measurement"
  - Markdown below: "High K means the filter is uncertain and leans heavily on
    new price data; low K means it is confident in its model prediction"
- Interactive widget:
  - Slider for initial uncertainty P0 (10 to 10000, logarithmic, initial 1000),
    label "Initial Uncertainty P0 (investor confidence at start)"
  - Slider for measurement noise R (0.5 to 50, step 0.5, initial 5.0), label
    "Observation Noise R"
  - Output: redraws both panels; note that P always converges regardless of P0
- Key insights: P converges to a steady state independent of P0; K decreases
  as the filter learns; this mirrors how an investor becomes more confident
  over time; high initial P0 models high uncertainty at market open

---

## Cell 4: Extended Kalman Filter - Time-Varying Beta in Pairs Trading

- Purpose: Introduce the EKF for tracking a nonlinear, time-varying hedge
  ratio (beta) between two correlated stocks in a pairs trading strategy
- Display:
  - Three-panel column:
    - Panel 1: Two simulated stock price series (Stock A in blue, Stock B in
      orange) over 200 trading days; title "Cointegrated Stock Pair (e.g.,
      XOM vs CVX)"; ylim fixed to [80, 160]
    - Panel 2: True time-varying beta (blue solid), EKF estimated beta (green
      dashed), naive rolling OLS beta (red dotted); title "True vs EKF
      Estimated Hedge Ratio beta"; ylim fixed to [0.3, 2.5]; legend with RMSE
      for EKF and rolling OLS
    - Panel 3: Spread = Stock A - beta * Stock B, using EKF beta (green) and
      rolling OLS beta (red); title "Pairs Trading Spread"; annotated
      horizontal lines at +2 sigma and -2 sigma as trading signals; ylim
      fixed to [-15, 15]
  - Markdown: "EKF tracks beta in real time as the relationship evolves;
    cleaner spread means fewer false trading signals"
- Interactive widget:
  - Slider for beta drift volatility sigma_beta (0.001 to 0.2, step 0.005,
    initial 0.05), label "How Fast Beta Changes (beta drift)"
  - Slider for measurement noise R (0.1 to 10, step 0.1, initial 1.0), label
    "Price Observation Noise R"
  - Output: redraws all three panels; RMSE updates in legend
- Key insights: Beta is not constant; EKF adapts faster than rolling OLS to
  regime changes; a better beta estimate reduces spread noise and improves
  signal quality; the Jacobian linearizes the nonlinear beta-price relationship

---

## Cell 5: Extended Kalman Filter - Linearization in Log-Return Space

- Purpose: Visually show how EKF linearizes the nonlinear log transformation
  used in log-return models and where the approximation breaks down
- Display:
  - Three-panel row:
    - Panel 1: Nonlinear function h(x) = log(x) (true curve, blue) and its
      tangent line at operating point x0 (Jacobian, green dashed); vertical
      dashed line at x0; title "EKF Linearizes log(x) at Current Estimate x0";
      xlim fixed to [0.1, 5]; ylim fixed to [-3, 2]; annotated divergence arrow
      showing error grows away from x0
    - Panel 2: Input Gaussian p(x) centered at x0 (blue shaded bell curve);
      title "Input Distribution: Price Uncertainty"; xlim fixed to [0.1, 5]
    - Panel 3: True propagated distribution through h (blue, from Monte Carlo),
      vs EKF Gaussian approximation (green dashed); title "True vs EKF
      Approximated Output Distribution"; comment box explaining that EKF
      underestimates skewness; xlim fixed to [-4, 2]
  - Markdown: "When price uncertainty is large or the log function is highly
    curved near x0, EKF introduces significant bias"
- Interactive widget:
  - Slider for operating point x0 (0.2 to 4.0, step 0.1, initial 1.0), label
    "Current Price Estimate x0"
  - Slider for input uncertainty sigma (0.05 to 1.5, step 0.05, initial 0.3),
    label "Price Uncertainty sigma"
  - Output: redraws all three panels; shows increased error for large sigma
- Key insights: EKF error grows with uncertainty and curvature; log-return
  models are mildly nonlinear near x0=1; far from x0 (e.g., crash scenarios)
  EKF can diverge; this motivates UKF for more volatile regimes

---

## Cell 6: Unscented Kalman Filter - Stochastic Volatility Estimation

- Purpose: Show how UKF uses sigma points to estimate latent volatility from
  observed returns without computing Jacobians
- Display:
  - Three-panel row:
    - Panel 1: 2D input distribution of [log-price, log-volatility] as a
      scatter of light blue Monte Carlo samples with UKF sigma points overlaid
      as red X marks; title "Sigma Points in [log-price, log-vol] State Space";
      xlim and ylim fixed to [-4, 4]
    - Panel 2: Sigma points after propagation through nonlinear dynamics (red X
      marks) with true Monte Carlo propagated samples (light blue dots); title
      "Propagated Sigma Points vs True Distribution"; xlim and ylim fixed
      to [-4, 4]
    - Panel 3: Time series of true latent volatility (blue), UKF estimated
      volatility (green dashed), EKF estimated volatility (red dotted), and
      GARCH(1,1) estimate (grey dashed); title "Volatility Estimation: UKF vs
      EKF vs GARCH"; ylim fixed to [0, 0.6]; RMSE in legend for each method
  - Markdown: "Volatility is not directly observable; UKF infers it from price
    returns without linearization"
- Interactive widget:
  - Slider for sigma-point spread alpha (0.001 to 1.0, step 0.01, initial 0.1),
    label "Sigma Point Spread alpha"
  - Slider for volatility of volatility kappa (0.01 to 0.5, step 0.01,
    initial 0.1), label "Volatility of Volatility kappa"
  - Output: redraws all three panels; RMSE updates
- Key insights: UKF captures the asymmetry of log-volatility without Jacobians;
  2n+1 = 5 sigma points span the state distribution; UKF outperforms EKF when
  kappa (volatility of volatility) is high

---

## Cell 7: UKF vs EKF - Volatility Estimation Under Market Stress

- Purpose: Directly compare UKF and EKF on stochastic volatility estimation
  during a simulated market stress event (sudden volatility spike)
- Display:
  - Three-panel row:
    - Panel 1: Simulated log-returns time series with a volatility spike in the
      middle; annotated box showing the stress period; title "Simulated Returns
      with Volatility Spike"; ylim fixed to [-0.3, 0.3]
    - Panel 2: EKF estimated volatility (green) vs true volatility (blue),
      with RMSE in legend; vertical shaded region during stress period; title
      "EKF Volatility Estimate"; ylim fixed to [0, 0.6]; annotated divergence
      region showing EKF falling behind
    - Panel 3: UKF estimated volatility (green) vs true volatility (blue),
      with RMSE in legend; vertical shaded region during stress period; title
      "UKF Volatility Estimate"; ylim fixed to [0, 0.6]
  - Comment box below: "UKF tracks the volatility spike more accurately
    because sigma points capture the nonlinear amplification during stress;
    EKF linearization underestimates tail risk"
- Interactive widget:
  - Slider for spike magnitude (1.0 to 10.0, step 0.5, initial 4.0), label
    "Volatility Spike Size (x normal vol)"
  - Slider for spike duration in days (5 to 40, step 1, initial 10), label
    "Stress Period Duration (days)"
  - Output: redraws all three panels; RMSE updates in each legend
- Key insights: EKF degrades faster than UKF under nonlinear stress regimes;
  UKF RMSE advantage grows with spike magnitude; both filters recover after
  the stress period; for linear periods their performance is identical

---

## Cell 8: Ensemble Kalman Filter - Portfolio Risk Scenario Analysis

- Purpose: Show EnsembleKalmanFilter as a Monte Carlo tool where each particle
  represents a possible portfolio state (price + volatility + correlation)
- Display:
  - Three-panel row:
    - Panel 1: Scatter of ensemble particles (small colored dots) at each time
      step, colored by time from blue (early) to red (late), with ensemble
      mean overlaid as a bold white line; x-axis "Price", y-axis "Volatility";
      title "Ensemble Particles: Portfolio State Distribution Over Time";
      xlim fixed to [80, 140]; ylim fixed to [0, 0.5]
    - Panel 2: Time series of ensemble mean price (green) vs true price (blue),
      with shaded 1-sigma and 2-sigma ensemble spread bands; title "Ensemble
      Price Estimate with Risk Bands"; ylim fixed to [80, 140]; annotated
      label "VaR region" on the lower 5th-percentile band
    - Panel 3: Histogram of ensemble particle prices at the final time step;
      vertical dashed lines at 5th percentile (Value at Risk) and 95th
      percentile; title "Final Portfolio Price Distribution (Value at Risk)";
      shaded red region below 5th percentile
  - Markdown: "The ensemble spread is a natural measure of tail risk; unlike
    Gaussian approximations, the ensemble can capture skewed and fat-tailed
    distributions"
- Interactive widget:
  - Slider for ensemble size N (10 to 500, step 10, initial 100), label
    "Ensemble Size N (number of scenarios)"
  - Slider for process noise Q (0.001 to 1.0, step 0.01, initial 0.1), label
    "Process Noise Q (market uncertainty)"
  - Output: reruns EnKF with new N and Q, redraws all three panels; VaR value
    updates in panel 3 title
- Key insights: Larger N gives a more accurate distribution but costs more
  computation; the ensemble naturally produces a full distribution not just a
  mean and variance; this is the bridge between Kalman filtering and Monte
  Carlo risk simulation

---

## Cell 9: All Four Filters - Financial State Estimation Comparison

- Purpose: Directly compare KF, EKF, UKF, and EnKF on the same financial
  state estimation problem (price trend + stochastic volatility) and
  summarize when to use each filter in practice
- Display:
  - Four-panel row (one per filter: KF, EKF, UKF, EnKF):
    - Each panel shows: true price trend (blue), noisy observed prices (orange
      scatter), filter estimate (green), 1-sigma band (light green shaded)
    - RMSE printed in each panel legend
    - x-axis "Trading Day"; y-axis "Price ($)"; xlim fixed to [0, 100];
      ylim fixed to [80, 140]
  - Summary comment box below the four panels:
    - "KF: optimal for linear models; fastest; fails for nonlinear dynamics"
    - "EKF: good for mildly nonlinear; requires Jacobians; fast"
    - "UKF: better for strongly nonlinear; no Jacobians; 3x slower than EKF"
    - "EnKF: most flexible; handles any distribution; slowest; scales with N"
  - Bar chart below comparing RMSE of all four filters; colors matching the
    panels above
  - Decision flowchart: "Is the model linear? -> KF; Is nonlinearity mild?
    -> EKF; Is nonlinearity strong? -> UKF; Do you need a full distribution?
    -> EnKF"
- Interactive widget: None
- Key insights: No single filter dominates all scenarios; KF is the baseline
  for linear financial models like simple trend + noise; EKF and UKF cover
  most nonlinear financial models (stochastic volatility, regime switching);
  EnKF is the tool of choice for portfolio-level Monte Carlo risk management
