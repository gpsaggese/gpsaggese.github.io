# FilterPy API Tutorial Script

Tutorial covering the FilterPy Python library for Bayesian state estimation.
Topics: Linear Kalman Filter, Extended Kalman Filter (EKF), Unscented Kalman
Filter (UKF), and Ensemble Kalman Filter (EnKF).

References:
- https://filterpy.readthedocs.io/en/latest/
- Roger Labbe, "Kalman and Bayesian Filters in Python"

<start>

## Cell 1: Introduction - The Predict-Update Cycle
- Purpose: Introduce Bayesian filtering and the core predict-update cycle that
  all Kalman variants share
- Display:
  - Markdown explanation of the two-step cycle:
    - Predict: x_prior = F @ x + B @ u; P_prior = F @ P @ F.T + Q
    - Update: K = P @ H.T @ inv(H @ P @ H.T + R); x = x_prior + K @ (z - H @
      x_prior); P = (I - K @ H) @ P_prior
  - Static diagram using matplotlib arrows showing: True State -> [Process
    Noise Q] -> Predicted State -> [Measurement Noise R] -> Measurement z ->
    [Kalman Gain K] -> Updated State
  - Table comparing what each matrix controls: F (dynamics), H (sensor), Q
    (process noise), R (measurement noise), P (uncertainty)
- Interactive widget: None
- Key insights: All Kalman variants share this predict-update structure; they
  differ only in how they handle nonlinearity

## Cell 2: Linear Kalman Filter - 1D Position and Velocity Tracking
- Purpose: Show KalmanFilter API to track a moving object under constant
  velocity with noisy position-only measurements
- Display:
  - 3-line time series plot:
    - True position (smooth curve, blue)
    - Noisy measurements (scatter, orange dots)
    - KF estimate (smooth estimate, green)
  - Shaded 1-sigma confidence band around KF estimate
  - y-axis fixed to [-5, 35] to avoid jumps when changing widget controls
- Interactive widget:
  - Slider for measurement noise R (0.1 to 50, step 0.5, initial 5)
  - Slider for process noise Q (0.001 to 1, step 0.01, initial 0.1)
  - Output: redraws plot with updated R and Q values
- Key insights: Smaller R trusts measurements more; larger Q allows the filter
  to follow rapid changes; the R/Q ratio is the key tuning parameter

## Cell 3: Linear Kalman Filter - Uncertainty Evolution
- Purpose: Show how the state covariance P evolves during filtering and
  converges over time
- Display:
  - 2 panels on one row:
    - Panel 1: Time series of diagonal elements of P (position variance and
      velocity variance) converging to steady state
    - Panel 2: Time series of Kalman gain K showing how trust in measurements
      decreases as P converges
  - y-axis for Panel 1 fixed to [0, 200]; y-axis for Panel 2 fixed to [0, 1]
- Interactive widget:
  - Slider for initial covariance P0 (1 to 10000, logarithmic, initial 1000)
  - Slider for R (0.1 to 50, initial 5)
  - Output: redraws both panels
- Key insights: P always converges to a steady-state value regardless of P0;
  K decreases as the filter becomes more confident

## Cell 4: Extended Kalman Filter - Radar Tracking with Polar Measurements
- Purpose: Show ExtendedKalmanFilter API for nonlinear measurement function
  converting polar (range, bearing) to Cartesian state
- Display:
  - 2D scatter plot with 3 elements:
    - True circular path (blue solid line)
    - Noisy radar measurements converted from polar to Cartesian (orange
      scatter)
    - EKF estimate trajectory (green dashed line)
  - xlim and ylim fixed to [-15, 15] to avoid jumps
- Interactive widget:
  - Slider for range noise sigma_r (0.1 to 5, initial 0.5)
  - Slider for bearing noise sigma_b in degrees (0.5 to 10, initial 2)
  - Output: redraws 2D scatter with updated noise levels
- Key insights: EKF uses Jacobians to linearize; works well for mildly
  nonlinear systems; polar-to-Cartesian is a classic EKF use case

## Cell 5: Extended Kalman Filter - Jacobian Linearization Visualization
- Purpose: Visually show how linearization approximates a nonlinear function
  and where error is introduced
- Display:
  - 3 panels on one row:
    - Panel 1: Nonlinear function h(x) = atan2(y, x) and its tangent line
      (Jacobian) at operating point x0; show divergence away from x0
    - Panel 2: Input Gaussian centered at x0 (blue shaded)
    - Panel 3: True propagated distribution through h (blue) vs EKF linearized
      approximation (green dashed); comments box explaining the error
  - x-axis for Panel 1 fixed to [-pi, pi]; y-axis fixed to [-pi, pi]
- Interactive widget:
  - Slider for operating point x0 in radians (-pi to pi, initial 0.5)
  - Slider for input uncertainty sigma (0.1 to 2, initial 0.5)
  - Output: redraws all three panels
- Key insights: Linearization error grows with uncertainty and curvature;
  EKF underestimates true posterior variance in highly nonlinear regions

## Cell 6: Unscented Kalman Filter - Sigma Points Intuition
- Purpose: Introduce sigma points and the Unscented Transform for propagating
  distributions through nonlinear functions
- Display:
  - 3 panels on one row:
    - Panel 1: 2D input Gaussian with sigma points overlaid (red X marks)
    - Panel 2: Sigma points after nonlinear transformation (red X marks) with
      true Monte Carlo propagated samples (light blue dots)
    - Panel 3: Comments box comparing UT mean/covariance vs true Monte Carlo
      vs EKF linearization
  - xlim and ylim for Panels 1 and 2 fixed to [-4, 4]
- Interactive widget:
  - Slider for alpha (sigma point spread, 0.001 to 1, initial 0.1)
  - Slider for input uncertainty sigma_x (0.1 to 2, initial 1.0)
  - Output: redraws all three panels with new sigma points
- Key insights: UT is accurate to 3rd order for nonlinear functions; requires
  no Jacobian; sigma points = 2n+1 deterministic samples

## Cell 7: UKF vs EKF - Side-by-Side Tracking Comparison
- Purpose: Compare UKF and EKF on the same nonlinear tracking problem to show
  when UKF outperforms EKF
- Display:
  - 3 panels on one row:
    - Panel 1: True trajectory + noisy measurements
    - Panel 2: EKF estimate with RMSE in legend
    - Panel 3: UKF estimate with RMSE in legend
  - xlim and ylim fixed to [-15, 15] for all panels
- Interactive widget:
  - Slider for curvature (controls nonlinearity, 0.1 to 2.0, initial 0.5)
  - Slider for measurement noise R (0.1 to 20, initial 1.0)
  - Output: redraws all three panels and updates RMSE
- Key insights: UKF typically achieves lower RMSE than EKF in nonlinear
  regimes; EKF can diverge under high curvature; both converge for linear
  systems

## Cell 8: Ensemble Kalman Filter - Particle Ensemble Visualization
- Purpose: Show EnsembleKalmanFilter API using Monte Carlo particles to
  represent state uncertainty
- Display:
  - 2 panels on one row:
    - Panel 1: Scatter plot of ensemble particles (small dots) at each time
      step colored by time; ensemble mean overlaid as a bold line
    - Panel 2: Time series of ensemble mean (green) vs true state (blue);
      shaded band showing ensemble spread (std)
  - ylim for Panel 1 fixed to [-5, 35]; ylim for Panel 2 fixed to [-5, 35]
- Interactive widget:
  - Slider for ensemble size N (10 to 500, logarithmic, initial 100)
  - Slider for process noise Q (0.001 to 1, initial 0.1)
  - Output: reruns EnKF with new N and Q, redraws both panels
- Key insights: Ensemble represents the full posterior distribution; N=100+
  usually sufficient; computational cost scales linearly with N

## Cell 9: Filter Comparison - All Four Filters on One Problem
- Purpose: Directly compare all four filters on the same synthetic 1D
  tracking problem and tabulate RMSE
- Display:
  - 4 panels on one row (KF, EKF, UKF, EnKF), each showing:
    - True state (blue), noisy measurements (orange scatter), filter estimate
      (green), 1-sigma band (shaded)
  - RMSE printed in each panel legend
  - Comments panel below summarizing which filter wins and why
  - xlim fixed to [0, 50]; ylim fixed to [-5, 35] across all panels
- Interactive widget: None
- Key insights: KF is optimal for linear systems; EKF is fastest for mildly
  nonlinear; UKF more accurate for strongly nonlinear; EnKF most flexible but
  computationally heavier

<end>
