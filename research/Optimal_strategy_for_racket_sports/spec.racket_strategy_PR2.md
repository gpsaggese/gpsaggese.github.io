# `racket_strategy_PR2` Implementation Spec: Closed-Form Trajectory Model

- `PR2` of `plan.racket_strategy.md`: implement trajectory feasibility and error
  propagation (paper Section III-C) in `racket_trajectory.py`, pure `numpy`
- Roadmap position: depends on `PR1` (`racket_params.py`); `PR3` builds on it
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **Closed-form launch speed per angle**
  - Fact: the paper defines $\mathcal{F}(T)$ as the $(v_0, \theta)$ pairs landing
    within $\pm\epsilon$ of $d_T$, clearing the net, with $v_0 \le v_{\max}$
  - Decision: for each $\theta$ of a fixed grid, solve the landing at $d_T$ exactly
    $$v_0(\theta) = \frac{d_T}{\cos\theta}\sqrt{\frac{g}{2(h_0 + d_T\tan\theta)}}$$
    valid when $h_0 + d_T \tan\theta > 0$; keep angles with $v_0 \le v_{\max}$
    and positive net clearance
  - Why: for $\epsilon \to 0$ the set is a curve in $\theta$, so a 1D grid gives
    it without a 2D search; the cell size of `PR3` plays the role of $\epsilon$
  - Check: $\theta = 8$ deg, $d_T = 20$ m, $h_0 = 1$ m gives $v_0 = 22.9$ m/s,
    $T_f = 0.88$ s, clearance 0.40 m, as in Figure 1 (verified numerically)
- **Return every feasible angle**
  - Fact: the paper does not say which point of $\mathcal{F}(T)$ the striker aims
  - Decision: return all candidates; `PR3` keeps the best one per cell
  - Why: `notes.md` trade-off: a lob flies longer (reachable), a flat shot risks
    the net; one fixed rule (e.g., minimum flight time) would hide it
- **Exact per-sample net check in 2D**: each sample flies along azimuth
  $\alpha = \psi + \phi$ (from the `+y` axis) and crosses the net at distance
  $-y_S / \cos\alpha$ and lateral position $x_S - y_S \tan\alpha$, where
  `get_net_height()` gives the net height
  - Why: lateral error moves the crossing point; the closed form costs $O(1)$
- **Vectorized, explicit RNG**
  - Functions broadcast over arrays of shape (`L` angles, `K` samples), radians
  - `sample_shot_errors()` takes an `np.random.Generator`, so `PR3` reuses one
    draw for every cell (common random numbers)

## Trade-off and Alternative Design

- 2D numerical search over $(v_0, \theta)$: slower and approximate
- Local linearization of the errors (Section III-C): the Gaussian footprint
  breaks near the net and the lines, where the in-bounds indicator jumps
- Trade-off: the angle grid adds a factor `L` to the cost of `PR3`, in exchange
  for modeling the lob versus flat choice

## Out of Scope

- Drag, Magnus lift, error growing with target difficulty (Section VIII): follow-up
- Volleys and bounce physics after the landing (Section III-B)

## Current State

- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py:39-65`
  computes the nominal and 4 perturbed trajectories inline, not reusable

## Implementation

### `racket_trajectory.py`

- Interface (one new file):
  ```python
  class FeasibleLaunches:  # theta_rad, v0_mps, flight_time_s, net_clearance_m
  class ShotErrors:        # d_theta_rad, d_v_frac, phi_rad: shape (K,)
  class Landings:          # x_m, y_m, flight_time_s, clears_net: shape (L, K)
  def get_flight_time(v0, theta, h0) -> np.ndarray: ...
  def get_height_at(x, v0, theta, h0) -> np.ndarray: ...
  def solve_launch_speed(d_target, theta, h0) -> np.ndarray:  # NaN if infeasible
  def get_feasible_launches(striker_xy, target_xy, h0, sport, *,
                            theta_grid_rad) -> FeasibleLaunches: ...
  def sample_shot_errors(error, n_samples, rng) -> ShotErrors: ...
  def simulate_landings(striker_xy, target_xy, h0, launches, errors) -> Landings:
  ```

## Interaction with Existing Code

- Imports `GRAVITY_MPS2`, `SportParams`, `get_net_height()` from `racket_params.py`

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- `test/test_racket_trajectory.py`:
  - `Test_solve_launch_speed`: Figure 1 speed; NaN when $h_0 + d\tan\theta \le 0$
  - `Test_get_flight_time`: $h_0 = 0$ gives $2 v_0 \sin\theta / g$; Figure 1 value
  - `Test_get_height_at`: 0.40 m net clearance at 12 m (Figure 1)
  - `Test_get_feasible_launches`: all meet $v_{\max}$ and clearance; tiny
    $v_{\max}$ gives none
  - `Test_sample_shot_errors`: same seed gives same draws; zero sigmas give zeros
  - `Test_simulate_landings`: zero error lands on target; lateral-only symmetric

## Risks and Limitations to Call Out

- A coarse angle grid misses narrow feasible windows (low $v_{\max}$, short
  distance): default step 1 deg; the range includes negative angles for serves
- NaN for infeasible angles must not leak into the averages of `PR3`
- Shots with $\cos\alpha \le 0$ never reach the net: guard with `hdbg.dassert`

## Result (to Fill in Once Implemented)

- TBD
