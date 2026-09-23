# `racket_strategy_PR3` Implementation Spec: Grid-Based Monte Carlo Scoring

- `PR3` of `plan.racket_strategy.md`: implement paper Section IV (grid, Monte Carlo
  $P_{\mathrm{in}}$, reachability, composite score, serve) in `racket_scoring.py`
- Roadmap position: depends on `PR2`; `PR4`-`PR6` build on it
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **$P_{\mathrm{in}}$ is the in-bounds probability**
  - Fact: the Section IV-B formula counts samples landing in cell $c$; the
    abstract, Section I, and Table II (0.65-0.95) call it the in-bounds probability
  - Decision: a sample counts if it clears the net and lands in the legal region
    (half court or service box); the cell hit rate is kept as `p_hit_cell`
  - Why: the per-cell reading scales with cell area, so the argmax would depend
    on grid resolution; `PR7` fixes the formula in the paper
- **Reachability as in the paper**: cell center, nominal $T_f$ (Section IV-C)
- **Best angle per cell**: $S(c) = \max_\ell P_{\mathrm{in}}(c,\ell)\,(1 -
  R(c,\ell))$ over the feasible angles $\ell$ of `PR2`; the max comes after $R$
  because the best angle depends on the opponent
- **Split opponent-independent sampling from scoring**
  - `estimate_launch_table()`: the costly step, $O(MLK)$, no opponent input
  - `score_targets()`: reachability and max over angles for one opponent, $O(ML)$
  - Why: `PR4` re-scores $N$ recovery positions without new sampling
- **Common random numbers**: one seeded `ShotErrors` draw shared by all cells and
  angles, so cell differences reflect geometry, not sampling noise
- **Serve is a situation, not a code path** (Section IV-E): the server stands
  behind the baseline and the legal region is the service box; presets
  `SERVE_PLAYER_TENNIS` ($h_0 = 2.7$ m) and `SERVE_PLAYER_PICKLEBALL` ($h_0 = 0.7$
  m, underhand) are illustrative

## Trade-off and Alternative Design

- Sample-level score $\mathbb{E}_k[\mathbb{1}_{\mathrm{in}}(1 -
  R(\mathrm{land}_k))]$: more physical, same cost, but departs from the paper
- Adaptive grids or importance sampling: premature; Section IV-A uses a uniform grid
- Trade-off: `pd.DataFrame` outputs for notebook readability, `numpy` internally

## Out of Scope

- Graded reachability (future work iii), second-bounce timing, volleys

## Current State

- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py:101-117`
  computes the toy $S$ from a precomputed `reach_radius`; no sampling exists

## Implementation

### `racket_scoring.py`

- Interface (one new file):
  ```python
  class ShotSituation:   # striker: PlayerParams, striker_xy, legal_region
  class ScoringConfig:   # n_samples=2000, theta_grid_deg=(-10, ..., 45), seed=1
  def make_target_grid(region, n_x, n_y) -> pd.DataFrame:
      # cell_id, x_m, y_m, x_min, x_max, y_min, y_max
  def estimate_launch_table(sport, situation, targets, config) -> pd.DataFrame:
      # One row per (cell_id, theta_deg): x_m, y_m, v0_mps, flight_time_s,
      # p_in, p_in_se, p_hit_cell
  def compute_reachability(dist_m, flight_time_s, returner) -> np.ndarray: ...
  def compute_score(p_in, reachable) -> np.ndarray: ...
  def score_targets(launch_table, returner, returner_xy) -> pd.DataFrame:
      # One row per cell_id: best theta_deg, p_in, reachable, score
  def select_best_cell(scores) -> pd.Series: ...
  def make_rally_situation(sport, striker, striker_xy) -> ShotSituation: ...
  def make_serve_situation(sport, server, serve_side) -> ShotSituation: ...
  ```
- A cell with no feasible angle gets `p_in = 0`, `score = 0`, `theta_deg = NaN`
- `p_in_se` $= \sqrt{p(1-p)/K}$ feeds the resolution study of `PR6`

## Interaction with Existing Code

- Calls the `PR2` functions and the `CourtRegion` getters of `PR1`

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- `test/test_racket_scoring.py`:
  - `Test_make_target_grid`: 2x3 grid on a unit region: centers and bounds
  - `Test_compute_reachability`: Table II $R$ columns ($T_f$ 0.8 s and 0.2 s)
  - `Test_compute_score`: Table II $S$ columns; argmax $c_3$ and $c_1$
  - `Test_estimate_launch_table`: zero error: `p_in` 1 inside, 0 outside; lateral
    error only at the sideline: 0.5 within 3 SE
  - `Test_score_targets`: tiny grid, fixed seed, whole frame via `assert_equal`
  - `Test_make_serve_situation`: every cell center lies in the service box

## Risks and Limitations to Call Out

- The $P_{\mathrm{in}}$ reading departs from the paper formula: needs sign-off
- Cost: 400 cells x 56 angles x 2000 samples is about 45M closed-form
  evaluations; loop over cells, vectorize over (angles, samples)
- A step-function $R$ makes the argmax jump between neighbor cells (Section VIII)

## Result (to Fill in Once Implemented)

- TBD
