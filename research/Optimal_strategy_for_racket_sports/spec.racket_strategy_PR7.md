# `racket_strategy_PR7` Implementation Spec: Example Notebook with Results

- `PR7` of `plan.racket_strategy.md`: turn the hand-worked example of paper
  Section VI into computational results, and run the studies Sections VII-VIII
  leave open
- Scope: the Jupytext pair `racket_strategy.example.ipynb` / `.py`, plus
  experiment helpers in `racket_strategy_utils.py`
- Roadmap position: depends on `PR5`; `PR8` reports its outputs in the paper
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **One notebook part per experiment**:
  1. Reproduce Table II from the paper inputs with `compute_reachability()` and
     `compute_score()`: must match exactly
  2. Replace the illustrative $P_{\mathrm{in}}$ (0.95-0.65) with Monte Carlo
     values for 5 targets at $d(O,c)$ = 0.3-1.9 m, tennis baseline geometry
  3. Full grid, tennis baseline rally versus pickleball non-volley zone exchange:
     heatmaps of $P_{\mathrm{in}}$, $R$, $S$, and the best angle
  4. Sensitivity: $t_r \in \{0.15, 0.2, 0.25\}$ s, $v_p \in \{1.5, 3, 4.5\}$
     m/s, error scale $\in \{0.5, 1, 2\}$; metrics: distance of $c^\star$ from
     $O$, $S(c^\star)$, reachable share
  5. Grid resolution versus variance (Section VIII): grids 5x5, 10x10, 20x20 and
     $K \in \{250, 1000, 4000\}$; metrics: runtime, max `p_in_se`, argmax
     stability over 10 seeds
  6. Game: equilibrium versus pure argmax on the rally grid (value,
     exploitability, support size); 3x3 tennis serve game (wide, body, T)
  7. Unified parametrization (Section VII): one table of regime metrics for both
     sports from the same code path
- **Helpers in utils, loops out of cells**: sweeps return tidy `pd.DataFrame`s
  that the cells plot
- **Serve directions**: wide, body, and T targets 1 m inside the service line,
  at the sideline, box center, and center line offsets
- **Exported results**: `save_results()` writes PNG and CSV files to `results/`
  only when `SAVE_RESULTS = True`, so the Docker test writes nothing
- **Runtime budget**: under 5 minutes in Docker; part 5 runs its full sizes only
  when `FULL_RUN = True`; all seeds fixed

## Trade-off and Alternative Design

- A standalone script: loses the notebook narrative that research projects use
  (e.g., `research/Causal_Analysis_of_Agent_Skill_And_Luck/`)
- Writing figures straight into `papers/`: couples research code to the paper
  layout; `PR8` copies selected files instead
- Trade-off: small default sizes keep tests fast; paper numbers need `FULL_RUN`

## Out of Scope

- Editing `paper.md`: `PR8`
- Real tracking data and calibration (future work ii)

## Current State

- `racket_strategy_utils.py` exists after `PR5`, with plotting only
- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py:101-140`
  draws the toy scores of Figure 2 from hard-coded arrays

## Implementation

### `racket_strategy_utils.py`: Experiment Helpers

- Interface (added to the `PR5` file; the notebook pair is new):
  ```python
  def run_sensitivity_sweep(sport, situation, base_returner, grid,
                            config) -> pd.DataFrame: ...
  def run_resolution_study(sport, situation, returner, grid_sizes, n_samples,
                           seeds) -> pd.DataFrame: ...
  def make_serve_direction_targets(court, serve_side) -> pd.DataFrame: ...
  def save_results(figures, tables, out_dir) -> None: ...
  ```

## Interaction with Existing Code

- Calls `PR3` and `PR4` public functions only; `README.md` lists the notebook

## Configuration and Secrets

- Notebook flags `SAVE_RESULTS` and `FULL_RUN`, both `False` by default

## Unit Test Plan

- `test/test_racket_strategy_utils.py`:
  - `Test_run_sensitivity_sweep`: tiny sizes: one row per parameter combination
  - `Test_run_resolution_study`: tiny sizes: expected columns and row count
  - `Test_make_serve_direction_targets`: 3 targets in the deuce box, ordered
    wide, body, T
- `test/test_docker_racket_strategy.py`: `test2` runs the example notebook

## Risks and Limitations to Call Out

- Results may contradict the toy example (e.g., Monte Carlo $P_{\mathrm{in}}$
  near 1 for all interior cells): report them as found and feed them into `PR8`
- The paper's pickleball exchange at the non-volley zone line is in practice a
  volley, which Section III-B excludes; the notebook states this caveat
- The 3x3 serve game only reproduces the equilibrium structure of Walker and
  Wooders; it is not a validation on their data

## Result (to Fill in Once Implemented)

- TBD
