# `racket_strategy_PR6` Implementation Spec: Interactive Shot-Placement
Exploration Notebook

- `PR6` of `plan.racket_strategy.md`: a dedicated notebook where the user
  drives `PR3`'s scoring with click-to-run controls for the shot and player
  parameters, to explore where hitter A can place the ball, how likely it is
  to land in, and how likely returner B is to miss it (a "winner")
- Scope: the Jupytext pair `racket_strategy.exploration.ipynb` / `.py`, plus
  one new widget-builder function in `racket_strategy_utils.py`
- Roadmap position: depends on `PR3` (scoring) and `PR5` (`draw_court()`,
  `plot_court_heatmap()`); independent of `PR4` (the zero-sum game) and `PR7`
  (the paper's computational-results notebook)
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **No new scoring math**: the "probability B misses the shot" (a winner) the
  user asked for is exactly `PR3`'s composite score
  $S(c) = P_{\mathrm{in}}(c) \, (1 - R(c))$; this PR is a presentation layer
  only, calling `racket_scoring.make_target_grid()`,
  `estimate_launch_table()`, and `score_targets()` as-is
- **Click-to-run, not live sliders**: `ipywidgets.interact_manual()` gathers
  all control values and only recomputes on a "Run" click
  - Why: `max_ball_speed_mps`, the shot error scale, and the striker position
    all feed `estimate_launch_table()`, the costly Monte Carlo step ($O(MLK)$,
    per `PR3`); a live slider would resample on every drag
  - Contrast with `PR5`'s widget: that one only touches
    `score_targets(launch_table, returner, returner_xy)`, the cheap re-score
    over a cached table, so it stays a live slider; this notebook is separate
    specifically because it needs the expensive path too
- **Four controls, one Monte Carlo re-sample per click**:
  - Ball speed: `max_ball_speed_mps`, overriding the sport preset
  - Shot dispersion: a single `error_scale` multiplier on
    `DEFAULT_ERROR`'s three sigmas (mirrors `PR7`'s sensitivity sweep), not
    three separate sigma sliders, to keep the panel to one control
  - Player move speed: `move_speed_mps`, the returner's `PlayerParams` field
  - Player positions: `striker_xy` and `returner_xy`, each an (x, y) pair
    restricted to sport-legal ranges
- **Fast defaults**: an 8x8 grid and $K = 500$ samples, smaller than `PR5`'s
  already-fast 12x12/1000 defaults, so a click-to-run completes in a few
  seconds rather than needing to feel instantaneous
- **Reuse, don't duplicate, the court plot**: calls `draw_court()` and
  `plot_court_heatmap()` from `PR5`; adds no new plotting primitives, only
  `build_exploration_widget()` to wire the controls to them

## Trade-off and Alternative Design

- Folding this into `PR5`'s existing widget: rejected (user decision) to keep
  `PR5`'s widget fast and live, and to give this heavier, click-to-run panel
  its own notebook rather than mixing two update speeds in one cell
- Splitting the four controls into "cheap" (move speed, returner position,
  live) and "expensive" (ball speed, error scale, striker position,
  click-to-run): more efficient, but two update paths in one widget is
  harder to reason about than one "Run" button for all four
- Three separate sigma sliders instead of one `error_scale`: more granular,
  but triples the control count for a dimension `PR7`'s sweep already treats
  as a single scale factor

## Out of Scope

- The zero-sum placement game (`PR4`): this notebook is single-shot
  placement only, not the repeated/mixed-strategy game
- Parameter sweeps, sensitivity tables, and paper-numbers reproduction:
  `PR7`
- Calibrating `max_ball_speed_mps` or `error_scale` from data (future work ii)

## Current State

- `racket_scoring.py` (`PR3`) and `racket_strategy_utils.py` (`PR5`) exist;
  neither exposes a control panel over ball speed, error scale, or striker
  position
- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py` has no
  interactive component

## Implementation

### `racket_strategy_utils.py`: New Widget

- Interface (added to the `PR5` file):
  ```python
  def build_exploration_widget(sport, config) -> ipywidgets.Widget:
      # Sliders: max_ball_speed_mps, error_scale, move_speed_mps,
      # striker_xy, returner_xy; interact_manual re-runs
      # make_rally_situation() -> estimate_launch_table() -> score_targets()
      # and re-draws the P_in, (1 - R), and S heatmaps plus court markers.
      ...
  ```

### `racket_strategy.exploration.ipynb`

- Sections, following `.claude/templates/API_notebook.template.ipynb`:
  1. Setup: pick a sport (`TENNIS` or `PICKLEBALL`), build the target grid
  2. Controls: `build_exploration_widget()`
  3. Reading the output: one heatmap panel per run, `P_{\mathrm{in}}`,
     $1 - R$ (winner probability for A), and $S$, with striker/returner
     markers drawn by `draw_court()`
  4. Two or three worked scenarios (e.g., a fast ball against a slow mover,
     a wide error scale) with a one-line takeaway each

## Interaction with Existing Code

- Read-only use of the `PR3` and `PR5` public functions; `README.md` lists
  the notebook

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- `test/test_racket_strategy_utils.py`:
  - `Test_build_exploration_widget`: constructs the widget without running
    the notebook UI; asserts on the returned control names and types
- `test/test_docker_racket_strategy.py`: extend with `test3`, running
  `racket_strategy.exploration.ipynb` end to end in Docker
  (`@pytest.mark.slow`), as in `PR5`'s `test1`

## Risks and Limitations to Call Out

- The widget does not run under `nbconvert`: the cell must also render one
  static default run so the Docker end-to-end test has something to check
- A single `error_scale` hides which of the three sigmas drives a given
  change; call this out in the notebook text
- Moving the striker or returner outside the sport's legal region gives
  degenerate scores (e.g., no feasible launches); guard with `hdbg.dassert`
  and clamp the sliders' ranges to the court bounds

## Result (to Fill in Once Implemented)

- Implemented `build_exploration_widget()` in `racket_strategy_utils.py`
  and the Jupytext pair `racket_strategy.exploration.ipynb` / `.py`, as
  specced
- Click-to-run is a `Run` button plus `ipywidgets.Output()` (the same
  pattern `notebook_utils_template.py` and `PR5`'s `build_score_widget()`
  use), rather than the `ipywidgets.interact_manual()` decorator named in
  the spec; behavior matches the spec's requirement (one Monte Carlo
  re-sample per click, not per slider drag), and the function returns the
  widget container so `Test_build_exploration_widget` can assert on it
  without running the notebook UI
- The widget also renders one static default run at build time, so the
  Docker end-to-end test has output even though the `Run` button itself
  is never clicked programmatically
- `racket_strategy.exploration.ipynb` was executed end to end locally via
  `jupyter nbconvert --execute` (outside Docker) with no errors; the
  `@pytest.mark.slow` Docker test (`test3`) is written but not run in this
  session
- `test/test_racket_strategy_utils.py::Test_build_exploration_widget` and
  the full local test suite (49/49) pass
