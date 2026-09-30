# `racket_strategy_PR5` Implementation Spec: Plotting Utils and API Notebook

- `PR5` of `plan.racket_strategy.md`: add the presentation layer and a notebook
  that walks through the API of `PR1`-`PR4`, one layer per section
- Scope: `racket_strategy_utils.py` (plot and widget helpers), the Jupytext pair
  `racket_strategy.API.ipynb` / `.py`, and its Docker end-to-end test
- Roadmap position: depends on `PR2`, `PR3`, `PR4`; `PR6` and `PR7` reuse the
  utils
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **One shared utils file**: both notebooks import `racket_strategy_utils.py`,
  named after the topic (`notebook.rules.md` "Shared-utils exception")
- **Utils present, modules compute**: plot helpers call the `PR1`-`PR4` modules
  and never re-implement physics or scoring
- **Figure 1 from the library first**: `plot_trajectory_fan()` rebuilds the paper
  Figure 1 from `racket_trajectory.py`, the first paper-versus-code check
- **One interactive widget**: sliders for returner position, $t_r$, $v_p$ update
  the score heatmap through `score_targets()` on a cached launch table, with no
  new sampling (`notebook.rules.md` "Interactive Idiom for Notebooks")
- **Fast by default**: $K \le 1000$ samples and grids up to 12x12 so the notebook
  runs in minutes inside Docker

## Trade-off and Alternative Design

- `plotly`: extra dependency; `matplotlib` and `ipywidgets` are in the template
- One notebook per module: fragments the story; sections mirror the layers instead
- Trade-off: a small grid keeps runtime low but hides fine-grid effects (`PR7`)

## Out of Scope

- The interactive exploration notebook with ball speed / shot std dev / player
  position controls: `PR6`
- Computational results and parameter sweeps: `PR7`
- Paper figures: `PR8`

## Current State

- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py:44-95`
  draws Figure 1 with hand-coded physics
- `class_project/project_template/template.API.ipynb` and
  `test/test_docker_template.py` are the structure to mirror

## Implementation

### `racket_strategy_utils.py`

- Interface (new file):
  ```python
  def draw_court(ax, court) -> None: ...
  def plot_trajectory_fan(sport, player, striker_xy, target_xy, theta_deg, *,
                          ax=None) -> None: ...
  def plot_court_heatmap(scores, column, sport, *, markers=None, ax=None) -> None:
  def plot_launch_tradeoff(launch_table, cell_id, *, ax=None) -> None: ...
  def plot_mixed_strategy(solution, targets, sport, *, ax=None) -> None: ...
  def build_score_widget(sport, launch_table, returner) -> ipywidgets.Widget: ...
  ```

### `racket_strategy.API.ipynb`

- Sections, following `.claude/templates/API_notebook.template.ipynb`:
  1. Sport parameters: Table I as a `pd.DataFrame` built from `TENNIS`,
     `PICKLEBALL`
  2. Trajectory feasibility: Figure 1 numbers and `plot_trajectory_fan()`
  3. Feasible set $\mathcal{F}(T)$: $v_0$, $T_f$, clearance versus $\theta$
  4. Error propagation: landing scatter for 3 targets
  5. Grid and $P_{\mathrm{in}}$: court heatmap
  6. Reachability and score: heatmaps plus the widget
  7. Serve: deuce service box heatmap
  8. Placement game: 5x5 grid, equilibrium versus pure argmax

## Interaction with Existing Code

- Read-only use of the `PR1`-`PR4` modules; `README.md` lists the notebook

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- `test/test_racket_strategy_utils.py` (Agg backend):
  - `Test_draw_court`: adds the expected court lines to the axes
  - `Test_plot_court_heatmap`: draws one patch per cell on a 2x2 grid
  - `Test_plot_trajectory_fan`: runs on the Figure 1 setup without error
- `test/test_docker_racket_strategy.py`: `Test_docker.test1` runs
  `racket_strategy.API.ipynb` in Docker (`@pytest.mark.slow`), as in
  `class_project/project_template/test/test_docker_template.py`

## Risks and Limitations to Call Out

- The widget does not run under `nbconvert`: the cell must also render a static
  default heatmap
- Notebook runtime inside Docker: the reviewer checks it stays in minutes

## Result (to Fill in Once Implemented)

- Partially implemented ahead of `PR4`, to unblock `PR6`:
  - `racket_strategy_utils.py`: `draw_court()`, `plot_trajectory_fan()`,
    `plot_court_heatmap()`, `plot_launch_tradeoff()`, `build_score_widget()`
  - `racket_strategy.API.ipynb` / `.py`: Parts 1-7 (sport parameters
    through serve); Part 8 (placement game) is not written
  - `test/test_racket_strategy_utils.py`, `test/test_docker_racket_strategy.py`
    (`test1`, running `racket_strategy.API.ipynb`)
- Not yet implemented: `plot_mixed_strategy()` (needs `racket_game.py`'s
  `GameSolution`) and the notebook's Part 8
- `racket_strategy.API.ipynb` was executed end to end locally via
  `jupyter nbconvert --execute` (outside Docker) with no errors; the
  `@pytest.mark.slow` Docker test is written but not run in this session
- A local-only gotcha found while building `PR6`'s widget, not specific to
  this PR: an inline-backend `matplotlib` figure left open inside an
  `ipywidgets.Output()` context can stall the kernel's idle handshake on
  the next update; every widget update function here calls `plt.close(fig)`
  right after `plt.show()`
