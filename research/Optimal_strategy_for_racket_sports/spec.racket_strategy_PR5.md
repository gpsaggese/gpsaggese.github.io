# `racket_strategy_PR5` Implementation Spec: Plotting Utils and API Notebook

- `PR5` of `plan.racket_strategy.md`: add the presentation layer and a notebook
  that walks through the API of `PR1`-`PR4`, one layer per section
- Scope: `racket_strategy_utils.py` (plot and widget helpers), the Jupytext pair
  `racket_strategy.API.ipynb` / `.py`, and its Docker end-to-end test
- Roadmap position: depends on `PR2`, `PR3`, `PR4`; `PR6` reuses the utils
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
- Trade-off: a small grid keeps runtime low but hides fine-grid effects (`PR6`)

## Out of Scope

- Computational results and parameter sweeps: `PR6`
- Paper figures: `PR7`

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

- TBD
