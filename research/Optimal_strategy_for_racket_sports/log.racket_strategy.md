# Racket Strategy Project Log

- What was done in `research/Optimal_strategy_for_racket_sports/`, newest entry
  first, with links to the issues and PRs
- Parent issue: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
  (Implement solution for optimal strategy for racket sports)
- The milestones are defined by GP in a
  [comment on #590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143)
  (2026-10-09):
  - **Milestone 1**: a notebook with the tennis / pickleball court where player 1
    serves to (`x1`, `y1`), player 2 moves there and returns toward (`x2`, `y2`)
    with a 2D Gaussian spread (`std_x2`, `std_y2`), and the probability that the
    return is in
  - **Milestone 2**: the probability that player 1 reaches the return, given a move
    speed, and the probability of a winner from player 2
  - In a later
    [comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6128247188)
    of 2026-10-12, GP asked to do the heatmap / next milestone next (#624). How
    the heatmap combines with the reachability and winner probability of
    Milestone 2 is still to be agreed with GP
- Dates are in UTC, as GitHub reports them
- The plan of GP's initial implementation is `plan.racket_strategy.md`, with one
  `spec.racket_strategy_PR*.md` file per planned PR

## Open Work

- [#624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624): Milestone 2,
  the reachability and winner probability from #590, shown as a heatmap over every
  target on player 1's half (details to agree with GP)
- [#625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625): this project
  log
- [#626](https://github.com/gpsaggese/gpsaggese.github.io/issues/626): a short doc
  on how to create a PR with the Claude Code skills, for new students and RAs
- [#627](https://github.com/gpsaggese/gpsaggese.github.io/issues/627): the paper,
  adding Aahan Rembersu as an author and the Milestone 1 model
- Manual widget checks of the Milestone 1 notebook, noted as in progress in #654

## 2026-10-14: Fix Private Renames, Serve Direction, and Fault Label

- **Issue**: [#623](https://github.com/gpsaggese/gpsaggese.github.io/issues/623),
  closed by the PR
- **PR**: [#654](https://github.com/gpsaggese/gpsaggese.github.io/pull/654), by
  Aahan Rembersu, opened 2026-10-14, merged 2026-10-14
- **Milestone**: follow-up to Milestone 1
- **What was done**:
  - The 5 functions made private without updating their callers are public again:
    `solve_launch_speed()`, `get_flight_time()`, `get_height_at()`,
    `compute_reachability()`, `compute_score()`
    - Only the 5 definitions and their 7 calls changed
    - This fixed the 14 failing tests
  - `make_serve_situation()` now serves diagonally: the server stands at the box
    center mirrored across the center line
    - E.g., a tennis deuce serve starts at (2.06, -11.885) instead of
      (-2.06, -11.885)
    - The server stays on the baseline, so the library does not depend on the
      notebook's stance
    - The API notebook's Cell 7.1 marks the server, so the diagonal is visible
  - The Milestone 1 "Fault" label no longer covers the serve arrow: it goes beside
    the arrow, on the half where the serve did not land
    - Checked for every fault the sliders can reach, both sports and both sides:
      the closest gap is about 0.76 m
- **Testing**:
  - Full suite (without the Docker test file): 110 passed, 0 failed (before: 93
    passed, 14 failed)
  - 3 new tests for the "Fault" label; the serve tests check the exact server
    position
  - Both API notebooks, the exploration notebook, and the shot probability notebook
    run top to bottom with no errors; each `.py` / `.ipynb` pair is in sync
  - Milestone 1 end-to-end check: 64 passed, 0 failed
  - Linters: `ruff`, `ruff format`, `docformatter`, `ssort` clean; `mypy` reports 5
    `no-any-return` errors in the renamed functions, the same 5 as on master
- **CI note**: the failing `fast_tests.no_docker` check on #619 was not caused by
  these changes
  - It fails on `class_scripts/test/test_count_lecture_slides_uv_run.py`: the `uv`
    environment has only `tabulate`, but the import chain needs `tqdm`
  - Master fails the same way
  - The check runs only `book_springer`, `book_cs_refreshers`, `class_project`,
    and `class_scripts`, so none of the `research/` tests run in CI
- **Note**: written with help from Claude Code; the author reviewed every change

## 2026-10-12: Milestone 1, Serve and Return Probability Notebook

- **Issue**: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **PR**: [#619](https://github.com/gpsaggese/gpsaggese.github.io/pull/619), by
  Aahan Rembersu, opened 2026-10-12 as a draft, merged 2026-10-12
- **Milestone**: Milestone 1
- The sections below are the PR description, condensed

### Summary

- A notebook that shows the probability that a return lands in, given where it is
  aimed and how much it spreads:
  - Pick deuce or ad; player 1 serves from behind the baseline to (`x1`, `y1`),
    following tennis and pickleball serve rules
  - If the serve is legal, player 2 moves instantly to (`x1`, `y1`) and returns
    toward (`x2`, `y2`); if it is a fault, the notebook says why and there is no
    return
  - The return lands with a 2D Gaussian spread (`std_x2`, `std_y2`)
  - The notebook shows $P_{\mathrm{in}}$, computed exactly and checked by Monte
    Carlo
  - Every input is an `ipywidgets` control; it works for tennis and pickleball
- A standalone model, separate from the physics-based scoring in
  `racket_scoring.py`: the Gaussian sits directly on the landing point, so
  $P_{\mathrm{in}}$ has a closed form

### Model

- $X \sim N(x_2, \sigma_x^2)$ and $Y \sim N(y_2, \sigma_y^2)$, independent, with
  $\sigma_x$ = `std_x2` and $\sigma_y$ = `std_y2`
- In means landing in player 1's half: $|x| \le W/2$ and $-L/2 \le y \le 0$
- Independence plus a rectangular region gives a product of two 1D terms:
  $$
  P_{\mathrm{in}} =
  \left[\Phi\left(\frac{W/2 - x_2}{\sigma_x}\right)
  - \Phi\left(\frac{-W/2 - x_2}{\sigma_x}\right)\right]
  \cdot
  \left[\Phi\left(\frac{0 - y_2}{\sigma_y}\right)
  - \Phi\left(\frac{-L/2 - y_2}{\sigma_y}\right)\right]
  $$

### Serve Rules

- The side (deuce or ad) sets the server's stance: 0.5 m beside the center mark,
  0.3 m behind the baseline (a typical stance, not a rule value)
- Only the box diagonal from the server is legal; lines are in
- Faults are named: wide, into the wrong box, long, or in the kitchen (pickleball,
  including the kitchen line)
- The box stops at the baseline (see the pickleball note in Pre-existing Issues)

### Commits and Files

- 5 commits:
  1. Make `draw_court()` public again (it was renamed private but is called from
     other modules; this fixed 2 failing tests)
  2. Gaussian in-court probability utils and tests
  3. Shot probability widget and notebook
  4. Input validation and comment cleanup
  5. Serve follows tennis and pickleball rules
- Files:
  - `notebooks/racket_shot_probability_utils.py`: the model, the widget, and the
    drawing helpers
  - `notebooks/racket_shot_probability.ipynb` and `.py`: the notebook (Jupytext
    pair)
  - `test/test_racket_shot_probability_utils.py`: the tests
  - `racket_strategy_utils.py`: `draw_court()` rename only
  - `README.md`: one row in the notebooks table

### Testing

- New tests: 43 tests and 19 subtests pass, including random order runs
- Full suite: 93 passed; the 14 failures are pre-existing (see below)
- Independent math check: the closed form matches `scipy`'s bivariate normal CDF
  and 2D numerical integration within 1e-13 over 2,300 random cases
- Monte Carlo agrees with the closed form within chance over 4,000 cases
- Serve legality: 0 mismatches against an independently written rule over 859,452
  points (both sports, both sides)
- Widget stress: 1,235 slider states per sport with no errors or memory leaks,
  plus 200 random states per sport after the serve rule change
- Mutation check: 8 of 8 deliberately planted bugs are caught by the tests
- Bad inputs (`nan`, `inf`, std <= 0, `n_samples` <= 0) raise clear errors
- The notebook runs top to bottom for both sports; outputs are reproducible
- Final end-to-end run from a fresh clone: 98 checks, 0 failures

### Design Choices

- The standard error shown uses $P_{\mathrm{in}}$, since $\hat{P}$ gives 0 when
  every sample lands on one side
- Slider values are rounded to 6 decimals, so the "+" and "-" buttons land exactly
  on 0
- Ellipses are dark and dots are see-through, the reverse of the notebook styling
  rule, because light dotted ellipses disappear inside the dot cloud
- In 2D, the 1 sigma ellipse holds about 39% of the samples and 2 sigma about 86%

### Limitations

- The net is ignored for the return: a short return counts as out only if it lands
  on the wrong side
  - Net clearance could be added by reusing `racket_trajectory.py`
- The `x` and `y` errors are independent (no tilted ellipse)
- The serve point decides whether there is a return, but does not change
  $P_{\mathrm{in}}$ for a legal serve
- Singles width only

### Pre-existing Issues Found

- 5 functions were made private without updating their callers:
  `solve_launch_speed()`, `get_flight_time()`, `get_height_at()`,
  `compute_reachability()`, `compute_score()`
  - This caused the 14 failing tests and broke `plot_trajectory_fan()` and the two
    API notebooks
  - Fixed later in #654
- `racket_scoring.make_serve_situation()` placed a deuce server on the same side as
  the deuce box, so its serves went straight down the court instead of diagonally
  - `get_service_box_region()` itself is correct, and Milestone 1 uses the
    diagonal rule
  - Fixed later in #654
- Pickleball `service_line_m` is 6.71 m but the half length is 6.705 m; the
  Milestone 1 serve check caps the box at the baseline
- Pickleball net height and max ball speed differ from the paper's Table I
- Environment:
  - `pytest` is not in the Docker image
  - `git` reports "dubious ownership" inside the container
  - The partial clone remote format breaks `hgit.py`
  - Test runs leave root-owned files
  - `isort` and `ruff format` disagree on wrapped imports

### Questions and GP's Answers

- In
  [GP's comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6128247188)
  of 2026-10-12:
  - Fix the 5 private renames and `make_serve_situation()` in a separate PR? A
    quick PR is fine: done in #654
  - Next step, a $P_{\mathrm{in}}$ heatmap over every target or net clearance with
    `racket_trajectory.py`? The heatmap (#624); `racket_trajectory.py` is set
    aside for now, since the 3D problem adds complexity without adding much over
    2D
- **Note**: written with help from Claude Code; the author reviewed every change

## 2026-10-09: GP's Initial Implementation

- **Issue**: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **PR**: [#591](https://github.com/gpsaggese/gpsaggese.github.io/pull/591), by GP,
  opened 2026-09-23, merged 2026-10-09
- **Milestone**: none; the base that the milestones build on
- **What was done**: the PR description is only "#590", so this list comes from the
  PR's files and the status in `plan.racket_strategy.md`:
  - Project scaffold: Docker files from `class_project/project_template/`,
    `README.md`, `requirements.txt` (plan PR1)
  - `racket_params.py`: sport, player, and error parameters, court geometry, and
    the Table I presets `TENNIS` and `PICKLEBALL` (plan PR1)
  - `racket_trajectory.py`: closed-form ball flight, feasible launches, error
    sampling, and landing simulation (plan PR2)
  - `racket_scoring.py`: target grid, Monte Carlo launch table, reachability,
    composite score, and the serve situation (plan PR3)
  - `racket_strategy_utils.py`: court drawing, trajectory fan, heatmaps, and
    widgets (plan PR5, partly done)
  - Notebooks: `racket_strategy.API`, `racket_trajectory.API`, and
    `racket_strategy.exploration` (plan PR5 and PR6)
  - Tests for each module, plus a Docker test
  - The plan and the specs: `plan.racket_strategy.md` and
    `spec.racket_strategy_PR1.md` to `spec.racket_strategy_PR7.md`
- Still open in the plan: PR4 (placement game), PR7 (example notebook), and PR8
  (paper results)
