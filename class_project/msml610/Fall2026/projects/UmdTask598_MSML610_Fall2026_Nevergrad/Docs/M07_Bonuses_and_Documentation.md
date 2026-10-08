# Milestone 7 — Bonuses & Documentation

> **GitHub Milestone**
> **Title:** `M7 — Bonuses & Documentation`
> **Timeframe:** Week 8 (Nov 16 – Nov 22)
> **Depends on:** `M5`, `M6`
> **Description:**
> Add the bonus analyses, get the full example notebook running end-to-end (**spec Milestone 3**), and begin the documentation. The two spec bonuses are a **95% CVaR** risk objective (replacing volatility, fronts compared) and **maximum drawdown as a third objective** (a 3-D Pareto surface). In parallel, assemble `nevergrad.example.ipynb` as a complete narrative — data → folds → baselines → objectives → NGOpt/RandomSearch fronts → out-of-sample → robustness → bonus — and start the README (setup, usage, API description, design decisions) (§6.7, §8). The bonuses plus the out-of-sample and robustness depth are what move the Complexity rubric item, which raw code volume does not. Exit criterion: at least one bonus delimited and interpreted, the example notebook running top-to-bottom, and the README underway.

## Notes
- **CVaR bonus:** replace volatility with the **95% Conditional Value-at-Risk** of portfolio returns as the risk objective; recompute the front and compare it to the volatility-based front — does optimizing tail risk change the chosen portfolios? (§6.7)
- **Third-objective bonus:** add **maximum drawdown** as a *third* loss, making the front a 3-D Pareto surface; discuss how `pareto_front()` behaves with three objectives and how to visualize/select from it (§6.7).
- **Keep the bonus clearly delimited from the required run,** and interpret the comparison (CVaR-vs-vol fronts, or the effect of the third objective) — don't just plot it (§6.7).
- **Example notebook is spec Milestone 3:** it should read as a narrative with markdown explaining *why*, not just *what*, and include the spec-required **brief comparison with Optuna and `tutorials/Ax_Multi_Objective_Optimization/`** from the hyperparameter-search point of view (§8).
- **README must cover:** Docker setup/build (incl. the base image chosen and why, and any Dockerfile modifications), how to run the notebooks, an API description of the Nevergrad features used, and the §7 design decisions with justification (§8). Use the course `tutorial_in_60_mins.*` and `notebook.*` Claude skills and note which produced which artifact.

---

## Issues

### `M7-1` — Bonus: 95% CVaR risk objective
- **Assignee:** @jake
- **Labels:** `bonus`, `cvar`, `optimization`
- **Blocked by:** `M5-3`
- **Blocks:** `M8-3`

**Context.** Swap the volatility objective for the 95% Conditional Value-at-Risk of portfolio returns, recompute the Pareto front under the same setup, and compare it to the volatility-based front to see whether optimizing tail risk changes the selected portfolios (§6.7).

**Deliverables**
- A CVaR (95%) risk objective reusing the existing loss/optimization machinery (`scipy` as needed).
- A recomputed front with CVaR in place of volatility, overlaid/compared against the volatility-based front.
- A written comparison: does the CVaR front pick different portfolios, and why?

**Definition of Done**
- [ ] The CVaR objective is implemented and the front recomputed under an otherwise-identical setup.
- [ ] CVaR-vs-volatility fronts compared in one figure/table.
- [ ] The comparison is interpreted, not just plotted; the bonus is delimited from the required run.

---

### `M7-2` — Bonus: maximum drawdown as a third objective
- **Assignee:** @jake
- **Labels:** `bonus`, `drawdown`, `optimization`
- **Blocked by:** `M5-3`
- **Blocks:** `M8-3`

**Context.** Add maximum drawdown as a third loss so the front becomes a 3-D Pareto surface, and discuss how `pareto_front()` behaves with three objectives and how to visualize and select from it (§6.7).

**Deliverables**
- A third objective (maximum drawdown) added to the loss vector; the optimizer run to produce a 3-D front.
- A visualization of the 3-D Pareto surface (or informative 2-D projections) and a note on selection with three objectives.
- A discussion of how the third objective reshapes the trade-off.

**Definition of Done**
- [ ] The three-objective front is produced and non-dominated.
- [ ] The 3-D front is visualized (or projected) and selection-with-three-objectives is discussed.
- [ ] The bonus is interpreted and clearly delimited from the required run.

---

### `M7-3` — Example notebook end-to-end (spec Milestone 3)
- **Assignee:** @jake
- **Labels:** `notebooks`, `tutorial`, `core`
- **Blocked by:** `M6-2`, `M7-1`, `M7-2`
- **Blocks:** `M8-1`

**Context.** Assemble `nevergrad.example.ipynb` so the whole cost-aware portfolio application runs top-to-bottom as a readable walkthrough — completing the spec's Milestone 3 (§8).

**Deliverables**
- The example notebook as a narrative: data + walk-forward → baselines → objectives → Nevergrad fronts (NGOpt vs RandomSearch) → out-of-sample evaluation → robustness → bonus, with markdown explaining *why* between cells.
- The spec-required brief **Optuna / Ax multi-objective** comparison from the hyperparameter-search point of view.
- The notebook paired to a `.py` via `jupytext` and runnable headless via `./docker_cmd.sh python .../nevergrad.example.py`.

**Definition of Done**
- [ ] The example notebook runs top-to-bottom in the container with fixed seeds (completes spec Milestone 3).
- [ ] It reads as a narrative and includes the Optuna/Ax comparison.
- [ ] Paired `.py` runs end-to-end headless.

---

### `M7-4` — Begin the README / documentation
- **Assignee:** @jake
- **Labels:** `documentation`, `readme`
- **Blocked by:** `M6-2`
- **Blocks:** `M8-2`

**Context.** Start the README while the work is fresh, covering the required sections, so documentation isn't left to the final crunch (§8).

**Deliverables**
- A README draft: Docker setup/build (base image chosen and why, any Dockerfile modifications), how to run the notebooks, an API description of the Nevergrad features used, and the §7 architectural/design decisions with justification.
- Docstrings added to `nevergrad_utils.py` functions (PEP 8); notebooks/`.py` linted with `linters2/lint_cc.py`.
- A note on which course Claude skills (`notebook.*`, `tutorial_in_60_mins.*`) produced which artifacts.

**Definition of Done**
- [ ] README draft covers setup, usage, API description, and design decisions.
- [ ] Utility functions documented with docstrings; PEP 8 / `lint_cc.py` clean.
- [ ] README committed to the branch with the Claude-skills usage note.
