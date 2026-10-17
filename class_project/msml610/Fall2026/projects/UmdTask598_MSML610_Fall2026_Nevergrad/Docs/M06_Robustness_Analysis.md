# Milestone 6 — Robustness Analysis

> **GitHub Milestone**
> **Title:** `M6 — Robustness Analysis`
> **Timeframe:** Week 7 (Nov 9 – Nov 15)
> **Depends on:** `M5`
> **Description:**
> Show how stable the out-of-sample result is to randomness and to the cost assumption. Repeat the full walk-forward pipeline for **five random seeds** and **three cost levels — 0, 10, and 25 basis points** — and, for each cost level, report the **mean and standard deviation across seeds of the out-of-sample Sharpe ratio** (and optionally turnover). Present as a small table and/or an error-bar chart over cost level (§6.6). This separates optimizer noise (the seed spread) from cost sensitivity (how much the edge depends on frictions) — together a direct lever on the Complexity and Depth rubric items. Exit criterion: a robustness table/figure of mean ± std OOS Sharpe for {0, 10, 25} bps across five seeds, with an interpretation.

## Notes
- **Vary only seed and cost;** keep everything else fixed (basket, window, budget, selection rule) so the sweep is clean (§6.6).
- **Seed handling per §4.7:** set the seed **before the parametrization's first use** in every run, or the "fixed seed" won't fix the run.
- **Reuse the whole M3→M5 pipeline** — data, folds, baselines, objectives, optimization, OOS evaluation — parameterized by seed and `cost_bps`; no duplicated logic.
- **Interpret both axes:** the seed spread (optimizer stability) and the cost sensitivity (how fragile the net-of-cost edge is) (§6.6).

---

## Issues

### `M6-1` — Robustness sweep harness (5 seeds × {0, 10, 25} bps)
- **Assignee:** @jake
- **Labels:** `robustness`, `cost-model`, `analysis`, `core`
- **Blocked by:** `M5-2`
- **Blocks:** `M6-2`

**Context.** Wrap the full walk-forward pipeline in a sweep over five seeds and three cost levels, re-running optimization and out-of-sample evaluation for each combination with everything else held fixed (§6.6).

**Deliverables**
- A sweep harness that runs the M3→M5 pipeline for each `(seed, cost_bps)` in `{5 seeds} × {0, 10, 25}`, seeding before the parametrization's first use.
- Collected out-of-sample Sharpe (and optionally turnover) per combination.
- Everything but seed and cost held fixed; the harness reuses existing utils (no duplicated math).

**Definition of Done**
- [ ] All 15 `(seed, cost)` runs complete and record OOS Sharpe.
- [ ] Seeds set before parametrization use; each run is individually reproducible.
- [ ] Only seed and cost vary across runs; the harness reuses the M3→M5 code.

---

### `M6-2` — Robustness table/figure & interpretation
- **Assignee:** @jake
- **Labels:** `robustness`, `visualization`, `reporting`, `core`
- **Blocked by:** `M6-1`
- **Blocks:** `M7-3`, `M8-3`

**Context.** Summarize the sweep as the mean and standard deviation of out-of-sample Sharpe across seeds, per cost level, and interpret what the spreads mean (§6.6).

**Deliverables**
- A table of mean ± std OOS Sharpe for {0, 10, 25} bps across the five seeds (optionally turnover too).
- An error-bar chart over cost level.
- A written interpretation separating the seed spread (optimizer stability) from the cost sensitivity (how much the edge depends on frictions).

**Definition of Done**
- [ ] Mean ± std OOS Sharpe reported per cost level across five seeds.
- [ ] Table and/or error-bar figure generated reproducibly.
- [ ] Interpretation addresses both optimizer noise and cost sensitivity.

---

### `M6-3` — One-at-a-time weight sensitivity `[stretch]`
- **Assignee:** @jake
- **Labels:** `sensitivity`, `analysis`, `stretch`
- **Blocked by:** `M6-1`
- **Blocks:** —

**Context.** Optional complementary view if the week has slack: for a chosen portfolio (e.g. the selected max-training-Sharpe portfolio from a representative fold), perturb each asset's weight by a fixed delta, renormalize, and record the change in return and risk — a per-asset sensitivity that enriches the seed/cost robustness story. Clearly gated; reuse the portfolio-stats helpers (no duplicated math) (§6.6).

**Deliverables**
- A perturbation routine: nudge each weight by ±delta, renormalize, recompute `(return, risk)` via the shared stats helper.
- A per-asset bar chart of the resulting return/risk deltas, with a short interpretation.
- Optionally, finite-difference gradients or a Monte-Carlo weighting spread as a deeper variant.

**Definition of Done**
- [ ] Each asset's weight perturbed and renormalized correctly; per-asset return/risk deltas recorded.
- [ ] A sensitivity bar chart generated; directions of change are economically sensible and explained.
- [ ] Clearly gated as stretch — core M6 does not depend on it.
