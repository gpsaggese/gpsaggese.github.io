# Milestone 4 — Nevergrad Multi-Objective Optimization & API Notebook

> **GitHub Milestone**
> **Title:** `M4 — Nevergrad Multi-Objective Optimization & API Notebook`
> **Timeframe:** Weeks 4–5 (Oct 19 – Nov 1)
> **Depends on:** `M3`
> **Description:**
> The core technical work, spanning two weeks, and **spec Milestone 2** (the API notebook). Formalize the cost-aware objectives and the 40% cap; build the Nevergrad search space over a `Dict` of **weights *and* a rebalancing-frequency choice**; run the explicit `ask`/`tell` loop for **both `NGOpt` and `RandomSearch` under one budget** to produce converged, non-dominated Pareto fronts; finish the API teaching notebook; and submit **Checkpoint PR #1** (data + baselines + working fronts + API notebook) around Nov 2 (§6.3–§6.4, §8, §5.6). The teaching point: a general-purpose black-box optimizer solves a mixed continuous/discrete, non-smooth, multi-objective problem where the textbook closed form no longer applies. If the checkpoint review is slow, keep working the next incremental branch rather than waiting.

## Notes
- **Search space is a `Dict`:** `param = ng.p.Dict(weights=ng.p.Array(shape=(n,)), rebal=ng.p.Choice(["monthly", "quarterly", "annual"]))`. A candidate's `.value` is `{"weights": np.ndarray, "rebal": <chosen item>}` (§6.4).
- **Map weights to the simplex** (softmax or clip-and-renormalize) so they're long-only and sum to 1. Note the caveat in the tutorial: softmax can't reach exact-zero weights and mildly biases toward the interior — though the hard 40% cap already removes single-asset corner solutions (§6.3–§6.4).
- **40% cap via `register_cheap_constraint`** returning a float **≤ 0 when satisfied**: `lambda d: float(np.max(to_weights(d["weights"])) - 0.40)` (§6.3).
- **Objectives are `[volatility, −net_return]`; both minimized.** Volatility = `√(wᵀΣw)·√252`; net return = gross annualized return − `cost_bps · turnover`, with **cost = 10 bps** for the main run. Confirm the sign — a flipped sign makes the optimizer reward the wrong thing (§6.3, §11 risks).
- **Optimizers:** `NGOpt` (meta-optimizer) vs `RandomSearch` (honest baseline), constructed with the `parametrization=` keyword (not the deprecated `instrumentation=`), kept at the **same budget** (~3,000 each; raise if the front is sparse) so the tutorial can ask whether the meta-optimizer beats random search at equal compute (§6.4, §7).
- **Explicit `ask`/`tell` loop** (the spec asks for this, not `minimize`): `cand = opt.ask()` → decode `w`, `freq` → `losses = portfolio_losses(w, freq, train_returns, cost_bps)` → `opt.tell(cand, losses)`. Optionally set an upper-bound reference first via `opt.tell(ng.p.MultiobjectiveReference(), [vol_ub, negret_ub])` ("highly advised," else auto-inferred) (§6.4).
- **Read the front with `optimizer.pareto_front()`** → a list of `Parameter` objects, each with `.value` (→ weights + chosen frequency) and `.losses` (§6.4).
- **Fix random seeds so the front reproduces:** call `np.random.seed(...)` **before the parametrization's first use** — the parametrization pulls its state from numpy's global seed at that moment, so the wrong order means the seed won't actually fix the run (§4.7).
- **After Checkpoint PR #1, continue on an incremental branch (`..._1`)** without waiting for review (§5.6).

---

## Issues

### `M4-1` — Search space, cost-aware objectives & the 40% constraint
- **Assignee:** @jake
- **Labels:** `nevergrad`, `optimization`, `cost-model`, `core`
- **Blocked by:** `M3-2`
- **Blocks:** `M4-2`

**Context.** Build the `Dict` search space (weights `Array` + rebalancing `Choice`), the simplex mapping, the cost-aware two-objective loss, and the 40% constraint, so the optimizer has a clean, correct objective. Turnover is accumulated across the rebalances implied by the chosen frequency (§6.3–§6.4).

**Deliverables**
- `ng.p.Dict(weights=ng.p.Array(shape=(n,)), rebal=ng.p.Choice([...]))` with a `to_weights` simplex mapping, in `nevergrad_utils.py`.
- `portfolio_losses(weights, rebal_freq, train_returns, cost_bps) → [volatility, −net_return]`, with turnover `Σ|w_new − w_old|` across the chosen frequency's rebalances; net return = gross − `cost_bps · turnover`.
- The 40% cap registered via `register_cheap_constraint` (float ≤ 0 when satisfied); the risk-free rate from M3-3 reused.
- A documented softmax/simplex caveat for the tutorial narrative.

**Definition of Done**
- [ ] Simplex-mapped weights are non-negative and sum to 1 for arbitrary input vectors.
- [ ] `portfolio_losses` returns `[volatility, −net_return]` with the sign confirmed; an equal-weight portfolio yields plausible numbers.
- [ ] The constraint rejects any portfolio with a weight > 40% (float > 0 there, ≤ 0 otherwise).
- [ ] Turnover is unit-tested on a known weight change.

---

### `M4-2` — Optimize with `NGOpt` & `RandomSearch` → Pareto fronts
- **Assignee:** @jake
- **Labels:** `nevergrad`, `optimization`, `pareto`, `core`
- **Blocked by:** `M4-1`
- **Blocks:** `M4-3`, `M5-1`

**Context.** Run the explicit `ask`/`tell` loop to a fixed budget for both optimizers under the same budget, optionally seeding the reference point via `tell`, then read `optimizer.pareto_front()` and decode each candidate back to weights, rebalancing frequency, and `(volatility, net_return)`. This is the object the whole project centers on (§6.4).

**Deliverables**
- `NGOpt` and `RandomSearch` optimizers, both constructed with `parametrization=` and the **same budget** (~3,000; raised if sparse); seeded before the parametrization's first use.
- The `ask`/`tell` loop decoding `cand.value["weights"]`/`["rebal"]`, computing `portfolio_losses`, and telling the loss list; optional `optimizer.tell(ng.p.MultiobjectiveReference(), [vol_ub, negret_ub])` set first.
- Two Pareto fronts (one per optimizer) from `optimizer.pareto_front()` → `Parameter` objects (`.value` → weights + frequency, `.losses`), decoded to portfolios.
- A convergence check: increasing the budget stops materially changing the front.

**Definition of Done**
- [ ] Multiple distinct, non-dominated portfolios per optimizer; all respect the 40% cap.
- [ ] Front stops changing as the budget increases (convergence shown).
- [ ] Each Pareto `Parameter` decodes (`.value`) to weights + chosen frequency + `(volatility, net_return)`.
- [ ] NGOpt's front is at least as good as RandomSearch's at equal budget — or, if not, that is discussed (budget too small, noisy objective), not hidden.
- [ ] Seeds fixed **before parametrization use**; a re-run reproduces both fronts.

---

### `M4-3` — Finish `nevergrad.API.ipynb` (spec Milestone 2)
- **Assignee:** @jake
- **Labels:** `notebooks`, `nevergrad`, `api`, `documentation`
- **Blocked by:** `M2-3`, `M4-2`
- **Blocks:** `M4-4`

**Context.** Build the API teaching notebook so a reader learns Nevergrad's building blocks in isolation on a *toy* function before seeing the portfolio application — short, self-contained cells, each with a one/two-sentence explanation. This notebook's coverage maps exactly to the spec's Milestone 2 (§8).

**Deliverables**
- Cells demonstrating, on a toy function: parametrization (`ng.p.Array`, `ng.p.Choice`, composed in `ng.p.Dict`; optimizers built with `parametrization=`), optimizer choice (**`NGOpt` vs `RandomSearch`**), single- vs multi-objective losses, the **`ask`/`tell` loop**, **constraints via `register_cheap_constraint`** (float ≤ 0 when satisfied), the optional reference point via `optimizer.tell(ng.p.MultiobjectiveReference(), [upper_bounds])`, and **`optimizer.pareto_front()`** (`Parameter` objects with `.value`/`.losses`).
- Prose before each cell explaining the concept; no deprecated `instrumentation=` or attribute-style `pareto_front`.
- The notebook runs top-to-bottom in the container; scaffold it with the course `notebook.*` Claude skills and note which produced what (§8).

**Definition of Done**
- [ ] All API concepts demonstrated in isolated, runnable cells (`ng.p.Dict`/`Array`/`Choice`, `ask`/`tell`, constraints, multi-objective, `pareto_front`).
- [ ] Each concept has a short written explanation; no deprecated idioms.
- [ ] Notebook executes end-to-end with no manual steps (completes spec Milestone 2).

---

### `M4-4` — Checkpoint PR #1
- **Assignee:** @jake
- **Labels:** `github`, `pr`, `checkpoint`
- **Blocked by:** `M2-4`, `M4-2`, `M4-3`
- **Blocks:** —

**Context.** Submit Checkpoint PR #1 — data pipeline + walk-forward folds + baselines + working fronts + completed API notebook — for intermediate review around Nov 2, then move onto the next incremental branch without stalling if review is slow (§5.6).

**Deliverables**
- Checkpoint PR #1 pushed off the branch with the data/folds/baselines pipeline, both converged fronts, and the API notebook.
- Reviewers pinged; review feedback tracked; commits incremental and cross-linked to #598.
- Next incremental branch (`..._1`) started for M5 work.

**Definition of Done**
- [ ] Checkpoint PR #1 open with data + baselines + fronts + API notebook; reviewers pinged.
- [ ] Commits are incremental and scoped (no single dump).
- [ ] Work continues on the `..._1` branch regardless of review latency.

---

### `M4-5` — Analytical efficient-frontier convergence cross-check `[stretch]`
- **Assignee:** @jake
- **Labels:** `validation`, `scipy`, `stretch`
- **Blocked by:** `M4-2`
- **Blocks:** —

**Context.** Optional, high-leverage validator for the Depth & understanding rubric item: within one training fold, overlay the plain mean-variance efficient frontier computed analytically (closed-form min-variance + a `scipy.optimize` sweep across target returns) and report the max gap to the Nevergrad front. This bounds how far the optimizer is from the no-cost frontier. Clearly gated — it ignores the turnover cost and the rebalancing choice, so it validates the volatility/gross-return structure only, not the net-of-cost front (§6.4, §7).

**Deliverables**
- A `scipy` sweep across target returns → the analytical mean-variance curve for a chosen training fold.
- An overlay of that curve on the Nevergrad front, with the max front-to-analytical gap reported as a convergence number.
- A note on exactly what this cross-check does and does not validate (no-cost, single-fold).

**Definition of Done**
- [ ] Analytical frontier overlaid on the Nevergrad front for a stated fold.
- [ ] Max gap reported; the no-cost / single-fold caveat stated.
- [ ] Clearly gated as stretch — core M4 does not depend on it.
