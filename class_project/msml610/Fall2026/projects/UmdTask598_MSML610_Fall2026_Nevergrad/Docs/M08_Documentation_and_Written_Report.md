# Milestone 8 — Documentation & Written Report

> **GitHub Milestone**
> **Title:** `M8 — Documentation & Written Report`
> **Timeframe:** Week 9 (Nov 23 – Nov 29) *(light week: Thanksgiving Nov 26)*
> **Depends on:** `M7`
> **Description:**
> Freeze the deliverable content. Polish both notebooks so they read as a genuine **60-minute tutorial**, finish the README, and write the full narrative report per the §14 structure (§8, §14). This is intentionally a lighter week (Thanksgiving falls Nov 26), and it is where the notebooks and report are frozen so the video can be recorded cleanly next week. Exit criterion: README complete and the notebooks read end-to-end as a 60-minute tutorial.

## Notes
- **The tutorial IS a graded deliverable.** A reader who knows Python/ML but not Nevergrad should follow both notebooks end-to-end in ~60 minutes (§8).
- **The example notebook reads as a narrative:** data + walk-forward → baselines → objectives → NGOpt/RandomSearch fronts → out-of-sample → robustness → bonus, with markdown explaining *why*, not just *what*, and the **Optuna/Ax comparison** included (§8).
- **The report follows the §14 twelve-section structure** and states assumptions/limitations (in-sample vs out-of-sample gap, estimation risk, cost-model simplifications, scope exclusions: long-only, no forecasting, no taxes/slippage, no intraday) (§14, §3, §12).
- **Follow the tutorial workflow** in `tutorials/README.gp.md` and the quality bar in `tutorials/tutorials_checklist.md` — avoid "AI slop," professional peer-level tone, say each thing once; note which course Claude skills produced which artifacts (§8).
- **Freeze notebooks before recording** — record only after the content is frozen (§10).

---

## Issues

### `M8-1` — Polish the notebooks into a 60-minute tutorial
- **Assignee:** @jake
- **Labels:** `documentation`, `notebooks`, `tutorial`
- **Blocked by:** `M7-3`
- **Blocks:** `M9-1`

**Context.** Make `nevergrad.example.ipynb` read as a coherent narrative walkthrough and confirm the API notebook flows cleanly, so the pair works as a ~60-minute tutorial — the graded teaching deliverable (§8).

**Deliverables**
- The example notebook as a narrative: data + walk-forward → baselines → objectives → fronts (NGOpt vs RandomSearch) → out-of-sample → robustness → bonus, with markdown explaining *why* between cells, and the Optuna/Ax comparison present.
- Both notebooks run top-to-bottom in the container with fixed seeds; both paired to `.py` via `jupytext` and lint-clean (`lint_cc.py`).
- A timing sanity-check that a reader can complete the pair in ~60 minutes.

**Definition of Done**
- [ ] The example notebook reads as a coherent narrative, not just code, and includes the Optuna/Ax comparison.
- [ ] Both notebooks execute end-to-end, no manual steps, seeds fixed.
- [ ] Content frozen for recording.

---

### `M8-2` — Finalize the README / documentation
- **Assignee:** @jake
- **Labels:** `documentation`, `readme`
- **Blocked by:** `M7-4`
- **Blocks:** `M9-1`

**Context.** Complete the README from the M7 draft so setup, usage, API description, and design decisions are all present and clear enough for a new reader to run the project from the README alone (§8).

**Deliverables**
- A finished README: Docker setup/build (base image chosen and why, any Dockerfile modifications), run steps, the Nevergrad API description, the §7 design decisions, and the Optuna/Ax comparison.
- Reproducibility notes: `version.log`, cached-data provenance (fetch date + checksum), fixed seeds, and the fixed data window.
- A note on which course Claude skills (`notebook.*`, `tutorial_in_60_mins.*`) produced which artifacts.

**Definition of Done**
- [ ] README covers all required sections completely (setup, usage, API, design decisions, Optuna/Ax).
- [ ] Reproducibility notes included.
- [ ] A new reader could set up and run the project from the README alone.

---

### `M8-3` — Write the narrative report
- **Assignee:** @jake
- **Labels:** `report`, `writing`
- **Blocked by:** `M7-1`, `M7-2`, `M6-2`
- **Blocks:** `M9-1`

**Context.** Write the full narrative report following the §14 twelve-section structure, interpreting the results and stating limitations honestly (§14).

**Deliverables**
- Report sections 1–12 per §14: introduction; data; problem formulation; method; results; **out-of-sample evaluation**; **robustness**; bonus; discussion & limitations; **comparison with Optuna/Ax**; conclusion; references & appendix.
- Figures embedded and captioned (Pareto fronts NGOpt vs RandomSearch, cumulative net-return vs baselines, robustness error-bar, bonus comparison).
- Assumptions and limitations stated (in-sample vs OOS gap, estimation risk, cost-model simplifications, scope exclusions).

**Definition of Done**
- [ ] All twelve report sections written, including out-of-sample, robustness, and the Optuna/Ax comparison.
- [ ] Figures embedded and captioned.
- [ ] Limitations and assumptions explicitly stated.
