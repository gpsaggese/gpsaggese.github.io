# Milestone 9 — Video & Final Submission

> **GitHub Milestone**
> **Title:** `M9 — Video & Final Submission`
> **Timeframe:** Final week + buffer (Nov 30 – Dec 6)
> **Depends on:** `M8`
> **Description:**
> The finale. With the notebooks and docs frozen, run a final top-to-bottom verification, record the **10–20 minute video** following the required structure, submit the clean final PR, and upload the video to the course Google Drive — all by the **Monday, December 7, 2026** deadline (§10, §12, §5.6). The strength of the submission is that everything runs end-to-end with fixed seeds and the Nevergrad portfolios are tested **honestly out of sample** — walk-forward against equal-weight and minimum-variance baselines, with a robustness sweep over seeds and cost levels — so the video demonstrates a defensible result, not just a plausible story. Exit criterion: final PR submitted **and** video in the course Google Drive, by Mon Dec 7.

## Notes
- **Record only after notebooks and docs are frozen** (M8); rehearse once — a clean 12–15 minute run beats an unrehearsed 20 (§10).
- **The video structure is prescribed:** intro → file showcase → Docker execution → open Jupyter → full walkthrough → discuss results → documentation review; steps 1–4 take ~1–2 min, and the majority of time goes to the walkthrough (§10).
- **The §12 Definition-of-Done gates the submission:** builds/runs cleanly; walk-forward folds non-leaking; both baselines and both optimizers (NGOpt, RandomSearch) produce per-fold results; fronts non-dominated and respect the 40% cap; selected portfolios evaluated out of sample (Sharpe, max drawdown, turnover, cumulative net return) vs baselines; robustness sweep (5 seeds × {0,10,25} bps) reported; all six required tasks + chosen bonus produce labeled outputs; docs complete; PR clean; limitations stated (§12).
- **Dec 5–6 is buffer:** address any last-minute review comments or re-record if needed before the Mon Dec 7 deadline (§9).
- **Don't leave the merge-ready check to the last hour;** keep commits scoped and the PR clean (§5.6).

---

## Issues

### `M9-1` — Final verification & clean-up
- **Assignee:** @jake
- **Labels:** `verification`, `qa`
- **Blocked by:** `M8-1`, `M8-2`, `M8-3`
- **Blocks:** `M9-2`, `M9-3`

**Context.** Run the full Definition-of-Done gate before recording: everything builds and runs top-to-bottom in the container with fixed seeds, and the PR is clean (§12).

**Deliverables**
- A clean container rebuild (`--no-cache`); both notebooks run top-to-bottom with no manual steps and fixed seeds.
- A checklist pass against §12: non-leaking folds; both baselines + both optimizers per fold; fronts non-dominated and ≤ 40% per weight; out-of-sample metrics vs baselines; robustness sweep reported; all six tasks + chosen bonus produce labeled outputs; docs complete (incl. Optuna/Ax comparison).
- The PR tidied: scoped commits, meaningful messages, `Fixes …#598` line, reviewers added, no stray debug/temp files.

**Definition of Done**
- [ ] Image builds and both notebooks run cleanly from the documented steps.
- [ ] §12 Definition-of-Done checklist fully green (OOS vs baselines + robustness, not an analytical-curve claim).
- [ ] PR clean (scoped commits, linked issue, no clutter).

---

### `M9-2` — Record the video
- **Assignee:** @jake
- **Labels:** `video`, `presentation`
- **Blocked by:** `M9-1`
- **Blocks:** `M9-3`

**Context.** Record the 10–20 minute video following the prescribed structure, after content is frozen and rehearsed once (§10).

**Deliverables**
- A video covering: introduction (name, UID, tool + difficulty, title); file showcase + naming conventions; Docker build/run success (and any workaround if Docker gave trouble); opening Jupyter; a full cell-by-cell walkthrough with verbal explanation; results discussion (fronts, out-of-sample metrics vs baselines, robustness — and how Nevergrad addressed the cost-aware allocation problem); documentation review.
- One rehearsal pass; aim for a tight 12–15 minutes.

**Definition of Done**
- [ ] Video recorded covering all seven required segments.
- [ ] Docker build/run success shown; the full walkthrough demonstrates correctness.
- [ ] Length within 10–20 min, rehearsed.

---

### `M9-3` — Final PR & video upload
- **Assignee:** @jake
- **Labels:** `github`, `pr`, `submission`
- **Blocked by:** `M9-1`, `M9-2`
- **Blocks:** —

**Context.** Submit the final deliverables by the deadline: the clean final PR (fork branch → upstream `master`) and the video uploaded to the course Google Drive (§10, §5.6).

**Deliverables**
- The final PR submitted (all deliverables, reviewers added, issue linked via the `Fixes …#598` line).
- The video uploaded to the course Google Drive folder.
- Submission confirmed before Mon Dec 7.

**Definition of Done**
- [ ] Final PR submitted and clean.
- [ ] Video uploaded to the course Google Drive.
- [ ] Everything complete by **Monday, December 7, 2026**.
