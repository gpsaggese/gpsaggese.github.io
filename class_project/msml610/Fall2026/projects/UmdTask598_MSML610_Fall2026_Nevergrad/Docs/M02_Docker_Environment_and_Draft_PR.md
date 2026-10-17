# Milestone 2 — Docker Environment & Draft PR

> **GitHub Milestone**
> **Title:** `M2 — Docker Environment & Draft PR`
> **Timeframe:** Week 2 (Oct 5 – Oct 11)
> **Depends on:** `M1`
> **Description:**
> Turn the copied template into a working, reproducible environment and open the draft PR. Choose the base image, pin project dependencies, configure `.dockerignore` so cached data never enters the image, build the container, launch JupyterLab, rename the notebook templates to `nevergrad.*`, and open the draft PR with reviewers added. The guiding principle is **change only what's necessary** — `requirements.txt`, the base-image choice, and `.dockerignore` are essentially all that should move (§4). Getting Docker and the PR working now protects two rubric items (Working Docker, PR quality) and starts the Agile review cadence early.

## Notes
- **Keep the Python 3.12-slim default** unless a dependency fails to build on slim, then fall back to Ubuntu 24.04 + pip (§4.3).
- **Do NOT add `jupyterlab`, `jupyterlab_vim`, or `ipywidgets`** to `requirements.txt` — the Dockerfile installs them as infrastructure (§4.4).
- **Pin `nevergrad==1.0.12`** (latest release, Apr 2025) — its multi-objective API is already confirmed against this release (§4.4), and it's published for Python 3.6+, so it's compatible with the 3.12-slim base. Later notebook work (M4) uses the confirmed idioms rather than re-verifying from scratch.
- **Do not ship data into the image** — exclude `data/`, `*.csv`, `*.pkl`, `*.parquet` in `.dockerignore` (§4.5).
- **Open the draft PR early** (from your fork's branch → upstream `master`), add TAs + `@gpsaggese` as **reviewers** (not assignees), and keep working even if review is slow (§5.6, §5.0).

---

## Issues

### `M2-1` — Base image, `requirements.txt`, and `.dockerignore`
- **Assignee:** @jake
- **Labels:** `docker`, `dependencies`, `config`, `core`
- **Blocked by:** `M1-3`
- **Blocks:** `M2-2`

**Context.** Configure the single Dockerfile's active base-image section, declare only project libraries (pinned) in `requirements.txt`, and set `.dockerignore` so cached price data never enters the image. Change only what's necessary — everything else inherits from the template.

**Deliverables**
- Python 3.12-slim base kept active; the other two base-image sections commented out (§4.3).
- `requirements.txt` pinning `yfinance==1.7.0`, `nevergrad==1.0.12`, `numpy`, `pandas`, `matplotlib`, `scipy` (closed-form min-variance baseline + the optional CVaR bonus) (§4.4). Optionally keep a `requirements.in` and `pip-compile` for transitive pinning. Add `pandas-datareader` only if you automate the Ken French fallback fetch (§4.8).
- `.dockerignore` extended to exclude `data/`, `*.csv`, `*.pkl`, `*.parquet` alongside the template's `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `.git/`, `*.log` (§4.5).

**Definition of Done**
- [ ] Exactly one base-image section active in the Dockerfile.
- [ ] `requirements.txt` lists only project libs, all pinned; no `jupyterlab`/`ipywidgets`.
- [ ] `nevergrad==1.0.12` pinned (API confirmed against this release; §4.4).
- [ ] Pins confirmed against the live index before freezing (`pip index versions yfinance`, `pip index versions nevergrad`); if a newer release supersedes `yfinance==1.7.0` / `nevergrad==1.0.12`, update the pin and re-check the dependent API notes (§4.4, §6.1, §6.3).
- [ ] `.dockerignore` excludes data artifacts and caches.

---

### `M2-2` — Build the container & launch Jupyter
- **Assignee:** @jake
- **Labels:** `docker`, `jupyter`, `core`
- **Blocked by:** `M2-1`
- **Blocks:** `M2-3`, `M2-4`

**Context.** Build the image and confirm the container runs and serves JupyterLab. This is the "Working Docker" rubric item, so it must build and run cleanly from the documented steps, with no host-side installs beyond the container engine.

**Deliverables**
- Successful `./docker_build.sh` (run `--no-cache` once for a clean build); smoke test `./docker_bash.sh ls`.
- JupyterLab reachable via `./docker_jupyter.sh` at `http://localhost:8888`.
- `version.log` reviewed after the build (auto-generated at Stage 8 via `version.sh`) and kept with the project (§4.7).

**Definition of Done**
- [ ] `docker_build.sh` completes without errors.
- [ ] `docker_bash.sh ls` smoke test passes.
- [ ] JupyterLab loads at `:8888`.
- [ ] `version.log` generated and reviewed.

---

### `M2-3` — Rename notebooks to `nevergrad.*` and mirror the reference layout
- **Assignee:** @jake
- **Labels:** `notebooks`, `structure`
- **Blocked by:** `M2-2`
- **Blocks:** `M4-3`

**Context.** Rename the template notebooks to the project's `nevergrad.*` names (the tool name, not the branch/directory name) and model the directory on a reference tutorial (`tutorials/Autogen`) so the layout matches the expected `*.API.ipynb` / `*.example.ipynb` / `*_utils.py` convention (§4.2, §5.5).

**Deliverables**
- `template.API.ipynb` → `nevergrad.API.ipynb`; `template.example.ipynb` → `nevergrad.example.ipynb`; the template utils module → `nevergrad_utils.py`.
- Empty `nevergrad_utils.py` helper module created for shared functions, importable inside the container.
- Directory reviewed against `tutorials/Autogen` (and `tutorials/Ax_Multi_Objective_Optimization/`, the Fall2025 Optuna project, and `msml610/tutorials/L03_knowledge_representation/`) for structure parity (§5.5).

**Definition of Done**
- [ ] Notebooks renamed to `nevergrad.API.ipynb` and `nevergrad.example.ipynb`; utils to `nevergrad_utils.py`.
- [ ] `nevergrad_utils.py` present and importable inside the container.
- [ ] Layout matches the reference tutorials' file convention.

---

### `M2-4` — Open the draft PR with reviewers
- **Assignee:** @jake
- **Labels:** `github`, `pr`, `core`
- **Blocked by:** `M2-2`
- **Blocks:** `M4-4`

**Context.** Open the draft PR (from your fork's branch → upstream `master`), named after the branch, reference the issue, and add reviewers so the Agile checkpoint cadence can start. Opening it now — even before it's "ready" — protects the PR-quality rubric item and gets feedback flowing early (§5.6).

**Deliverables**
- Draft PR opened (fork branch → upstream `master`), named the same as the branch, with `Fixes gpsaggese/gpsaggese.github.io#598` at the start of a line in the description (§5.6).
- TAs and `@gpsaggese` added in the **Reviewers** field; PR self-assigned.
- First meaningful commits pushed (environment + renamed notebooks), each subject ending with `(gpsaggese/gpsaggese.github.io#598)` — not a single end-of-term dump.

**Definition of Done**
- [ ] Draft PR open with the correct name, `Fixes …#598` line, and issue link.
- [ ] Reviewers (TAs + `@gpsaggese`) added; PR self-assigned.
- [ ] Environment + notebook-rename commits pushed to the fork branch, cross-linked to #598.
