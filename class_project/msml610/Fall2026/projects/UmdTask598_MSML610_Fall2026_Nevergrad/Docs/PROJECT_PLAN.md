# MSML610 — Project 1 (Fall2026): Cost-Aware Multi-Objective Portfolio Allocation
## Comprehensive Project Plan

**Course:** MSML610 — Advanced Machine Learning (UMD, MS in Applied Machine Learning)
**Project type:** "Learn X in 60 Minutes" tutorial — build a cost-aware portfolio optimizer **using Nevergrad**
**Difficulty:** Hard (confirm the exact rating against the sign-up sheet)
**Final submission due:** **Monday, December 7, 2026** (from the course class schedule — *not* stated in the Project 1 spec or `class_project/README.md`, both of which defer the date to the schedule's "Class assignment" column)
**Spec of record:** `class_project/project_descriptions/MSML610/nevergrad_Project_Description.md` — **Project 1 (Fall2026)** plus the shared **Tutorial** section at the top of that file
**Prepared:** September 2026 (revised October 2026 to match the updated Project 1 spec)

---

## 1. Project Overview

Build a system that selects weights for a **multi-asset ETF portfolio** by simultaneously **maximizing net-of-cost return** and **minimizing risk (return volatility)**, using the **multi-objective optimization** facilities of the **Nevergrad** library. Because these two goals conflict, the output is a *set* of non-dominated portfolios — the **Pareto front** — not a single portfolio. The project then **tests the Pareto portfolios out of sample**, with walk-forward evaluation, against two simple baselines (equal-weight and closed-form minimum-variance).

Two features make this a *cost-aware* problem rather than a textbook Markowitz exercise: returns are charged a **transaction cost tied to turnover**, and the optimizer also chooses a **rebalancing frequency**. Both push the problem out of clean quadratic-programming territory and into exactly the regime a gradient-free optimizer is built for.

**Framing that governs every deliverable:** per the course guidelines, the end product is a **tutorial that teaches a curious computer scientist how to use Nevergrad in ~60 minutes**, using cost-aware portfolio allocation as the worked example. The goal is *two-fold*: (a) correctly solve the allocation problem and test it honestly out of sample, and (b) explain Nevergrad clearly enough that a reader learns the tool from the notebooks and documentation. Design decisions should always be legible and justified, not black-box.

**One-line objective:** *Given daily prices for nine ETFs since 2006, use Nevergrad to search weights (and a rebalancing frequency) that trade off net-of-cost return against risk; walk-forward-test the selected Pareto portfolios out of sample against equal-weight and minimum-variance baselines; analyze robustness across seeds and cost levels; and package the whole thing as a runnable, well-documented 60-minute tutorial contributed via a proper GitHub PR.*

---

## 2. Conceptual Framing

Four ideas up front make every task fall into place and make the tutorial defensible.

- **Markowitz mean-variance, made cost-aware.** A portfolio is a weight vector `w` (long-only, sums to 1). Its expected return is `w · μ` and its risk is `wᵀ Σ w`, where `μ` and `Σ` are the annualized mean-return vector and covariance matrix. Here the return objective is charged a cost proportional to **turnover** (how much the weights move when rebalancing), so the optimizer trades off raw return against the cost of achieving it.
- **Multi-objective optimization & Pareto dominance.** Portfolio A *dominates* B if it is at least as good on both objectives and strictly better on one. The portfolios that nothing dominates form the **Pareto front**. There is no single "best" answer in-sample — the front is the answer, and we pick *one* point from it by a stated rule (highest training Sharpe) and then judge it out of sample.
- **Why Nevergrad, not quadratic programming.** Classical mean-variance is solved analytically or by QP. This problem is not clean QP: the **turnover-cost term is non-smooth and path-dependent**, and the **rebalancing frequency is a discrete choice**. Nevergrad — a **gradient-free / evolutionary** optimizer — handles this mixed continuous/discrete, non-smooth, multi-objective problem directly through one `ask`/`tell` interface. That is the real teaching point: a general-purpose black-box optimizer solves a problem where the textbook closed form no longer applies.
- **Out-of-sample reality.** In-sample Pareto-optimality does **not** guarantee out-of-sample performance; costs and estimation error erode it. Walk-forward evaluation, naive baselines, and a robustness sweep over seeds and cost levels are how the project tests — honestly — whether the Nevergrad portfolios are actually worth the complexity.

---

## 3. Scope, Constraints & Deliverables

### In scope
- Historical price acquisition for the nine-ETF basket since 2006, with returns and **walk-forward windows** (5 training years + 1 test year, rolled yearly).
- Two baselines: **equal-weight** and **closed-form minimum-variance**, reported with out-of-sample Sharpe and turnover.
- A **cost-aware multi-objective loss**: negative annualized return *net of* a turnover cost, and annualized volatility; with a **per-weight cap of 40%** enforced as a constraint.
- A Nevergrad optimizer over a `Dict` of weights **and a rebalancing-frequency choice**, run via `ask`/`tell`, returning a Pareto front; compared for **`NGOpt` vs `RandomSearch` under the same budget**.
- **Out-of-sample evaluation**: select the highest-training-Sharpe front point, apply it to the next year, and report cumulative net return, Sharpe, maximum drawdown, and turnover against the baselines.
- **Robustness analysis**: five seeds × costs {0, 10, 25 bps}, reporting mean and standard deviation of out-of-sample Sharpe.
- **Bonus:** replace volatility with 95% CVaR and compare fronts; add maximum drawdown as a third objective.
- A **60-minute tutorial** (two notebooks + a utils module + documentation) teaching Nevergrad through the above.

### Constraints (from course guidelines)
- **Python only**, plus configuration files for the tools. No other languages.
- **Everything runs locally in Docker — no cloud compute.** (`yfinance` fetches *public* market data over HTTP, which is fine; there is no paid/cloud service or AWS resource. Cache fetched data locally.)
- **No paid services.** Nothing in this project requires them.
- Do the work yourself: AI/search are aids, but copy-pasted code you can't explain defeats the purpose (and the video/PR review will expose it).

### Out of scope (state explicitly in the report)
- Short-selling / leverage (assume **long-only, fully invested**).
- Return *forecasting* — historical estimates are proxies, with limitations noted.
- Taxes and market-impact/slippage beyond the simple per-unit-turnover cost model.
- Intraday and live/real-time trading.

> **Note on what is now *in* scope.** Transaction costs (via turnover) and rebalancing dynamics (via the rebalancing-frequency choice) are **required** by this project — they are *not* out of scope. An earlier draft of this plan treated them as exclusions; that was for a different (plain-Markowitz) project and has been removed.

### Deliverables (all submitted via one PR to `gpsaggese.github.io`; see §5)
1. **`nevergrad.API.ipynb`** — a notebook that teaches Nevergrad's API in isolation (`ng.p.Dict`, `ng.p.Array`, `ng.p.Choice`; the `NGOpt` and `RandomSearch` optimizers; the `ask`/`tell` loop; constraints via `register_cheap_constraint`; multi-objective losses and `optimizer.pareto_front()` on a small toy function), mirroring the reference `*.API.ipynb` layout.
2. **`nevergrad.example.ipynb`** — the full cost-aware portfolio application (all six tasks + chosen bonus), written as a readable walkthrough.
3. **`nevergrad_utils.py`** — a helper module holding reusable functions (data, walk-forward splitting, portfolio stats, losses, optimization, evaluation, plotting), imported by both notebooks.
4. **Dockerized environment** — `Dockerfile`, `requirements.txt`, and the template's `docker_*.sh` scripts, building and running cleanly.
5. **Documentation / README** — setup steps, usage examples, an API description of the Nevergrad features used, the architectural/design decisions (see §7–§8), and a results narrative (see §14).
6. **Generated figures** — Pareto fronts (NGOpt vs RandomSearch), cumulative net return vs baselines, and the robustness summary.
7. **Video recording (10–20 min)** — uploaded to the course Google Drive (see §10).
8. **`version.log`** — auto-generated build record for reproducibility.
9. **Paired `.py` files** (via `jupytext`) for each notebook, so the code lints and runs end-to-end headless (see §8).

---

## 4. Environment & Setup

This project runs inside the **course Docker project template** (`class_project/project_template`). All work happens *in the container*, driven by the provided `docker_*.sh` helper scripts; nothing is installed on the host beyond a container engine. The template's guiding principle is **change only what's necessary** — here that means `requirements.txt` and (if needed) the base-image choice; everything else inherits from the template.

### 4.1 — Prerequisites
- A container engine: **Docker** (Linux, or Docker Desktop on macOS) or **Apple's `container`** tool (macOS default). Scripts pick the engine from `DOCKER_ENGINE` — `docker` on Linux, `apple` on macOS by default. Force Docker on macOS with `DOCKER_ENGINE=docker ./docker_build.sh`.
- On Windows, use VMware or dual-boot Linux (or a UMD lab machine) since Docker is expected to run natively.
- Multi-architecture builds are not needed; if ever used they require `docker buildx` (`DOCKER_BUILD_MULTI_ARCH=1`).

### 4.2 — Create the project from the template
- Copy the template into your project directory (path per §5.4) and `cd` into it:
  - `cp -r class_project/project_template <your-project-dir>` → `cd <your-project-dir>`
  - *(Alternative scaffolder)* `create_project.py --dst_dir <your-project-dir>` copies the template, renames the `template.*` files, and sets `IMAGE_NAME=umd_project_<dir-name>` in `docker_name.sh`. **Caveat:** its rename keys off the destination **directory name** (your `UmdTask598_…` branch), so it would emit `UmdTask598_….API.ipynb`, not the required names — so do the rename below regardless. **Do not** use its `create_links` action: symlinking the Docker files back to the template would leave the submission non-self-contained.
- **Rename the three deliverables to the tool name `nevergrad`** (spec requirement; mirrors `tutorials/Autogen` → `autogen.*`): `template.API.ipynb` → **`nevergrad.API.ipynb`**, `template.example.ipynb` → **`nevergrad.example.ipynb`**, and the template utils module → **`nevergrad_utils.py`**. The notebook/utils prefix is the **tool name**, *not* the project-directory/branch name.
- This provides the `Dockerfile`, helper scripts (`docker_build.sh`, `docker_bash.sh`, `docker_jupyter.sh`, `docker_cmd.sh`, `docker_clean.sh`, `docker_exec.sh`, `docker_push.sh`, `run_jupyter.sh`, `utils.sh`, `version.sh`), config files (`bashrc`, `etc_sudoers`, `docker_name.sh`), `.dockerignore`, notebook templates (`template.example.ipynb`, `template.API.ipynb`), and the test harness (`test/test_docker_all.py`). Follow `class_project/project_template/docker_scripts.README.md` carefully.

### 4.3 — Choose the base image
The single `Dockerfile` ships three base-image sections; exactly one is active.

| Option | Best for |
|---|---|
| **Python 3.12-slim** (default, active) | Minimal image; pure-Python / wheel packages |
| Ubuntu 24.04 + pip | Needs system tools (git, curl, graphviz, etc.) |
| Ubuntu 24.04 + uv | Fast dependency resolution with `uv` |

- **Recommendation:** keep the **Python 3.12-slim default** — this project needs only wheel-available packages, so no system toolchain is required. Fall back to **Ubuntu 24.04 + pip** only if a dependency fails to build on slim.
- Uncomment the chosen section; comment out the other two.

### 4.4 — Declare project dependencies (`requirements.txt`)
List **only project libraries, with pinned versions**:
- `yfinance==1.7.0` (data; latest release, confirmed current Oct 2026), `nevergrad==1.0.12` (optimization; latest release, Apr 23 2025), `numpy==2.5.3`, `pandas==2.3.3`, `matplotlib==3.11.2`, and `scipy==1.18.1` (closed-form minimum-variance baseline and the optional CVaR bonus). Versions are the latest stable on PyPI as of Oct 7 2026, except `pandas` — see the next bullet. All four require Python ≥ 3.12 or lower, so they build on the Python 3.12-slim base image (§4.3); none of them, `yfinance`, or `nevergrad` declares an upper-bound cap that conflicts with these pins.
- **`pandas` is deliberately pinned to the latest 2.x (`2.3.3`), not 3.x (`3.0.6`).** pandas 3.0 is a major release (Copy-on-Write by default, new default string dtype); this project uses pandas only as the I/O boundary (yfinance fetch → CSV cache → date alignment), while the numeric core is NumPy, so 3.0 buys nothing here. The one real risk is `yfinance==1.7.0`'s internal DataFrame handling under 3.0 — it declares no cap (`pandas>=1.3.0`), but "no declared cap" is not "tested on 3.0," and this is a must-run-cleanly-in-Docker tutorial (§4.7, §12). 2.3.3 needs `numpy>=1.26.0` with no upper cap, so it stays compatible with `numpy==2.5.3`. Revisit 3.x only after a full headless container run confirms yfinance behaves.
- *(Optional)* `pandas-datareader` only if you automate the Ken French fallback fetch (§4.8); otherwise download that CSV once and cache it.
- **Do NOT** add `jupyterlab`, `jupyterlab_vim`, or `ipywidgets` — installed by the Dockerfile (Stage 5) as infrastructure.
- For clean pinning, optionally keep a `requirements.in` and generate `requirements.txt` with `pip-compile` (pip-tools) for full transitive pinning.
- **Pin `nevergrad==1.0.12`** — every API idiom in this plan is confirmed against it (verified against the live docs so the notebook doesn't copy a stale example):
  - **Compose the search space with `ng.p.Dict`** of named parts, e.g. `ng.p.Dict(weights=ng.p.Array(shape=(n,)), rebal=ng.p.Choice([...]))`. A candidate's `.value` is the dict `{"weights": np.ndarray, "rebal": <chosen item>}`.
  - **`ng.p.Array(shape=(n,))`** supports `.set_bounds(lower, upper)` and `.set_integer_casting()`; **`ng.p.Choice([...])`** is an unordered categorical whose selected item is read from `.value`.
  - **Constraints go through `parametrization.register_cheap_constraint(fn)`**, where `fn` receives the parameter value and returns a **float that is ≤ 0 when the constraint is satisfied** (docs idiom: a constraint `x[0] >= 1` is written `lambda x: x[0] - 1`). For the 40% cap: `lambda d: float(np.max(to_weights(d["weights"])) - 0.40)`.
  - **Optimizers are constructed with the `parametrization=` keyword** (e.g. `ng.optimizers.NGOpt(parametrization=param, budget=...)`); the older `instrumentation=` keyword is deprecated.
  - **Multi-objective:** `tell` a *list* of losses, e.g. `optimizer.tell(cand, [volatility, neg_net_return])`; optionally set an upper-bound reference first via `optimizer.tell(ng.p.MultiobjectiveReference(), [vol_ub, negret_ub])` ("highly advised," else auto-inferred); read the front with `optimizer.pareto_front()` → a list of `Parameter` objects, each carrying `.value` and `.losses`. `pareto_front(n, subset=...)` returns a representative subset.
  - **Python compatibility:** 1.0.12 is published for Python 3.6+, so it is fully compatible with the Python 3.12-slim base image (§4.3). Upstream marks multi-objective support as *unstable*, so re-verify these idioms only if you change the pin.

### 4.5 — Configure `.dockerignore`
Keep the template's exclusions and add project-specific ones. Critically, **do not ship data into the image**: exclude `data/`, `*.csv`, `*.pkl`, `*.parquet`, alongside the usual `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `.git/`, `*.log`. Cached price data lives in a mounted `data/` directory, not in the image.

### 4.6 — Build, run, and work in the container
- **Build:** `./docker_build.sh` (add `--no-cache` for a clean rebuild). Smoke-test with `./docker_bash.sh ls`.
- **Interactive shell:** `./docker_bash.sh` — mounts the current directory as `/data` inside the container.
- **Run a script / command:** `./docker_cmd.sh python <script>.py ...`, or `./docker_cmd.sh pytest test/`.
- **JupyterLab:** `./docker_jupyter.sh`, then open `http://localhost:8888`. Use `-p <port>` for a custom port and `-u` for vim keybindings.
- Every script accepts `-h` (help) and `-v` (verbose/trace).

### 4.7 — Reproducibility
- **Pinned `requirements.txt`** fixes package versions.
- **`version.log`** is auto-generated at build time (Dockerfile Stage 8 via `version.sh`), recording Python/pip/Jupyter and every installed package — review after each build and keep it with the submission.
- **Fix random seeds** for Nevergrad and any Monte-Carlo sampling. Call `np.random.seed(...)` **before the parametrization's first use** — Nevergrad's parametrization pulls its random state from numpy's global seed at that moment, so getting the order wrong means the "fixed seed" won't actually fix the run. The robustness sweep (§6.6) deliberately varies this seed across five values.
- **Fix the data window** with explicit `start`/`end` dates so results don't drift as new prices arrive.
- **Treat the cached CSV as the source of truth.** Fixed dates alone don't guarantee identical numbers on re-fetch: `yfinance` serves *adjusted* history that can be restated later (splits, dividend adjustments, backfills), so a fetch months from now can differ from today's. Commit a **data-provenance note** (fetch date + a checksum of the cached file) and have the notebooks load the cached CSV rather than re-fetching, or the video-day re-run may produce slightly different figures than the report.

### 4.8 — Data choices to lock early
- **Basket:** the nine ETFs from the spec — **SPY, EFA, EEM, TLT, IEF, LQD, GLD, VNQ, DBC** (US + developed + emerging equity, long + intermediate Treasuries, investment-grade credit, gold, REITs, commodities). A deliberately cross-asset basket so the covariance structure is interesting.
- **Window:** **daily data from 2006 onward** — long enough to support many walk-forward folds of 5 training years + 1 test year. (2006 start gives ~13 test years through the mid-2020s.)
- **Frequency:** daily returns, annualized by a 252-trading-day factor.
- **Walk-forward split:** 5 training years, 1 test year, **rolled one year at a time**; every estimate (`μ`, `Σ`, the optimization, baseline weights) uses **only** the training window of its fold — no test-year data leaks into estimation.
- **Static fallback (required by the spec):** the **Ken French "10 Industry Portfolios (Daily)"** library, used if Yahoo blocks requests. Note this fallback provides daily *returns* for ten industry portfolios (not the nine ETFs), so the loader should consume returns directly and skip the price→return step; keep the fallback path documented and cached with the same provenance discipline.
- **Caching:** save fetched prices/returns into the mounted `data/` directory (git/docker-ignored) so re-runs don't re-hit the API.

---

## 5. Course Workflow, Repository Structure & Submission

The project is delivered the way open-source contributions are: via a GitHub **issue → branch → commits → PR** flow into the `gpsaggese.github.io` repository, with intermediate reviews (Agile). Set this up **first**; the technical work lives inside it. This section is confirmed against `class_project/README.md`, section *Contribution to the Repo*, **and** `class_project/how_to_contribute.md` (the fork-and-PR mechanics in §5.0).

> **Term note:** the deadline (Dec 7, 2026) places this in **Fall 2026**. The guideline's tag pattern literally shows `Spring{year}`; adapt the term to your actual semester and **confirm the exact label (`Fall2026`) and course path with your TA/instructor** before creating the issue and branch.

### 5.0 — Contribution model: fork-and-PR
Two course docs describe the workflow and they assume **different access levels**. `class_project/README.md` (*Contribution to the Repo*) shows cloning the upstream repo directly and pushing a branch to it — the **collaborator** model, which needs write access. `class_project/how_to_contribute.md` describes the **fork-and-PR** model. You currently have **pull-only** access to upstream (not yet a collaborator), so **fork-and-PR is the flow that applies** — build on it, not README's direct-clone example.

- **Clone your fork, not upstream**, and wire the upstream remote once:
  - `GIT_LFS_SKIP_SMUDGE=1 git clone --recursive git@github.com:jkelle11-source/msml610.github.io.git` (the skip-smudge flag gets past upstream's broken LFS object).
  - `git remote add upstream git@github.com:gpsaggese/gpsaggese.github.io.git`
- **Sync before every new branch:** `git fetch upstream && git checkout master && git merge upstream/master`, so each branch is cut off the latest upstream `master` (`how_to_contribute.md`, step 3). Re-sync and rebase/merge `master` into your branch if conflicts arise before a review.
- **Push branches to your fork (`origin`), never to upstream `master`.** The PR is opened **from your fork's branch into upstream** `master`.

### 5.1 — Prerequisites (do once, up front)
- **Watch, star, and fork** the repos `gpsaggese/gpsaggese.github.io` (the course repo, called `umd_classes` in the guidelines) and `causify-ai/helpers`.
- **Install Docker** (see §4.1).
- **Confirm your GitHub issue assignment** on the `gpsaggese.github.io` issues tracker — make sure you are assigned.

### 5.2 — Project tag
Format: `{Class}_{Term}{Year}_{project_title_without_spaces}`.
- Working example: `MSML610_Fall2026_Nevergrad`.

### 5.3 — GitHub issue
- Create an issue titled with your **project tag**.
- Paste the project description and a link to the spec document (`.../MSML610/nevergrad_Project_Description.md`).
- Assign it to yourself (or tag the individuals working on it). This issue is the discussion thread.

### 5.4 — Branch & file location
- Branch name: `UmdTask{issue_number}_{project_tag}` (e.g., `UmdTask598_MSML610_Fall2026_Nevergrad`).
- Put files **only** under your project directory (note the **lowercase** course code):
  `{GIT_ROOT}/class_project/msml610/Fall2026/projects/{branch_name}`.
- Create the branch off `master` in your clone; never push to `master`.

### 5.5 — Reference layout and tutorials to study
Model the directory on the reference tutorials. The canonical layout exemplar is `tutorials/Autogen` (`docker_build.sh`, `autogen.API.ipynb`, `autogen.example.ipynb`, `autogen_utils.py`). The Project 1 spec additionally directs you to study, before structuring yours:
- `tutorials/Ax_Multi_Objective_Optimization/` (the closest existing multi-objective optimization tutorial — read its `README.md`).
- The **Fall2025 Optuna project**: `class_project/msml610/Fall2025/projects/UmdTask60_Fall2025_Optuna_Customer_Segmentation_Using_Clustering/README.md` (the closest existing optimizer/hyperparameter-search project).
- `msml610/tutorials/L03_knowledge_representation/` — **make your project *look like* this** in structure and style.

Your analogues: `nevergrad.API.ipynb`, `nevergrad.example.ipynb`, `nevergrad_utils.py`, plus the Docker files from the template.

### 5.6 — Pull Request workflow
- Open a **PR** (from your fork's branch → upstream `master`, per §5.0) named the same as your branch; reference the issue number.
- **Auto-close the issue on merge:** put `Fixes gpsaggese/gpsaggese.github.io#598` at the **start of a line** in the PR description (`how_to_contribute.md`) — the `Fixes` keyword only triggers from line start, and the `owner/repo#` prefix is required because the PR lives in a fork, not in the issue's repo.
- Add your **TAs and `@gpsaggese`** as reviewers — in the PR's **Reviewers** field, not Assignees (`how_to_contribute.md`); assign the PR to yourself.
- **Link each commit to the upstream issue:** end the commit subject with `(gpsaggese/gpsaggese.github.io#598)` so fork commits cross-link to issue #598. (The guideline writes this prefix as `gpsaggese/umd_classes#…`; the real repo slug is `gpsaggese.github.io`.)
- **Commit regularly** (not one dump at the end); expect intermediate review feedback.
- **Don't let reviewer latency stall you.** The Agile checkpoint model assumes `@gpsaggese` and the TAs review on a useful cadence, which you don't control. Open the draft PR **early**, ping reviewers proactively, and if Checkpoint PR #1 (~Nov 2) sits unreviewed, **keep working on the next incremental branch anyway** — never treat "waiting for review" as idle time.
- After a checkpoint PR is merged, continue on an **incremental branch** (`..._1`, `..._2`, …).
- Keep PRs clean: meaningful commit messages, scoped changes, no stray debug/temp files.

---

## 6. Work Breakdown Structure

Each task lists its **objective**, **approach**, **key decisions**, **outputs**, and **acceptance criteria**. The six tasks map one-to-one to the spec's Project 1 task list.

### 6.1 — Acquire the Data (+ walk-forward windows)
- **Objective:** Obtain clean, aligned daily returns for the nine-ETF basket and define the walk-forward folds.
- **Approach:** Download prices via `yf.download(tickers, start="2006-01-01", end=..., auto_adjust=True)`. **Mind the current column shape:** `download()` returns **MultiIndex columns by default** (price-type-first then ticker, e.g. `('Close', 'SPY')`), *even for a single ticker* — select the `Close` level to get a (dates × tickers) frame, or pass `multi_level_index=False`. Align all tickers on common trading dates, drop/handle missing values, and compute daily returns. Then build **walk-forward folds**: for each fold, 5 consecutive training years and the following 1 test year, rolled forward by one year; expose them as a generator so every downstream estimate is fed only its fold's training slice.
- **Key decisions:** **Pass `auto_adjust=True` explicitly** — with it the OHLC are split/dividend-adjusted and **there is no separate `Adj Close` column** (the adjusted series *is* `Close`); pinning the argument keeps intent explicit and reproducible. Simple vs. log returns (simple is standard). How to handle ETFs with shorter history than 2006 (shorten the common window to the latest inception, or document the trimmed start). Fallback to the Ken French returns dataset if Yahoo blocks (§4.8).
- **Outputs:** A cached returns table (dates × assets) with a provenance note (fetch date + checksum, per §4.7); a walk-forward fold iterator; a brief data-quality note.
- **Acceptance criteria:** No missing values; all assets share one date index; sane return ranges; folds are contiguous, non-leaking (training strictly precedes test), and rolled yearly.

### 6.2 — Define the Baselines
- **Objective:** Establish the naive comparators the Nevergrad portfolios must beat.
- **Approach:** For each fold, compute (i) the **equal-weight** portfolio (`1/N`) and (ii) the **closed-form minimum-variance** portfolio (`w ∝ Σ⁻¹ 1`, renormalized; long-only via `scipy.optimize` if the unconstrained solution goes short). Apply each to the test year and record its **out-of-sample Sharpe ratio and turnover**.
- **Key decisions:** How min-variance handles the long-only constraint (clip-and-renormalize vs. a small QP/`scipy.optimize` solve). Turnover definition (sum of absolute weight changes at each rebalance; see §6.3). Risk-free rate for Sharpe — the **average T-bill rate over the training window**, not 0 (short rates moved from ~0% to ~5% across 2006–2024, which materially changes Sharpe rankings); state it and reuse it everywhere Sharpe appears.
- **Outputs:** A table of per-test-year out-of-sample Sharpe and turnover for both baselines — the comparison backbone reused in §6.5.
- **Acceptance criteria:** Baseline weights are valid (long-only, sum to 1); the per-fold table is populated for every test year.

### 6.3 — Define the Objectives
- **Objective:** Formalize the two losses and the weight cap.
- **Approach:** Define two objectives to *minimize*: (1) **annualized volatility** `√(wᵀ Σ w) · √252`, and (2) **negative annualized return net of cost**, where cost = **10 basis points per unit of turnover**. Turnover is `Σ |w_new − w_old|` accumulated across the rebalances implied by the chosen frequency; net return = gross annualized return − `cost_bps · turnover`. Cap **each weight at 40%** via a Nevergrad constraint.
- **Key decisions:** Weight representation — an `ng.p.Array` mapped to the simplex (softmax, or clip-and-renormalize) so weights are long-only and sum to 1; the **40% cap enforced with `register_cheap_constraint(lambda d: np.max(to_weights(d["weights"])) - 0.40)`** (≤ 0 when satisfied). Annualization factor (252). Cost level (10 bps for the main run; 0 and 25 bps appear in the robustness sweep, §6.6). Risk-free rate per §6.2.
- **Outputs:** A documented `portfolio_losses(weights, rebal_freq, train_returns, cost_bps)` routine returning `[volatility, -net_return]`, plus the constraint function — the backbone reused by the optimizer and the evaluator.
- **Acceptance criteria:** Correct shapes and signs (volatility and *negative* net return, both minimized); an equal-weight portfolio yields plausible numbers; the constraint rejects any portfolio with a >40% weight.

### 6.4 — Optimize with Nevergrad
- **Objective:** Search weights *and* rebalancing frequency with Nevergrad and return a Pareto front, for two optimizers under one budget.
- **Approach:**
  - Build the search space as **`param = ng.p.Dict(weights=ng.p.Array(shape=(n,)), rebal=ng.p.Choice(["monthly", "quarterly", "annual"]))`** and register the 40% constraint on it (§6.3).
  - Run the **`ask`/`tell` loop** (the spec asks for this explicitly, not `minimize`): `cand = optimizer.ask()`; decode `w = to_weights(cand.value["weights"])`, `freq = cand.value["rebal"]`; compute `losses = portfolio_losses(w, freq, train_returns, cost_bps)`; `optimizer.tell(cand, losses)`. Optionally set an upper-bound reference first via `optimizer.tell(ng.p.MultiobjectiveReference(), [vol_ub, negret_ub])`.
  - After the budget is spent, read `optimizer.pareto_front()` — a list of non-dominated `Parameter` objects, each with `.value` (→ weights + chosen frequency) and `.losses`.
  - Do this for **both `NGOpt` and `RandomSearch` under the *same* budget**, so the tutorial can ask whether the meta-optimizer actually beats random search for a given compute budget.
- **Key decisions:** Optimizers are **`NGOpt`** (Nevergrad's meta-optimizer) and **`RandomSearch`** (the honest baseline) — constructed with the `parametrization=` keyword. **Budget** (start ~3,000 per optimizer; raise if the front is sparse) kept identical across optimizers for a fair comparison. Reference-point values (tune toward realistic maxima). The simplex-mapping caveat — softmax can't reach exact-zero weights and mildly biases toward the interior; note it, though the hard 40% cap already removes single-asset corner solutions. API confirmed against `nevergrad==1.0.12` (§4.4).
- **Outputs:** Two Pareto fronts (one per optimizer), each a list of portfolios with weights, rebalancing frequency, and `(volatility, net_return)`.
- **Acceptance criteria:** Multiple distinct, non-dominated portfolios per optimizer; increasing the budget stops materially changing the front (convergence); NGOpt's front is at least as good as RandomSearch's at equal budget (or, if not, that is discussed).

### 6.5 — Evaluate Out of Sample
- **Objective:** Test the selected Pareto portfolios on held-out years against the baselines.
- **Approach:** In each fold, from the training-data Pareto front select the **single point with the highest training Sharpe ratio**, then **apply its weights and rebalancing frequency to the next (test) year**. Compute the realized **cumulative net return, out-of-sample Sharpe ratio, maximum drawdown, and turnover**, alongside the equal-weight and minimum-variance baselines from §6.2. Plot the training Pareto fronts and the **cumulative net-return curves of the selected portfolio vs. the baselines**.
- **Key decisions:** The selection rule (highest training Sharpe) is fixed and stated so the choice is reproducible, not cherry-picked. Risk-free rate per §6.2. Maximum-drawdown definition (largest peak-to-trough decline of the cumulative net-return series). Whether to aggregate metrics across folds (mean per metric) in addition to per-year reporting.
- **Outputs:** Per-fold out-of-sample metric table (Sharpe, max drawdown, turnover, cumulative net return) for the Nevergrad portfolio and both baselines; the front and cumulative-return figures.
- **Acceptance criteria:** Metrics are computed on test-year data only; the comparison against baselines is explicit; the report states clearly whether and when the Nevergrad portfolio beats the baselines out of sample.

### 6.6 — Analyze the Robustness
- **Objective:** Show how stable the out-of-sample result is to randomness and to the cost assumption.
- **Approach:** Repeat the full walk-forward pipeline for **five random seeds** and for **three cost levels — 0, 10, and 25 basis points**. For each cost level, report the **mean and standard deviation across seeds of the out-of-sample Sharpe ratio** (and, optionally, turnover). Present as a small table and/or an error-bar chart over cost level.
- **Key decisions:** Keep everything else fixed across runs (basket, window, budget, selection rule) so only seed and cost vary. Seed handling per §4.7 (seed set before the parametrization's first use). Whether to also vary the budget as a sensitivity (optional).
- **Outputs:** A robustness table/figure: mean ± std out-of-sample Sharpe for {0, 10, 25} bps across five seeds.
- **Acceptance criteria:** Results are reproducible per seed; the write-up interprets both the seed spread (optimizer stability) and the cost sensitivity (how much the edge depends on frictions).

### 6.7 — Bonus (Optional)
- **CVaR objective:** Replace volatility with the **95% Conditional Value-at-Risk** of portfolio returns as the risk objective; recompute the front and compare it to the volatility-based front (does optimizing tail risk change the chosen portfolios?).
- **Third objective:** Add **maximum drawdown** as a *third* loss, making the front a 3-D Pareto surface; discuss how `pareto_front()` behaves with three objectives and how to visualize/select from it.
- **Acceptance criteria:** The bonus is clearly delimited from the required run; the comparison (CVaR-vs-vol fronts, or the effect of the third objective) is interpreted, not just plotted.

---

## 7. Methodology & Key Design Decisions (Consolidated)

| Decision | Choice | Rationale |
|---|---|---|
| Universe & window | 9 ETFs, daily, 2006→ | Cross-asset basket with rich covariance; long history supports many walk-forward folds |
| Validation protocol | Walk-forward, 5y train / 1y test, rolled yearly | Honest out-of-sample test; no look-ahead — every estimate uses only its fold's training window |
| Weight representation | `ng.p.Array` → simplex (softmax / renormalize) | Long-only, fully invested weights |
| Weight cap | ≤ 40% via `register_cheap_constraint` (float ≤ 0 when satisfied) | Spec-required diversification constraint; also the "constraints" teaching point for the API notebook |
| Rebalancing | `ng.p.Choice` of frequency (e.g. monthly/quarterly/annual) | A discrete decision variable the optimizer trades off against turnover cost |
| Objectives | Minimize `[volatility, −(net return)]`, net = gross − 10 bps · turnover | Cost-aware mean-variance; sign-flip turns return maximization into minimization |
| Return/covariance | Annualized (×252) | Interpretable, standard finance convention |
| Optimizers | **`NGOpt` vs `RandomSearch`**, same budget | Spec-required; tests whether the meta-optimizer beats random search at equal compute |
| Multi-objective read-out | `ask`/`tell` loop → `optimizer.pareto_front()` | Explicit loop per spec; front of `Parameter` objects with `.value`/`.losses` |
| Front → one portfolio | Highest **training** Sharpe, applied to next year | Fixed, reproducible selection rule for the out-of-sample test |
| Baselines | Equal-weight + closed-form minimum-variance | Naive comparators; reported with OOS Sharpe and turnover |
| Risk-free rate (Sharpe) | Average T-bill over the training window | A zero rate misranks portfolios over a rate-moving window |
| Robustness | 5 seeds × {0, 10, 25} bps | Separates optimizer noise from cost sensitivity |

---

## 8. Documentation & the 60-Minute Tutorial

The tutorial *is* a graded deliverable, split across two notebooks plus a README. Aim for a reader who knows Python/ML but not Nevergrad to follow it end-to-end in ~60 minutes. Follow the tutorial workflow in `tutorials/README.gp.md` and the quality bar in `tutorials/tutorials_checklist.md`.

- **`nevergrad.API.ipynb` (teach the tool):** short, self-contained cells demonstrating Nevergrad's building blocks in isolation on a *toy* function — parametrization (`ng.p.Array`, `ng.p.Choice`, composed in an `ng.p.Dict`; construct optimizers with `parametrization=`), choosing an optimizer (**`NGOpt` vs `RandomSearch`**), single- vs. multi-objective losses, the **`ask`/`tell` loop**, **constraints via `register_cheap_constraint`** (float ≤ 0 when satisfied), the optional reference point via `optimizer.tell(ng.p.MultiobjectiveReference(), [upper_bounds])`, and **`optimizer.pareto_front()`** (returns `Parameter` objects with `.value` and `.losses`). Each concept gets a one- or two-sentence explanation before the cell. This notebook's coverage maps exactly to the spec's Milestone 2.
- **`nevergrad.example.ipynb` (apply the tool):** the cost-aware portfolio project as a narrative — data + walk-forward → baselines → objectives → Nevergrad fronts (NGOpt vs RandomSearch) → out-of-sample evaluation → robustness → bonus — with markdown between cells explaining *why*, not just *what*. Include the spec-required **brief comparison with Optuna and `tutorials/Ax_Multi_Objective_Optimization/` from the hyperparameter-search point of view** (how Nevergrad's `ask`/`tell` and multi-objective `pareto_front` relate to Optuna studies and Ax's Bayesian multi-objective loop).
- **`nevergrad_utils.py`:** reusable, documented functions (PEP 8, docstrings) so notebooks stay readable and logic isn't duplicated. Pair each notebook to a `.py` with `jupytext` (`jupytext.py --action pair`), lint with `linters2/lint_cc.py`, and run end-to-end headless via `./docker_cmd.sh python .../nevergrad.example.py`.
- **Use the course Claude skills and document how you used them** (spec requirement): the `.claude/skills/notebook.*` skills (`notebook.create_api_intro`, `notebook.create_outline`, `notebook.implement_outline`) to scaffold the notebooks, and the `.claude/skills/tutorial_in_60_mins.*` skills (`.rules.md`, `.create`, `.format`) to build and format the tutorial. Note in the README which skills produced which artifacts.
- **Quality principles (`tutorials_checklist.md`):** avoid "AI slop"; write in a professional, peer-level tone; state each thing once in the right place; every sentence and example must earn its place. Include test coverage patterned on `class_project/project_template/test/test_docker_all.py`.
- **README / project documentation must include:** setup & build steps (Docker) — including the **base image chosen and why** and **any Dockerfile modifications** (per `docker_scripts.README.md`, "Document customizations in your project README") — how to run the notebooks, an **API description** of the Nevergrad features used, the **architectural/design decisions** (from §7) with brief justification, and the results narrative (§14).

---

## 9. Timeline & Milestones (to Dec 7, 2026)

Expected effort is **~6–8 full days (~40 hours)**, spread across the semester with at least one intermediate checkpoint PR and a final submission. Dates below work back from the **Monday, December 7, 2026** deadline (US Thanksgiving falls Nov 26, so that week is intentionally light).

The spec defines three high-level milestones — **M1** (container + data + baselines), **M2** (API notebook), **M3** (example notebook end-to-end). The weekly plan below is finer-grained for personal pacing but hits those three as checkpoints. The highest-leverage, easy-to-underweight items are the **out-of-sample evaluation and robustness sweep** (§6.5–§6.6) and making the notebooks read as a genuine **60-minute tutorial** (§8) — together these move the Depth & understanding, Complexity, and Documentation rubric items (5+ pts each), which raw code volume does not.

| Week | Dates | Focus | Checkpoint / exit criterion |
|---|---|---|---|
| 1 | Sep 28 – Oct 4 | Prereqs (fork/star/Docker/issue), create issue with project tag, branch, copy template into project dir | Issue + branch created; template copied to correct path |
| 2 | Oct 5 – Oct 11 | Choose base image, `requirements.txt`, build container, launch Jupyter; rename notebooks to `nevergrad.*`; open **draft PR** | `docker_build.sh` succeeds; Jupyter on :8888; PR open with reviewers |
| 3 | Oct 12 – Oct 18 | Task 6.1 (data + walk-forward folds) and 6.2 (baselines, OOS Sharpe + turnover) — **spec Milestone 1** | Baseline OOS Sharpe/turnover table populated for each test year |
| 4–5 | Oct 19 – Nov 1 | Task 6.3 (cost-aware losses + 40% constraint) and 6.4 (NGOpt & RandomSearch fronts); finish `nevergrad.API.ipynb` — **spec Milestone 2** | Converged, non-dominated fronts for both optimizers; API notebook complete |
| — | ~Nov 2 | **Checkpoint PR #1** (data + baselines + working fronts + API notebook); address review; continue on branch `..._1` **without waiting** if review is slow | Intermediate review passed (or next branch already underway) |
| 6 | Nov 2 – Nov 8 | Task 6.5 (out-of-sample evaluation vs baselines) | Per-fold OOS metric table + cumulative-return figure |
| 7 | Nov 9 – Nov 15 | Task 6.6 (robustness: 5 seeds × {0,10,25} bps) | Robustness table/figure complete |
| 8 | Nov 16 – Nov 22 | Bonus 6.7 (CVaR and/or 3rd objective); begin documentation; `nevergrad.example.ipynb` running end-to-end — **spec Milestone 3** | Example notebook runs top-to-bottom; bonus delimited |
| 9 | Nov 23 – Nov 29 | Documentation + tutorial polish + results narrative (light week: Thanksgiving Nov 26) | README complete; notebooks read as a 60-min tutorial |
| Final | Nov 30 – Dec 4 | **Record video (10–20 min)**; final review; final PR; upload video | Everything runs top-to-bottom; PR clean |
| Buffer | Dec 5 – Dec 6 | Weekend buffer: address any last-minute review comments; re-record if needed | — |
| **Due** | **Mon Dec 7** | **Final submission complete** | Final PR submitted + video in Google Drive |

**Milestone checklist:**
- [ ] Repos forked/starred; Docker installed; issue assigned
- [ ] Project tag, issue, branch created; template in correct path
- [ ] Container builds; Jupyter runs; draft PR open with reviewers
- [ ] Data clean; walk-forward folds non-leaking; **baseline OOS Sharpe + turnover table** (spec M1)
- [ ] Cost-aware losses + 40% constraint; NGOpt & RandomSearch fronts converged; **API notebook complete** (spec M2)
- [ ] **Checkpoint PR #1 reviewed**
- [ ] Out-of-sample evaluation vs baselines (Sharpe, max drawdown, turnover, cumulative net return)
- [ ] Robustness sweep (5 seeds × {0,10,25} bps) with mean/std OOS Sharpe
- [ ] Bonus (CVaR and/or third objective)
- [ ] **`nevergrad.example.ipynb` runs end-to-end** (spec M3)
- [ ] Documentation/README complete; results narrative written; Optuna/Ax comparison included
- [ ] Video recorded and uploaded
- [ ] Final PR submitted by **Dec 7**

---

## 10. Video Recording (Required Deliverable)

A **10–20 minute** video, uploaded to the course Google Drive folder, presenting the project professionally. Required structure (from `class_project/README.md`):

1. **Introduction** — name, UID, tool + difficulty, project title.
2. **File showcase** — show all files in the PR and confirm naming conventions.
3. **Docker execution** — build/run the image and show the success message (if Docker gave trouble, explain the issue and workaround).
4. **Open Jupyter** — (steps 1–4 should take ~1–2 minutes total).
5. **Full walkthrough** — run every required cell with clear verbal explanation of what each does; demonstrate correctness. Spend the majority of the time here.
6. **Discuss results** — interpret outputs (fronts, out-of-sample metrics vs baselines, robustness) and explain how Nevergrad addressed the cost-aware allocation problem.
7. **Documentation review** — show how docs are organized and how a non-technical reader could follow the project.

*Plan to record after the notebooks and docs are frozen (Final week), and rehearse once — a clean 12–15 minute run usually beats an unrehearsed 20.*

---

## 11. Risks, Pitfalls & Mitigations

| Risk / Pitfall | Impact | Mitigation |
|---|---|---|
| Look-ahead leakage (test-year data in `μ`/`Σ`/optimization) | Inflated, invalid out-of-sample results | Walk-forward folds where training strictly precedes test; feed each estimate only its fold's training slice (§6.1) |
| Turnover-cost sign/accounting error | Optimizer rewarded for the wrong thing | Net return = gross − `cost_bps · turnover`; unit-test turnover on a known weight change; confirm losses are `[volatility, −net_return]` (§6.3) |
| Infeasible or ignored 40% cap | Over-concentrated portfolios; spec constraint missed | Enforce via `register_cheap_constraint` returning a float ≤ 0 when satisfied; assert no weight > 0.40 on the front (§6.3) |
| Treating the front as one "best" portfolio | Misreads multi-objective output | Use `pareto_front()`; select by the stated training-Sharpe rule and discuss the front, not a single point (§6.4–§6.5) |
| NGOpt not beating RandomSearch at equal budget | Weak or surprising result | Keep budgets identical; if NGOpt doesn't win, *discuss why* (budget too small, noisy objective) rather than hide it (§6.4) |
| Sparse Pareto front | Weak frontier | Increase budget; set/tighten the loss reference point (§6.4) |
| Nevergrad API drift across versions | Notebook built on stale method names | Pinned to `nevergrad==1.0.12`; `ng.p.Dict`/`Array`/`Choice`, `ask`/`tell`, `register_cheap_constraint`, `optimizer.pareto_front()` confirmed against 1.0.12 (§4.4). Upstream marks multi-objective unstable — re-verify only if the pin changes |
| Copying a stale Nevergrad example (`instrumentation=`, wrong constraint sign) | Deprecated/incorrect code that misleads readers | Use `parametrization=` and a constraint that is ≤ 0 when satisfied, per §4.4 |
| Assuming an `Adj Close` column exists | `KeyError` / wrong data | With `auto_adjust=True` (now the default) there is **no** `Adj Close` — the adjusted series *is* `Close`; pass it explicitly and use `Close` (§6.1) |
| Multi-ticker `download()` returns MultiIndex columns | Flat-column code breaks | Select the `Close` level, or pass `multi_level_index=False`; confirmed against yfinance 1.7.0 (§6.1) |
| Yahoo blocks requests | No data on run/video day | Cache locally and keep the Ken French 10-Industry returns fallback wired and documented (§4.8) |
| Zero risk-free rate over a rate-moving window | Max-Sharpe selection misranked | Use the average training-window T-bill rate; state it and reuse it (§6.2) |
| In-sample optimism | Overstated performance | The walk-forward OOS protocol *is* the mitigation; report the in-sample vs OOS gap honestly (§6.5) |
| Leaving Docker/PR to the end | Lost points on two rubric items | Build container and open the draft PR in Week 2; commit regularly |
| One giant end-of-term commit | Poor PR score; no review feedback | Commit incrementally; hit the checkpoint PR |
| Reviewer latency | Incremental-branch plan stalls | Open the draft PR early; ping reviewers; keep working the next branch regardless (§5.6) |
| Video rushed at the last minute | Lost presentation points | Freeze notebooks first; rehearse once |

---

## 12. Validation & Definition of Done

Complete when:
- The notebooks/scripts run top-to-bottom in the container with no manual intervention and fixed seeds.
- **The Docker image builds and runs cleanly** following the documented steps.
- Walk-forward folds are non-leaking; both **baselines** and both **Nevergrad optimizers** (NGOpt, RandomSearch) produce results per fold.
- The Nevergrad Pareto fronts are non-dominated and respect the 40% cap; the selected portfolios are evaluated **out of sample** (Sharpe, maximum drawdown, turnover, cumulative net return) against the baselines (§6.5).
- The **robustness sweep** (5 seeds × {0, 10, 25} bps) reports mean/std out-of-sample Sharpe (§6.6).
- All six required tasks (+ chosen bonus) produce clearly labeled outputs and figures.
- Documentation covers setup, usage, API description, and design decisions, and includes the Optuna/Ax comparison; code is PEP 8 with docstrings and has test coverage.
- The **PR is clean** (scoped commits, meaningful messages, linked issue, reviewers added).
- The **video** is recorded and uploaded.
- Assumptions and limitations are stated in the report.

---

## 13. Grading Rubric & Alignment

### Official rubric (from `class_project/README.md`)

| Item | Points | What it rewards |
|---|---|---|
| All deliverables delivered | 10 | Complete, standard-structured, runnable by others without extra files |
| Working Docker | 5 | Builds without errors; runs as expected; ports/deps correct |
| Documentation quality | 5 | Clear setup/run/understand; well-written; required sections present |
| Actual project complexity | 5 | Depth beyond the base scope; edge cases; creative problem-solving |
| Code quality | 5 | Clean, modular, documented; consistent style (PEP 8) |
| PR quality | 5 | Organized PR; good commit messages; linked issue; no clutter |
| Depth and understanding | 5 | Justified design trade-offs; can defend the implementation |
| Late submission | −5 | Penalty if late without approved extension |
| Incomplete work | −5 | Penalty for missing/broken major parts |

### How this plan targets each item
- **Deliverables (10):** §3 deliverable list + §9 checklist.
- **Working Docker (5):** §4 setup; build/PR opened by Week 2; §12 gate.
- **Documentation (5):** §8 (README: setup, usage, API, architecture, Optuna/Ax comparison) + §14 narrative.
- **Complexity (5):** walk-forward out-of-sample protocol, NGOpt-vs-RandomSearch comparison, the robustness sweep, and the bonus (CVaR / third objective).
- **Code quality (5):** `nevergrad_utils.py` with docstrings, PEP 8, no duplication; `jupytext`-paired `.py` linted with `lint_cc.py`; test coverage.
- **PR quality (5):** §5.6 workflow — incremental commits, checkpoint PR, reviewers.
- **Depth & understanding (5):** §2 framing + §7 justified decisions, defended in the §10 video.
- **Avoid penalties:** finish by **Dec 7** (§9) with no missing parts (§12).

### Assignment-requirement traceability

| Spec task (Project 1) | Covered by |
|---|---|
| Acquire the Data (+ walk-forward windows) | §6.1 |
| Define the Baselines | §6.2 |
| Define the Objectives (net-of-cost return, volatility, 40% cap) | §6.3, §7 |
| Optimize with Nevergrad (`ng.p.Dict`/`Choice`, `ask`/`tell`, NGOpt & RandomSearch, `pareto_front`) | §6.4 |
| Evaluate Out of Sample | §6.5 |
| Analyze the Robustness | §6.6 |
| Bonus: CVaR / max-drawdown third objective | §6.7 |
| Spec Milestone 1 (container + data + baselines) | §9 Week 3 |
| Spec Milestone 2 (API notebook) | §9 Weeks 4–5, §8 |
| Spec Milestone 3 (example notebook end-to-end) | §9 Week 8, §8 |

---

## 14. Results Narrative & Report Structure

The narrative lives in the README/docs (a standalone report is not separately required, but the documentation rubric expects a clear write-up). Suggested structure:

1. **Introduction** — problem, objectives, why cost-aware multi-objective optimization applies, and the tutorial goal.
2. **Data** — the nine ETFs, 2006→ window, source, cleaning, walk-forward folds, summary statistics, fallback dataset.
3. **Problem Formulation** — objectives (net-of-cost return, volatility), the 40% cap, the rebalancing-frequency choice, weight representation.
4. **Method** — Nevergrad setup (`ng.p.Dict`/`Array`/`Choice`), NGOpt vs RandomSearch, budget, reference point, selection rule.
5. **Results** — Pareto fronts (NGOpt vs RandomSearch), headline portfolios, baseline comparison, tables/figures.
6. **Out-of-Sample Evaluation** — cumulative net return, Sharpe, max drawdown, turnover vs equal-weight and minimum-variance.
7. **Robustness** — seeds × cost levels; mean/std out-of-sample Sharpe; interpretation.
8. **Bonus** — CVaR-vs-volatility fronts and/or the third-objective experiment.
9. **Discussion & Limitations** — in-sample vs out-of-sample gap, estimation risk, cost-model simplifications, scope exclusions.
10. **Comparison with Optuna / Ax** — Nevergrad vs these tools from a hyperparameter-search point of view.
11. **Conclusion** — what was demonstrated (a black-box multi-objective optimizer building cost-aware portfolios, honestly tested out of sample).
12. **References & Appendix** — sources, reproducibility notes (`version.log`; cached-data provenance — fetch date + checksum).

---

## 15. References & Resources

- **Project 1 spec of record** — `class_project/project_descriptions/MSML610/nevergrad_Project_Description.md` (Project 1 Fall2026 + the shared Tutorial section: skills, reference tutorials, Optuna/Ax comparison, deliverables).
- **Class project guidelines** — `class_project/README.md` (project types, *Contribution to the Repo* workflow, tag/branch/PR conventions, video requirements, grading rubric).
- **Tutorial workflow** — `tutorials/README.gp.md` (`create_project.py` skeleton, `jupytext` pairing, `linters2/lint_cc.py`, end-to-end run via `docker_cmd.sh`, and the notebook/tutorial skills).
- **Tutorial quality bar** — `tutorials/tutorials_checklist.md` (avoid AI slop, professional tone, no redundancy, study `project_template` and `test/test_docker_all.py`).
- **Course Claude skills** — `.claude/skills/tutorial_in_60_mins.*` (`.rules.md`, `.create`, `.format`) and `.claude/skills/notebook.*` (`create_api_intro`, `create_outline`, `implement_outline`); document how each was used.
- **Reference tutorials/projects** — `tutorials/Autogen` (canonical `*.API.ipynb` / `*.example.ipynb` / `*_utils.py` layout), `tutorials/Ax_Multi_Objective_Optimization/` (closest multi-objective tutorial), the Fall2025 Optuna project (`class_project/msml610/Fall2025/projects/UmdTask60_Fall2025_Optuna_Customer_Segmentation_Using_Clustering/`), and `msml610/tutorials/L03_knowledge_representation/` (structure/style to emulate). Also the `project_template` and `docker_scripts.README.md`.
- **`gpsaggese.github.io`** and **`helpers`** repositories (fork/star; optional intern dev-setup guide).
- **Nevergrad 1.0.12** — multi-objective docs: parametrization via `ng.p.Dict`/`Array`/`Choice` with `parametrization=`, the `ask`/`tell` loop, `register_cheap_constraint` (float ≤ 0 when satisfied), optional `MultiobjectiveReference` via `tell`, and `optimizer.pareto_front()`. API usage confirmed against this release.
- **yfinance 1.7.0** — historical market data retrieval (`download(..., auto_adjust=True)`; MultiIndex columns via `multi_level_index`); confirmed current as of Oct 2026.
- **Ken French Data Library** — "10 Industry Portfolios (Daily)", static fallback dataset.
- **Markowitz mean-variance / Modern Portfolio Theory** — mean-variance, Sharpe-ratio, and minimum-variance background; CVaR and maximum-drawdown risk measures for the bonus.

---

*End of plan. This document defines what will be built, how it is delivered (Docker + PR + video), and how success is judged; implementation (code) is a separate step to be undertaken against this plan.*
