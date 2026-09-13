# Notebook Rules Fix Plan

Source: audit of all `msml610/tutorials/**/*.ipynb` against
`.claude/skills/notebook.rules.md` (2026-09-13).

## How to use this plan

- Edit the paired `.py` file (jupytext `percent` format), not the `.ipynb`
  directly. After each edit: `uvx jupytext --sync <file>.py`.
- Work top to bottom per notebook: structure fixes first (cheap, mechanical),
  content-paradigm rewrite next (needs judgment), formatting sweep last.
- After finishing a notebook, verify with `docker_cmd.sh "python
  <path>.py"` (see notebook.rules.md `## Testing Notebook`).
- Do not commit without asking first.

## Fix-action legend (tag used in each checklist item)

| Tag | Action |
|-----|--------|
| S1 | Skill `notebook.lint_numbered_cells`: rebuild `# Part N:` / `## Cell <P>.<ID>:` headers, sequential numbering, sync `cellN_*()` names |
| S2 | Skill `notebook.split_header_cells`: one header per markdown cell |
| S3 | Skill `notebook.split_cells`: one logical task per code cell |
| S4 | Skill `notebook.refactor_to_utils`: move inline viz/widget code to `*_utils.py`; create the file if missing |
| S5 | Skill `notebook.format_interactive_cells`, then hand-upgrade its output: the skill still targets the older Goal/Plots/Key-observations triplet, so after running it, rewrite to the required `**Goal**` -> `**Implementation**: \`fn(...)\`` -> `**Usage**` (`- Inputs` / `- Panels`) -> `**Guided usage**` structure per `## Visualization Cell Triplet Details` |
| S6 | Skill `notebook.format_md_cells` for bullet/wrap, plus manual check for sentence case, ASCII-only, emdash-to-colon (skill doesn't guarantee these) |
| S7 | Skill `notebook.fix_packages`: remove inline `!pip install`, reconcile `requirements.txt`/`Dockerfile` |
| S8 | Skill `notebook.delete_dead_code`: drop unused functions, leftover commented/debug cells |
| S9 | Skill `git.move`: rename a `*_utils.py` file and update its notebook's import when naming breaks convention |
| manual | No skill covers it; fix by hand against the quoted rule |

Suggested order across the whole repo: S1 -> S7 -> S8 -> S9 -> S4 -> S3 ->
S5 -> S6/manual. S5 is the most labor-intensive step (0 of 34 notebooks
currently pass it) - budget the most time there.

---

## L03_knowledge_representation (5 notebooks) - lightest lift

### L03_01_entailment_implication_inference
- [ ] [S6] Cell headers to sentence case (e.g. "Cell 1.1: Possible worlds, models, and satisfaction")
- [ ] manual: add `%load_ext autoreload` / `%autoreload 2` to setup cell (missing, unlike sibling files)
- [ ] [S7] remove commented `# !pip install -q sympy==1.14.0`; confirm `sympy` in `requirements.txt`

### L03_02_wumpus_world
- [ ] [S1] fix Part 3 numbering gap (Cell 3.2 with no Cell 3.1)
- [ ] [S6] wrap the ~401-char unbulleted paragraph (~line 203) into nested bullets; replace raw `α` with `$\alpha$`
- [ ] [S5] move `**Goal**` to directly follow the cell title in Cell 2.1/2.2 (prose/lists currently precede it)
- [ ] [S6] sentence-case Title-case headers (e.g. "Cell 1.1: The Wumpus World Grid and the Knowledge Base")
- [ ] [S6] bullet-ify stray sentence in "Example of Reasoning" section
- [ ] [S7] remove commented pip-install line

### L03_03_rule_based_expert_systems
- [ ] [S1] group the pre-Part-1 cells (Winston-rules print, "The Rule Base") under a `# Part N:` header
- [ ] manual: move inline `print()` feature/species catalog to a `utils.show_...()` helper, consistent with the very next cell's pattern
- [ ] [S7] remove commented pip-install line

### L03_04_ontology_reasoning
- [ ] [S7] remove commented pip-install line (only issue found)

### L03_06_logic_solvers
- [ ] [S1] group "The Three Engines" comparison table under a `# Part N:` header
- [ ] [S6] sentence-case cell/section headers
- [ ] [S4]/manual: replace hand-formatted `"%-26s -> %s"` print table (~lines 152-156) with a pandas `DataFrame` + `display()`; move to a `utils.show_...()` helper
- [ ] [S6] wrap long comment lines (~144, 148, 169) to 85 chars, add trailing periods
- [ ] [S7] remove commented pip-install line

---

## L05_statistical_learning (6 notebooks)

All 6 need both structural fixes below - listed once, apply to every file:
- [ ] [S1] (all 6) add real `# Part N:` grouping header with `<PART>.<ID>` numbering (currently missing, or mislabeled `# Cell N:`, or mixed bare/numbered)
- [ ] [S5] (all 6) rebuild markdown to Goal -> Implementation -> Usage(Inputs/Panels) -> Guided usage (currently Goal+Plots/Key-observations, or nothing)

Per-file extras:

### L05_01_01_hoeffding_inequality
- [ ] [S6] wrap line ~86 (minor, only outlier)

### L05_01_02_bin_analogy_ml
- [ ] [S1] unify inconsistent numbering (bare `Cell 1/2/3` mixed with `2.1`/`3.1`)
- [ ] [S6] wrap 25 lines over 85 chars

### L05_01_03_vc_dimension
- [ ] [S1] convert all 6 level-1 `# Cell N` headers to the Part/Cell scheme
- [ ] [S6] wrap 13 lines over 85 chars (up to 169 chars)

### L05_01_04_growth_function
- [ ] [S1] convert all 10 level-1 `# Cell N` headers to the Part/Cell scheme
- [ ] manual: replace hand-built ASCII tables (~lines 116, 246, 353-365) with pandas + `display()`
- [ ] [S3] split Cell 3 (growth-curve calc + breakpoint search), Cell 9 (2 examples: positive rays, perceptron), Cell 10 (2 edge cases: N=1, collinear points)
- [ ] [S6] wrap 4 lines over 85 chars
- [ ] note: API/library-call cells correctly stay inline per "Library Calls vs. Visualization" carve-out - keep as is

### L05_02_01_bias_variance
- [ ] [S1] unify mixed level-1/level-2 headers, add missing Part header
- [ ] [S6] wrap 49 lines over 85 chars (worst in repo, up to 195 chars)

### L05_02_02_overfitting
- [ ] [S1] fix both level-1 `# Cell 1/2` headers, add Part header
- [ ] [S6] wrap 22 lines over 85 chars (up to 198 chars)
- [ ] [S6] replace ALL-CAPS emphasis ("HIGH BIAS", "LOW VARIANCE", etc., ~lines 127-133) with bold/italic

---

## L06_bayesian_networks (2 notebooks)

### L06_01_exact_inference
- [ ] [S1] remove/retitle stray level-1 "# Exact Inference" header (not a `# Part N:` header)
- [ ] [S5] rebuild all 4 Part's worth of cells from Goal/Plots/Parameters/Key-observations to Goal/Implementation/Usage/Guided-usage; add `hintros.print_obj_info()`; use bold+backtick for Panel labels
- [ ] [S6] wrap line ~77

### L06_02_approximate_inference
- [ ] [S1] remove duplicate untitled level-1 header (~line 34)
- [ ] [S5] rebuild all 9 cells to the required triplet structure
- [ ] [S6] sentence-case all headers (currently Title Case); resolve leftover `# TODO(ai_gp)` cell-split comment (~line 94) and wrap that line
- [ ] manual: remove redundant `ipywidgets.Label()` before controls (all 9 widgets)
- [ ] manual: move the seed widget to first position in all 9 widgets (currently last)

---

## L07_prob_programming (6 notebooks) - biggest lift in the repo

### L07_01_bayesian_coin (has utils file)
- [ ] [S1] add Part/Cell numbering (none exists; bare topic headers)
- [ ] [S5] add full Goal/Implementation/Usage/Guided-usage triplets (currently none)
- [ ] [S6] wrap line ~111 (~175 chars); replace unicode `alpha`/`beta`/`theta` in widget labels with ASCII
- [ ] manual: label bare `print(data)` calls (~lines 93, 138, 197) with the variable name
- [ ] [S3] split cells that combine def+docstring+print+slider+interact
- [ ] [S4] move `sample_bernoulli`/`sample_binomial`/`sample_beta` + their slider/interact setup from the notebook into utils
- [ ] manual: rebuild `beta_prior_interactive` to 1xN layout + Comments panel via `add_fitted_text_box()`; move seed widget first; switch matplotlib to seaborn
- [ ] [S7]/[S8] remove 5 install/labextension cells (~lines 26, 27, 30, 33, 36)

### L07_02_probabilistic_programming (no utils file - fix pairing first)
- [ ] [S4] create `L07_02_probabilistic_programming_utils.py`; move `iqr()`, `plot_models()`, `posterior_grid()`, `metropolis()` and other inline helpers into it
- [ ] [S1] add Part/Cell headers (none exist)
- [ ] [S5] add Goal/Implementation/Usage/Guided-usage triplets throughout
- [ ] [S6] sentence-case title (~line 17); replace unicode `pi` (~line 455) with ASCII
- [ ] [S3] split multi-step cells (load+transform+plot+save combined)
- [ ] manual: wrap bare trailing-expression tables in `print()`/`display()`; use `_ = statement` to suppress noisy output
- [ ] [S8] delete (not just comment) the labextension/install cell (~lines 23, 26-27)

### L07_02_robust_modeling (no utils file)
- [ ] [S4] create utils file; move the Student-t PDF sweep (~lines 120-142) into it
- [ ] [S1] add Part/Cell headers
- [ ] [S5] add triplets
- [ ] [S6] sentence-case 2 Title-case headers (~17, 65); wrap 4 long lines
- [ ] manual: label bare `print(len(data), data)` (~line 72); wrap `az.summary()` bare expressions in `display()`; add trailing periods to comments
- [ ] [S3] split the load+print+plot+process cell (~71-79)
- [ ] [S8] delete labextension (~26-27) + 2 install cells (~30, 33)

### L07_03_hierarchical_models (no utils file)
- [ ] [S4] create utils file; move the forest-plot + manual-vlines comparison (~151-167) into it
- [ ] [S1] add Part/Cell headers
- [ ] [S5] add triplets
- [ ] [S6] sentence-case title (~line 17, inconsistent with line 102 which is already sentence case); wrap line ~156
- [ ] [S3] split 2 multi-step cells (~76-87, 110-119)
- [ ] manual: label bare `print(tip[:10])` (~line 78); suppress `sns.boxplot(...)` (~line 73) via `_ = `
- [ ] [S8] delete labextension (~26-27) + 2 install cells (~30, 33)

### L07_04_generalized_linear_models (no utils file)
- [ ] [S4] create utils file; move `plot_data_and_model()` (~172-206) and `scatter_plot()` (~525-541) into it; fix `scatter_plot()`'s 2x2 grid to 1xN
- [ ] [S1] add Part/Cell headers
- [ ] [S5] add triplets
- [ ] [S6] sentence-case title (~line 17)
- [ ] manual: label bare `print(x_plot.shape)` / `print(mean_line.shape)` (~209, 212); add comments to prior-definition cells; switch matplotlib to seaborn/pandas where used for plotting
- [ ] [S8] delete labextension (~25-27) + 2 install cells (~29-30, 32-33)

### L07_05_evaluating_models (no utils file)
- [ ] [S4] create utils file; move `plot_models()`, `iqr()`, and the model-comparison loops into it
- [ ] [S1] add Part/Cell headers (drop non-spec `###` level)
- [ ] [S5] add triplets
- [ ] [S6] sentence-case title (~line 17)
- [ ] [S3] split multi-step cells (load+transform+plot+save; 2 models defined+sampled together; postprocess+print+plot)
- [ ] manual: wrap 4 bare trailing-expression tables (`waic_l`, `waic_q`, `loo_l`, `loo_q`, ~288-300) in labeled `print()`/`display()`; change the vertical 2x1 subplot (~line 182) to horizontal 1xN
- [ ] [S8] delete labextension cell (~23-27)

---

## L08_causal_inference (5 notebooks) - worst header compliance

### L08_04_01_causal_inference
- [ ] [S1] replace level-1 `# Cell 1/2` with `# Part N:` + `## Cell N.M:`
- [ ] [S5] replace `**Purpose**`/`**What it shows**`/`**Key insight**` with Goal/Implementation/Usage/Guided-usage
- [ ] [S6] wrap lines ~162, 178, 179, 215; sentence-case all Title-case headers
- [ ] manual: label bare `print(data.shape)` (~71), `print(df.shape)` (~250), `print(z)` (~388); wrap bare trailing-expression tables (~72, 251, 232, 240, 390, 474) in `display()`
- [ ] [S3]/[S4] split and move the inline boxplot/scatterplot block (~266-292) into utils; switch to seaborn where numpy/matplotlib is used directly
- [ ] [S8] delete the live (uncommented) `hmodule.install_module_if_not_present("dataframe_image", ...)` call (~49-52)

### L08_04_02_causal_inference
- [ ] [S1] replace level-1 `# Cell 1/2/3` with the Part/Cell scheme
- [ ] [S5] rename `**Plots**`/`**Parameters**`/`**Key observations**` to Implementation/Usage(Inputs/Panels)/Guided-usage; add `hintros.print_obj_info()`
- [ ] [S6] sentence-case 3 Title-case headers
- [ ] manual: add a variable-name label to the repeated `print(not dag.is_dconnected(...))` pattern (~113-147, 6x)

### L08_04_05_propensity_score
- [ ] [S1] add Part/Cell headers entirely (none exist)
- [ ] [S5] add full triplets for all 5 visualization calls (~105, 109, 112, 118, 186, 255)
- [ ] [S6] fix emdash (~line 71: "outcome-average..." -> "outcome: average..."); sentence-case "Stabilized Propensity Weights"
- [ ] manual: label bare `print(model.params["intervention"])` (~169); wrap bare `.summary()`/`.head()` trailing expressions (~102, 159, 177) in `display()`; remove duplicate `plot_engagement_vs_intervention(df)` call (~105 vs 109) and dead commented call (~115)
- [ ] [S9] rename utils import from `L08_04_05_causal_inference_utils` to `L08_04_05_propensity_score_utils` to match the modern naming convention

### L08_04_07_metalearners
- [ ] [S1] add Part/Cell headers (currently 5 bare level-1 dividers, no per-cell numbering)
- [ ] [S5] add triplets for the 4 metalearner sections (T/X/S-learner, Double ML/R-learner)
- [ ] manual: replace semicolon-suppression (`m1.fit(...);` ~113, `sns.lineplot(...);` ~212) with `_ = ...`; wrap bare `.head()`/bare `m0` trailing expressions in `display()`
- [ ] [S3] split the cell that computes effects + fits + plots together (~140-151)
- [ ] [S8] delete the live install call `hmodule.install_module_if_not_present(["lightgbm", "fklearn"], ...)` (~63-68)

### L08_04_08_difference_in_difference
- [ ] [S1] add Part/Cell headers (only a bare `# Load data`; "Canonical DiD" is a code comment, not a markdown header)
- [ ] [S6] sentence-case notebook title "Difference In Difference"; move dataset explanation (~84-88, currently code comments) into a markdown cell
- [ ] manual: wrap every bare trailing-expression result (~92-96, 106, 116, 119) in `print()`/`display()`; add trailing periods to comments (~110-111)
- [ ] [S8] delete the dead commented-out utils import (~line 34) and the leftover lightgbm/fklearn install block (~46-66) copy-pasted from another notebook

---

## L09_kalman_filter (5 notebooks) + L09_multi_armed_bandits (2 notebooks)

### L09_04_gh_filter
- [ ] [S1] rename level-1 `# Cell 1/2` to `# Part 1/2:`
- [ ] [S5] add Goal/Implementation/Usage/Guided-usage triplets to the 5 widget cells (1.4, 2.1, 2.5, 2.9)
- [ ] [S4] move inline `df.plot()`/`plt.savefig()` (cells 1.1, 2.7, 2.8) into utils
- [ ] manual: add 1xN layout + Comments panel (`add_fitted_text_box`) to the 4 widget functions in utils (currently single-axis, no Comments panel)

### L09_05_01_discrete_bayes_dog
- [ ] [S1] rename level-1 `# Cell 1/2` to the Part scheme
- [ ] [S5] convert existing "Left plot"/"Right plot" prose + "movements1/2" general facts into proper Usage(Panels)/Guided-usage bullets
- [ ] [S6] wrap 15 long lines (~107, 341, 378, 432); replace `±` (~line 285) with ASCII
- [ ] manual: label bare `print(belief)` (~line 70)
- [ ] [S4] move the inline `interact(show_prior, ...)` call + `show_prior()` callback (~345-374) into utils
- [ ] [S7] remove commented `!pip install --quiet filterpy` (~line 29)

### L09_05_02_univariate_kalman_filter
- [ ] [S1] rename level-1 `# Cell 1/2`; restore missing `Cell 2.X` numbering under "Bad Initial Estimate"/"Extreme amount of noise"/"Too much belief in the model"
- [ ] [S5] rename Goal/Plots/Parameters/Key-observations to Implementation/Usage/Guided-usage (3 cells); add `hintros.print_obj_info()`
- [ ] [S6] wrap lines ~290, 294, 406; replace `²` (~line 392) with `^2`
- [ ] manual: label bare `print(x)`, `print(z)` (x4), `print(zs)` (~49, 74, 143, 158, 169, 274); replace semicolon-suppression (~79, 150, 162, 173) with `_ = ...`
- [ ] manual: add 1xN layout + Comments panel to `cell2_interactive_dog_simulation()` in utils

### L09_05_03_multivariate_kalman_filter
- [ ] [S1] fully rebuild header scheme (worst of the 5: "Cell 1" reused for 2 different topics, later sections drop the Cell/Part prefix entirely)
- [ ] [S5] add triplets for the 2 widget cells (`cell_dog_tracking_interactive`, `cell_hidden_variable_comparison_interactive`)
- [ ] [S6] wrap lines ~67, 146
- [ ] manual: label bare `print(P2)` (~124); replace semicolon-suppression `plt.plot(...);` (~199) with `_ = ...`
- [ ] manual: add a Comments panel (`add_fitted_text_box`) to both interactive utils functions (currently none)
- [ ] [S7] remove commented `!pip install --quiet filterpy` (~line 29)

### L09_05_04_non_linear_kalman_filter
- [ ] [S1] add Part/Cell structure (currently only the notebook title + one incidental `### Unscented transform`)
- [ ] [S5] add Goal/Implementation/Usage/Guided-usage markdown throughout (currently none)
- [ ] [S4] move inline `plt.subplot`/`plt.scatter` blocks (~71-76, 121-125) into utils, matching the `time_ut.plot_*` calls already used elsewhere in the same file
- [ ] [S7] remove commented `!pip install --quiet filterpy` (~line 29)

### L09_03_01_multi_armed_bandits_sim_API (near-compliant API notebook)
- [ ] [S1] rename Part 6/7 sub-headers from `## Example N:`/`## Pattern N:` to `## Cell 6.N:`/`## Cell 7.N:`
- [ ] [S9] resolve shared-utils naming: rename `L09_03_multi_armed_bandits_sim.py`/`L09_03_multi_armed_bandits_utils.py` to a per-notebook `_utils.py`, or explicitly document the shared-utils exception (used by both L09_03_01 and L09_03_02)

### L09_03_02_multi_armed_bandits
- [ ] [S1] convert level-1 `# Cell N:` (no sub-numbering) to `# Part N:`/`## Cell N.M:`; fix numbering gap (Cell 3 -> Cell 5, missing Cell 4)
- [ ] [S5] convert `**Goal**` + italic `_Panel name_:` lines + `**Key observations**` into proper `**Usage**` (Inputs/Panels) + `**Guided usage**`; add `**Implementation**` bullets
- [ ] [S6] wrap lines ~86, 90, 96, 98
- [ ] [S8] delete commented `#!apt-get update && apt-get install -y git` (~line 43); remove stray `# TODO(ai_gp)` (~57) and `//`-style comment (~101)
- [ ] [S9] same shared-utils naming resolution as L09_03_01

---

## L10_causal_discovery (1 notebook) + L12_reinforcement_learning (2 notebooks)

### L10_2_causal_discovery (worst file in the whole repo)
- [ ] [S1] replace all 10 level-1 `# Cell N:` headers with `# Part N:`/`## Cell N.M:`
- [ ] [S5] add Goal/Implementation/Usage/Guided-usage triplets (currently Goal/Plots/Parameters/Key-observations only)
- [ ] [S6] wrap 22 long lines; replace unicode `-> <- _|_` for `→ ← ⊥`; convert inline comma-lists (~line 98) to bullets; sentence-case Title-case headers
- [ ] manual (in `_utils.py`): replace raw `ipywidgets.FloatSlider`/`IntSlider` with `htutori.build_widget_control()`/`build_log_widget_control()` (adds the required +/- buttons)
- [ ] manual: add a seed widget, placed first, replacing hardcoded `np.random.seed(42)` (4x)
- [ ] manual: add `add_fitted_text_box()` Comments panels (currently 0 uses)
- [ ] manual: replace post-plot ALL-CAPS `print("KEY INSIGHT: ...")` with an in-plot `ax.text()`/Comments panel
- [ ] manual: fix the 2x2 grid in `cell9_domain_knowledge` to 1xN

### L12_01_gridworld_4x3 (mostly compliant)
- [ ] [S1] remove the stray duplicate level-1 title "# Gridworld 4x3" (~line 30); rename `## Cell 0:` to fit `<PART>.<ID>`; fix out-of-order Part 2 numbering (`2.1a` -> `2.2` -> `2.1` -> `2.3`)
- [ ] [S5] add Implementation/Usage/Guided-usage (currently only Goal/Key-observations)
- [ ] [S6] wrap lines ~74, 222, 359, 363; replace 8 emdashes (~67-74) with `:`; replace unicode `π γ ∑` + ALL-CAPS "LINEAR"/"NONLINEAR" in a code comment (~319-333) with ASCII/normal case
- [ ] [S3] split the cell (~113-129) that combines a baseline Q-values example with a "favor right" scenario
- [ ] [S8] remove the leftover debug cell with GitHub source-line pointers (~356-363)

### L12_02_gridworld_4x3_gymnasium (cleanest file in the repo)
- [ ] [S1] remove the stray duplicate level-1 title "# Gridworld 4x3 Gymnasium" (~line 31)
- [ ] [S5] add Implementation/Usage/Guided-usage (currently only Goal/Key-observations)
- [ ] [S6] wrap 8 long lines (~94, 97, 125, 166, 201, 230, 254, 288)

---

## Cross-cutting items (do once, not per-notebook)

- [ ] Decide the shared-utils exception for `L09_03_01`/`L09_03_02` (see S9 items above) before touching either file, so both get the same answer
- [ ] After the S1 pass repo-wide, re-run `notebook.lint_numbered_cells`'s own verification checklist per file (sequential numbering, `cellN_*` name sync, jupytext sync)
- [ ] After the S5 pass repo-wide, spot-check a few files with `skill.check_against_rules` against `notebook.rules.md` to confirm the new triplet structure actually matches `## Visualization Cell Triplet Details` (the skill itself still describes the older pattern)
