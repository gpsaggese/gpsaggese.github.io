# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Entailment, Implication, and Inference: the Rain and Wet Ground World
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpsaggese/gpsaggese.github.io/blob/gp/msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.ipynb)
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gpsaggese/gpsaggese.github.io/gp?filepath=msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.ipynb)
#
# - This notebook stays on the lecture's own smallest examples, rain and wet
#   ground, and $x = 0$ implies $x \cdot y = 0$, to make the model-theoretic
#   definitions concrete without a larger running project
# - The pedagogical arc:
#   - Models, possible worlds, and satisfaction
#   - Entailment as model inclusion, verified by model checking
#   - The same definition applied to a non-Boolean world
#   - Implication vs entailment vs inference, three views of one example
#   - Soundness and completeness, seen by deliberately breaking a reasoner

# %% [markdown]
# ## Imports

# %%
import os
import sys

# Detected once, reused below and in the next cell: autoreload watches this
# repo's own files for live edits, which Colab/Binder don't have until the
# next cell clones/locates them, so skip it there.
ON_COLAB = "google.colab" in sys.modules
ON_BINDER = "BINDER_LAUNCH_HOST" in os.environ

if not (ON_COLAB or ON_BINDER):
    # `get_ipython()` is `None` when the paired .py runs as a plain script
    # (outside a notebook kernel); skip the magics rather than crash.
    from IPython.core.getipython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Neither Colab nor Binder hosts the whole repo, only this one notebook file,
# so `helpers` (source, not pip-installed) and the paired `_utils.py` file
# are missing unless we fetch them. Docker gets both for free (helpers_root
# on PYTHONPATH, cwd = this notebook's dir); this cell reproduces that.
# Plain Python (`subprocess`, not `!`/`get_ipython()`), so the paired .py
# script still runs standalone outside a notebook.
import subprocess

if ON_COLAB:
    # Colab starts empty: clone the repo, then point Python at it.
    REPO_DIR = "gpsaggese.github.io"
    BRANCH = "gp"
    NB_DIR = "msml610/tutorials/L03_knowledge_representation"

    if not os.path.exists(REPO_DIR):
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                BRANCH,
                f"https://github.com/gpsaggese/{REPO_DIR}.git",
            ],
            check=True,
        )
    sys.path.insert(0, os.path.abspath(f"{REPO_DIR}/helpers_root"))
    sys.path.insert(0, os.path.abspath(f"{REPO_DIR}/{NB_DIR}"))
    os.chdir(f"{REPO_DIR}/{NB_DIR}")
    subprocess.run(
        ["pip", "install", "-q", "-r", "requirements.txt"], check=True
    )
elif ON_BINDER:
    # Binder already clones the repo and builds requirements.txt into the
    # image; cwd is already this notebook's dir, only PYTHONPATH is missing.
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sys.path.insert(0, os.path.join(git_root, "helpers_root"))

# %%
import helpers.hnotebook as hnotebook

import L03_01_entailment_implication_inference_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# %% [markdown]
# # Part 1: Models and Satisfaction

# %% [markdown]
# ## Cell 1.1: Possible Worlds, Models, and Satisfaction
#
# **Goal**:
# - Ground "model" as one full true/false assignment to every variable, using
#   the $(Rain, WetGround)$ world
# - Introduce $M(\alpha)$, the set of models where a sentence $\alpha$ is
#   true
# - _Model table_: all 4 rows of the truth table over `Rain` and `WetGround`
# - _Comments_: chosen $\alpha$, its truth value in each named model, $|M(\alpha)|$

# %%
# Shade M(alpha) over the 4 named models of the rain world.
utils.cell1_1_models_and_satisfaction()

# %% [markdown]
# **Key observations**:
# - $M(Rain) = \{m_1, m_2\}$ regardless of `WetGround`: satisfaction depends
#   only on the variables a sentence mentions
# - "The model satisfies the sentence" reads backwards at first
#   - The model (the world) is what varies across rows
#   - The sentence's truth is read off each fixed row
# - $M(Rain \implies WetGround)$ excludes only $m_2$: implication is false in
#   exactly one of the 4 models

# %% [markdown]
# # Part 2: Entailment as Model Inclusion

# %% [markdown]
# ## Cell 2.1: Entailment as Model Inclusion, by Model Checking
#
# **Goal**:
# - Define $KB \models \alpha$ as $M(KB) \subseteq M(\alpha)$, and verify that
#   $KB = \{Rain, Rain \implies WetGround\}$ entails $WetGround$
# - Run the model-checking algorithm explicitly: enumerate every model, find
#   $M(KB)$, check $\alpha$ in each of those rows
# - _Model table_: the same 4-row table, with $M(KB)$ shaded blue and
#   $M(\alpha)$ outlined in dashed orange
# - _Comments_: which `KB` sentences are toggled on, the query $\alpha$,
#   and the entailment verdict

# %%
# Toggle KB sentences and alpha, and read off the entailment verdict.
utils.cell2_1_entailment_model_checking()

# %% [markdown]
# **Key observations**:
# - With both `KB` sentences on and $\alpha = WetGround$, every row of
#   $M(KB)$ falls inside $M(\alpha)$: no counterexample, so entailment holds
# - Turning off `Rain => WetGround` leaves `KB = {Rain}`, which does not
#   entail `WetGround`: $m_2 = (Rain=T, WetGround=F)$ satisfies `KB` and
#   violates `WetGround`
# - Entailment is a property of the whole shaded set: one counterexample row
#   is enough to break it, no matter how many rows agree

# %% [markdown]
# ## Cell 2.2: The Same Definition on a Non-Boolean World
#
# **Goal**:
# - Show that $M(KB) \subseteq M(\alpha)$ does not require Boolean
#   variables, by checking the lecture's "sitting table" example:
#   $\alpha$: "$x = 0$" entails $\beta$: "$x \cdot y = 0$", for any $y$
# - Reinforce that a model here is a pair $(x, y)$, not a truth assignment
# - _Model grid_: every integer pair $(x, y)$ in a small range as a
#   scatter grid, points where $\alpha$ holds shaded blue and points
#   where $\beta$ holds outlined in dashed orange
# - _Comments_: the current range, $|M(\alpha)|$, $|M(\beta)|$, and the
#   entailment verdict

# %%
# Check alpha |= beta over integer pairs instead of truth assignments.
utils.cell2_2_nonboolean_world()

# %% [markdown]
# **Key observations**:
# - Every blue point ($x = 0$) is also outlined ($x \cdot y = 0$), for any
#   $y$: exactly $M(\alpha) \subseteq M(\beta)$ on this world
# - Switching $\alpha$ to `x = 1` produces blue points with no outline
#   (e.g., $(1, 2)$): a visible counterexample, so entailment fails
# - Model, satisfaction, and entailment are the same three definitions as
#   Part 1 and Cell 2.1; only the shape of a "world" changed

# %% [markdown]
# # Part 3: Implication, Entailment, and Inference

# %% [markdown]
# ## Cell 3.1: Implication, Entailment, and Inference: Three Views
#
# **Goal**:
# - Separate three ideas the lecture distinguishes on one running example:
#   implication inside a single sentence, entailment across all models, and
#   inference as a procedure that tries to track it
# - _Implication view_: the truth table of $Rain \implies WetGround$
#   alone, with its one false row highlighted
# - _Entailment view_: the same shaded model table from Cell 2.1, for
#   $KB = \{Rain, Rain \implies WetGround\} \models WetGround$
# - _Inference view_: a step-by-step modus ponens trace, run forward from
#   the facts or backward from the goal
# - _Comments_: which view is active and its one-line definition

# %%
# Toggle between the four views of the same rain and wet-ground example.
utils.cell3_1_three_views()

# %% [markdown]
# **Key observations**:
# - Implication ($A \implies B$) is a single sentence's truth value in one
#   model; it says nothing about whether $A$ or $B$ actually holds
# - Entailment ($KB \models \alpha$) is a claim about every model at once;
#   it is what forward and backward chaining both try to compute correctly
# - Forward chaining starts from `Rain` and `Rain => WetGround` and derives
#   `WetGround`; backward chaining starts from the goal `WetGround`, finds
#   the rule, and reduces the goal to proving `Rain`
# - Both traces reach the same conclusion because model checking already
#   confirmed the entailment holds: a trace is only as good as the
#   entailment it tracks

# %% [markdown]
# # Part 4: Soundness and Completeness

# %% [markdown]
# ## Cell 4.1: Soundness and Completeness by Breaking a Reasoner
#
# **Goal**:
# - Make soundness (no false positives) and completeness (no false
#   negatives) concrete by extending the `KB` by one hop,
#   $KB = \{Rain, Rain \implies Puddle, Puddle \implies WetGround,
#   Sprinkler \implies WetGround\}$, and running three reasoners against the
#   model-checking ground truth
# - _Verdict table_: one row per query, comparing each reasoner's answer
#   to the model-checking verdict, colored correct, false positive, or
#   false negative
# - _Comments_: which reasoner is active, and its failure counts

# %%
# Run a correct, an unsound, and an incomplete reasoner on the same KB.
utils.cell4_1_soundness_completeness()

# %% [markdown]
# **Key observations**:
# - Model checking is both sound and complete here: the model space is
#   finite, so enumerating it settles every query correctly
# - The unsound reasoner sees `WetGround` and the rule
#   `Sprinkler => WetGround`, and wrongly affirms `Sprinkler`: a false
#   positive, since `Sprinkler` is genuinely undetermined by the `KB`
# - The incomplete reasoner applies modus ponens once, deriving `Puddle`
#   from `Rain` but never re-applying it to derive `WetGround`: a false
#   negative on a query model checking confirms is entailed
# - Soundness and completeness are independent failures: one reasoner
#   asserts too much, the other too little, and fixing one does not fix the
#   other

# %% [markdown]
# # Summary: The Mental Model
#
# - A model is one full assignment to every variable; $M(\alpha)$ is the set
#   of models where $\alpha$ is true, whether the variables are Boolean
#   (`Rain`, `WetGround`) or numeric ($x$, $y$)
# - $KB \models \alpha$ means $M(KB) \subseteq M(\alpha)$: every model
#   consistent with what is known also makes $\alpha$ true; one
#   counterexample model is enough to break it
# - Implication is syntax inside one sentence, entailment is semantics
#   across all models, and inference is the algorithm, forward or backward,
#   that tries to track entailment
# - A sound reasoner never asserts more than entailment supports (no false
#   positives), a complete reasoner never misses what entailment supports
#   (no false negatives), and the two failures are independent
# - Model checking is sound and complete whenever the model space is finite,
#   because it implements the definition of entailment directly
