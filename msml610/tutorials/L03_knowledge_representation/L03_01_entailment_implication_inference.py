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
# - [Open in Binder](https://mybinder.org/v2/gh/gpsaggese/gpsaggese.github.io/gp?filepath=msml610/tutorials/L03_knowledge_representation/L03_01_entailment_implication_inference.ipynb)
# - For Google Colab, use the standalone copy in
#   [`examples/L03_01_entailment_implication_inference_colab.ipynb`](https://colab.research.google.com/github/gpsaggese/gpsaggese.github.io/blob/gp/msml610/tutorials/L03_knowledge_representation/examples/L03_01_entailment_implication_inference_colab.ipynb)
#
# - This notebook stays on the lecture's own smallest examples, rain and wet
#   ground, and $x = 0$ implies $x \cdot y = 0$, to make the model-theoretic
#   definitions concrete without a larger running project
# - The pedagogical arc:
#   - Models, possible worlds, and satisfaction
#   - Entailment as model inclusion, verified by model checking
#   - The same definition applied to a non-Boolean world
#   - Implication vs entailment vs inference, three views of one example

# %% [markdown]
# ## Imports

# %%
# Binder builds its image straight from this GitHub repo, so it already
# has the whole repo on disk, just not always on `sys.path`. Docker gets
# `helpers` and the paired `_utils.py` file for free (helpers_root on
# PYTHONPATH, cwd = this notebook's dir); `colab_setup` reproduces that
# for Binder too, shared by every tutorial notebook so this cell stays the
# same everywhere except the `setup()` argument. Plain Python
# (`subprocess`, not `!`/`get_ipython()`), so the paired .py script still
# runs standalone outside a notebook.
#
# This notebook does not run on Google Colab: Colab starts with none of
# the repo on disk, so `class_scripts.colab_setup` itself is not
# importable there. Use the standalone copy in `examples/` for Colab.
import os
import sys

ON_BINDER = "BINDER_LAUNCH_HOST" in os.environ

if ON_BINDER:
    import subprocess

    # `colab_setup` lives inside the repo; Binder already has it on disk,
    # just not always on `sys.path`.
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Docker gets this for free via PYTHONPATH; Binder needs it.
    sys.path.insert(0, repo_root)

import class_scripts.colab_setup as colab_setup

colab_setup.setup("msml610/tutorials/L03_knowledge_representation")
colab_setup.maybe_enable_autoreload()

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# !pip install -q sympy==1.14.0

import sympy
print("sympy version: ", sympy.__version__)

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
#
# **Explanation of Widget**
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
#
# **Explanation of Widget**
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
#
# **Explanation of Widget**
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
# - Separate three ideas the lecture distinguishes on one running example
#   - Implication inside a single sentence
#   - Entailment across all models
#   - Inference as a procedure that tries to track it
#
# **Explanation of Widget**
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
