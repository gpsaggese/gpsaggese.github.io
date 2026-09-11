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
# # Wumpus World: From Percepts to Proofs
#
# - This notebook builds a knowledge-based agent for the classic wumpus world
#   and uses it to make the model-theoretic definition of entailment concrete
# - The pedagogical arc:
#   - Percepts and a knowledge base (`KB`) built with `TELL`
#   - Models, `M(KB)`, and entailment as set inclusion
#   - Implication vs entailment vs inference
#   - Soundness and completeness, seen by deliberately breaking each
#   - Model checking vs a SAT solver as the state space grows
#   - Propositional rules rewritten as first-order sentences
#   - A full agent loop that ties every piece together

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %%
import helpers.hnotebook as hnotebook

import L03_01_wumpus_world_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# %% [markdown]
# # Part 1: Percepts and the Knowledge Base

# %% [markdown]
# ## Cell 1.1: The Wumpus World Grid and the Knowledge Base
#
# **Goal**:
# - Give students a concrete grid to reason about before any logic is
#   introduced
# - Show that a knowledge-based agent knows only what it has been `TELL`-ed,
#   not the hidden truth of the world
#
# _Hidden world_: 4x4 grid with pits, the wumpus, and gold shown as icons on
# one panel
# _KB view_: the agent's current `KB` (told percepts only) on the adjacent
# panel
# _Comments_: current cell clicked, percept just told, total number of facts
# in the `KB`

# %%
# Click a cell, then TELL its percept, and watch the KB panel catch up.
utils.cell1_1_world_and_kb()

# %% [markdown]
# **Key observations**:
# - The left panel (truth) and the right panel (`KB`) disagree almost
#   everywhere at first: the agent starts knowing almost nothing
# - Every `TELL` adds one sentence to the `KB`, and nothing is ever removed
# - The gap between the two panels is exactly the reasoning problem the rest
#   of the notebook solves

# %% [markdown]
# # Part 2: Models and Entailment

# %% [markdown]
# ## Cell 2.1: Models and the Breeze Axiom
#
# **Goal**:
# - Introduce a model as one full true/false assignment to every pit variable
# - Encode the breeze axiom
#   $B_{1,2} \Leftrightarrow (P_{1,1} \lor P_{2,2} \lor P_{1,3})$ and
#   enumerate every model it can be checked against
#
# _Model table_: all $2^n$ rows of a truth table over the pit variables, with
# rows satisfying the `KB` shaded
# _Model count_: bar chart of satisfying vs non-satisfying model counts
# _Comments_: number of pit variables $n$, number of models $2^n$, number
# shaded

# %%
# Enumerate every model of the breeze axiom and shade the ones satisfying it.
utils.cell2_1_models_and_axiom()

# %% [markdown]
# **Key observations**:
# - $M(KB)$ is not a formula, it is the shaded subset of rows in the table
# - The number of models to check doubles with each added variable
# - A model that violates the biconditional is never shaded, regardless of
#   how plausible it looks informally

# %% [markdown]
# ## Cell 2.2: Entailment as Model Inclusion
#
# **Goal**:
# - Build on Cell 2.1 to define $KB \models \alpha$ as
#   $M(KB) \subseteq M(\alpha)$
# - Answer "is cell $(2,2)$ provably safe?" by checking model inclusion
#   directly
#
# _Model table_: same table as Cell 2.1, now with a second shading color for
# $M(\alpha)$, so overlap and gaps between the two sets are visible
# _Inclusion counts_: bar chart of $M(KB)$, the overlap, and the
# counterexamples
# _Comments_: query sentence $\alpha$, whether $M(KB) \subseteq M(\alpha)$
# holds, entailment verdict

# %%
# Decide KB |= alpha by checking whether M(KB) sits inside M(alpha).
utils.cell2_2_entailment()

# %% [markdown]
# **Key observations**:
# - $KB \models \alpha$ holds only when every `KB`-shaded row is also
#   $\alpha$-shaded: a single counterexample row breaks entailment
# - Some queries are entailed, some are contradicted, and some are simply
#   undetermined by the current `KB`
# - Entailment is a property of the full model sets, never of one row alone

# %% [markdown]
# ## Cell 2.3: Implication, Entailment, and Inference
#
# **Goal**:
# - Separate three ideas that are easy to conflate: a sentence's internal
#   structure, a semantic guarantee across models, and a computational
#   procedure
#
# _Implication view_: the biconditional sentence itself, with its logical
# connectives highlighted
# _Entailment view_: the same shaded model table from Cell 2.2
# _Inference view_: a step-by-step trace of the procedure walking from `KB`
# to $\alpha$
# _Comments_: which of the three views is active, and its one-line definition

# %%
# Toggle between the three views of the same breeze example.
utils.cell2_3_three_views()

# %% [markdown]
# **Key observations**:
# - Implication lives inside one sentence, entailment lives across all
#   models, inference is a procedure a computer actually runs
# - A correct inference procedure produces exactly the conclusions entailment
#   predicts, no more and no fewer
# - Cell 3.1 shows what happens when a procedure gets this correspondence
#   wrong

# %% [markdown]
# # Part 3: Soundness, Completeness, and Scaling

# %% [markdown]
# ## Cell 3.1: Soundness and Completeness by Breaking Them
#
# **Goal**:
# - Make soundness (no false positives) and completeness (no false
#   negatives) concrete by running two broken reasoners against the Cell 2.2
#   ground truth
#
# _Verdict table_: conclusions from each reasoner, marked against the true
# entailed set
# _Failure modes_: count of false positives (unsound) and false negatives
# (incomplete)
# _Comments_: which reasoner is active, and its failure counts

# %%
# Run a correct, an unsound, and an incomplete reasoner on the same queries.
utils.cell3_1_soundness_completeness()

# %% [markdown]
# **Key observations**:
# - The unsound reasoner reports facts that are not actually entailed: false
#   positives
# - The incomplete reasoner misses facts that are entailed: false negatives
# - Soundness and completeness are independent properties: a reasoner can
#   fail either one without failing the other

# %% [markdown]
# ## Cell 3.2: Model Checking Doesn't Scale
#
# **Goal**:
# - Measure how brute-force model checking degrades as the grid grows, and
#   compare it against a SAT solver on the same `KB`
#
# _Runtime curve_: log-scale runtime vs grid size (2x2 to 6x6) for model
# checking and for a SAT solver on the same query
# _Comments_: current grid size, number of variables, measured runtime for
# each method

# %%
# Compare brute-force model checking against a SAT solver as the grid grows.
utils.cell3_2_scaling()

# %% [markdown]
# **Key observations**:
# - Model-checking runtime grows exponentially with grid size, matching the
#   $2^n$ model count from Cell 2.1
# - The SAT solver answers the same query far faster by never enumerating
#   all models explicitly
# - This is the expressiveness vs tractability trade-off from the lecture,
#   made visible as a runtime curve rather than a claim

# %% [markdown]
# # Part 4: First-Order Logic and the Full Agent

# %% [markdown]
# ## Cell 4.1: From Propositional Rules to First-Order Sentences
#
# **Goal**:
# - Rewrite the propositional breeze axiom as a single first-order sentence
#   with a universal quantifier, and instantiate it for a specific cell
#
# _First-order rule_: the sentence
# $\forall x, y \; Breeze(x,y) \Leftrightarrow \exists x', y' \; Adjacent(x,y,x',y') \land Pit(x',y')$
# _Grounded instance_: the grid with the chosen cell's ground literals
# highlighted after universal and existential instantiation
# _Comments_: chosen cell, the ground sentence produced

# %%
# Rewrite the breeze axiom as one quantified sentence, then ground it.
utils.cell4_1_first_order()

# %% [markdown]
# **Key observations**:
# - One first-order sentence replaces 16 separate propositional axioms, one
#   per grid cell
# - Universal instantiation grounds $x, y$ to the chosen cell; existential
#   instantiation then picks a witness among its neighbors
# - The grounded sentence is exactly the propositional axiom from Cell 2.1
#   for that cell

# %% [markdown]
# ## Cell 4.2: The Agent Loop: KB Grows, Candidate Models Shrink
#
# **Goal**:
# - Tie every prior cell together in a stepping agent that `TELL`s new
#   percepts and re-`ASK`s safety at each step
#
# _Agent grid_: current agent position and every percept told so far
# _Candidate model count_: line plot of $|M(KB)|$ after each step, always
# non-increasing
# _Comments_: current step number, last percept told, current $|M(KB)|$

# %%
# Step the agent through provably safe cells and watch |M(KB)| shrink.
utils.cell4_2_agent_loop()

# %% [markdown]
# **Key observations**:
# - Each `TELL` can only remove models from $M(KB)$, never add them
# - Confidence about a cell's safety increases exactly as $M(KB)$ shrinks
#   toward models that agree on that cell
# - This closes the loop from Cell 1.1's raw percepts to a working
#   knowledge-based agent

# %% [markdown]
# # Summary: The Mental Model
#
# - A knowledge base is a set of sentences built with `TELL`; `ASK` answers a
#   query by model checking, comparing $M(KB)$ against $M(\alpha)$
# - $KB \models \alpha$ means $M(KB) \subseteq M(\alpha)$: every model
#   consistent with what is known also makes $\alpha$ true
# - Implication is syntax inside one sentence, entailment is semantics across
#   all models, and inference is the algorithm that tries to track entailment
# - A sound reasoner never asserts more than entailment supports, and a
#   complete reasoner never misses what entailment supports; the two failures
#   are independent
# - Model checking implements entailment directly but costs $2^n$ time; a SAT
#   solver answers the same query without ever enumerating all models
# - First-order logic compresses one axiom per grid cell into a single
#   quantified sentence, grounded by instantiation
