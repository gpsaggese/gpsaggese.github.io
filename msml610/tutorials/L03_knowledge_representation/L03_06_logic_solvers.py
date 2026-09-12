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
# # Three Engines for Logic: Sympy, PySAT, and Z3
#
# - This notebook runs the lecture's own examples through three solvers:
#   - `sympy` for symbolic rewriting
#   - `PySAT` for propositional satisfiability
#   - `z3` for first-order, quantified reasoning
# - The pedagogical arc:
#   - Parsing and evaluating propositional sentences symbolically with
#     `sympy`
#   - Rewriting sentences with equivalences, and normal forms (CNF, DNF)
#   - Encoding a CNF sentence as integer clauses for a real SAT solver
#   - `PySAT` solving real instances, and why NP-complete does not mean
#     always slow
#   - Entailment by refutation, and the phase transition of random 3-SAT
#   - First-order logic with `z3`: quantifiers, predicates, and witnessing
#     models
#   - Routing one entailment question to whichever engine can answer it

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
# !pip install -q networkx==3.6.1 python-sat==1.9.dev15 sympy==1.14.0 z3-solver==5.1.0.0

import networkx
print("networkx version: ", networkx.__version__)
import pysat
print("pysat version: ", pysat.__version__)
import sympy
print("sympy version: ", sympy.__version__)
import z3
print("z3 version: ", z3.get_version_string())

# %%
import helpers.hnotebook as hnotebook

import L03_06_logic_solvers_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# %% [markdown]
# ## The Three Engines
#
# - The same question, "does this sentence follow?", is answered by
#   different machinery in each library:
#
# | Engine | Layer of logic | Core call | What it returns |
# |--------|----------------|-----------|-----------------|
# | `sympy` | propositional | `to_cnf()`, `satisfiable()` | a rewritten sentence, or one model |
# | `PySAT` | propositional | `Solver().solve()` | `SAT` with a model, or `UNSAT` |
# | `z3` | first-order | `Solver().check()` | `sat` with a model, or `unsat` |
#
# - The layer is what decides the choice: `sympy` and `PySAT` stop at
#   symbols and connectives, while `z3` also handles quantifiers,
#   predicates, and functions

# %% [markdown]
# # Part 1: Propositional Logic With `sympy`

# %%
# TODO(ai_gp): Explain how sympy builds sentences using weather_symbols and adding pointers to the code using the hintrospection stuff.

# %%
# Show the weather sentences every propositional cell reuses.
for label, sentence in utils.weather_sentences().items():
    print("%-26s -> %s" % (label, utils.format_sentence(sentence)))
# Outcome: 8 sentences covering every connective of the lecture, from a
# plain conjunction to a biconditional.

# %% [markdown]
# ## Cell 1.1: Sentences as Symbols, Parsed and Evaluated
#
# **Goal**:
# - Represent the lecture's weather symbols ($Rain$, $Cold$, $Sunny$,
#   $Snow$, $Cloudy$) as `sympy.logic` boolean symbols, and build atomic and
#   complex sentences from them
# - Read the truth table of a chosen sentence as one row per model $m$, the
#   object semantics is defined over
#
# **Explanation of Widget**
# - _Truth table_: `sympy`-generated table for the current sentence, one row
#   per model, with the row matching the toggled model outlined
# - _Parse tree_: the sentence's tree, connectives at internal nodes and
#   symbols at the leaves, each node colored by its truth value in the
#   toggled model
# - _Comments_: current sentence, number of atoms, number of models, truth
#   value at the toggled model

# %%
# TODO(ai_gp): Add an option to cell1_1_sentences_and_truth_tables to use all the variables instead of only the ones that are of use given the sentence 

# %%
# Parse a weather sentence and evaluate it in every model.
utils.cell1_1_sentences_and_truth_tables()

# %% [markdown]
# **Key observations**:
# - `sympy`'s boolean symbols carry no meaning by themselves
#   - Grounding `Rain` to "it's raining" is a step the notebook does, not
#     `sympy`
#   - Renaming every symbol would leave every truth table unchanged
# - $P \implies Q$ (`Implies`) is true whenever $P$ is false
#   - `Rain => Sunny` is true in every model where `Rain` is false, the
#     vacuous-truth surprise from the lecture
#   - Toggle `Rain` off and watch the root of the parse tree turn green,
#     whatever `Sunny` is
# - The biconditional `Sunny <=> ~Cloudy` is true in exactly the two models
#   where `Sunny` and `Cloudy` disagree in sign
# - Truth propagates bottom-up: each internal node's color is a function of
#   its children's colors, which is compositional semantics made visible

# %% [markdown]
# ## Cell 1.2: Equivalences and Normal Forms
#
# **Goal**:
# - Verify the lecture's equivalences (De Morgan, distributivity,
#   contraposition, double negation, implication elimination, biconditional
#   elimination) by checking $M(\alpha) = M(\beta)$, not by trusting the
#   algebra alone
# - Convert a sentence to conjunctive normal form and to disjunctive normal
#   form, and watch the clause count change
#
# **Explanation of Widget**
# - _Equivalence check_: the two sides of the law, their model counts, and
#   the verdict from testing $\lnot(\alpha \iff \beta)$
# - _Size of each form_: bar chart of the connectives in the input against
#   the CNF clauses and the DNF terms it produces
# - _Comments_: chosen law, equivalence verdict, the CNF and DNF sentences,
#   and their clause counts

# %%
# Check an equivalence by model sets, and convert a sentence to CNF and DNF.
utils.cell1_2_equivalences_and_normal_forms()

# %% [markdown]
# **Key observations**:
# - Every equivalence comes back with $\lnot(\alpha \iff \beta)$
#   unsatisfiable, which is exactly the lecture's definition of $\equiv$
#   - The check is about model sets, not about matching strings
#   - Both sides always report the same model count, which is the same fact
#     seen from the other side
# - Double negation is the odd one out: `sympy` applies it while the
#   sentence is being built, so the two sides are already the same object
# - `to_cnf()` output is a valid CNF but not the only one: it only has to
#   share the input's models, not its shape
# - The biconditional-heavy sentence produces more CNF clauses than it had
#   connectives, the blowup a SAT solver has to deal with in Part 2

# %% [markdown]
# # Part 2: Propositional Satisfiability With `PySAT`

# %% [markdown]
# ## Cell 2.1: From Formula to Clauses, in DIMACS
#
# **Goal**:
# - Turn a `sympy` CNF sentence into `PySAT`'s clause format: one integer
#   per symbol, negative for a negated literal
# - Read the resulting DIMACS file, the exchange format every SAT solver
#   competition since the 1990s has used
#
# **Explanation of Widget**
# - _Symbol map and clauses_: each symbol with its DIMACS id, then the CNF
#   sentence as one row per clause of signed integers
# - _DIMACS file_: the same clauses as written to disk, header line included
# - _Comments_: number of variables, number of clauses, the header line
#   `p cnf <vars> <clauses>`, and the file size

# %%
# Encode a CNF sentence as integer clauses and write it out as DIMACS.
utils.cell2_1_cnf_to_dimacs()

# %% [markdown]
# **Key observations**:
# - A clause is a disjunction of literals, and the CNF sentence is the
#   conjunction of all its clauses
#   - A solver therefore only has to satisfy every row at once
#   - Nothing in the format can express an implication directly: the CNF
#     rewrite of Cell 1.2 is what makes the encoding possible
# - DIMACS drops the symbol names entirely
#   - Nothing in the file says `Rain`, only variable `1`
#   - The symbol map has to travel with the file, or the model that comes
#     back cannot be read
# - This formula-to-clauses-to-integers step is what every encoding in the
#   next cells reuses, from the pigeonhole formula to random 3-SAT

# %% [markdown]
# ## Cell 2.2: A Real Solver Against Model Checking
#
# **Goal**:
# - Hand the encoded clauses to a `PySAT` solver and read back a satisfying
#   model, or `UNSAT`
# - Compare solve time against enumerating every model, on the pigeonhole
#   formula, as $n$ grows
#
# **Explanation of Widget**
# - _Solve time_: `Minisat22` against model checking ($O(2^n)$) on the same
#   instance, measured while the $2^n$ rows still fit and projected past
#   that, on a log scale
# - _What the solver returns_: the satisfying assignment for $n$ pigeons in
#   $n$ holes, and the `UNSAT` verdict for $n$ pigeons in $n - 1$ holes
# - _Comments_: current $n$, variables, clauses, models to enumerate, and
#   the two solve times

# %%
# Solve the pigeonhole formula both ways, and time them.
utils.cell2_2_solver_vs_model_checking()

# %% [markdown]
# **Key observations**:
# - Model checking explodes exactly as $O(2^n)$ predicts, while the solver
#   stays milliseconds away
#   - At $n = 7$ the formula has 42 variables, so a truth table would need
#     $2^{42}$ rows
#   - This is why a knowledge-based agent asks a solver, not a table
# - The solver curve is not flat either: pigeonhole with $n$ pigeons and
#   $n - 1$ holes is unsatisfiable for every $n$, and every resolution-based
#   solver pays an exponential price for it (Haken, 1985)
#   - NP-complete does not mean "always slow", but it does mean some
#     instances stay hard for every algorithm
# - The solver returns one satisfying assignment, where model checking would
#   have listed every one of them: a decision procedure answers "is there
#   one?", not "how many?"

# %% [markdown]
# ## Cell 2.3: Refutation, and the 3-SAT Phase Transition
#
# **Goal**:
# - Prove $KB \models \alpha$ the lecture's way, by showing that
#   $KB \land \lnot \alpha$ is unsatisfiable, using `PySAT` instead of
#   enumerating models
# - Locate the phase transition of random 3-SAT, the region where
#   satisfiability and solver difficulty both change sharply
#
# **Explanation of Widget**
# - _Entailment by refutation_: the `KB`, two queries, the verdict on
#   $KB \land \lnot \alpha$, and the counterexample model when there is one
# - _Random 3-SAT phase transition_: fraction of satisfiable instances and
#   median solve time, both against the clause-to-variable ratio, with the
#   ratio $4.26$ marked
# - _Comments_: the two entailment verdicts, the sweep parameters, and the
#   fraction satisfiable and median time at the selected ratio

# %%
# Decide entailment by refutation, then sweep the clause-to-variable ratio.
utils.cell2_3_refutation_and_phase_transition()

# %% [markdown]
# **Key observations**:
# - $KB \land \lnot WetGround$ coming back `UNSAT` is the same fact as
#   $M(KB) \subseteq M(WetGround)$, reached without enumerating a single
#   model
#   - When the query is not entailed, the solver hands back the
#     counterexample directly: the world where the `KB` holds and the query
#     fails
# - The satisfiable fraction drops from near 1 to near 0 in a narrow band
#   around ratio $4.26$, and the band sharpens as `n_vars` grows
# - Median solve time peaks at the same ratio where the satisfiable fraction
#   is near $0.5$
#   - Under-constrained instances have many models, so a solver finds one
#     fast
#   - Over-constrained instances contradict themselves early, so a solver
#     refutes them fast
#   - The hardest instances sit at the transition, not at the extremes

# %% [markdown]
# # Part 3: First-Order Logic With `z3`

# %% [markdown]
# ## Cell 3.1: Quantifiers Over a Real Domain
#
# **Goal**:
# - Declare a finite domain, predicates, and quantified formulas in `z3`,
#   for the lecture's own $Loves$ and Aristotle examples
# - Watch `z3` return a concrete model witnessing $\exists$, instead of a
#   truth table that cannot even be written once the domain grows
#
# **Explanation of Widget**
# - _Domain and relation from the model_: the objects as nodes, with the
#   $Loves$ edges or the predicate membership that `z3` chose
# - _Query and verdict_: the formula being checked, whether the sentence or
#   its negation was asserted, and `z3`'s answer
# - _Comments_: domain size, quantifier pattern, verdict, and what the
#   drawn model contains

# %%
# Check a quantified sentence with z3 and draw the model it returns.
utils.cell3_1_z3_quantifiers()

# %% [markdown]
# **Key observations**:
# - `z3` does not enumerate $2^n$ propositional models: it searches for a
#   witnessing object directly
#   - This is exactly why first-order logic needs quantifiers instead of
#     writing $P(a) \lor P(b) \lor P(c) \dots$ by hand
#   - The propositional encoding would have to grow with the domain, while
#     the quantified sentence does not change at all
# - $\forall x\, \exists y\, Loves(x, y)$ and
#   $\exists y\, \forall x\, Loves(x, y)$ are both satisfiable, but not by
#   the same relation
#   - The third query asserts the first and the negation of the second, and
#     `z3` still finds a model: quantifier order changes the claim
#   - Raise `domain_size` and the witness gets bigger, but the verdict does
#     not change
# - The Aristotle syllogism and the quantifier duality rules come back
#   `unsat` when `z3` is asked to falsify them, on every domain size tried:
#   a refutation is how a prover states "this always holds"

# %% [markdown]
# ## Cell 3.2: One Question, Three Engines
#
# **Goal**:
# - Take one entailment question at the propositional layer and at the
#   first-order layer, and route it to the engine that can answer it
# - See why `sympy` and `PySAT` stop at propositional logic while `z3` alone
#   checks the quantified version
#
# **Explanation of Widget**
# - _Three engines, one question_: one row per engine, with the layer it
#   handles, the call it makes, its verdict, and its time
# - _What each engine reaches_: propositional logic drawn inside first-order
#   logic, with each engine placed in the box it covers
# - _Comments_: the current question, which engines answer "entailed", and
#   each engine's verdict

# %%
# Run the same entailment question through all three engines.
utils.cell3_2_three_engines()

# %% [markdown]
# **Key observations**:
# - On the propositional query the three engines agree: `sympy`'s
#   satisfiability check, `PySAT`'s refutation, and `z3` are three routes to
#   one verdict
#   - The timings differ by orders of magnitude, and none of them matters at
#     this size
# - On the first-order query the propositional engines answer "not
#   entailed", and they are not wrong about what they were asked
#   - $\forall x\, (Human(x) \implies Mortal(x))$ cannot be written down
#     with symbols and connectives alone
#   - `Human_Socrates` and `Mortal_Socrates` become two unrelated atoms, so
#     nothing links them
#   - Losing the quantifier silently changes the question, which is the real
#     failure mode
# - The right tool follows the expressiveness the problem needs, not the
#   speed of the engine:
#   - `sympy` for symbolic rewriting and normal forms
#   - `PySAT` when the question is propositional satisfiability and size is
#     the problem
#   - `z3` when the sentence has quantifiers, functions, or theories on top

# %% [markdown]
# # Summary: The Mental Model
#
# - A propositional sentence is syntax, a model is one assignment to every
#   symbol, and a truth table is the complete semantics: `sympy` manipulates
#   the syntax, and every equivalence is a claim about the model sets
# - CNF is the shape solvers agree on, and DIMACS is the format they
#   exchange: once a sentence is clauses of signed integers, any SAT solver
#   can decide it, and the symbol map is the only thing tying the answer
#   back to `Rain` and `Snow`
# - Entailment is decided by refutation: $KB \models \alpha$ iff
#   $KB \land \lnot \alpha$ is unsatisfiable, which turns a claim about all
#   models into a single solver call that also returns the counterexample
#   when the claim fails
# - NP-complete describes the worst case, not the usual one: real solvers
#   cross instances that a truth table could never finish, while the
#   pigeonhole formula and the ratio $4.26$ region stay hard for everyone
# - Quantifiers are a different layer, not a harder propositional problem:
#   `z3` states $\forall$ and $\exists$ directly and answers with a model or
#   a refutation, which is what neither `sympy` nor `PySAT` can do at all
