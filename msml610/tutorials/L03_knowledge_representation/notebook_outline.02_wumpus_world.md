# Wumpus World: From Percepts to Proofs

- Source: `Lesson03.1-Knowledge_representation.smd`,
  `Lesson03.2-Propositional_and_first_order_logic.smd`
- This notebook builds a knowledge-based agent for the classic Wumpus World and uses
  it to make the model-theoretic definition of entailment concrete
- The pedagogical arc:
  - Percepts and a knowledge base (`KB`) built with `TELL`
  - Models, `M(KB)`, and entailment as set inclusion
  - Implication vs entailment vs inference
  - Soundness and completeness, seen by deliberately breaking each
  - Model checking vs a SAT solver as the state space grows
  - Propositional rules rewritten as first-order sentences
  - A full agent loop that ties every piece together

## Cell 1: the Wumpus World Grid and the Knowledge Base

**Goal**:
- Give students a concrete grid to reason about before any logic is introduced
- Show that a knowledge-based agent knows only what it has been `TELL`-ed, not the
  hidden truth of the world
**Plots and their descriptions**:
- _Wumpus grid_: 4x4 grid with pits, the wumpus, and gold shown as icons on one
  panel, and the agent's current `KB` (told percepts only) on an adjacent panel
- _Comments_: current cell clicked, percept just told, total number of facts in the
  `KB`
**Widgets**:
- `cell`: click any grid cell to `TELL` its percept (breeze, stench, glitter, or
  none) into the `KB`
- `seed`: random seed for the pit/wumpus/gold layout
**Key observations** (post-visualization):
- The left panel (truth) and the right panel (`KB`) disagree almost everywhere at
  first: the agent starts knowing almost nothing
- Every click adds one sentence to the `KB`; nothing is ever removed
- The gap between the two panels is exactly the reasoning problem the rest of the
  notebook solves
**Implementation**: `matplotlib` grid patches for both panels, `ipywidgets` click
handling via a `Button` grid, `sympy.symbols` for percept propositions

## Cell 2: Models and the Breeze Axiom

**Goal**:
- Introduce a model as one full true/false assignment to every pit variable
- Encode the breeze axiom
  $B_{1,2} \Leftrightarrow (P_{1,1} \lor P_{2,2} \lor P_{1,3})$ and enumerate every
  model it can be checked against
**Plots and their descriptions**:
- _Model table_: all $2^n$ rows of a truth table over the pit variables, with rows
  satisfying the `KB` shaded
- _Model count bar_: bar chart of satisfying vs non-satisfying model counts
- _Comments_: number of pit variables $n$, number of models $2^n$, number shaded
**Widgets**:
- `n`: number of pit variables enumerated (3-6), on a logarithmic slider since the
  model count doubles with each step
**Key observations** (post-visualization):
- $M(KB)$ is not a formula, it is the shaded subset of rows in the table
- The number of models to check doubles with each added variable
- A model that violates the biconditional is never shaded, regardless of how
  plausible it looks informally
**Implementation**: `sympy.logic` for the biconditional and truth-table enumeration,
`pandas.DataFrame` for the shaded table, `matplotlib` bar chart

## Cell 3: Entailment as Model Inclusion

**Goal**:
- Build on Cell 2 to define $KB \models \alpha$ as $M(KB) \subseteq M(\alpha)$
- Answer "is cell $(2,2)$ provably safe?" by checking model inclusion directly
**Plots and their descriptions**:
- _Model table_: same table as Cell 2, now with a second shading color for
  $M(\alpha)$, so overlap and gaps between the two sets are visible
- _Comments_: query sentence $\alpha$, whether $M(KB) \subseteq M(\alpha)$ holds,
  entailment verdict
**Widgets**:
- `alpha`: dropdown of candidate queries (e.g., "no pit at $(2,2)$", "pit at
  $(2,2)$", "no pit at $(3,1)$")
**Key observations** (post-visualization):
- $KB \models \alpha$ holds only when every `KB`-shaded row is also $\alpha$-shaded:
  a single counterexample row breaks entailment
- Some queries are entailed, some are contradicted, and some are simply undetermined
  by the current `KB`
- Entailment is a property of the full model sets, never of one row alone
**Implementation**: `sympy.logic.inference.satisfiable`, `pandas` for the two-color
shaded table

## Cell 4: Implication, Entailment, and Inference

**Goal**:
- Separate three ideas that are easy to conflate: a sentence's internal structure, a
  semantic guarantee across models, and a computational procedure
**Plots and their descriptions**:
- _Implication view_: the biconditional sentence itself, with its logical connectives
  highlighted
- _Entailment view_: the same shaded model table from Cell 3
- _Inference view_: a step-by-step trace of the procedure walking from `KB` to
  $\alpha$
- _Comments_: which of the three views is active, and its one-line definition
**Widgets**:
- `view`: toggle between "implication", "entailment", and "inference"
**Key observations** (post-visualization):
- Implication lives inside one sentence, entailment lives across all models,
  inference is a procedure a computer actually runs
- A correct inference procedure produces exactly the conclusions entailment predicts,
  no more and no fewer
- Cells 5 and 6 show what happens when a procedure gets this correspondence wrong
**Implementation**: `matplotlib` text panels for the three views, `sympy` for the
sentence and its trace

## Cell 5: Soundness and Completeness by Breaking Them

**Goal**:
- Make soundness (no false positives) and completeness (no false negatives) concrete
  by running two broken reasoners against the Cell 3 ground truth
**Plots and their descriptions**:
- _Unsound reasoner_: conclusions from a reasoner that affirms the consequent, with
  wrong conclusions marked against the true entailed set
- _Incomplete reasoner_: conclusions from a modus-ponens-only reasoner (no
  resolution), with missed entailed facts marked
- _Comments_: count of false positives (unsound) and false negatives (incomplete)
**Widgets**:
- `reasoner`: dropdown for "correct", "unsound (affirms consequent)", "incomplete
  (modus ponens only)"
**Key observations** (post-visualization):
- The unsound reasoner reports facts that are not actually entailed: false positives
- The incomplete reasoner misses facts that are entailed: false negatives
- Soundness and completeness are independent properties: a reasoner can fail either
  one without failing the other
**Implementation**: `sympy.logic.inference` for the correct reasoner, custom rule
sets in Python for the unsound and incomplete variants

## Cell 6: Model Checking Doesn't Scale

**Goal**:
- Measure how brute-force model checking degrades as the grid grows, and compare it
  against a SAT solver on the same `KB`
**Plots and their descriptions**:
- _Runtime curve_: log-scale runtime vs grid size (2x2 to 6x6) for model checking and
  for a SAT solver on the same query
- _Comments_: current grid size, number of variables, measured runtime for each
  method
**Widgets**:
- `grid_size`: slider from 2x2 to 6x6
**Key observations** (post-visualization):
- Model-checking runtime grows exponentially with grid size, matching the $2^n$ model
  count from Cell 2
- The SAT solver (`python-sat`) answers the same query far faster by never
  enumerating all models explicitly
- This is the expressiveness vs tractability trade-off from the lecture, made visible
  as a runtime curve rather than a claim
**Implementation**: `sympy.logic.inference.satisfiable` for model checking,
`pysat.solvers` (DPLL/CDCL) for the SAT solver, `matplotlib` log-scale plot

## Cell 7: From Propositional Rules to First-Order Sentences

**Goal**:
- Rewrite the propositional breeze axiom as a single first-order sentence with a
  universal quantifier, and instantiate it for a specific cell
**Plots and their descriptions**:
- _First-order rule_: the sentence
  $\forall x, y \; Breeze(x,y) \Leftrightarrow \exists x', y' \; Adjacent(x,y,x',y') \land Pit(x',y')$
- _Grounded instance_: the grid with the chosen cell's ground literals highlighted
  after universal and existential instantiation
- _Comments_: chosen cell, the ground sentence produced
**Widgets**:
- `cell`: dropdown to pick the cell to instantiate the universal rule for
**Key observations** (post-visualization):
- One first-order sentence replaces 16 separate propositional axioms, one per grid
  cell
- Universal instantiation grounds $x, y$ to the chosen cell; existential
  instantiation then picks a witness among its neighbors
- The grounded sentence is exactly the propositional axiom from Cell 2 for that cell
**Implementation**: `sympy` symbolic display for the quantified sentence,
`matplotlib` grid highlighting for the grounded instance

## Cell 8: the Agent Loop: KB Grows, Candidate Models Shrink

**Goal**:
- Tie every prior cell together in a stepping agent that `TELL`s new percepts and
  re-`ASK`s safety at each step
**Plots and their descriptions**:
- _Agent grid_: current agent position and every percept told so far
- _Candidate model count_: line plot of $|M(KB)|$ after each step, always
  non-increasing
- _Comments_: current step number, last percept told, current $|M(KB)|$
**Widgets**:
- `step`: button to advance the agent by one move
- `seed`: random seed for the layout, reused from Cell 1
**Key observations** (post-visualization):
- Each `TELL` can only remove models from $M(KB)$, never add them
- Confidence about a cell's safety increases exactly as $M(KB)$ shrinks toward models
  that agree on that cell
- This closes the loop from Cell 1's raw percepts to a working knowledge-based agent
**Implementation**: `ipywidgets.Button` for stepping, `matplotlib` grid and line
plot, `sympy.logic` for the shrinking model count