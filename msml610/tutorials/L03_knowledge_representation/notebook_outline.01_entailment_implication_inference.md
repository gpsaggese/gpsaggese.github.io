# Entailment, Implication, and Inference: the Rain and Wet Ground World

- Source: `Lesson03.1-Knowledge_representation.smd`, section "Entailment and
  Inference"
- This notebook stays on the lecture's own smallest examples, rain and wet
  ground, and $x = 0$ implies $xy = 0$, rather than a larger running project
- The pedagogical arc:
  - Models, possible worlds, and satisfaction
  - Entailment as model inclusion, verified by model checking
  - The same definition applied to a non-Boolean world
  - Implication vs entailment vs inference, three views of one example
  - Soundness and completeness, seen by deliberately breaking a reasoner

## Cell 1: Possible Worlds, Models, and Satisfaction

**Goal**:
- Ground "model" as one full true/false assignment to every variable, using
  the lecture's own $(Rain, WetGround)$ world
- Introduce $M(\alpha)$, the set of models where a sentence $\alpha$ is true,
  as the object entailment is built from in Cell 2

**Plots and their descriptions**:
- _Model table_: all 4 rows of the truth table over `Rain` and `WetGround`,
  the lecture's own enumeration ($m_1 = (T,T)$, $m_2 = (T,F)$, $m_3 = (F,T)$,
  $m_4 = (F,F)$), with the rows satisfying the chosen $\alpha$ shaded
- _Model count_: bar chart of $|M(\alpha)|$ vs the rows that do not satisfy
  $\alpha$
- _Comments_: chosen $\alpha$, its truth value in each of the 4 named models,
  $|M(\alpha)|$

**Widgets**:
- `alpha`: dropdown over `Rain`, `WetGround`, `Rain and WetGround`,
  `Rain => WetGround`, `not Rain`; picks the sentence whose model set is
  shaded

**Key observations** (post-visualization):
- $M(Rain) = \{m_1, m_2\}$ regardless of `WetGround`, exactly the lecture's
  example: satisfaction depends only on the variables a sentence mentions
- "The model satisfies the sentence" reads backwards at first: the model
  (the world) is what varies across rows, the sentence's truth is read off
  each fixed row
- $M(Rain \implies WetGround)$ excludes only $m_2$: implication is false in
  exactly one of the 4 models

**Implementation**: `numpy` for the $2^2$ enumeration, `sympy.logic` for the
sentence, `matplotlib` for the table and bar chart, `ipywidgets.Dropdown` for
`alpha`

## Cell 2: Entailment as Model Inclusion, by Model Checking

**Goal**:
- Build on Cell 1 to define $KB \models \alpha$ as $M(KB) \subseteq M(\alpha)$,
  and verify the lecture's claim that
  $KB = \{Rain, Rain \implies WetGround\}$ entails $WetGround$
- Run the model-checking algorithm explicitly: enumerate every model, find
  $M(KB)$, check $\alpha$ in each of those rows

**Plots and their descriptions**:
- _Model table_: the same 4-row table, now with $M(KB)$ shaded blue and
  $M(\alpha)$ outlined in dashed orange, so inclusion or a counterexample row
  is visible directly
- _Inclusion counts_: bar chart of $|M(KB)|$, the overlap with $M(\alpha)$,
  and the counterexample rows
- _Comments_: which `KB` sentences are toggled on, the query $\alpha$, and the
  entailment verdict

**Widgets**:
- `Rain` and `Rain => WetGround`: checkboxes that add or remove each sentence
  from the `KB`
- `alpha`: dropdown over `WetGround`, `not WetGround`, `Rain or WetGround`

**Key observations** (post-visualization):
- With both `KB` sentences on and $\alpha = WetGround$, every row of $M(KB)$
  falls inside $M(\alpha)$: no counterexample exists, so the lecture's
  entailment claim holds
- Turning off `Rain => WetGround` leaves `KB = {Rain}`, which does not entail
  `WetGround`: $m_2 = (Rain=T, WetGround=F)$ satisfies `KB` and violates
  `WetGround`, exactly the lecture's counterexample
- Entailment is a property of the whole shaded set, never of a single row: one
  counterexample row is enough to break it, no matter how many rows agree

**Implementation**: `sympy.logic` sentence substitution over the 4 enumerated
models for the model-checking step, `matplotlib` for the table and bar chart,
`ipywidgets.Checkbox` and `ipywidgets.Dropdown` for the controls

## Cell 3: The Same Definition on a Non-Boolean World

**Goal**:
- Show that $M(KB) \subseteq M(\alpha)$ does not require Boolean variables, by
  checking the lecture's "sitting table" example: $\alpha$: "$x = 0$" entails
  $\beta$: "$x \cdot y = 0$", for any $y$
- Reinforce that a model here is a pair $(x, y)$, not a truth assignment

**Plots and their descriptions**:
- _Model grid_: every integer pair $(x, y)$ in a small range as a scatter
  grid, points where $\alpha$ holds shaded blue and points where $\beta$ holds
  outlined in dashed orange
- _Comments_: the current range of $x, y$, $|M(\alpha)|$, $|M(\beta)|$, and the
  entailment verdict

**Widgets**:
- `alpha`: dropdown over `x = 0`, `y = 0`, `x = 1`, so students can also see
  entailment fail (`x = 1` does not entail `x*y = 0`)
- `range`: slider for how far $x, y$ range (e.g., $-3$ to $3$, $-5$ to $5$),
  showing the same verdict holds at any grid size

**Key observations** (post-visualization):
- Every blue point ($x = 0$) is also outlined ($x \cdot y = 0$), for any $y$,
  which is exactly $M(\alpha) \subseteq M(\beta)$ on this world
- Switching $\alpha$ to `x = 1` produces blue points with no outline
  (e.g., $(1, 2)$): a visible counterexample, so entailment fails
- Model, satisfaction, and entailment are the same three definitions as
  Cell 1 and Cell 2; only the shape of a "world" changed, from a truth
  assignment to a numeric pair

**Implementation**: `numpy` for the $(x, y)$ grid, `matplotlib` scatter for
the model grid, `ipywidgets.Dropdown` and `ipywidgets.IntSlider` for the
controls

## Cell 4: Implication, Entailment, and Inference: Three Views

**Goal**:
- Separate the three ideas the lecture distinguishes on one running example:
  implication as a connective inside a single sentence, entailment as a
  guarantee across all models, inference as a procedure that tries to track it

**Plots and their descriptions**:
- _Implication view_: the truth table of $Rain \implies WetGround$ alone,
  with its one false row highlighted, and the note that the sentence can be
  true even when `Rain` is false
- _Entailment view_: the same shaded model table from Cell 2, for
  $KB = \{Rain, Rain \implies WetGround\} \models WetGround$
- _Inference view_: a step-by-step modus ponens trace,
  "1. Rain, 2. Rain => WetGround, 3. modus ponens on 1, 2: WetGround", next to
  the same trace run backward from the goal `WetGround`
- _Comments_: which view is active and its one-line definition

**Widgets**:
- `view`: dropdown over `implication`, `entailment`, `inference (forward)`,
  `inference (backward)`

**Key observations** (post-visualization):
- Implication ($A \implies B$) is a single sentence's truth value in one
  model; it says nothing about whether $A$ or $B$ actually holds
- Entailment ($KB \models \alpha$) is a claim about every model at once; it is
  what forward and backward chaining both try to compute correctly
- Forward chaining starts from `Rain` and `Rain => WetGround` and derives
  `WetGround`; backward chaining starts from the goal `WetGround`, finds the
  rule `Rain => WetGround`, and reduces the goal to proving `Rain`
- Both traces reach the same conclusion because model checking already
  confirmed the entailment holds; a trace is only as good as the entailment it
  tracks

**Implementation**: `matplotlib` text panels for the three views,
`ipywidgets.Dropdown` for `view`

## Cell 5: Soundness and Completeness by Breaking a Reasoner

**Goal**:
- Make soundness (no false positives) and completeness (no false negatives)
  concrete by extending the `KB` by one hop,
  $KB = \{Rain, Rain \implies Puddle, Puddle \implies WetGround,
  Sprinkler \implies WetGround\}$, and running three reasoners against the
  model-checking ground truth

**Plots and their descriptions**:
- _Verdict table_: one row per query (`Rain`, `Puddle`, `WetGround`,
  `Sprinkler`), comparing each reasoner's answer to the model-checking
  verdict, colored correct, false positive, or false negative
- _Comments_: which reasoner is active, and its false positive / false
  negative counts

**Widgets**:
- `reasoner`: dropdown over `correct (model checking)`,
  `unsound (affirms the consequent)`, `incomplete (single-hop modus ponens)`

**Key observations** (post-visualization):
- Model checking is both sound and complete here: the model space is finite,
  so enumerating it settles every query correctly
- The unsound reasoner sees `WetGround` and the rule
  `Sprinkler => WetGround`, and wrongly affirms `Sprinkler`: a false positive,
  since `Sprinkler` is genuinely undetermined by the `KB`
- The incomplete reasoner applies modus ponens once, deriving `Puddle` from
  `Rain` but never re-applying it to derive `WetGround` from `Puddle`: a false
  negative on a query that model checking confirms is entailed
- Soundness and completeness are independent failures: one reasoner asserts
  too much, the other too little, and fixing one does not fix the other

**Implementation**: `sympy.logic` for the extended `KB` and query sentences,
model checking by enumeration for the ground truth, hand-written derivation
rules for the two broken reasoners, `matplotlib` for the verdict table,
`ipywidgets.Dropdown` for `reasoner`
