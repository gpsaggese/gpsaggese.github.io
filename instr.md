When there is a widget separate the goal of the cell from the explanation of the
different parts of the widget

```
**Goal**:
- Represent the lecture's weather symbols ($Rain$, $Cold$, $Sunny$,
  $Snow$, $Cloudy$) as `sympy.logic` boolean symbols, and build atomic and
  complex sentences from them
- Read the truth table of a chosen sentence as one row per model $m$, the
  object semantics is defined over
- _Truth table_: `sympy`-generated table for the current sentence, one row
  per model, with the row matching the toggled model outlined
- _Parse tree_: the sentence's tree, connectives at internal nodes and
  symbols at the leaves, each node colored by its truth value in the
  toggled model
- _Comments_: current sentence, number of atoms, number of models, truth
  value at the toggled model
```

```
**Goal**:
- Represent the lecture's weather symbols ($Rain$, $Cold$, $Sunny$,
  $Snow$, $Cloudy$) as `sympy.logic` boolean symbols, and build atomic and
  complex sentences from them
- Read the truth table of a chosen sentence as one row per model $m$, the
  object semantics is defined over

**Explanation of Widget**
- _Truth table_: `sympy`-generated table for the current sentence, one row
  per model, with the row matching the toggled model outlined
- _Parse tree_: the sentence's tree, connectives at internal nodes and
  symbols at the leaves, each node colored by its truth value in the
  toggled model
- _Comments_: current sentence, number of atoms, number of models, truth
  value at the toggled model
```

Update all the msml610/tutorials/L03_knowledge_representation/*.ipynb
to follow this

Update the notebook.rules.md

## Plan
- [x] Update `.claude/skills/notebook.rules.md` to codify the new convention:
  split a widget cell's `**Goal**` bullets from a new `**Explanation of
  Widget**` bullet block (was: one tight combined list)
  - [x] `## Keep One Bullet List Tight (No Blank Lines Between Items)`
  - [x] `## Visualization Cell Triplet Details` >
    `### Markdown Cell (Before the Visualization)`
- [ ] For each paired `.py` file in
  `msml610/tutorials/L03_knowledge_representation/`, split every
  `**Goal**` markdown cell that describes widget panels into `**Goal**`
  + `**Explanation of Widget**`, fixing missing `- ` bullet dashes where
  present
  - [x] `L03_01_entailment_implication_inference.py` (4 cells)
  - [x] `L03_02_wumpus_world.py` (7 cells)
  - [x] `L03_03_rule_based_expert_systems.py` (8 cells, also missing
    bullet dashes)
  - [x] `L03_04_ontology_reasoning.py` (7 cells)
  - [x] `L03_06_logic_solvers.py` (6 remaining cells; 1 already done)
- [x] Regenerate each `.ipynb` from its `.py` via `jupytext --sync`,
  preserving existing outputs
- [x] Verify each notebook is still valid JSON and outputs are preserved
- [x] `git add` the modified files (no commit)

## Result
- Done:
  - Updated `.claude/skills/notebook.rules.md`: the two sections
    `## Keep One Bullet List Tight` (renamed
    `## Separate Goal from Widget Explanation, Keep Each List Tight`)
    and `## Visualization Cell Triplet Details` now document `**Goal**`
    and `**Explanation of Widget**` as two separate tight bullet lists
    with a blank line between, instead of one combined list
  - Split all 33 widget markdown cells across the 5 notebooks in
    `msml610/tutorials/L03_knowledge_representation/` into `**Goal**` +
    `**Explanation of Widget**` (4 + 7 + 8 + 7 + 7 cells, one of the 7 in
    `L03_06` already matched the target format before this task)
  - Fixed missing `- ` bullet dashes and 2-space continuation indent in
    all 8 `L03_03_rule_based_expert_systems` cells while splitting them
  - Re-synced every `.py` pair from its `.ipynb` via `jupytext --sync`;
    verified by stable cell `id` that only the intended markdown
    `source` fields changed and all code-cell `outputs` are untouched by
    this task (an unrelated pre-existing output diff in `L03_06`, from
    before this task started, was left as-is)
  - Staged all 9 changed files (5 `.ipynb`, 4 `.py`, `notebook.rules.md`)
    with `git add`, not committed
- Not done: n/a
