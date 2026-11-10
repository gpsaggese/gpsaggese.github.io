# Tukey Decisions

- `proposed` decisions wait for the reviewers' agreement

## Problem and Scoring

### `D1` (Score against the graph the data can identify)

- **Decision**: score against the equivalence class (CPDAG) for linear-Gaussian
  data, and against the DAG where the model identifies it (non-Gaussian noise,
  nonlinear additive noise)
- **Why**: linear-Gaussian data cannot reveal every edge direction, so Key
  Example 1 is read as "directions where identifiable"
- **Alternatives**: the true DAG everywhere (reported as a diagnostic)
- **Status**: proposed

### `D2` (Hidden confounders are scored against a PAG)

- **Decision**: an observed-confounder variant, where tests can remove the false
  edge, and a hidden-confounder variant scored against a PAG
- **Why**: conditioning on observed variables cannot remove a hidden common cause
- **Alternatives**: the DAG over the observed variables
- **Status**: proposed

### `D3` (Combine structure and prediction with `lam = 0.5`)

- **Decision**: score = -`lam` x (target SHD / true edges) - (1 - `lam`) x
  (prediction shortfall on the test rows)
- **Why**: prediction alone rewards dense graphs (`lam = 0` collapses to them);
  structure alone (`lam = 1`) ended within 0.003 (not tested)
- **Alternatives**: structure only
- **Status**: proposed

## Data

### `D4` (Standardize columns with training statistics)

- **Decision**: standardize each column with training-row statistics; keep the
  graph, the generation order and the test rows out of the agent's view
- **Why**: raw simulated data leak the causal order through variance (Reisach et
  al. 2021)
- **Alternatives**: raw data, which a variance sort solves
- **Status**: accepted

## Baselines and Evaluation

### `D5` (The prototype's rule is the baseline to beat)

- **Decision**: the prototype's threshold heuristic is the scripted baseline; PC,
  GES and DirectLiNGAM are references
- **Why**: the spec's M3 asks to beat the scripted baseline; the rule is not PC,
  so PC is reported too
- **Alternatives**: a hand-tuned rule (a secondary comparison)
- **Status**: accepted

### `D6` (The policy runs under a call budget)

- **Decision**: cap the policy's calls at the rule's median calls on 200
  validation graphs of the same size; the rule runs uncapped; classical methods
  have no tool budget
- **Why**: keeps the policy near the rule's typical effort; it is a cap, not equal
  calls or runtime
- **Alternatives**: exactly matched calls per graph
- **Status**: accepted

### `D7` (Sealed test sets are used once)

- **Decision**: create test sets once, record their hashes, evaluate once per
  registered comparison after freezing; report development results separately
- **Why**: tuning after seeing test results overstates them
- **Alternatives**: development results only
- **Status**: accepted

## Training

### `D8` (Behaviour cloning, then REINFORCE)

- **Decision**: clone the rule, then alternate REINFORCE on the decoder
  (leave-one-out baseline) and on the tool head (value baseline, KL to the clone,
  entropy bonus)
- **Why**: the spec allows REINFORCE or PPO; cloning avoids a random start
- **Alternatives**: PPO from scratch
- **Status**: accepted

## Process

### `D9` (Submit one milestone per PR)

- **Decision**: one reviewed PR at a time; AVICI, listed in #628, is dropped from
  the reference methods
- **Why**: #629 (93 files) was too large to review; AVICI was never implemented
- **Alternatives**: one PR at the end
- **Status**: accepted
