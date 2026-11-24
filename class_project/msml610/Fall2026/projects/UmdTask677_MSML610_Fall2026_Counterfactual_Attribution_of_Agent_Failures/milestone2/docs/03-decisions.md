# Milestone 2 Decisions

## Experiment and Publication

### `D1` (Use the Direct-Arm Fallback)

- **Decision**: Estimate each reply's repair gain directly from keep-original-reply
  and redecide arms at the same before-step checkpoint.
- **Why**: The first-stage sample sizes did not validate the checkpoint-value
  shortcut within the predeclared equivalence tolerance.
- **Alternatives**: Rely on the unvalidated shortcut or change the tolerance after
  seeing results. Neither supplies defensible labels.
- **Status**: Accepted before the direct-arm collection; the versioned protocol
  and first-stage results preserve the chronology.

### `D2` (Force the Complete Original Reply Once)

- **Decision**: The keep arm restores the same prefix and forces the complete
  original reply once, then samples subsequent replies normally.
- **Why**: Both prose and command enter later context. Holding only the move fixed
  would define a different intervention.
- **Alternatives**: Move-only forcing or discarding the original explanation.
- **Status**: Accepted; state/context/limit checks and replay audits are retained.

### `D3` (Publish the Frozen M2 Result Independently)

- **Decision**: Export the committed M2 tree under `milestone2/`, with a relative
  link to the existing M1 project for the frozen import paths.
- **Why**: Preserve experiment-source identities and avoid duplicating or editing
  Linu's M1 runner. Exclude uncommitted later-milestone work.
- **Alternatives**: Copy all M1 files again, refactor the experiment source during
  publication, or include ongoing M3 changes in the same PR.
- **Status**: Accepted for this publication; all 50 referenced M1 files match.
