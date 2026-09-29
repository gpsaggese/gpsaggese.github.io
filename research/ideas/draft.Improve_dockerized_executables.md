# Improve Transparency of Dockerized Executables

## Status
- **Status:**: draft
- **Complete Specs:**: 20%
- **Assignee:**: TBD

## Core Idea
- Dockerized CLI executables (tools packaged and run entirely inside a
  container) are convenient for reproducibility but tend to become opaque
  black boxes: hard to tell what a run actually did, why it failed, or what
  image/config produced a given output
- "Hard" because the improvements aren't a single feature but a set of
  overlapping concerns: better logging, dry-run/explain modes, and
  introspectable build provenance (which image, which layer, which
  dependency versions actually ran)
- Different concern from [[docker.shrink_container]] /
  [[docker.shrink_requirements]] (which target size/speed): this is about
  runtime behavioral transparency and debuggability

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **Opaque failure today**: a dockerized executable exits non-zero with a
  generic traceback; unclear which config/env/mounted volume caused it,
  requiring `docker exec` spelunking to reproduce
- **Target behavior**: `--dry-run` prints the exact command, mounts, and
  environment that would be used without executing; `--explain` on failure
  dumps image digest, dependency versions, and the effective resolved config
- **Provenance example**: two runs produce different output; a
  `--show-provenance` flag reveals that the image digest silently changed
  between runs (e.g., `:latest` tag drift): the actual root cause

## Questions
1. What's the minimal set of "transparency" features (dry-run, explain,
   provenance) that covers most real debugging sessions with dockerized
   tools, without needing a full observability stack?
2. Can build/run provenance (image digest, base image, key dependency
   versions) be captured automatically and cheaply, or does it require
   deliberate instrumentation in every tool?
3. How much of this generalizes into a reusable wrapper/library applicable to
   any dockerized executable in this repo, vs. needing per-tool customization?

## Research Topics
- Reproducible build provenance (image digests vs. mutable tags, SBOM-style
  dependency capture)
- Dry-run/explain UX patterns from other CLI tools (`terraform plan`,
  `kubectl diff`)
- Structured logging conventions for containerized CLI tools

## Next steps
- [ ] Inventory the dockerized executables in this repo and their current
      failure/debugging experience
- [ ] Prototype a `--dry-run` / `--explain` wrapper for one tool
- [ ] Add provenance capture (image digest, key dependency versions) to that
      tool's output
- [ ] Generalize into a reusable pattern if the prototype proves useful

## Implementation plan

- Milestone 1: inventory the repo and pick a target tool
  - List the dockerized executables in this repo and, for each, note the
    current failure/debugging experience: what a non-zero exit looks like
    today
  - Select one representative tool matching the opaque-failure example
    (generic traceback, unclear root cause) as the prototype target
  - This is the result: an inventory table of dockerized executables, with
    one tool chosen and its current debugging pain points documented

- Milestone 2: prototype `--dry-run` and `--explain`
  - Add a `--dry-run` flag that prints the exact `docker run` command,
    mounts, and environment without executing it
  - Add an `--explain` flag that, on failure, dumps the image digest,
    dependency versions, and the effective resolved config
  - This is the result: the chosen tool supports both flags, with a
    captured transcript showing them on a real failing run

- Milestone 3: add provenance capture and reproduce tag drift
  - Capture image digest and key dependency versions automatically at run
    time, via `docker inspect` and an in-image manifest/SBOM-style dump
  - Reproduce the provenance example (two runs differing because a
    `:latest` tag drifted) and confirm `--show-provenance` surfaces the
    digest change as the root cause
  - This is the result: a reproduced tag-drift case where provenance
    capture correctly identifies the digest change

- Milestone 4: generalize into a reusable pattern
  - Extract the dry-run/explain/provenance logic from the prototype into a
    small wrapper/library usable by other dockerized executables
  - Apply the wrapper to a second, different dockerized tool and note what
    was reusable versus what needed per-tool customization
  - This is the result: a reusable wrapper applied to two tools, with a
    short list of what generalized and what did not

## References
- (none yet)
