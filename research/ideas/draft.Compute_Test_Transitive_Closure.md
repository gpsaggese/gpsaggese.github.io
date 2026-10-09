# Compute the Transitive Closure of Test-to-Code Dependencies

## Status
- **Status**: draft
- **Complete Specs**: 15%

## Core Idea
- CI runtime grows with codebase size unless tests can be selectively run
  based on what actually changed; doing this correctly requires knowing which
  tests transitively depend on which code (not just direct imports, but the
  full call/import closure)
- Use static analysis (e.g., `pyan`, or Python's own `ast`/import graph) to
  build a code dependency graph, compute its transitive closure, and map it to
  "which tests exercise this function/module", enabling a `run only affected
  tests` mode for CI
- Validate/refine the static closure with dynamic analysis (coverage.py
  per-test line coverage) where static analysis is imprecise (dynamic
  imports, duck typing, monkeypatching)

## Formalization
- Build directed graph `G = (V, E)` where `V` = modules/functions, `E` =
  "imports" or "calls" edges
- Test-to-code map: `T(m) = { t : t transitively depends on m }`, i.e. the
  transitive closure of `G` restricted to test entry points
- On a diff touching modules `M_changed`, affected tests =
  `∪_{m ∈ M_changed} T(m)`

## Key Examples
- **Static-only precision loss**: a test imports a module dynamically
  (`importlib.import_module`): static analysis misses the edge, so the test
  is wrongly excluded from the affected set. Dynamic coverage data catches
  this
- **Validation approach**: run the full suite once with coverage instrumentation,
  compare the coverage-derived test-to-code map against the static closure,
  and quantify false-negative rate (tests static analysis would have skipped
  but that actually exercise the changed code)

## Questions
1. How large is the precision gap between static-only and
   coverage-validated test-to-code maps in a real, large Python codebase?
2. Is a hybrid (static graph + periodic coverage-based correction) enough to
   make "run only affected tests" safe, or does it need to always fall back to
   the full suite on some class of changes (e.g., conftest.py, fixtures)?
3. How much CI time does affected-test-selection actually save in practice
   once correctness safeguards are added?

## Research Topics
- Python static call-graph tools (`pyan`, `pyflakes`, `ast`-based custom
  analysis)
- Coverage-based validation of static dependency graphs
- Existing "test impact analysis" tools/literature (e.g., in Java/C# CI
  ecosystems) for cross-checking approach

## Next steps
- [ ] Build a static import/call graph for a representative subset of the repo
- [ ] Compute transitive closure and derive a test-to-code map
- [ ] Cross-validate against coverage.py data on a full test run
- [ ] Prototype a `pytest --affected-by <diff>` mode

## Implementation plan

- **Milestone 1: build the static import/call graph**
  - Build `G = (V, E)` for a representative repo subset (e.g.,
    `helpers_root`), using `ast`-based analysis (or `pyan`) to extract
    module/function nodes and import/call edges
  - Report basic graph statistics (node/edge counts, connected components)
    as a sanity check on the extraction
  - This is the result: a working graph-builder producing `G` for the
    chosen subset, with reported node/edge counts

- **Milestone 2: compute the transitive closure and test-to-code map**
  - Compute the transitive closure of `G` restricted to test entry points
    to derive `T(m)` for each module/function `m`
  - Build a function that, given a set of changed modules `M_changed`,
    returns the affected test set `∪_{m ∈ M_changed} T(m)`
  - This is the result: a test-to-code map `T(m)` and a working
    `M_changed -> affected tests` function

- **Milestone 3: cross-validate against coverage.py**
  - Run the full test suite once with coverage instrumentation to derive
    a coverage-based test-to-code map
  - Compare it against the static map from Milestone 2 and compute the
    false-negative rate (tests static analysis would skip but that
    actually exercise the changed code, e.g., via dynamic imports)
  - This is the result: a quantified precision gap (false-negative rate)
    between the static and coverage-based test-to-code maps

- **Milestone 4: prototype `pytest --affected-by`**
  - Combine the static graph with periodic coverage-based correction, and
    add safe fallback rules (e.g., always run the full suite on
    `conftest.py` or fixture changes)
  - Measure CI time saved by running this mode on a sample of real diffs,
    versus running the full suite
  - This is the result: a working `pytest --affected-by <diff>` prototype
    with measured CI runtime reduction and documented fallback rules

## References
- `pyan`: Python static call graph generator
