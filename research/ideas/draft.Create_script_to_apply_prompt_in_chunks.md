# Chunked Prompt Execution: Externalizing Control Flow from the Agent

## Status

- **Status**: draft
- **Complete Specs**: 30%

## Core Idea

- Many prompts are really programs, not questions:
  - A loop over files, e.g., "apply this rule to every file in the directory"
  - A sequence of actions, e.g., "run these commands in order and fix what
    breaks"
- The LLM is asked to be both the interpreter of the control flow and the
  worker inside the body, and it reliably fails at the first role: it
  processes a prefix of the files, drifts on the later ones, and declares the
  task done
- Proposal: move the control flow out of the prompt and into a deterministic
  driver script
  - The script owns iteration, ordering, retries, and state
  - The model is called once per unit of work with a small, fixed context
- The non-obvious claim: this does not just improve reliability, it changes
  what the failure looks like
  - A driver that misses an item fails loudly (the item is still in the queue)
  - An agent that misses an item fails silently (it reports success)

## Formalization

- A chunked prompt is a triple:
  ```text
  (enumerate, body, verify)
  ```
  - `enumerate(input) -> [u_1, ..., u_N]`: produce the work units
  - `body(u_i) -> patch_i`: one model call per unit, with fresh context
  - `verify(u_i, patch_i) -> pass | fail`: deterministic check before commit
- The driver loop, run outside the model:
  ```python
  for u in enumerate(input):
      for attempt in 1..K:
          p = body(u)
          if verify(u, p): commit(p); break
      else:
          mark_failed(u)
  ```
- Cost model: single-shot vs chunked
  ```text
  cost_single  = 1 call with context O(sum |u_i|)
  cost_chunked = N calls with context O(max |u_i|) + shared preamble
  ```
  - Chunked costs more tokens in total (the preamble is resent) but each call
    stays far from the context limit
  - The trade is tokens for recall, so the comparison must be at equal recall,
    not equal cost
- Recall guarantee: with a deterministic `enumerate`, recall over units is 1
  by construction, and quality per unit becomes the only variable

## Key Examples

- **Loop over files**: "apply `markdown.rules.md` to every file in
  `research/ideas/`" becomes a shell loop calling the model once per file,
  with the rule set as a fixed preamble
- **Sequence of commands**: "run the linter, fix errors, re-run until clean"
  becomes a driver that runs the command, feeds only the error output to the
  model, and re-runs, with a hard iteration cap
- **Loop over sites within a file**: a file with 40 rule violations is
  processed one violation at a time, which is the mitigation tested in
  `draft.Create_LLM_Benchmark_for_thoroughness.md`
- **Failure mode**: units that are not independent, e.g., renaming a symbol in
  file A breaks file B, so the driver needs either a dependency order or a
  final global consistency pass
- **Failure mode**: `enumerate` itself needs the model (e.g., "find all the
  places where this pattern applies"), which reintroduces the recall problem
  at the enumeration step

## Questions

1. Which prompts are mechanically detectable as loops or sequences, so the
   driver can be generated rather than hand-written per task?
2. At equal recall, what is the token overhead of chunking vs single-shot, and
   how does it scale with $N$?
3. When units are not independent, is a final global pass enough, or does the
   driver need an explicit dependency graph?
4. If `enumerate` must be a model call, can it be made reliable by asking only
   for locations (cheap, verifiable against the source) rather than edits?
5. If externalized control flow dominates for this class of task, agent
   frameworks should expose loops as first-class primitives rather than hoping
   the model simulates them

## Research Topics

- **Task taxonomy**: which prompt shapes (map, reduce, fixed-point iteration,
  pipeline) cover the common cases, and what driver each needs
- **Driver generation**: inferring `enumerate` and `verify` from a natural
  language prompt, versus a small declarative spec written by hand
- **Verification**: what per-unit checks are cheap enough to run every
  iteration (lint, tests, diff scope)
- **Cost/recall frontier**: measuring both strategies on the same benchmark
- **Existing infrastructure**: the repo already has multi-file prompt
  application scripts and skill files, so the prototype should wrap those
  rather than start from scratch

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the map-style driver
  - Implement a script taking a prompt, a file glob, and a verifier, then
    running one model call per file with retries and per-unit logging
  - This is the result: a reusable driver plus a run log recording which
    units passed, failed, or were retried

- Milestone 2: measure against single-shot
  - Pick a task with deterministic ground truth (rule application with known
    violation sites)
  - Compare recall, precision, tokens, and wall-clock for single-shot vs
    chunked
  - This is the result: the cost/recall frontier for one task class

- Milestone 3: handle sequences and fixed points
  - Add the command-sequence driver (run, capture failure, patch, re-run)
    with an iteration cap and convergence detection
  - This is the result: a second driver shape validated on a lint-until-clean
    task

- Milestone 4: dependent units
  - Add dependency ordering and a final global consistency pass
  - Test on a cross-file rename where units interact
  - This is the result: evidence on whether independence can be relaxed
    safely, and what it costs

## References

- Khot et al., _Decomposed Prompting: A Modular Approach for Solving Complex
  Tasks_. (2023)
- Wu et al., _PromptChainer: Chaining Large Language Model Prompts through
  Visual Programming_. (2022)
- Schick et al., _PEER: A Collaborative Language Model_. (2022)
