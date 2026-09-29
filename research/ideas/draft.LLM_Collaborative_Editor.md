# LLM Collaborative Editor

## Status

- **Status:**: draft
- **Complete Specs:**: 20%
- **Assignee:**: TBD

## Core Idea

- Design a new editor paradigm where humans and LLMs co-edit documents,
  code, and notebooks in real time, similar to Google Docs but with an LLM
  as an active collaborator
- Goal: the LLM is a first-class co-author that tracks document state via
  autoreload, proposes incremental edits, and responds to human edits in
  near-real time
  - This enables a tighter human-machine editing loop than current tools
    like Cursor or Copilot provide

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. How do you resolve conflicts between human and LLM edits in real time?
2. What is the right granularity for autoreload: per keystroke, per cell,
   or per save?
3. How should the LLM handle retracting a suggestion the human partially
   accepted?

## Research Topics

- **Autoreload mechanisms**: let the LLM see live changes to the
  document/code and re-evaluate its suggestions without manual re-prompting
- **Cell-based editing**: Jupyter-like cells authored by the human, the
  LLM, or interactively negotiated between both
- **UI patterns for LLM edits**: inline diffs, side-by-side views, or
  annotation layers for showing LLM-suggested edits alongside human edits
- **Latency and context management**: keep the LLM's state coherent across
  long editing sessions without losing earlier context
- **Streaming integration**: incorporate streaming LLM output into
  incremental document edits rather than replacing full blocks
- **Data sources**:
  - GitHub Copilot interaction logs (synthetic): reproduced via VS Code
    extension telemetry studies, e.g., accept/reject rates for inline
    suggestions and edit distances between the suggestion and the final
    code
  - Jupyter notebook revision history: sequential cell edits, execution
    order, and outputs from public GitHub notebooks, useful for modeling
    the human editing loop
  - Google Docs collaborative editing dataset: multi-author revision
    sequences with timestamps, e.g., the CoEdIT dataset
    (`grammarly/coedit` on HuggingFace)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build a minimal autoreload prototype
  - Pick a concrete substrate to start from (a plain text/code file watched
    via filesystem events, or a Jupyter notebook watched via cell
    execution), and implement a watcher that diffs the document on every
    save/execution
  - Feed each diff, with surrounding document context, to an LLM and
    capture its proposed incremental edit as a structured patch
  - This is the result: a working prototype that observes a live-edited
    file and returns a contextual proposed edit after each change

- Milestone 2: define the edit protocol and conflict resolution
  - Represent LLM edits as structured, line-range patches with an
    accept/reject/partial-accept state machine
  - Handle the case where a human edits a region the LLM has an
    outstanding proposal on, including retracting a partially accepted
    suggestion
  - This is the result: an edit-state protocol validated against a
    scripted sequence of interleaved human and LLM edits, including
    conflicting ones

- Milestone 3: evaluate suggestion quality against interaction logs
  - Replay Jupyter notebook revision histories and synthetic
    Copilot-style accept/reject sequences through the prototype
  - Measure edit distance between the LLM's proposed edit and the eventual
    accepted code, plus acceptance rate and end-to-end latency
  - This is the result: quantitative suggestion-quality and latency
    numbers for the prototype across the replayed logs

- Milestone 4: build a UI and ablate autoreload granularity
  - Build a thin UI (a VS Code extension or a small web app) that shows
    LLM edits as inline diffs or an annotation layer alongside human edits
  - Ablate the autoreload trigger granularity (per keystroke, per cell,
    per save) and measure its effect on latency and how often it
    interrupts the human's editing flow
  - This is the result: a recommendation for the granularity/UI pattern
    that best balances latency against disruption, backed by the ablation

## References

- Cursor IDE: LLM-integrated editor, but not true co-editing (human
  drives, LLM responds)
- GitHub Copilot: inline completion, with no document-level state tracking
- Raheja et al., _CoEdIT: Text Editing by Task-Specific Instruction
  Tuning_. (2023)
- GPT-4 Code Interpreter: executes code, but does not co-edit the notebook
  live
