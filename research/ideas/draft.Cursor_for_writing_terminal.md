# Cursor-Style AI Autocomplete for Prose Writing and the Terminal

## Status
- **Status**: draft
- **Complete Specs**: 15%
- **Assignee**: TBD

## Core Idea
- Cursor brought inline, context-aware AI autocomplete to code editing; the
  same interaction pattern (predict the next few tokens/lines, accept with
  Tab, keep typing to reject) is under-explored for two adjacent domains:
  prose writing (e.g., in an editor like Zed) and shell/terminal usage
- **Prose variant**: inline next-sentence/next-clause suggestions conditioned
  on the document so far, tuned for suggestion *acceptance rate* and
  *disruption to flow* rather than code-correctness metrics
- **Terminal variant**: inline next-command suggestions conditioned on shell
  history, current directory state, and recent command output, plus a safety
  layer that flags/confirms before suggesting destructive commands
  (`rm -rf`, force-push)

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **Prose**: writer types "The experiment showed that", tool suggests a
  plausible continuation grounded in the rest of the document (not a generic
  LLM completion): measure how often suggestions are accepted verbatim vs.
  edited vs. rejected
- **Terminal**: after `git status` shows conflicts, tool suggests
  `git mergetool` or a targeted `git checkout --ours <file>` rather than a
  generic "next command" guess
- **Safety example**: user starts typing `rm -rf`, tool surfaces a warning
  and requires explicit confirmation before offering to autocomplete the
  rest, rather than fluently completing the destructive command

## Questions
1. What context window (document history, shell history, cwd state, recent
   errors) is actually predictive of a useful next suggestion, vs. noise?
2. How do you measure "good autocomplete" for prose, where there's no
   analogue to code's pass/fail tests: acceptance rate? edit distance from
   suggestion to final text?
3. For the terminal variant, how do you keep the safety layer from being
   either too aggressive (annoying false positives) or too permissive (misses
   a real destructive command)?

## Research Topics
- Inline-suggestion UX patterns (Cursor, GitHub Copilot, fig.io for terminal)
- Context-selection strategies for autocomplete (what to include in the
  prompt, and how to keep latency low enough for inline use)
- Destructive-command detection and confirmation UX for terminal assistants

## Next steps
- [ ] Prototype the terminal variant first (narrower scope, clearer safety
  requirements) as a shell wrapper or Zed/editor plugin
- [ ] Define acceptance-rate and edit-distance metrics for evaluation
- [ ] Build a minimal destructive-command safety layer
- [ ] Extend to the prose-writing variant if the terminal prototype validates
  the interaction pattern

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- fig.io / Warp: existing terminal-autocomplete products (prior art)
