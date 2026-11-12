# Tutorial: Building Agents with the Claude Agent SDK

## Status
- **Status:**: draft
- **Complete Specs:**: 15%

## Core Idea

- Write a practical, hands-on tutorial (following the
  `tutorial_in_60_mins.create` conventions already used in this repo) that
  teaches engineers to build a custom agent with the Claude Agent SDK: the
  agentic loop, tool definitions, memory/context management, and deployment
  patterns (CLI, server, scheduled job)
- Distinguish clearly from the raw Messages API tool-use loop: show what the
  SDK buys you (built-in loop, tool runner, session handling) vs. what you'd
  hand-roll otherwise

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **Minimal agent**: single custom tool (e.g., a calculator or file-search
  tool), showing the full request/response/tool-execution loop
- **Multi-tool agent with memory**: an agent that persists state across
  turns (e.g., a simple task tracker), showing session/context patterns
- **Deployment variant**: same agent exposed as a CLI vs. as a long-running
  server process, showing what changes and what doesn't

## Questions

1. What's the smallest example that demonstrates the SDK's value over a
   hand-rolled tool-use loop?
2. Where do first-time SDK users get stuck (tool schema definition?
   streaming? error handling in tool execution?), and can the tutorial
   front-load exactly those?
3. How does the SDK's agent loop compare to Claude Code's own harness: what
   design choices carry over, what's different?

## Research Topics

- **Claude Agent SDK architecture**: agentic loop, tool runner,
  session/state
- **Comparison against manual Messages-API tool-use loops**
- **Common pitfalls**: tool schema validation, streaming, error propagation

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Read the Claude Agent SDK docs and existing examples
- Draft an outline (follow `tutorial_in_60_mins.create` skill conventions)
- Build a minimal working agent end-to-end
- Build a second, slightly more complex example (memory or multi-tool)
- Write up gotchas encountered while building the examples

## References

- Anthropic, _Claude Agent SDK documentation_
