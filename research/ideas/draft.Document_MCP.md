# Document Model Context Protocol (MCP)

## Status
- **Status:**: draft
- **Complete Specs:**: 15%
- **Assignee:**: TBD

## Core Idea
- Model Context Protocol (MCP) is becoming the standard way LLM apps (Claude
  Desktop, Claude Code, IDEs) connect to external tools and data sources, but
  the ecosystem docs are scattered across the spec repo, SDK READMEs, and
  scattered blog posts
- Write a single, hands-on tutorial/reference that takes an engineer from zero
  to: understanding the client/server/transport architecture, building a
  minimal MCP server exposing a tool and a resource, wiring it into an MCP
  client (Claude Desktop/Code), and the common gotchas (auth, stdio vs. SSE
  transport, schema validation errors)

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples
- **Minimal server**: a "hello world" MCP server exposing one tool (e.g.
  `get_weather`) and one resource, in both Python and TypeScript SDKs
- **Real integration**: wrap an existing internal API (e.g., a helpers-repo
  utility) as an MCP tool and connect it to Claude Code
- **Common failure mode**: schema mismatch between declared tool input schema
  and what the server actually validates, causing silent tool-call failures

## Questions
1. What's the minimal mental model needed to reason about MCP (client, server,
   transport, capability negotiation) without reading the full spec?
2. Where do most first-time implementers get stuck (auth flow? transport
   choice? schema validation?), and can a tutorial front-load exactly those?
3. How does MCP compare to writing a plain function-calling tool schema
   directly: when is the extra protocol layer worth it?

## Research Topics
- MCP spec (transports, capability negotiation, resources vs. tools vs.
  prompts)
- Existing SDKs (Python, TypeScript) and their idioms
- Security/auth patterns for MCP servers (local stdio vs. remote HTTP/SSE)

## Next steps
- [ ] Look for existing MCP tutorials/docs to avoid duplicating content
- [ ] Draft an outline (follow `tutorials_in_60_mins.create` skill conventions)
- [ ] Build and test a minimal end-to-end example server + client
- [ ] Write up gotchas encountered while building the example

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Anthropic, _Model Context Protocol specification_
