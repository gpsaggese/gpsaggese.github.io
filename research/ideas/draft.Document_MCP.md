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

- Milestone 1: survey existing MCP docs and scope the tutorial outline
  - Catalog what the spec repo, SDK READMEs, and blog posts already cover,
    and where they are incomplete or scattered
  - Define the target reader and the minimal mental model to teach: client,
    server, transport, capability negotiation
  - Draft a tutorial outline following the `tutorials_in_60_mins.create`
    skill conventions
  - This is the result: an outline document listing sections, target
    reader, and the gaps in existing MCP docs the tutorial will fill

- Milestone 2: build the minimal end-to-end server and client example
  - Implement a "hello world" MCP server exposing one tool (`get_weather`)
    and one resource, in the Python SDK
  - Repeat the same server in the TypeScript SDK
  - Connect each server to an MCP client (Claude Desktop or Claude Code)
    and verify a full tool-call round trip
  - This is the result: two working minimal MCP servers (Python,
    TypeScript), each verified against a live client with a successful
    tool call

- Milestone 3: build the real integration example and catalog gotchas
  - Wrap an existing helpers-repo utility as an MCP tool exposed to a
    client
  - Deliberately reproduce the common failure modes: tool input schema
    mismatch, stdio vs. SSE transport confusion, and an auth failure
  - Write a minimal reproduction and fix for each gotcha
  - This is the result: a working helpers-repo MCP integration plus a
    documented, reproducible list of gotchas with fixes

- Milestone 4: write and validate the tutorial
  - Assemble the outline, examples, and gotchas into the full tutorial per
    `tutorials_in_60_mins.create` conventions
  - Have a second engineer follow the tutorial from a clean environment,
    noting where they get stuck
  - Revise the tutorial based on the friction points observed
  - This is the result: a published tutorial that a new engineer can
    follow, unassisted, to build and connect a working MCP server

## References
- Anthropic, _Model Context Protocol specification_
