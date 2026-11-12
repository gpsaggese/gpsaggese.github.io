# MCP Server for Extracting and Injecting Markdown Chunks

## Status

- **Status**: draft
- **Complete Specs**: 30%

## Core Idea

- Skill files and documentation repeat the same blocks of markdown: rule
  snippets, conventions, examples
  - Copies drift from the source, so a rule fixed in one place stays wrong in
    the five files that copied it
  - Pure references (a link instead of the text) do not work for LLM
    consumption, because the model needs the content inline in its context
- Proposal: transclusion for markdown, modeled on the existing
  `render_images.py` workflow in this repo
  - The text is physically copy-pasted into the destination file, so it is
    inline for both humans and LLMs
  - Delimiters record where the chunk came from, so it can be regenerated
  - An updater script refreshes every injected block from its source
- Expose the extract and inject operations as an MCP server so an agent can
  pull a named chunk on demand instead of loading a whole rules file
- The extraction code already exists in the repo, so the work is the
  addressing scheme, the idempotent update, and the staleness detection, not
  the parsing

## Formalization

- A chunk is addressed by (`file`, `anchor`), where the anchor is a header
  path or an explicit named marker in the source:
  ```text
  markdown.rules.md#Lists and Items/Bullet Lists
  ```
- The injected block in the destination is delimited and carries the source
  address plus a content hash:
  ```markdown
  <!-- BEGIN INCLUDE: markdown.rules.md#Bullet Lists sha=a1b2c3 -->
  ...copied text...
  <!-- END INCLUDE -->
  ```
- The update operation must be idempotent:
  ```text
  update(update(D)) == update(D)
  ```
- Staleness is detected by comparing hashes, without rewriting the file:
  ```text
  stale(block) = sha(current_source_text) != block.sha
  ```
- The MCP surface is three tools:
  - `list_chunks(file)`: enumerate the addressable anchors
  - `get_chunk(address)`: return the text for one anchor
  - `check_stale(paths)`: report blocks whose source changed
- Editing policy must be explicit: the injected region is generated, so local
  edits inside the delimiters are overwritten by the next update and must be
  reported as a conflict rather than silently discarded

## Key Examples

- **Shared convention block**: the bullet-point rules live once in
  `text.rules.md` and are injected into every skill that needs them, so fixing
  the rule once updates all consumers on the next run of the updater
- **Selective loading by an agent**: instead of reading a whole rules file, an
  agent calls `get_chunk` for the one section relevant to the current task,
  which reduces context and is the same motivation as
  `draft.How_to_measure_LLM_compliance.md`
- **CI staleness check**: `check_stale` runs in CI and fails when a source
  chunk changed without the dependents being regenerated, which is the
  mechanism that actually prevents drift
- **Failure mode**: a source header is renamed, breaking every address that
  points at it, so the updater must fail loudly rather than leave the stale
  copy in place
- **Failure mode**: an injected chunk is edited in place by a well-meaning
  contributor, and the edit is lost on the next update unless conflicts are
  detected

## Questions

1. Should anchors be header paths (readable, fragile under renames) or
   explicit named markers in the source (stable, but they clutter the source)?
2. Is transclusion better than the alternative of having the agent fetch
   chunks at runtime via MCP and never materialize them into files? The
   trade-off is reviewability of the checked-in text vs duplication
3. How should nested includes be handled, i.e., a chunk that itself contains
   an include, and how are cycles detected?
4. Does per-chunk loading measurably reduce token cost and improve rule
   compliance, or does the agent just miss rules it never fetched?

## Research Topics

- **Addressing scheme**: header paths vs named markers vs line ranges, and
  their behavior under refactoring of the source
- **Existing implementations**: Obsidian block transclusion, Org-mode noweb
  references, `mdBook` includes, and literate programming tooling, as prior
  art for the delimiter and update design
- **Reuse of repo tooling**: the existing markdown extraction code and the
  `render_images.py` update pattern define the conventions to follow
- **MCP server design**: tool granularity, and whether the server should be
  read-only or also able to apply updates
- **Evaluation**: measure drift incidents before and after, plus context
  tokens saved per agent task

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: extractor and addressing
  - Reuse the existing markdown extraction code to expose `list_chunks` and
    `get_chunk` over a chosen addressing scheme
  - This is the result: a library that can resolve an address to text, with
    unit tests covering renames and missing anchors

- Milestone 2: injector and updater
  - Implement the delimiter format, the injection, and an idempotent
    `update` command following the `render_images.py` pattern
  - Add `check_stale` for a non-mutating CI check
  - This is the result: a working updater plus a CI job that fails on drift

- Milestone 3: MCP server
  - Wrap the library as an MCP server with the three tools
  - This is the result: an agent that can fetch a single rule section instead
    of a whole file

- Milestone 4: migrate and measure
  - Convert the duplicated blocks in the existing skill files to includes
  - Measure duplicated lines removed, drift incidents caught, and context
    tokens saved per task
  - This is the result: evidence on whether transclusion pays for its
    complexity

## References

- Model Context Protocol specification. (2024)
- Knuth, _Literate Programming_. (1984)
- Nelson, _Literary Machines_ (transclusion). (1981)
