# Knowledge Base Construction from Papers and Hacker News for LLMs

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Build knowledge bases from papers and Hacker News discussions that LLMs
  can use

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

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build ingestion pipelines for both sources
  - Build a paper ingestion pipeline pulling text and metadata from a
    source such as the arXiv or Semantic Scholar API
  - Build a Hacker News ingestion pipeline pulling threads and top
    comments via the Algolia HN Search API
  - This is the result: raw paper and HN datasets stored with consistent
    metadata (source, url, date, author)

- Milestone 2: extract structured knowledge from raw text
  - Chunk papers by section and extract claims/entities per chunk using an
    LLM extraction pass
  - Extract structured facts/claims from HN discussion threads, tracking
    which comment(s) each fact came from
  - This is the result: a set of extracted (source, claim, provenance)
    records for both papers and HN discussions

- Milestone 3: build the knowledge base store
  - Index extracted records in a vector store and/or knowledge graph,
    preserving source provenance and links between related paper and HN
    items
  - This is the result: a queryable KB where a topic query returns related
    items from both papers and HN discussions with source links

- Milestone 4: build and evaluate an LLM retrieval interface
  - Build a retrieval-augmented generation (RAG) interface over the KB for
    an LLM to query
  - Assemble a benchmark set of test queries and measure retrieval
    precision/recall and answer quality with citations, against a baseline
    LLM with no KB
  - This is the result: a benchmark comparison showing whether the KB
    improves grounded, cited answers over the no-KB baseline

## References
- Author(s), _Title_. (Year)
