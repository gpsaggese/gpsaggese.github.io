# Survey of Open Problems Across ML, Causal AI, and Finance

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- What are the most important open problems across:
  - Machine learning and statistical learning
  - Causal AI
  - Bayesian statistics
  - Finance and financial mathematics
  - DeFi and blockchain

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
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

- Milestone 1: define scope, taxonomy, and problem template
  - Fix the boundaries of each of the 5 domains (ML/statistical learning,
    causal AI, Bayesian statistics, finance/financial math, DeFi/blockchain)
    and the criteria for what counts as an "open problem" (unresolved,
    actively researched, admits a formal statement)
  - Design a per-problem template: problem statement, formalization, known
    partial results, why it matters, and open sub-questions
  - Draft a literature search plan per domain, listing the venues and
    survey papers to mine (e.g., NeurIPS/ICML for ML, UAI/CLeaR for causal
    AI, ISBA proceedings for Bayesian statistics, JFE/quant finance
    journals, and DeFi security/mechanism-design papers)
  - This is the result: a scoping document with the taxonomy template and a
    literature search plan for each domain

- Milestone 2: collect candidate problems per domain
  - For each domain, mine recent survey and "open problems" papers and
    shortlist candidate problems, filling in the template from Milestone 1
  - Cross-check candidates against the other research idea files already in
    `research/ideas/` (e.g., causal VC dimension, time as a feature) to
    avoid duplicating a problem already tracked as its own idea
  - This is the result: a per-domain list of candidate open problems, each
    with a short write-up, a formalization sketch, and references

- Milestone 3: synthesize cross-domain problems
  - Identify problems that recur across domains under different names
    (e.g., sample complexity under structural assumptions,
    non-stationarity, identifiability, incentive design) and write a
    unified statement for each
  - Rank all candidate problems by tractability and potential impact, and
    select a prioritized subset for the final survey
  - This is the result: a synthesis document that groups problems by
    cross-domain theme and ranks them

- Milestone 4: write and circulate the survey
  - Write the full survey with one subsection per selected problem,
    following the Milestone 1 template, and a short cross-domain
    introduction
  - Circulate the draft for feedback and revise based on comments
  - This is the result: a complete survey document, ready for review or
    submission as a paper or blog post

## References

- Author(s), _Title_. (Year)
