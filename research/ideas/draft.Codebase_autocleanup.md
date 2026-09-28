# Agentic Codebase Auto-Cleanup Driven by TODOs

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- A mature codebase accumulates TODOs that encode known debt, but they are
  never worked off because there is no cheap way to triage them and no owner
  for the boring ones
- Proposal: an agentic pipeline that treats the TODO set as a backlog and
  drives it to zero under human supervision:
  - Scan the code for TODOs and extract them with context
  - Rank them by complexity and risk
  - Have the user approve a batch
  - For each approved item, file a GitHub issue, produce a fix, run the
    regressions, and open a PR
- The non-obvious part is the ranking, not the fixing
  - Modern agents can fix most single-site TODOs
  - The value comes from ordering the work so that low-risk, high-confidence
    changes land first and build trust, while risky ones are escalated to a
    human rather than attempted
- Success is measured by TODOs retired per unit of human review time, not by
  raw PR count

## Formalization

- Let $T = \{t_1, \dots, t_n\}$ be the extracted TODO items
- Each item gets two independent scores in $[0, 1]$:
  - $c(t)$: complexity, i.e., predicted effort (files touched, symbols
    referenced, whether it needs a design decision)
  - $r(t)$: risk, i.e., blast radius if the fix is wrong (public API, test
    coverage of the touched lines, criticality of the module)
- Priority orders cheap and safe work first:
  ```
  score(t) = b(t) / (c(t) + eps) * (1 - r(t))
  ```
  - $b(t)$ is the estimated benefit (e.g., unblocks other TODOs, removes a
    known bug)
- Routing policy by risk band:
  ```
  r(t) < r_lo         -> auto-fix, human reviews the PR
  r_lo <= r < r_hi    -> agent proposes a plan, human approves, then fix
  r(t) >= r_hi        -> file an issue only, do not attempt
  ```
- Acceptance gate for any generated PR:
  ```
  accept iff (tests pass) and (lint clean) and (diff touches only planned files)
  ```
- Dependency ordering: TODOs referencing the same symbol form a cluster and
  are batched into one PR to avoid conflicting edits

## Key Examples

- **Low-risk mechanical TODO**: `# TODO: rename to _private` in a module with
  full test coverage, one call site, auto-fixed and merged after review
- **Clustered TODOs**: five TODOs in different files all asking for the same
  helper to be extracted, batched into one refactoring PR instead of five
  conflicting ones
- **Escalation case**: `# TODO: this is O(n^2), rewrite` in an uncovered
  module, which is filed as an issue with a proposed plan and never
  auto-fixed
- **Failure mode**: a stale TODO describing an intent that no longer matches
  the code, where the agent "fixes" a problem that does not exist, so a
  staleness check (blame age vs last edit of surrounding code) is required
- **Failure mode**: a TODO satisfied by deleting the code path it annotates,
  which passes tests while silently removing behavior

## Questions

1. Can complexity and risk be predicted accurately enough from static
   features (blame age, test coverage, fan-in, module criticality) that the
   ranking beats a random order, or does it need an LLM judgment per item?
2. What fraction of TODOs are stale or unactionable, and can staleness be
   detected before spending agent tokens on a fix?
3. What is the right batch size for human approval? Approving one item at a
   time defeats the purpose, approving 50 defeats the supervision
4. If the pipeline works, does the TODO count become a controllable metric
   rather than a passive one, i.e., does writing a TODO become a way to
   queue work rather than a way to defer it forever?

## Research Topics

- **TODO extraction and normalization**: parsing the varied conventions in
  the codebase (`TODO(user)`, `FIXME`, `XXX`, `TODO(ai_gp)`) into a schema,
  reusing the existing annotation conventions in this repo
- **Risk scoring**: static features (coverage, fan-in, blame age, public vs
  private API) vs LLM-judged severity, validated against human labels
- **Human-in-the-loop protocol**: batch size, approval UI, and the cost of
  review measured in minutes per retired TODO
- **Regression safety**: what test selection is sufficient before opening a
  PR, and whether the transitive test closure is needed, see
  `draft.Compute_Test_Transitive_Closure.md`
- **Related work**: automated program repair, Sapienz/SapFix-style industrial
  auto-fix pipelines, and Renovate/Dependabot as an existing model for
  bot-authored PRs at scale

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: inventory and scoring
  - Extract all TODOs in the repo into a structured inventory with
    surrounding context and git blame metadata
  - Implement static complexity and risk features plus an LLM-judged score
  - This is the result: a ranked TODO backlog and a report on how many items
    are stale

- Milestone 2: validate the ranking
  - Hand-label a sample of TODOs for complexity and risk
  - Measure rank correlation between predicted and human labels
  - This is the result: evidence that the ranking is better than random, or
    a decision to fall back to LLM-only scoring

- Milestone 3: end-to-end fix loop on the low-risk band
  - For approved low-risk items: file the issue, generate the fix, run the
    relevant tests, open the PR
  - This is the result: a working pipeline plus the first batch of merged
    bot-authored PRs

- Milestone 4: measure and scale
  - Track TODOs retired, human review minutes per TODO, PR acceptance rate,
    and regressions introduced
  - Extend to clustered and mid-risk items with plan approval
  - This is the result: a cost curve showing whether the pipeline is cheaper
    than doing the work by hand

## References

- Le Goues et al., _Automated Program Repair_. (2019)
- Marginean et al., _SapFix: Automated End-to-End Repair at Scale_. (2019)
- Storey et al., _TODO or To Bug: Exploring How Task Annotations Play a Role
  in the Work Practices of Software Developers_. (2008)
