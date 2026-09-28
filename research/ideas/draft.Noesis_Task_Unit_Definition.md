# Noesis: A Concrete Task-Unit Definition

## Status
- **Status**: draft
- **Complete Specs**: 15%
- **Assignee**: TBD

## Core Idea
- The Noesis protocol paper (`papers/Noesis/03_contracts_and_notation.tex`)
  defines a task, the unit of quantity in every contract, bid, and ask, as a
  fully abstract placeholder: "quantities are expressed in tasks without
  committing to a specific definition of `U`", requiring only that it
  support addition and a total order
- Every worked example in the paper (100 tasks, 10,000 tasks, `N_tasks` in
  [[draft.Intelligence_Market]]'s contract tuple) already treats this
  abstraction as settled, so the whole notation rests on a placeholder that
  is never actually pinned down
- This idea is to propose, formalize, and stress-test one concrete
  instantiation of the task unit, so that cross-provider comparability, the
  central property the abstraction is supposed to guarantee, is checked
  rather than assumed

## Formalization
- Candidate definitions flagged in the paper: raw token count, wall-clock
  compute duration, or a benchmark-normalized task-equivalent
- Proposed definition to work out in full:
  ```
  task_equivalent(x) = tokens(x) * capability_weight(model_used_for_x)
  ```
  where `capability_weight` is calibrated against a fixed reference model,
  e.g., cost or FLOPs per token relative to that reference, so that a token
  from a frontier model and a token from a small model are not treated as
  the same unit of work
- Required properties (from the paper's Definition 3): `task_equivalent`
  must support addition (summing tasks across requests in a contract) and a
  total order (comparing `N_tasks` across bids and asks); check both hold
  for the proposed definition, including edge cases like mixed-provider
  fulfillment within one contract

## Key Examples
- **Raw tokens vs. task-equivalent**: a contract for 10,000 tasks priced in
  raw tokens is filled entirely by a cheap model at the same nominal token
  count as a frontier-model fill; show whether `task_equivalent` treats
  these as the same delivered volume or not, and whether that matches
  buyer intent
- **Cross-tier comparison**: two asks at different capability tiers quote
  `N_tasks` in the same unit; verify the total order lets a matching engine
  compare them meaningfully within Problem `prob:clearing`
- **Sensitivity check**: re-clear the same synthetic order book once under
  raw-token denomination and once under `task_equivalent`; a difference in
  clearing price or matched volume would show the unit choice is not a
  second-order detail

## Questions
1. Does the task-unit choice (raw tokens vs. calibrated task-equivalent)
   change the auction's clearing price or matched volume enough to matter
   in practice, or is it a second-order detail?
2. How should `capability_weight` be calibrated, and does it need to be
   re-estimated as providers update their models mid-market?
3. Does a benchmark-normalized unit introduce a new gaming surface, e.g., a
   seller picking whichever benchmark minimizes its `capability_weight`?
4. Is a single scalar `capability_weight` per model sufficient, or does
   task difficulty within a tier vary enough that per-request weighting is
   needed?

## Research Topics
- Task-equivalence metrics: benchmark-normalized cost models for comparing
  tokens across models of different capability
- Cross-provider comparability in commodity markets, and how other bundled
  goods (e.g., natural gas calorific value) standardize a unit that differs
  physically across sellers
- Interaction with the clearing problem: does the choice of unit affect the
  existence or uniqueness of a clearing price (Proposition 1 in
  `04_noesis_market.tex`)?

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan
- Milestone 1: propose and write up `task_equivalent(x)`
  - Define `capability_weight` calibration against a fixed reference model
    and check the addition/total-order requirements from Definition 3
  - This is the result: one concrete, defensible task-unit definition,
    written up in the paper's own notation

- Milestone 2: sensitivity-test the definition against raw tokens
  - Using a small synthetic order book (or, once available, the simulator
    from [[draft.Noesis_Prototype_Validation]]), re-clear the same bids and
    asks under both units and compare clearing price and matched volume
  - This is the result: quantitative evidence for or against the unit
    choice mattering in practice

- Milestone 3: update the paper
  - Replace the abstract Definition 3 and its "left as an open question"
    framing in `papers/Noesis/03_contracts_and_notation.tex` and
    `09_open_questions.tex` with the concrete definition and its
    sensitivity result
  - This is the result: every contract example in the paper uses one
    concrete unit instead of an abstract placeholder

## References
- Noesis paper: `papers/Noesis/03_contracts_and_notation.tex`
  (`TODO(gp)` comment after Definition 3, the task unit)
- Related ideas: [[draft.Intelligence_Market]],
  [[draft.Noesis_Prototype_Validation]]
