# Measuring LLM Instruction-Following Compliance

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- Question: how do we measure the ability of an LLM to follow instructions, when the
  instruction set is large, heterogeneous, and partially implicit?
- Public instruction-following benchmarks score a handful of verifiable constraints
  per prompt (e.g., "answer in exactly 3 bullets", "do not use the letter e")
  - Real systems instead carry hundreds of standing instructions spread over system
    prompts, style guides, and skill files
- Hypothesis: compliance is not a single scalar but a curve
  - Compliance decays as the number of active instructions grows
  - Decay depends on instruction position, phrasing (prohibition vs prescription),
    and conflict with the model's priors
- This is non-obvious because a model can score high on any single instruction in
  isolation and still violate most of them when they are presented together, so
  per-instruction evaluation overstates real compliance

## Formalization

- Let $I = \{i_1, \dots, i_n\}$ be a set of instructions, each with a programmatic
  verifier $v_k$ returning pass or fail on an output
- Let $M(x, I)$ be the model output for input $x$ under instruction set $I$
- Per-instruction compliance:

  ```
  c_k(x, I) = 1[v_k(M(x, I)) == pass]
  ```

- Aggregate compliance and the decay curve:

  ```
  C(n) = E_x [ (1/n) * sum_k c_k(x, I_n) ]
  ```

  - $I_n$ is a random subset of size $n$ drawn from a pool of instructions
  - The object of study is the shape of $C(n)$, not $C$ at one value of $n$
- Isolation gap measures the cost of bundling:

  ```
  gap(n) = C(1) - C(n)
  ```

- Position effect: compliance of instruction $k$ as a function of its normalized
  position $p_k \in [0, 1]$ in the prompt
- Applicability must be separated from compliance
  - An instruction is scored only on inputs where it applies, otherwise vacuous
    passes inflate the score

## Key Examples

- **Style-guide compliance**: apply `.claude/skills/markdown.rules.md` to a document
  and check each rule mechanically (no em-dash, fenced blocks tagged, `:` instead of
  `-` in list items), with one verifier per rule
- **Prohibition vs prescription**: "always use `-` for bullets" is followed more
  reliably than "never use `*` for bullets", even though the two are equivalent on
  the same input
- **Conflict with priors**: an instruction that fights the model's default behavior
  (e.g., "do not add a summary section") fails more often than an arbitrary but
  neutral instruction (e.g., "start every section with a bullet"), which isolates
  capability from preference
- **Failure mode**: an instruction that is trivially satisfiable by an empty or
  degenerate output, which the verifier passes and a human would reject

## Questions

1. Is the decay $C(n)$ smooth, or does it show a knee, i.e., a working-set size
   beyond which the model effectively ignores most instructions?
2. How much of the loss is retrieval (the instruction was not attended to) vs
   conflict (it was attended to and overridden)? A counterfactual test: re-prompt
   with the single violated instruction and see whether it is then satisfied
3. Does compliance depend on where an instruction lives (system prompt, skill file,
   inline user turn) once content is held constant?
4. If compliance decays predictably in $n$, should instruction sets be retrieved
   per-task rather than loaded wholesale?

## Research Topics

- **Verifier construction**: which instruction classes admit programmatic checkers,
  and how to build LLM-judge verifiers with measured agreement for the rest
- **Decay measurement**: sampling protocol for $I_n$, seeds, and confidence
  intervals, separating applicability from compliance
- **Positional sensitivity**: shuffling instruction order and measuring the variance
  attributable to position (lost-in-the-middle effects)
- **Mitigations**: instruction chunking, retrieval of only applicable rules, and
  post-hoc self-check passes, measured against the same curve
- **Relation to thoroughness**: compliance is per-instruction, thoroughness is
  per-application-site of one instruction, see
  `draft.Create_LLM_Benchmark_for_thoroughness.md`

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation Plan

- Milestone 1: build the instruction pool and verifiers
  - Extract atomic, independently checkable instructions from an existing rule set
    (e.g., `.claude/skills/markdown.rules.md`)
  - Implement one deterministic verifier per instruction plus an applicability
    predicate
  - This is the result: a versioned pool of (`instruction`, `verifier`,
    `applicability`) triples with unit tests
- Milestone 2: measure the decay curve
  - Sample instruction subsets of increasing size $n$ and run a fixed set of inputs
  - Report $C(n)$ with confidence intervals across seeds
  - This is the result: the first compliance-decay curve for one model and one rule
    set
- Milestone 3: attribute the loss
  - For each violation, re-prompt with only that instruction to separate retrieval
    failure from conflict
  - Run the position-shuffling ablation
  - This is the result: a breakdown of non-compliance into retrieval, conflict, and
    position effects
- Milestone 4: compare models and mitigations
  - Run the same protocol across models and against chunked or retrieved instruction
    delivery
  - This is the result: a ranking of models by compliance at fixed $n$ and evidence
    on whether chunking recovers the isolation gap

## References

- Zhou et al., _Instruction-Following Evaluation for Large Language Models (IFEval)_
  (2023)
- Qin et al., _InFoBench: Evaluating Instruction Following Ability in Large Language
  Models_. (2024)
- Jiang et al., _FollowBench: A Multi-level Fine-grained Constraints Following
  Benchmark for LLMs_. (2024)
- Liu et al., _Lost in the Middle: How Language Models Use Long Contexts_. (2023)
