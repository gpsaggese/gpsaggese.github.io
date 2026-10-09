# Explainability via Distillation into an Interpretable Form

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Treat explainability as a form of regulation: after fitting a neural
  network, distill it into a second model that can be interpreted

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

1. What formulation of a neural network can be explained, e.g., what if the
   distilled form is a Python program?
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- Study existing procedures for distilling a neural network into an
  interpretable form

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey distillation targets and define evaluation criteria
  - Survey existing approaches to distilling a trained model into an
    interpretable form: decision trees, rule lists, symbolic regression,
    and program synthesis
  - Define what "interpretable" means for this project: a bounded-size
    symbolic or programmatic representation a human can simulate by hand
  - Shortlist 2-3 candidate target forms to implement and compare,
    including the program-synthesis form raised in the Questions section
  - This is the result: a short survey memo with a shortlist of target
    forms and the criteria used to score them

- Milestone 2: build the distillation pipeline on a controlled benchmark
  - Train a baseline neural network on a synthetic or small tabular task
    with a known, checkable ground-truth decision rule
  - Distill the trained network into the first target form (e.g., a
    decision tree) by training it on the teacher's outputs
  - Distill the same network into a program (a small domain-specific
    language plus a synthesis or search procedure) for the same task
  - This is the result: two distilled models (tree-based and
    program-based) that reproduce the teacher's predictions, with fidelity
    reported on held-out data

- Milestone 3: measure the fidelity vs. interpretability tradeoff
  - Define a fidelity metric (teacher/student agreement on held-out data)
    and an interpretability metric (program length, node count, or a small
    human comprehension-time study)
  - Sweep the complexity budget of each target form and plot the resulting
    fidelity vs. interpretability tradeoff curve
  - This is the result: a tradeoff curve comparing target forms and
    complexity budgets on the benchmark task

- Milestone 4: test whether distillation reveals or hides the true rule
  - Check whether the distilled form recovers the benchmark's known
    ground-truth decision rule, not just the teacher's outputs
  - Document any case where a distilled model is high-fidelity (matches
    the teacher's predictions) but still misrepresents the underlying rule
  - This is the result: a case study stating whether, and under what
    conditions, high-fidelity distillation actually reveals the network's
    true decision rule

## References
- Author(s), _Title_. (Year)
