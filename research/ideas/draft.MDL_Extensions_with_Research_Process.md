# MDL Extensions: Including the Research Process

## Status

- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Standard Minimum Description Length (MDL) and VC dimension frameworks
  evaluate the complexity of a final model, but ignore the complexity of
  the process that produced it
- In practice, researchers and AutoML pipelines search over vast numbers
  of models, architectures, and hyperparameters before selecting the final
  one
- This search process adds hidden complexity that must be accounted for in
  generalization bounds

## Formalization

- Total complexity is the sum of model complexity and search complexity:
  \[
  C = C_{\text{model}} + C_{\text{search}}
  \]
- Even if the "best" model has low VC dimension, the search process adds
  \(\log(N_{\text{attempts}})\) bits of complexity
  - E.g., AutoML trying 1000 architectures times 100 hyperparameter
    combinations adds \(\log(1000 \times 100) \approx 17\) bits,
    explaining why AutoML often overfits despite returning "simple" final
    models

## Key Examples

- **AutoML pipelines**: trying 1000 architectures with 100 hyperparameter
  combinations each, the winning model appears simple, but the search
  process adds about 17 bits of hidden complexity
- **Scientific publication**: a researcher tries 100 approaches but only
  publishes the best one, so the community systematically underestimates
  the true complexity
- **Kaggle competitions**: winning solutions often involve ensembling
  hundreds of models after extensive search, so the reported "final model"
  has low VC dimension, but the search process explains a significant
  portion of performance

## Questions

1. If a researcher tries 100 approaches but only publishes the best one,
   is the community systematically underestimating model complexity? How
   to fix this?
2. Can overfitting be detected from the research process itself, by
   analyzing the trajectory through model space rather than just the final
   model?
3. Should scientific venues require a "complexity tax" where papers must
   report \(C_{\text{search}}\) alongside model accuracy? Would this
   reveal that many "breakthroughs" are actually just overfitting?
4. If two researchers independently discover the same solution, does this
   reduce its effective complexity? Does convergent discovery serve as
   evidence against overfitting?
5. Can the idea that "surprising" results (low prior probability) should
   be penalized more heavily for multiple testing than "expected" results
   be formalized?
6. Is transfer learning a way to amortize search cost across tasks? If a
   pre-trained model is fine-tuned with minimal search, does this reduce
   \(C_{\text{search}}\) for the downstream task?

## Research Topics

- VC dimension of hyperparameter search
- Complexity of AutoML pipelines
- Generalization bounds that account for architecture search
- Detecting overfitting from research trajectory analysis
- Publication policy design informed by learning theory

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Derived from _Research_plan/paper.tex_ (Section: MDL Extensions /
  Including the Research Process)
