# MDL Extensions: Including the Research Process

## Status

- **Status:**: draft
- **Complete Specs:**: 10%

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

- Milestone 1: formalize $C_{\text{search}}$ and derive a bound
  - Define $C_{\text{search}}$ precisely, beyond the naive
    $\log(N_{\text{attempts}})$ count, to account for pruning, early
    stopping, and correlation between trials in a search
  - Derive or adapt a generalization bound (PAC-Bayes or MDL-style) that
    uses $C = C_{\text{model}} + C_{\text{search}}$, and check that it
    reduces to the standard model-only bound when $C_{\text{search}} = 0$
  - This is the result: a written formal bound with a proof sketch and a
    worked derivation of $C_{\text{search}}$

- Milestone 2: collect real AutoML search trajectories
  - Instrument an AutoML pipeline (random/grid search or an
    Optuna/NAS-style search) over architectures and hyperparameters on a
    benchmark dataset
  - Log every trial's held-out score and the full search trajectory
    (order, pruning decisions, early stops), and compute both the naive
    $\log(N_{\text{attempts}})$ estimate and the refined
    $C_{\text{search}}$ from Milestone 1 for each run
  - This is the result: a dataset of AutoML search trajectories with both
    complexity estimates and the observed train/test generalization gap
    attached to each run

- Milestone 3: test the bound's predictive power
  - Correlate the search-aware bound from Milestone 1 against the actual
    generalization gap across many independent AutoML runs, varying the
    search budget $N$ and the dataset
  - Compare its predictive accuracy against the naive VC/MDL bound that
    ignores search complexity entirely
  - This is the result: a plot/table showing whether the search-aware
    bound predicts overfitting better than the model-only bound

- Milestone 4: prototype a trajectory-based overfitting detector
  - Build a detector that flags likely overfitting from the trajectory
    through model space (e.g., rate of held-out-score improvement per
    trial) rather than only the final model's held-out score
  - Test the detector on a held-out subset of the runs from Milestone 2
  - This is the result: a working detector with measured precision and
    recall for flagging overfit runs on the held-out trajectories

## References

- Derived from _Research_plan/paper.tex_ (Section: MDL Extensions /
  Including the Research Process)
