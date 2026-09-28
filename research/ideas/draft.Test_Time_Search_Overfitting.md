# Test-Time Search as Backtest Overfitting

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- Test-time scaling methods (best-of-$N$, self-consistency, tree search, agent
  retries) all generate many candidates and keep the one that maximizes a proxy
  score: a reward model, a verifier, visible unit tests, or a vote
- Picking the maximum over $N$ noisy proxy evaluations is structurally identical to
  backtesting $N$ strategies and deploying the best one, so the effective complexity
  of the answer grows with the search, not with the model
  - This connects `draft.Backtesting_Complexity.md` and
    `draft.MDL_Extensions_with_Research_Process.md` directly to LLM inference
- The prediction is non-obvious and testable: true quality should rise, peak, and
  then decline with $N$, because the selection increasingly exploits proxy noise
  rather than real quality
- Practical payoff: a stopping rule for $N$ derived from measurable quantities, and a
  correction to reported test-time-scaling gains that are measured with the same
  proxy used for selection

## Formalization

- Sample candidates $y_1, \dots, y_N \sim \pi(\cdot \mid x)$
- Let $v(y)$ be true quality and $\hat{v}(y) = v(y) + \epsilon(y)$ the proxy, with
  noise scale $\sigma_\epsilon$ and proxy-truth correlation $\rho$
- Selection returns $y^{*} = \arg\max_{i} \hat{v}(y_i)$
- For light-tailed noise, the expected proxy score of the winner grows like:
  $$
  E[\hat{v}(y^{*})] \approx \mu + \sigma_{\hat{v}} \sqrt{2 \ln N}
  $$
  while the expected true quality grows only with the shared component:
  $$
  E[v(y^{*})] \approx \mu_v + \rho \, \sigma_v \sqrt{2 \ln N}
  $$
- The gap between reported and real gain is the overfitting term:
  $$
  \mathrm{Gap}(N) \approx (1 - \rho) \, \sigma_{\hat{v}} \sqrt{2 \ln N}
  $$
- $N^{*}$ is finite whenever $\rho$ decays in the upper tail, i.e., the proxy is
  least reliable exactly on the candidates it ranks highest

## Key Examples

- **Reward model best-of-$N$**: true win rate improves up to a moderate $N$, then
  degrades as the policy is pushed into the region where the reward model was never
  trained, the standard reward overoptimization curve
- **Unit-test-guided repair**: an agent generates patches until visible tests pass,
  and the fraction of patches that also pass held-out tests falls as $N$ grows, i.e.,
  the patch is fit to the visible test set
- **Self-consistency voting**: majority voting has $\rho$ close to 1 when errors are
  independent, so it should show a much later peak than reward model selection,
  giving a clean contrast for the theory
- **Failure mode**: a proxy with heavy-tailed noise (a verifier that occasionally
  assigns a huge score to a degenerate answer) makes $N^{*}$ very small, so more
  compute is strictly harmful

## Questions

1. Can $N^{*}$ be estimated from quantities observable at inference time (proxy score
   dispersion, agreement among independent verifiers) without any held-out ground
   truth?
2. Is the empirical penalty closer to $\sqrt{\ln N}$, $\ln N$, or something heavier
   once the proxy is a learned reward model rather than Gaussian noise?
3. Does the same correction apply at the agent level, where the unit of search is a
   whole trajectory rather than a single answer?
4. Does verifier ensembling with diverse verifiers raise $\rho$ enough to move
   $N^{*}$, or does it merely rescale the noise?
5. If true, how much of the published gain from test-time compute is real, and how
   much is measurement against the selection proxy?

## Research Topics

- **Empirical scaling curves**: measure true quality vs $N$ on tasks with cheap
  ground truth (math with checkable answers, code with held-out tests) across proxies
  of varying quality
- **Penalty estimation**: fit the functional form of $\mathrm{Gap}(N)$ and test
  whether the fitted $\rho$ predicts $N^{*}$ out of sample
- **Held-out verifier protocol**: always evaluate with a verifier disjoint from the
  one used for selection, in the same way a backtest needs an out-of-sample period
- **Conformal selection**: use distribution-free bounds to select candidates with a
  guaranteed error rate instead of a raw argmax
- **Comparison of search shapes**: best-of-$N$ vs voting vs tree search at matched
  token budget, with the overfitting penalty as the explanatory variable
- **Connection to MDL**: express the search as extra description length and check
  whether the MDL penalty predicts the observed degradation

## Next Steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation Plan

- Milestone 1: build the candidate generation harness
  - Tasks with cheap ground truth: math word problems and code tasks with
    visible/held-out test splits
  - Generate $N$ up to a few hundred candidates per task, caching proxy and true
    scores separately
  - This is the result: a dataset of (candidate, proxy score, true score) triples
    that supports offline resampling of any $N$
- Milestone 2: measure the overfitting curve
  - Plot true quality vs $N$ for proxies of deliberately varying quality (strong
    verifier, weak reward model, visible tests only)
  - This is the result: empirical evidence for or against a finite $N^{*}$ and an
    estimate of where it lies per proxy
- Milestone 3: predict $N^{*}$ without ground truth
  - Estimate $\rho$ and noise scale from verifier disagreement alone, then predict
    $N^{*}$ and validate against the measured curve
  - This is the result: an inference-time stopping rule with measured accuracy
- Milestone 4: extend to agent trajectories
  - Apply the same analysis where the search unit is a full agent run scored by a
    proxy (visible tests, self-critique)
  - This is the result: a correction factor for reported agent scaffold gains,
    feeding `draft.Skill_vs_Luck_in_Agent_Benchmarks.md`

## References

- Gao et al., _Scaling Laws for Reward Model Overoptimization_ (2022)
- Cobbe et al., _Training Verifiers to Solve Math Word Problems_ (2021)
- Bailey et al., _The Probability of Backtest Overfitting_ (2014)
- `draft.Backtesting_Complexity.md`
- `draft.MDL_Extensions_with_Research_Process.md`
- `draft.Skill_vs_Luck_in_Agent_Benchmarks.md`