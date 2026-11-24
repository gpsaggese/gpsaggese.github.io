# Test-Time Search as Backtest Overfitting

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- Test-time scaling methods (best-of-$N$, self-consistency, tree search, agent
  retries) all generate many candidates and keep the one that maximizes a proxy
  score: a reward model, a verifier, visible unit tests, or a vote
- Picking the maximum over $N$ noisy proxy evaluations is structurally identical to
  backtesting $N$ strategies and deploying the best one
  - The effective complexity of the answer grows with the search, not with the model
  - This connects `draft.Backtesting_Complexity.md` and
    `draft.MDL_Extensions_with_Research_Process.md` directly to LLM inference
- The prediction is non-obvious and testable: true quality should rise, peak, and
  then decline with $N$
  - Reason: the selection increasingly exploits proxy noise rather than real quality
- Practical payoff:
  - A stopping rule for $N$ derived from measurable quantities
  - A correction to reported test-time-scaling gains that are measured with the same
    proxy used for selection
- Focus of the empirical work: verifier co-fooling
  - The standard remedy for a noisy proxy is to ensemble diverse verifiers, in the
    same way a portfolio diversifies across assets
  - At budget $N$ the winner sits near each verifier's top $1/N$ of candidates, so
    raising $N$ probes the joint upper tail of the verifiers' errors rather than
    their average agreement
  - Two verifiers co-fool when both rank the same wrong answer near the top
  - An ensemble helps only if its members do not co-fool, and their correlation
    does not reveal whether they do
  - This is the blind spot of correlation-based diversification in finance, where
    assets that look uncorrelated on average still fall together in a crash

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
- $N^{*}$ cannot be identified from unlabeled proxy scores
  - $N^{*}$ depends on the joint distribution of $\hat{v}$ and $v$, while
    inference-time data reveal only the distribution of $\hat{v}$
  - Two settings with identical verifier scores can have different $N^{*}$
    - E.g., two verifiers that agree because both are right look the same as two
      verifiers that are fooled by the same wrong answer
  - A label-free estimate therefore needs an assumption such as independent
    verifier errors, which is exactly what co-fooling violates
- Tail co-fooling of verifiers $k$ and $l$ at level $q$ is:
  $$
  \lambda_{kl}(q) = P(u_l > q \mid u_k > q, \text{ wrong answer})
  $$
  where $u_k$ is the percentile of verifier $k$'s score among the wrong answers to
  the same problem
  - Independent verifiers give $\lambda_{kl}(q) = 1 - q$, so the lift
    $\lambda_{kl}(q) / (1 - q)$ is 1
  - A Gaussian copula with the pair's observed correlation gives the tail sharing
    that correlation alone predicts

## Key Examples

- **Reward model best-of-$N$**: true win rate improves up to a moderate $N$, then
  degrades
  - The policy is pushed into the region where the reward model was never trained
  - This is the standard reward overoptimization curve
- **Unit-test-guided repair**: an agent generates patches until visible tests pass
  - The fraction of patches that also pass held-out tests falls as $N$ grows
  - I.e., the patch is fit to the visible test set
- **Self-consistency voting**: majority voting has $\rho$ close to 1 when errors are
  independent
  - It should show a much later peak than reward model selection
  - This gives a clean contrast for the theory
- **Failure mode**: a proxy with heavy-tailed noise makes $N^{*}$ very small
  - E.g., a verifier that occasionally assigns a huge score to a degenerate answer
  - More compute is then strictly harmful

## Questions

1. How many labeled problems are needed to locate $N^{*}$?
2. Is the empirical penalty closer to $\sqrt{\ln N}$, $\ln N$, or something heavier
   once the proxy is a learned reward model rather than Gaussian noise?
3. Does the same correction apply at the agent level, where the unit of search is a
   whole trajectory rather than a single answer?
4. Do diverse verifiers share their worst mistakes in the upper tail
   that best-of-$N$ selection reaches?
   - Does the correlation between verifiers predict it?
   - Does it predict how much an ensemble gains at large $N$, i.e., whether
     ensembling raises $\rho$ enough to move $N^{*}$ or merely rescales the noise?
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
- **Tail co-fooling map**: for every pair of verifiers, measure how often both rank
  the same wrong answer in their top fraction, against independence and against a
  Gaussian copula with the same correlation
  - Relate it to verifier type (outcome reward model, process reward model, LLM
    judge), base model, and training data
- **Tail-aware ensembling**: with a small labeled split, test whether choosing or
  weighting ensemble members by tail co-fooling beats choosing them by accuracy or
  by correlation
- **Conformal selection**: use distribution-free bounds to select candidates with a
  guaranteed error rate instead of a raw argmax
- **Comparison of search shapes**: best-of-$N$ vs voting vs tree search at matched
  token budget, with the overfitting penalty as the explanatory variable
- **Connection to MDL**: express the search as extra description length and check
  whether the MDL penalty predicts the observed degradation
- **Related ideas**: `draft.Backtesting_Complexity.md`,
  `draft.MDL_Extensions_with_Research_Process.md`, and
  `draft.Skill_vs_Luck_in_Agent_Benchmarks.md`

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the candidate generation harness
  - Tasks with cheap ground truth: math word problems and code tasks with
    visible/held-out test splits
  - Generate $N$ up to a few hundred candidates per task, caching proxy and true
    scores separately
  - Score every candidate with several verifiers, so that verifier pairs can be
    compared on the same candidates
  - This is the result: a dataset of (candidate, proxy score, true score) triples
    that supports offline resampling of any $N$
- Milestone 2: measure the overfitting curve
  - Plot true quality vs $N$ for proxies of deliberately varying quality (strong
    verifier, weak reward model, visible tests only)
  - Include ensembles, and check whether averaging the top verifiers beats the best
    single verifier at large $N$
  - This is the result: empirical evidence for or against a finite $N^{*}$ and an
    estimate of where it lies per proxy
- Milestone 3: measure tail co-fooling across verifiers
  - Estimate $\lambda_{kl}(q)$ for every verifier pair, with confidence intervals
    from a bootstrap over problems, and compare it with independence and with the
    Gaussian copula benchmark
  - Explain the pattern by verifier type, base model, and training data
  - Test whether the tail co-fooling of an ensemble's members predicts its gain
    over its best member at large $N$ better than their correlation does
  - Test tail-aware ensembling with a small labeled split, and report how many
    labeled problems are needed to locate $N^{*}$
  - This is the result: a ground-truth measurement of where verifier diversity
    fails in the tail, and of whether that explains where ensembles stop improving
- Milestone 4: extend to agent trajectories
  - Apply the same analysis where the search unit is a full agent run scored by a
    proxy (visible tests, self-critique)
  - This is the result: a correction factor for reported agent scaffold gains,
    feeding `draft.Skill_vs_Luck_in_Agent_Benchmarks.md`

## References

- Gao et al., _Scaling Laws for Reward Model Overoptimization_. (2022)
- Cobbe et al., _Training Verifiers to Solve Math Word Problems_. (2021)
- Bailey et al., _The Probability of Backtest Overfitting_. (2014)
- Saad-Falcon et al., _Shrinking the Generation-Verification Gap with Weak
  Verifiers_. (2025)
- Lee et al., _FUSE: Ensembling Verifiers with Zero Labeled Data_. (2026)
- Afrin and Shihab, _Evaluator Ensembles Under Reward Hacking: Covariance Geometry
  and Finite-Search Guarantees_. (2026)
- Dorner et al., _ROC-n-reroll: How Verifier Imperfection Affects Test-Time
  Scaling_. (2025)
- Embrechts et al., _Correlation and Dependence in Risk Management: Properties and
  Pitfalls_. (2002)
