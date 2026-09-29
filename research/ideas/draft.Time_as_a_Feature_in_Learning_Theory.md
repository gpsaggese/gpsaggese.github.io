# Time as a Feature in Learning Theory

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Modern learning theory largely treats time as an external index over data
- In real systems, time is generative, causal, and structural: markets
  evolve, machines degrade, energy systems adapt, and agents respond to
  predictions
- Core question: _is time truly special, or have we artificially separated
  it from the feature space due to historical modeling choices?_
- Can time and features be unified into a single formal structure, removing
  the boundary between static learning and dynamical systems?

## Formalization

- Define a time-indexed hypothesis class:
  $$
  \mathcal{H}_{t} = \{h_{t}(x) : t \in \mathbb{T}\}
  $$
- Assume bounded drift between consecutive hypotheses:
  $$
  \|h_{t+1} - h_{t}\| \le \epsilon
  $$
- Classical PAC bound:
  $$
  n \gtrsim \frac{VC(\mathcal{H}) + \log(1/\delta)}{\epsilon^{2}}
  $$
- Time-aware extension:
  $$
  n \gtrsim \frac{VC(\mathcal{H}) + D_{T} + \log(1/\delta)}{\epsilon^{2}}
  $$
- where $D_{T}$ measures cumulative drift:
  $$
  D_{T} = \sum_{t=1}^{T-1} \|h_{t+1} - h_{t}\|
  $$

### Temporal VC Dimension (Complexity View)

- Merged from `draft.Temporal_VC_Dimension.md`
- Complementary to the drift-bound view above: instead of bounding sample
  complexity via cumulative drift $D_{T}$, bound it via the complexity of the
  _union_ of hypothesis classes visited over time:
  $$
  VC_{T} = VC\left( \bigcup_{t=1}^{T} \mathcal{H}_{t} \right)
  $$
- If $h_{t}$ can drift arbitrarily, $VC_{T}$ can grow unboundedly with $T$:
  this potentially makes long-term learning impossible, the complexity
  analogue of the drift bound blowing up as $D_{T} \to \infty$

## Key Examples

- **Stock market prediction**: $h_{t}(x)$ predicts returns from features
  where the relationship changes due to market regimes (bull/bear markets,
  crisis periods)
- **Adaptive recommendation systems**: user preferences drift over time,
  requiring hypothesis updates as tastes evolve
- **Climate modeling**: physical relationships shift due to anthropogenic
  forcing, violating stationarity assumptions
- **Medical diagnosis**: disease patterns and diagnostic criteria evolve with
  new treatments and emerging pathogens
- **NLP**: language models trained on pre-2020 data fail to understand
  pandemic-related terminology or evolving slang
- **Fraud detection**: fraudsters continuously adapt strategies to evade
  detection, creating adversarial drift where $P(y|x,t)$ is actively
  manipulated
- **Energy demand forecasting**: the relationship between temperature and
  electricity usage changes as solar panel adoption and electric vehicles
  increase

### Complexity Examples (Temporal VC Dimension)

- **Linear classifiers with drift**: for $h_{t}(x) = \text{sign}(w_{t}^{\top}
  x)$ where $w_{t}$ drifts smoothly, at any fixed $t$, $VC(\mathcal{H}) =
  d+1$. Over $T$ timesteps with unbounded drift, $VC_{T}$ can grow without
  bound as the union captures increasingly complex decision boundaries
- **Neural networks with time-varying weights**: if weights can change
  arbitrarily, $VC_{T}$ may equal the VC dimension of the universal function
  approximator class, even if each $h_{t}$ individually has bounded
  complexity
- **Seasonal models**: a retailer uses $h_{\text{holiday}}$ during December
  and $h_{\text{regular}}$ otherwise. Then $VC_{T}$ is at least
  $\max(VC(h_{\text{holiday}}), VC(h_{\text{regular}}))$ but could be larger
  if the union creates new decision boundaries

## Questions

1. Can a model that constantly adapts to time still learn anything
   generalizable? If $h_{t}$ changes at every timestep, is this learning or
   mere tracking?
2. Is forgetting necessary for generalization in non-stationary
   environments? Classical theory rewards more data, but in drifting
   environments, old data may hurt performance
3. Does the notion of "ground truth" even make sense in time-varying
   systems? If $y = f_{t}(x)$ where $f_{t}$ itself evolves, what are we
   actually trying to learn: the current $f_{t}$ or the meta-function that
   generates the sequence $\{f_{t}\}$?
4. Can we have PAC-style guarantees without stationarity? Or does
   non-stationarity fundamentally break the connection between empirical and
   true risk?
5. Should we penalize model complexity or model _stability_? A complex but
   stable model might generalize better over time than a simple but volatile
   one
6. Is the distinction between "learning from data" and "tracking a signal"
   just a matter of timescale? At what rate of change does learning become
   impossible?
7. _(from Temporal VC Dimension)_ Is complexity additive over time, or does
   the union bound $VC_{T}$ capture interaction effects that a simple sum
   would miss?
8. _(from Temporal VC Dimension)_ If $VC_{T} \to \infty$ as $T \to \infty$,
   does that mean all long-term predictions are impossible, or just that the
   _union_ bound is too loose to be useful?
9. _(from Temporal VC Dimension)_ Can two models have identical
   $VC(\mathcal{H})$ but vastly different $VC_{T}$ due to different drift
   patterns: does drift _pattern_ (not just magnitude) matter?

## Research Topics

- **Growth rate of $VC_T$**: as a function of drift smoothness, and when it
  grows sublinearly (learnable) vs. linearly (unlearnable)
- **$VC_T$ vs. cumulative drift**: whether PAC-style bounds parameterized by
  $VC_{T}$ and by cumulative drift $D_{T}$ are two views of the same bound,
  or genuinely different
- **Time-dependent Rademacher complexity**
- **Stability under evolving distributions**
- **Meta-learning for non-stationary tasks**

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

- Derived from `Research_plan/paper.tex` (Section: Time as a Feature of
  Machine Learning; Section: Quasi-Stationary Learning / Temporal VC
  Dimension)
