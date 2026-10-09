# Causal Analysis of Success in Modern Society

## Status

- **Status**: done
- **Complete Specs**: TBD

## Core Idea

- Conventional meritocracy assumes talent leads to success, but observed
  talent is normally distributed while observed success (wealth,
  publications) follows a Pareto (power-law) distribution
- Hypothesis: randomness (luck) is a critical, underappreciated driver of
  success, not just talent
- Goal: formalize and simulate a causal, Bayesian model of talent versus
  luck, to test whether luck alone can explain the mismatch between the two
  distributions

## Formalization

### Agents

- Population of $N = 100$ agents
- Each agent $i$ has a talent vector $\mathbf{T}_i = (t^{(1)}_i, t^{(2)}_i,
  \ldots, t^{(d)}_i)$, with $d \in \{3, 4\}$
- Dimensions:
  - **Intensity** ($t^{(1)}_i$): effort, grit, hours worked
  - **IQ** ($t^{(2)}_i$): cognitive skill
  - **Networking** ($t^{(3)}_i$): social capital
  - **Initial capital** ($t^{(4)}_i$)

### Events

- $M$ fixed events, split into positive (lucky) and negative (unlucky)
- Each event is a Bernoulli trial: $E_{ij} \sim \text{Bernoulli}(q_i)$,
  where $E_{ij} = 1$ if event $j$ hits agent $i$
- Impact distribution:
  - Positive events: $\Delta C \sim \mathcal{N}(\mu_+, \sigma^2)$ or $\sim
    \text{Exp}(\lambda)$
  - Negative events: $\Delta C \sim \mathcal{N}(\mu_-, \sigma^2)$ or $\sim
    \text{Exp}(\lambda)$

### Event Modifiers

- **Intensity** increases the surface area of luck: $q_i = \sigma(\alpha
  t^{(1)}_i)$
- **IQ** affects the probability of exploiting an event: $p_i =
  \sigma(\beta t^{(2)}_i)$
- **Networking** gives the probability of capturing another agent's event:
  $\Pr(\text{inherit event}) \propto t^{(3)}_i$
- **Initial capital** sets baseline wealth $C_{i,0}$, with no direct effect
  in the base model (see the dependency note under Assumptions)

### Dynamics

- Each agent has the same lifespan of $T$ rounds
- Capital evolves multiplicatively:
  - $C_{i,t+1} = C_{i,t}(1 + \Delta C_{i,t})$ on a lucky event
  - $C_{i,t+1} = C_{i,t}(1 - \Delta C_{i,t})$ on an unlucky event
  - $C_{i,t+1} = C_{i,t}$ otherwise

### Assumptions

- Attributes are independent, though this is unrealistic in practice
  - In reality, wealth increases networking ($t^{(4)} \to t^{(3)}$),
    improves education ($t^{(4)} \to t^{(2)}$), and enables outsourcing
    that raises intensity ($t^{(4)} \to t^{(1)}$)
- The number of events $M$ is fixed
- Capital effects are multiplicative, not additive

### Model Improvements

- **Talent evolution**: $t^{(k)}_{i,t+1} = t^{(k)}_{i,t} + f(\Delta
  C_{i,t}) - g(\text{burnout})$
- **Variable event magnitude**: continuous distributions for small versus
  transformative opportunities
- **Path dependence**: one event unlocks or blocks others
- **Feedback loops**: reputation and visibility amplify the probability of
  future events, via $q_{i,t+1} = q_{i,t} + \gamma \log(1 + C_{i,t})$
- **Externalities**: allow negative spillovers from monopolies or
  exploitation

## Key Examples

- **Inequality emerges from luck alone**: despite talent being normally
  distributed, final capital $C$ follows a Pareto distribution, $P(C > x)
  \sim x^{-\alpha}$
- **Top success does not mean top talent**: the most successful agents
  typically have average talent plus many lucky events, while
  exceptionally talented agents can remain unsuccessful without luck
- **Luck dominates the correlation with success**: $\text{corr}(\#
  \text{lucky events}, C_T) \gg \text{corr}(|\mathbf{T}_i|, C_T)$
- **Talent and luck interact**: success is not linear in talent, it
  requires both favorable randomness and capability

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Causal ML estimation**: use causal forests (treatment: number of lucky
  events, outcome: final capital, moderator: talent vector) to estimate
  conditional average treatment effects (CATEs)
- **Double machine learning**: use ML (e.g., Lasso) to partial out
  confounders in high dimensions, e.g., the causal effect of opportunities
  on income
- **Instrumental variables**: needed when luck is not random, using
  exogenous shocks (weather, lotteries) as instruments, via Deep IV or
  Orthogonal Random Forests
- **Uplift modeling**: estimate individual-level treatment effects to
  identify which agents gain most from additional opportunities or funding
- **Empirical calibration**: calibrate the stylized model against real
  wealth and income distributions, startup funding rounds, and scientific
  career trajectories (citations, grants), and check whether the simulated
  power-law exponents match observed data
- **Data requirements**: talent proxies (education, test scores, skills),
  opportunity data (funding, life events, network shocks), outcome data
  (income, patents, career milestones), and exogenous variation (lotteries,
  policy changes, weather shocks)
- **Policy implications**: compare egalitarian (broad small grants),
  meritocratic (reward past winners), and random allocation strategies, and
  measure how raising baseline talent or opportunity density affects
  outcome inequality

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- **Milestone 1: build the agent-based talent/luck simulator**
  - Implement the population of $N = 100$ agents with independent talent
    vectors $\mathbf{T}_i$, the Bernoulli event process, and the
    multiplicative capital update from the Formalization
  - Verify the base model reproduces the target stylized fact: talent
    normally distributed but final capital $C_T$ following a Pareto tail
  - This is the result: a working simulator whose output distribution can be
    inspected (histograms, log-log rank plots) to confirm Pareto-shaped
    inequality emerges from luck alone

- **Milestone 2: validate causal ML estimation on synthetic ground truth**
  - Apply causal forests (treatment: number of lucky events, outcome:
    $C_T$, moderator: $\mathbf{T}_i$) to estimate CATEs, and double machine
    learning to partial out talent confounders
  - Compare recovered luck/talent effects against the simulator's known
    true parameters to check estimator bias and variance
  - This is the result: a validated causal-estimation pipeline that
    recovers the simulator's known ground-truth effects within a measured
    error bound

- **Milestone 3: add model improvements and test the luck-dominance claim**
  - Extend the simulator with talent evolution, path dependence, and the
    reputation feedback loop $q_{i,t+1} = q_{i,t} + \gamma \log(1 +
    C_{i,t})$
  - Rerun the causal estimators from Milestone 2 on the richer model and
    track $\text{corr}(\#\text{lucky events}, C_T)$ versus
    $\text{corr}(|\mathbf{T}_i|, C_T)$ as complexity increases
  - This is the result: a report on whether luck's dominance over talent
    persists, weakens, or strengthens as feedback loops and path
    dependence are added

- **Milestone 4: calibrate against real data and compare policies**
  - Gather talent proxies, opportunity data, and outcome data (wealth or
    income distributions, startup funding rounds, or scientific career
    citations/grants), and fit simulator parameters so its Pareto exponent
    matches the empirical one
  - Simulate egalitarian, meritocratic, and random allocation policies on
    the calibrated model and measure resulting inequality
  - This is the result: a calibrated model matching an observed real-world
    Pareto exponent, plus a comparison table of how each allocation policy
    changes outcome inequality

## References

- Author(s), _Title_. (Year)
