# Decision by Committee Vs. Single Authority

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- The classic **wisdom of crowds vs. expert autocracy** problem: when should
  organizations delegate decisions to committees, and when should a single expert
  choose?
- The key tension emerges when agents share correlated errors (same training, shared
  biases, information cascades) or when coordination costs (deliberation time,
  negotiation overhead) dominate
- As these factors vary, optimal decision structure shifts non-monotonically: small
  independent committees beat solo experts, but high correlation or coordination
  costs flip the outcome
- The hypothesis: formal decision-theory models (Condorcet Jury Theorem, opinion
  dynamics, Bayesian aggregation) can reproduce this crossover empirically in
  simulation, and the intersection point is where practical insights live

## Formalization

### Single Decision-Maker

- Agent has signal accuracy $p$ (probability of choosing correct action) and possible
  bias $b$ (systematic error)
- Decision quality: $\text{accuracy} = p$
- Decision speed: $O(1)$ (no coordination cost)
- Decision variance: depends only on agent's noise distribution

### Committee with Majority Vote

- **Baseline**: Condorcet Jury Theorem
  - If $n$ independent agents each have accuracy $p > 0.5$, then majority vote
    accuracy $\to 1$ as $n \to \infty$
  - This is the "committees are good" baseline case
- **Realism via correlation**: agents' errors are correlated $\rho$ (shared bias,
  shared training, information cascades)
  - Accuracy degrades as $\rho$ increases
  - Often drops below solo expert's $p$ when $\rho$ is high
- **Coordination cost**: decision time $T(n)$ grows with committee size (deliberation
  rounds, negotiation overhead)
  - Metric: $\text{quality-per-unit-time} = \frac{\text{accuracy}}{T(n)}$, not just
    final accuracy

### Committee with Weighted Aggregation

- Agents weighted by competence $w_i$ (Bayesian pooling)
- Generalizes both extremes:
  - All weight to one agent $w_1 = 1, w_{i>1} = 0$: reduces to autocracy
  - Equal weights $w_i = 1/n$: pure committee majority
- Agents update beliefs sequentially after observing others (DeGroot opinion dynamics
  or French-Harary-Zeeman model)
  - Groupthink parameter $\lambda$ shrinks effective independence

### Agent-Based Model

- Each agent has noisy signal $s_i = \theta + \epsilon_i$ about true state $\theta$,
  with $\epsilon_i \sim N(0, \sigma_i^2)$
- Vary $\sigma_i$ to represent expertise heterogeneity
- Correlation $\rho$ controls error correlation across agents (shared bias or cascade
  effects)
- Communication topology:
  - Fully connected: everyone talks to everyone
  - Hub-and-spoke: one leader synthesizes opinions
  - Hierarchical: staged information flow

## Key Examples

- **Small independent committee beats solo expert**: $n = 5$ uncorrelated agents with
  $p = 0.7$ each beats solo expert with $p = 0.9$
  - Accuracy: committee -> 98.3%, solo -> 90%
  - But small time cost $T(5)$, so quality-per-unit-time still high
- **Correlated errors kill the committee**: same setup with error correlation
  $\rho = 0.8$ (shared training/bias)
  - Committee accuracy drops to 87%, now worse than solo expert
  - Captures "too many cooks with the same recipe"
- **Coordination cost crossover**: committee of 20 agents
  - With $T(20) = 20$ (units of time), quality-per-unit-time drops below solo expert
    even if committee accuracy is higher
  - Shows when deliberation overhead dominates
- **One bad apple**: solo expert performance robust to internal noise, but one
  incompetent committee member (low $\sigma_i$) with high weight corrupts group
  decision
  - Solo: insensitive to outliers
  - Committee: sensitive to one loud fool unless you weight by competence

## Questions

1. At what error correlation $\rho^*$ does committee accuracy fall below solo expert
   $p$, as a function of committee size $n$ and individual accuracy $p$? Is there a
   closed-form threshold?
2. How does sequential Bayesian updating (agents observe others' beliefs) change the
   correlation structure vs. simultaneous voting? Does it improve or worsen
   groupthink?
3. Can symbolic regression or analytic approximation capture the non-monotonic
   quality-per-unit-time curve as a function of $(n, \sigma_i, \rho, T(n))$, or is
   simulation the only route?
4. Do these patterns generalize to non-Gaussian signal noise (heavy tails, bimodal
   distributions) or only to Gaussian settings?

## Research Topics

- **Theoretical foundation**: derive threshold $\rho^*$ analytically; connect to
  Condorcet Jury Theorem and opinion dynamics literature
- **Simulation design**: agent architecture (state, signal, aggregation), parameter
  sweep ranges, and validation metrics
- **Coordination cost model**: is decision time linear in $n$, superlinear (debates
  get longer), or is there a plateau?
- **Robustness and outliers**: sensitivity analysis for incompetent agents,
  heterogeneous noise levels, and adversarial minority opinions
- **Extensions**: multi-round voting, vetoes/overrides by a leader, partial
  information (agents don't see all others' opinions), and adaptive topology (agents
  select conversation partners)

## Next steps

- [ ] Look for related research (Condorcet, opinion dynamics, organizational theory)
- [ ] Finalize the agent-based model and parameter space
- [ ] GP to review and approve the formalization
- [ ] Hack a quick end-to-end simulation (1-2 days) showing non-monotonic curve on
      committee size vs. accuracy
- [ ] Break into phases and milestones
- [ ] Execute one step at a time

## Implementation plan

- **Milestone 1**: Build baseline agent model and Condorcet majority vote
  - Implement agent with noisy signal and majority vote aggregation
  - Sweep committee size $n$ and individual accuracy $p$, verify Condorcet theorem
    (accuracy -> 1 as $n \to \infty$ for $p > 0.5$)
  - Output: validated simulation confirming textbook result
- **Milestone 2**: Add error correlation and measure degradation
  - Add correlation knob $\rho$ (shared bias or cascade model)
  - Measure committee accuracy as function of $(n, p, \rho)$
  - Find threshold $\rho^*$ where committee accuracy drops below solo expert
  - Output: characterization of crossover point
- **Milestone 3**: Integrate coordination cost and quality-per-unit-time
  - Assign time cost $T(n)$ to decision (linear, superlinear, or empirical model)
  - Compute quality-per-unit-time and identify optimal committee size
  - Output: non-monotonic curve showing sweet spot for committee size
- **Milestone 4**: Add sequential updates and opinion dynamics
  - Implement DeGroot or French-Harary-Zeeman opinion dynamics
  - Measure groupthink parameter $\lambda$ effect on effective independence
  - Compare simultaneous voting vs. sequential updating
  - Output: learning curves showing convergence and final opinion distribution
- **Milestone 5**: Robustness and outliers
  - Test sensitivity to incompetent agents (high $\sigma_i$), adversarial minority,
    vetoes
  - Compare weighted aggregation vs. majority vote
  - Output: robustness profiles for different decision structures

## References

- Surowiecki, _The Wisdom of Crowds_. (2004)
- DeGroot, _Reaching a Consensus_, Journal of the American Statistical Association.
  (1974)
- French et al., _The anatomy of a controversy_. (1968)
- Condorcet, _Essai sur l'application de l'analyse a la probabilite des decisions
  rendues a la pluralite des voix_. (1785)
