# Wealth Redistribution Policy in an AI-Driven Abundance Economy

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- With AI creating an unbounded amount of wealth, what should happen to it?
- One idea: recognize that getting rich involves luck, and just spread the
  wealth
- Proposal: cap net worth at $10M per person (adjusted for inflation), since
  this is enough to live a very nice life

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. Why do wealthy people often behave erratically once they have far more
   money than they need?

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: formalize the cap policy and build a toy wealth simulation
  - Define a formal model where agents accumulate wealth through a mix of
    luck (random shocks) and skill (deterministic drift), with a net-worth
    cap of $10M (inflation-adjusted) that triggers redistribution of
    everything above the cap
  - Specify a redistribution mechanism for wealth above the cap (e.g.,
    taxed at 100% above the cap and distributed as a UBI-style transfer or
    paid into a public AI-dividend fund)
  - Implement an agent-based simulation (Python) with heterogeneous agents
    and a pluggable redistribution policy layer
  - This is the result: a working agent-based simulation of wealth
    accumulation with a pluggable cap-and-redistribute policy

- Milestone 2: simulate policy variants and measure macro effects
  - Run the simulation under several variants: no cap (baseline), hard cap
    at $10M, soft cap (progressive taxation approaching a limit), and
    varying AI-driven wealth-growth rates
  - Measure outcomes per variant: Gini coefficient over time, total
    welfare, capital-formation rate, and simulated effort/risk-taking
    incentives
  - This is the result: comparative plots showing how each policy variant
    affects inequality, welfare, and incentives under AI-driven abundance
    growth

- Milestone 3: calibrate against real-world data and historical precedent
  - Calibrate simulation parameters using real wealth-distribution data
    (e.g., Forbes billionaire list, Federal Reserve Survey of Consumer
    Finances)
  - Research historical analogues: mid-20th-century US high marginal tax
    rates, Nordic wealth taxes, estate taxes, and documented capital
    flight in response to wealth taxes
  - Add an agent "exit" response (capital flight or relocation) to the
    simulation and re-run the Milestone 2 variants as a sensitivity check
  - This is the result: a calibrated simulation with a capital-flight
    sensitivity analysis grounded in historical wealth-tax precedent

- Milestone 4: analyze behavioral effects and write a policy brief
  - Survey behavioral-economics and psychology literature on why wealth
    holders behave erratically once wealth far exceeds their needs
    (hedonic adaptation, status-seeking, loss aversion at extreme wealth)
  - Synthesize the simulation results, historical case studies, and
    behavioral literature into a policy brief covering: cap enforcement,
    valuation of illiquid assets (e.g., equity), avoidance loopholes, and
    the choice of redistribution channel
  - This is the result: a policy brief stating the proposed mechanism, its
    expected macro and behavioral effects, and its main open risks

## References

- Hacker News discussion: https://news.ycombinator.com/item?id=49317760
