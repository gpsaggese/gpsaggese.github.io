# Backtesting Complexity and Overfitting

## Status

- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- A researcher or quantitative analyst backtests many strategies and selects the
  best performer
  - The effective complexity of the chosen strategy is far larger than the VC
    dimension of the strategy class alone
- The search over strategies adds a multiplicative complexity penalty
  - This penalty explains why impressive backtest performance so often fails to
    materialize in deployment

## Formalization

- Effective VC dimension accounting for strategy search:
  $$
  VC_{\text{eff}} = VC(\mathcal{H}) + \log(N_{\text{strategies tested}})
  $$
- E.g., a hedge fund backtests 10,000 trading strategies on 20 years of data and
  selects the top performer
  - The top performer may show a Sharpe ratio of 3.0 in-sample
  - The effective VC dimension includes the search over 10,000 strategies, which
    dramatically inflates the risk of overfitting
  - The "published" strategy may have $VC_{\text{eff}} \gg VC(\mathcal{H})$

## Key Examples

- **Quantitative trading**: strategies with impressive backtest performance
  degrade immediately upon deployment
  - A phenomenon directly explained by ignoring $C_{\text{search}}$ in complexity
    estimates
- **Scientific replication crisis**: many published findings fail to replicate
  - Researchers (or the scientific process itself) test many hypotheses but only
    report the significant ones
  - This is a failure to account for $C_{\text{search}}$ in reported p-values
- **Medical trial design**: testing multiple subgroups, endpoints, or analysis
  methods inflates the effective complexity of the "discovery"

## Questions

1. Is the "replication crisis" in science fundamentally a failure to account for
   $C_{\text{search}}$ in reported p-values?
2. Can we use learning theory to derive optimal publication policies? For
   instance, should journals require authors to report how many analyses they
   attempted?
3. If a strategy was discovered by "accident" (zero search cost) versus intensive
   backtesting (high search cost), should we trust the accidental discovery more?
4. Does cross-validation protect against overfitting the research process itself,
   or does it only protect against overfitting within a single model?
5. If we condition on "this backtest passed," we've selected from a larger
   hypothesis class than we realize: can we formalize this selection bias using
   VC theory?

## Research Topics

- VC dimension of trading strategies
- Multiple testing corrections via learning theory
- Overfitting detection using Rademacher complexity
- Formalizing selection bias in VC theory
- Designing publication and disclosure standards based on learning theory

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: simulate the search penalty
  - Generate $N$ random strategies with no true edge on synthetic returns and
    select the best one by in-sample Sharpe ratio
  - Measure the out-of-sample Sharpe ratio of the selected strategy as $N$
    grows from 1 to 10,000
  - Fit the in-sample vs out-of-sample gap against $\log N$ and compare it
    with the $VC_{\text{eff}}$ prediction
  - This is the result: a curve of the overfitting gap as a function of $N$
    and a measure of how well the $\log N$ term explains it

- Milestone 2: apply to real data and formalize
  - Run the same procedure on a public price dataset with a parameterized
    strategy family (e.g., moving-average crossovers), counting every tested
    parameter setting as one strategy
  - Compute a Rademacher-complexity bound and a multiple-testing-corrected
    significance level for the selected strategy
  - Write the selection-bias statement in VC terms
  - This is the result: a corrected performance estimate for the selected
    strategy and a first formal statement of the selection bias

## References

- Derived from `Research_plan/paper.tex` (Section: MDL Extensions / Backtesting
  Complexity)
