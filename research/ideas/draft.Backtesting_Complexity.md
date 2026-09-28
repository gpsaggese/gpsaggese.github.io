# Backtesting Complexity and Overfitting

**Status:** draft | **Specs:** 10% | **Assignee:** —

## Core Idea

When a researcher or quantitative analyst backtests many strategies and selects
the best performer, the effective complexity of the chosen strategy is far
larger than the VC dimension of the strategy class alone. The search over
strategies adds a multiplicative complexity penalty that explains why
impressive backtest performance so often fails to materialize in deployment.

## Formalization

Effective VC dimension accounting for strategy search:

\[
VC_{\text{eff}} = VC(\mathcal{H}) + \log(N_{\text{strategies tested}})
\]

A hedge fund that backtests 10,000 trading strategies on 20 years of data and
selects the top performer may find a strategy with Sharpe ratio 3.0
in-sample. But the effective VC dimension includes the search over 10,000
strategies, dramatically inflating the risk of overfitting. The "published"
strategy may have \(VC_{\text{eff}} \gg VC(\mathcal{H})\).

## Key Examples

- **Quantitative trading**: Strategies with impressive backtest performance
  degrade immediately upon deployment—a phenomenon directly explained by
  ignoring \(C_{\text{search}}\) in complexity estimates.
- **Scientific replication crisis**: Many published findings fail to
  replicate because researchers (or the scientific process itself) test many
  hypotheses but only report the significant ones. This is a failure to
  account for \(C_{\text{search}}\) in reported p-values.
- **Medical trial design**: Testing multiple subgroups, endpoints, or
  analysis methods inflates the effective complexity of the "discovery."

## Provocative Questions

1. Is the "replication crisis" in science fundamentally a failure to account
   for \(C_{\text{search}}\) in reported p-values?
2. Can we use learning theory to derive optimal publication policies? For
   instance, should journals require authors to report how many analyses they
   attempted?
3. If a strategy was discovered by "accident" (zero search cost) versus
   intensive backtesting (high search cost), should we trust the accidental
   discovery more?
4. Does cross-validation protect against overfitting the research process
   itself, or does it only protect against overfitting within a single model?
5. If we condition on "this backtest passed," we've selected from a larger
   hypothesis class than we realize—can we formalize this selection bias
   using VC theory?

## Research Topics

- VC dimension of trading strategies
- Multiple testing corrections via learning theory
- Overfitting detection using Rademacher complexity
- Formalizing selection bias in VC theory
- Designing publication and disclosure standards based on learning theory

## References

- Derived from *Research_plan/paper.tex* (Section: MDL Extensions /
  Backtesting Complexity)
