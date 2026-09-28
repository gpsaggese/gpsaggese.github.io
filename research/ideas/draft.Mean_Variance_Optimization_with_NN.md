# Solving Mean-Variance Portfolio Optimization with a Neural Network

## Status

- **Status**: draft
- **Complete Specs**: 10%
- **Assignee**: TBD

## Core Idea

- Mean-variance optimization is a clean instance of amortized optimization (the
  general framing is in [[draft.Minimizing_N_Dimensional_Functions_with_NN]]): the
  problem instance is `c = (mu, Sigma)` and the minimizer is the portfolio weights
  `w`
- Unconstrained, this has a closed form, so a NN is pointless: the case for learning
  appears only with constraints that break the closed form (long-only, cardinality
  limits, turnover/transaction costs, integer lot sizes), where each instance
  otherwise requires a fresh numerical solve
- The stronger version is **decision-focused learning**: instead of estimating
  `(mu, Sigma)` and then optimizing (the classic two-step, where estimation error in
  `mu` dominates the result), train the network end-to-end from raw features to
  weights, with realized portfolio utility as the loss
- The point is that the two-step pipeline optimizes the wrong objective: minimizing
  prediction error on `mu` is not the same as maximizing portfolio utility, and a
  small `mu` error in the wrong direction is amplified by `Sigma^{-1}`

## Formalization

\[ w^{\*}(c) = \arg\min\_{w} \; w^{T} \Sigma w - \lambda \mu^{T} w \quad \text{s.t.}
\quad \mathbf{1}^{T} w = 1, \; w \in \mathcal{C} \]
- **Amortized variant**: train `h_phi(c) ≈ w*(c)` across sampled instances `c`, so
  inference is one forward pass instead of a solve
- **Decision-focused variant**: train `h_phi(features) -> w` directly against
  realized utility, with a differentiable solver layer (`cvxpylayers`, OptNet) inside
  the network so the constraint set `C` is satisfied by construction

## Key Examples

- **Constrained rebalancing**: long-only with turnover and transaction-cost
  penalties, where the closed form does not apply and a solver runs per period
- **Amortization payoff**: many instances per day (per-account or per-strategy
  portfolios) so the training cost is spread over many solves
- **Failure mode**: mean-variance solutions are notoriously unstable in `mu`, so a NN
  can look great in-sample and produce extreme weights out of sample
- **Failure mode**: the "learned" allocator is beaten by equal weight or minimum
  variance, which are famously hard baselines: any result that does not report them
  is uninformative

## Questions

1. Does end-to-end decision-focused training beat the classical predict-then-optimize
   pipeline out of sample, or does it just overfit the backtest?
2. Under which constraint sets is the amortized NN actually faster than simply
   calling a convex solver, once solver time per instance is measured honestly?
3. How much of any measured improvement is the NN vs. implicit regularization
   (shrinkage-like effects) that a simpler estimator would also give?

## Research Topics

- Differentiable convex optimization layers (`cvxpylayers`, OptNet) and
  decision-focused / "smart predict-then-optimize" learning
- Covariance estimation and shrinkage (Ledoit-Wolf) as the baseline the NN has to
  beat
- Walk-forward evaluation against equal-weight and minimum-variance baselines, with
  the pitfalls in [[draft.Backtesting_Complexity]]
- Performance attribution of the resulting portfolios, connecting to
  [[draft.Causal_Analysis_of_Hedge_Fund_Performance]]

## Next Steps

- [ ] Look for related research (deep portfolio optimization, decision-focused
      learning literature)
- [ ] Set up a walk-forward harness with equal-weight and minimum-variance baselines
      before training anything
- [ ] Implement the amortized variant on a constrained instance family and measure
      speed vs. a convex solver
- [ ] Implement the decision-focused variant and compare out-of-sample utility

## References

- Markowitz, H., _Portfolio Selection_. (1952)
- Amos, B., & Kolter, J. Z., _OptNet: Differentiable Optimization as a Layer in
  Neural Networks_. (2017)
- Agrawal, A., et al., _Differentiable Convex Optimization Layers_. (2019)
- Elmachtoub, A. N., & Grigas, P., _Smart "Predict, then Optimize"_. (2022)
- Zhang, Z., Zohren, S., & Roberts, S., _Deep Learning for Portfolio Optimization_
  (2020)
- DeMiguel, V., Garlappi, L., & Uppal, R., _Optimal Versus Naive Diversification_
  (2009)
