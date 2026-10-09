# Measuring Human Ability to Predict Events: Skill or Luck

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- People are ranked constantly by their apparent ability to call outcomes:
  pundits, forecasters, analysts, hedge fund managers
  - The ranking is almost always computed on the realized track record, with
    no correction for how many candidates were searched over to find the
    winner
- Central claim: an observed track record is skill plus luck plus selection,
  and the third term is the one that is systematically ignored
  - With $N$ candidates and a short history, the best track record in the pool
    is mostly a record of the search, not of the forecaster
- Proposal: apply a capacity-control framing borrowed from statistical
  learning theory
  - Treat "picking the best forecaster out of $N$" as fitting a hypothesis
    class of size $N$
  - Charge the apparent skill a penalty that grows with the size of the search
- This is the human-forecaster version of the same decomposition applied to
  agents in `draft.Skill_vs_Luck_in_Agent_Benchmarks.md` and to funds in
  `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`, and it shares the
  multiple-testing machinery of `draft.Backtesting_Complexity.md`

## Formalization

- Let forecaster $j$ issue probabilistic predictions $p_{j,i}$ on events $i$
  with binary outcomes $y_i$
- Scoring uses a proper scoring rule, so honest reporting is optimal:
  ```text
  Brier_j = (1/n) * sum_i (p_{j,i} - y_i)^2
  ```
- Decompose the score into calibration, resolution, and irreducible
  uncertainty (Murphy decomposition):
  ```text
  Brier = reliability - resolution + uncertainty
  ```
  - Resolution is the part that measures discrimination, i.e., actual
    forecasting ability
  - Calibration alone can be achieved by always predicting the base rate
- Selection-adjusted skill, in the spirit of an effective VC dimension:
  ```text
  skill_eff = observed_skill - penalty(N_candidates, n_events)
  penalty ~ sqrt(log(N_candidates) / n_events)
  ```
  - The $\log N$ term is the price of searching over candidates
  - The $\sqrt{1/n}$ term is the price of a short track record
- Equivalent framing via multiple testing: report the deflated statistic after
  adjusting for the number of candidates examined, as done for the deflated
  Sharpe ratio in finance
- Persistence test, which is the strongest evidence and needs no penalty:
  ```text
  corr(rank in period 1, rank in period 2)
  ```
  - Skill implies persistence across independent periods
  - Luck implies rank correlation near zero
- Difficulty must be controlled: forecasters do not answer the same questions,
  so an item-response model estimating question difficulty and discrimination
  is needed before scores are comparable

## Key Examples

- **Pundit track record**: a commentator with 8 correct calls out of 10 looks
  skilled until the pool is counted, and out of 1000 commentators the best is
  expected to hit 9 or 10 by chance alone
- **Hedge fund manager ranking**: top-decile performance over three years, with
  survivorship bias removed and the size of the manager universe charged
  against the score, typically collapses toward the mean
- **Superforecaster case**: the Good Judgment Project result that a small
  group shows persistent, out-of-sample skill is the existence proof that the
  effect is not always zero, and gives a positive control
- **Question difficulty confound**: two forecasters with identical Brier
  scores where one answered only near-certain questions, so raw score
  comparison is meaningless without difficulty adjustment
- **Failure mode**: resolution criteria that are ambiguous after the fact, so
  the forecaster and the scorer disagree on whether the event happened, which
  quietly inflates measured skill

## Questions

1. What is the right penalty functional? The $\sqrt{\log N / n}$ form is a
   uniform-convergence bound, but the candidates are highly correlated (they
   see the same news), so the effective $N$ is much smaller than the nominal
   one. How is it estimated?
2. How many resolved questions are needed to separate the top forecaster from
   the median at a stated confidence, and is that number achievable in
   practice?
3. Does measured skill transfer across domains (geopolitics to markets to
   sports), or is it question-class specific?
4. Can luck be reduced by aggregation rather than selection, i.e., is a
   weighted crowd average reliably better than the best individual, and by
   how much?
5. If almost all observed ranking is selection, then hiring, promotion, and
   allocation decisions based on track record are close to random, which is a
   direct claim about how forecasting talent should be sourced

## Research Topics

- **Scoring**: proper scoring rules, Murphy decomposition, and calibration vs
  resolution as separate axes
- **Selection correction**: effective number of independent candidates,
  false-discovery-rate control, deflated performance statistics
- **Persistence and transfer**: split-period rank correlation, cross-domain
  rank correlation
- **Difficulty modeling**: item-response theory to make scores comparable
  across non-overlapping question sets
- **Data sources**: Good Judgment Open, Metaculus, prediction market prices,
  and published analyst forecasts, all with resolution dates and full
  candidate universes so the selection term can be computed
- **Aggregation**: extremized crowd averages as a baseline that any individual
  claim of skill must beat

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: assemble a forecast panel
  - Pull a public forecasting dataset with per-forecaster, per-question
    predictions and resolutions
  - Build the (`forecaster`, `question`, `prediction`, `outcome`, `date`)
    panel, including forecasters who dropped out, to avoid survivorship bias
  - This is the result: a clean panel plus a documented count of the full
    candidate universe

- Milestone 2: score and decompose
  - Compute Brier scores and the Murphy decomposition per forecaster
  - Produce a naive leaderboard as the baseline to be corrected
  - This is the result: raw rankings with calibration and resolution split out

- Milestone 3: charge the selection
  - Estimate the effective number of independent candidates via the
    correlation structure of predictions
  - Apply the penalty and report selection-adjusted rankings with intervals
  - This is the result: a corrected leaderboard showing which ranks survive

- Milestone 4: persistence, difficulty, and transfer
  - Split the history into independent periods and measure rank correlation
  - Fit an item-response model for question difficulty and rescore
  - Measure cross-domain transfer for forecasters active in several categories
  - This is the result: an estimate of how much of observed forecasting skill
    is real, and how many questions are needed to detect it

## References

- Tetlock and Gardner, _Superforecasting: The Art and Science of Prediction_.
  (2015)
- Mellers et al., _Psychological Strategies for Winning a Geopolitical
  Forecasting Tournament_. (2014)
- Murphy, _A New Vector Partition of the Probability Score_. (1973)
- Bailey and Lopez de Prado, _The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality_. (2014)
- Vapnik, _The Nature of Statistical Learning Theory_. (1995)
