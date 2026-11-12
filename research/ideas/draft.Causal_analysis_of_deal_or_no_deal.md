# Causal Analysis of Deal or No Deal

## Status

- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea

- Analyze the optimality of contestants' decisions in the game show Deal or No
  Deal
  - Each round the contestant chooses between accepting the banker's offer
    (deal) and opening more cases (no deal)
  - Round-level data on real contestants exists from several country editions
- Build an optimal strategy as the benchmark:
  - Compute the average amount of money that can be won by playing on
  - Compare it with the best offer
  - Add a risk appetite parameter, so that the benchmark covers risk-neutral
    and risk-averse contestants
- Measure how far observed decisions deviate from the benchmark, and which
  factors explain the deviation
  - The show is a natural experiment for decision-making under risk, mainly
    studied in behavioral economics

## Formalization

- Game state at round $t$:
  - $R_t$: set of case values still in play (including the contestant's own
    case)
  - $b_t$: banker's offer
  - $E_t = \frac{1}{|R_t|} \sum_{v \in R_t} v$: expected value of a random case
    in play
- Contestant decision at round $t$:
  - Deal: receive $b_t$ and the game ends
  - No deal: open more cases and move to round $t + 1$
- Risk-neutral optimal strategy via backward induction:
  ```
  V_t = max(b_t, E[V_{t+1} | R_t])
  ```
  - Deal is optimal if and only if $b_t \geq E[V_{t+1} | R_t]$
- Risk appetite via a utility function $u$ with parameter $\rho$:
  ```
  V_t = max(u(b_t), E[V_{t+1} | R_t]),  u(x) = x^(1 - rho) / (1 - rho)
  ```
  - $\rho = 0$: risk-neutral
  - $\rho > 0$: risk-averse
  - $\rho < 0$: risk-seeking
- Deviation from the benchmark:
  - Compare the observed deal/no-deal choice with the optimal choice for a given
    $\rho$
  - Estimate $\rho$ that best explains the observed choices

## Key Examples

- **Risk-neutral rejection (illustrative)**: three cases remain with values 1,
  100, and 10000, so $E_t = 3367$; a hypothetical offer of 2000 is below
  $E_t$, so a risk-neutral contestant keeps playing
- **Risk-averse acceptance (illustrative)**: with the same hypothetical offer
  of 2000, a sufficiently risk-averse contestant accepts the deal, since the
  10000 case has only a one-in-three chance
- **Identification edge case (illustrative)**: a single rejection of an offer
  below $E_t$ is consistent with both risk neutrality and risk seeking, so one
  decision alone cannot identify $\rho$

## Questions

1. Which factors explain deviations from the optimal strategy (e.g., the ratio
   of offer to expected value, the cases opened in earlier rounds)?
2. Do prior outcomes change the decision beyond what the current offer and the
   remaining cases imply?
3. How much money do contestants leave on the table relative to the optimal
   strategy at the same risk appetite?
4. Can banker offers be predicted from the remaining case values, and does the
   predictability change what the contestant should do?
5. Is one risk appetite parameter enough, or does it differ by contestant,
   round, or country edition?

## Research Topics

- **Data sources: contestant decisions**:
  - Post, van den Assem, Baltussen, and Thaler replication files: round-by-round
    case eliminations, banker offers, and deal/no-deal choices from the Dutch,
    German, and US versions
    - Replication files are normally available from the AEA data archive
      (openICPSR)
  - ZBW Journal Data Archive, "Decision making under risk in Deal or No Deal
    (replication data)": directly downloadable replication package
    - https://journaldata.zbw.eu/dataset/activity/decision-making-under-risk-in-deal-or-no-deal
  - Kaggle, "Deal or No Deal game data":
    - https://www.kaggle.com/datasets/jaredstock/deal-or-no-deal-game-data
  - Hobbyist US data, "I figured out how Deal or No Deal works (kind of)":
    - More than 100 US games documented round by round, almost 800 rounds in
      total
    - Check the article series for a link to the data
    - https://towardsdatascience.com/i-figured-out-how-deal-or-no-deal-works-kind-of-875e63a8cef6
  - Other country editions studied in economics papers (data links not checked,
    some authors share data on request):
    - Australia: de Roos and Sarafidis; Mulino et al.
    - Italy: Bombardini and Trebbi; Botti et al.
    - UK and other editions: Blavatskyy and Pogrebna
    - Mexico: Deck, Lee, and Reyes
- **Optimal strategy**: backward induction over the case values, expected
  winnings of the optimal strategy versus the best offer
- **Risk appetite**: choice of utility family and estimation of the risk
  parameter from observed decisions
- **Banker offer model**: whether offers are predictable from the remaining
  case values
  - The small UK sample of Geddes et al. tested this and is useful mainly as an
    example, not as a dataset
- **Causal analysis**: separating the effect of the offer from the effect of
  earlier outcomes on the decision
- **Name collision**: the Lewis et al. "Deal or No Deal" dataset is a negotiation
  dialogue dataset from Facebook AI Research and has nothing to do with the TV
  show
  - Do not use it for this idea

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: get and clean the contestant data
  - Start from the Post et al. replication files or the ZBW package, which
    have the cleanest round-level offer and decision records
  - Build one round-level table: case values still in play, banker offer,
    decision
  - This is the result: a clean dataset with a documented schema, one row per
    contestant per round

- Milestone 2: compute the optimal strategy under risk neutrality
  - Implement backward induction over the remaining case values
  - Compute the average amount that can be won by playing on and compare it
    with the best offer
  - This is the result: per-round optimal decision and expected winnings for
    every game in the dataset

- Milestone 3: add risk appetite
  - Add a utility function with a risk parameter $\rho$
  - Estimate $\rho$ from observed decisions and check how the optimal strategy
    changes with $\rho$
  - This is the result: the optimal strategy as a function of risk appetite and
    an estimated $\rho$ for the contestants

- Milestone 4: analyze deviations from the optimal strategy
  - Compare observed decisions with the optimal ones
  - Test which factors explain the deviations (offer-to-expected-value ratio,
    earlier outcomes)
  - This is the result: a quantified estimate of the money lost to suboptimal
    decisions and of the factors behind it

## References

- Post, van den Assem, Baltussen, and Thaler, _Deal or No Deal? Decision Making
  under Risk in a Large-Payoff Game Show_. (2008)
  - Published in _American Economic Review_
  - https://www.aeaweb.org/articles?id=10.1257/aer.98.1.38
- Geddes et al., _S1_3 Deal or No Deal?_. (2009)
  - Published in University of Leicester _Physics Special Topics_
  - https://journals.le.ac.uk/index.php/pst/article/view/2303
- Lewis et al., _Deal or No Deal? End-to-End Learning for Negotiation
  Dialogues_. (2017)
  - Unrelated to the TV show: same name only
  - https://arxiv.org/abs/1706.05125
