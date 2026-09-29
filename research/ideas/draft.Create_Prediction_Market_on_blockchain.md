# Create a Prediction Market on Blockchain

## Status
- **Status**: draft
- **Complete Specs**: 15%
- **Assignee**: TBD

## Core Idea
- Build a small, working prediction market as a smart contract: users bet on
  the outcome of a future event, the contract prices shares via an automated
  market maker (e.g., LMSR), and pays out based on a resolved outcome
- The interesting research question isn't "can you deploy a betting contract"
  (that's well-trodden), but **how much does market accuracy degrade under a
  weak or manipulable oracle**, i.e., how sensitive is the market's
  information-aggregation property to the trustworthiness of whoever reports
  the real-world outcome

## Formalization
- Logarithmic Market Scoring Rule (LMSR) cost function for a binary market
  with outcome shares `q_yes`, `q_no`, liquidity parameter `b`:
  ```text
  C(q) = b * log(exp(q_yes / b) + exp(q_no / b))
  ```
- Market-implied probability: `p_yes = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))`

## Key Examples
- **Well-behaved oracle**: a single trusted reporter resolves the market;
  measure how closely `p_yes` over time tracked the eventual outcome
  (calibration)
- **Adversarial oracle**: reporter has a stake in one outcome; measure how
  much manipulated resolution distorts pre-resolution trading incentives
- **Decentralized oracle (e.g., UMA-style dispute mechanism)**: compare
  accuracy/cost tradeoff against the single-trusted-reporter case

## Questions
1. How does resolution-source trust interact with market liquidity `b`:
   does a thin market amplify the effect of a bad oracle more than a deep one?
2. Can on-chain dispute mechanisms (bond-and-challenge) recover most of the
   accuracy lost to a first attempt at manipulation, and at what gas cost?
3. Is a simple LMSR AMM sufficient, or do the interesting failure modes only
   show up with order-book-style markets?

## Research Topics
- Automated market maker design (LMSR, CPMM) for binary/categorical markets
- Oracle design patterns (single trusted reporter, optimistic oracle,
  decentralized dispute resolution)
- Empirical calibration analysis of existing prediction markets (Polymarket,
  Manifold) as a baseline to compare a toy implementation against

## Next steps
- [ ] Look for related research (existing prediction-market mechanism-design
  literature, Polymarket/Augur/Manifold postmortems)
- [ ] Implement a minimal LMSR contract on a testnet
- [ ] Simulate trusted vs. adversarial oracle scenarios
- [ ] Break the problem into phases (contract, simulation, empirical
  comparison)

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Hanson, R. (2003). _Combinatorial Information Market Design_ (LMSR)
