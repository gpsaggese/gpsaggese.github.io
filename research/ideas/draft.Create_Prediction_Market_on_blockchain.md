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

- Milestone 1: implement a minimal LMSR contract on testnet
  - Write a binary-outcome LMSR AMM contract in Solidity, implementing the
    cost function $C(q)$ and the price function $p_{yes}$ from the
    Formalization
  - Deploy it to a testnet with buy, sell, and redeem functions
  - Write unit tests verifying the pricing and cost invariants match the
    formalization
  - This is the result: a deployed, tested LMSR contract on testnet with
    verified pricing behavior

- Milestone 2: build the oracle layer and a trading simulator
  - Implement single-trusted-reporter resolution and a bond-and-challenge
    dispute mechanism modeled on UMA's optimistic oracle
  - Build a trading simulation harness with bots that trade on noisy private
    signals, to generate synthetic order flow into the contract
  - This is the result: working resolution mechanisms plus a simulator that
    can drive realistic trading activity

- Milestone 3: run trusted vs. adversarial oracle experiments
  - Simulate the well-behaved-oracle scenario and measure how closely
    $p_{yes}$ tracks the eventual outcome over time (calibration)
  - Simulate the adversarial-oracle scenario, where the reporter has a stake
    in one outcome, and measure the resulting pre-resolution price
    distortion
  - Vary the liquidity parameter $b$ to test whether thin markets amplify
    the effect of a bad oracle more than deep ones
  - This is the result: calibration and distortion measurements across
    oracle-trust and liquidity conditions

- Milestone 4: evaluate the dispute mechanism, benchmark against real markets
  - Measure how much of the manipulation-induced accuracy loss the
    bond-and-challenge mechanism recovers, and at what gas cost
  - Compare the simulated calibration curves against published Polymarket
    and Manifold calibration data
  - This is the result: a quantified accuracy-recovery/gas-cost tradeoff for
    the dispute mechanism and a benchmark comparison to real markets

## References
- Hanson, R. (2003). _Combinatorial Information Market Design_ (LMSR)
