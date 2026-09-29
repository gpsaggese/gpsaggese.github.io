# Noesis: Resolving Periodic-Auction vs. Interactive-Latency Mismatch

## Status
- **Status**: draft
- **Complete Specs**: 10%
- **Assignee**: TBD

## Core Idea
- `NoesisMarket` clears every `T` minutes (default 5), but a large share of
  real LLM demand, chat agents, coding assistants, is interactive and needs
  a sub-second response. The paper (`papers/Noesis/04_noesis_market.tex`,
  `sec:rounds`) currently only gestures at this as a future "hybrid design"
  in the open-questions list, without resolving it in the main design
- This idea is to specify, in the paper's own notation, how an individual
  request inside a fulfillment window is actually served, i.e., that `T`
  governs when capacity and price are cleared, not the latency of any one
  request, and what happens on overage within a window or for a request
  that arrives between rounds
- The mismatch is load-bearing: without an explicit answer, it is not clear
  the protocol can serve its primary stated use case (autonomous agents
  consuming inference synchronously) at all

## Formalization
- Two-layer mechanism:
  - **Capacity layer** (unchanged): the existing batch call auction clears
    `(N_tasks, C_level, L_max, R_min, P)` every `T` minutes, exactly as in
    Problem `prob:clearing`
  - **Spot layer** (new): `NoesisServer` serves each individual request
    synchronously against the buyer's already-cleared `N_tasks` balance for
    the active contract, decrementing the balance per request
- Overage rule: once a contract's `N_tasks` balance is exhausted mid-window,
  the buyer either (a) waits for the next auction round, or (b) is served
  at a posted overage price `P_overage >= P` (analogous to real-time prices
  exceeding day-ahead prices in electricity markets, paper
  `sec:related_commodity_markets`)
  ```text
  spot_price(t) = P              if balance(contract, t) > 0
                = P_overage      otherwise
  ```
- Between-round arrivals: a request from a buyer with no active contract is
  either rejected, queued for the next round, or served at a standing
  posted price outside the auction entirely; which of these three the
  design adopts is the main open decision this idea has to settle

## Key Examples
- **Within-balance request**: a chat-agent request arrives mid-window; it
  is served immediately against the buyer's standing contract balance, with
  no wait for the next round
- **Overage request**: the same buyer's traffic spikes and exhausts its
  `N_tasks` balance before the window closes; the next request is served at
  `P_overage` rather than blocked
- **Between-round arrival**: a new buyer with no contract yet submits a
  request 30 seconds after a round closed; the design must specify whether
  it waits up to `T` minutes, is queued, or is served at a posted price

## Questions
1. Does the spot layer preserve the uniform-price fairness property of
   Proposition 1 (existence of a clearing price), or does it reintroduce
   the latency-arbitrage problem batch auctions were chosen to avoid
   (paper `sec:related_auctions`, citing Budish et al.)?
2. How should `P_overage` be set: a fixed multiple of `P`, a secondary
   auction, or a posted price set by the seller?
3. Should a between-round arrival with no contract be rejected, queued, or
   served at a standing posted price, and how does that choice affect the
   incentive to bid honestly in the next round?
4. Does this design need per-seller commitments about how much of their
   cleared `N_tasks` they can serve synchronously versus in a batch, given
   that provider-side latency still varies?

## Research Topics
- Market microstructure: batch vs. continuous auctions, and hybrid
  designs that layer a fast market over a periodic one, mirroring the
  day-ahead/real-time split in electricity markets
  (paper `sec:related_commodity_markets`)
- Frequent batch auctions and latency arbitrage (Budish et al.), and
  whether the argument for batching over continuous markets still holds
  once a spot layer is added underneath it
- Overage and congestion pricing schemes from other metered commodity
  markets (electricity, bandwidth)

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan
- Milestone 1: specify the spot layer and overage rule
  - Write up the two-layer mechanism above in the paper's notation, and
    decide the between-round-arrival policy
  - This is the result: a fully specified hybrid mechanism, not just a
    named future direction

- Milestone 2: simulate it
  - Add the spot path to the auction simulator from
    [[draft.Noesis_Prototype_Validation]], and test it against a synthetic
    interactive-latency workload (a stream of requests arriving between
    rounds)
  - This is the result: simulated evidence the design serves interactive
    requests without breaking the uniform-price property

- Milestone 3: update the paper
  - Replace the deferred "hybrid design... not evaluated" framing in
    `papers/Noesis/04_noesis_market.tex` (`sec:rounds`) and
    `09_open_questions.tex` (`sec:open_questions_market`, item 2) with the
    specified mechanism and simulation result
  - This is the result: the latency mismatch resolved in the main design
    rather than left as an open question

## References
- Noesis paper: `papers/Noesis/04_noesis_market.tex`
  (`TODO(gp)` comment at the end of `sec:rounds`)
- Related ideas: [[draft.Intelligence_Market]],
  [[draft.Noesis_Prototype_Validation]]
- Budish, E., Cramton, P., Shim, J., _The High-Frequency Trading Arms Race:
  Frequent Batch Auctions as a Market Design Response_. (2015)
