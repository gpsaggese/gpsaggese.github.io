# Noesis Marketing Plan

- TODO(gp):
  - Supply:
    - Talk to token providers to get blocks
  - Demand
    - Google adsense OpenRouter, tokens
    - Students
    - Promo
    - Refer and get discount
    - Crypto people from my linkedin

## Goal

- Lay out how to get initial supply and demand liquidity into \Noesis{} once
  [[plan.Noesis]]'s Milestone 9 (`NoesisPlatform`) exposes a public API, so the
  market is not just correct (Milestones 1-8) but actually has bids and asks flowing
  through it
- Background: [[plan.Noesis]], `papers/Noesis/01_introduction.tex` (protocol
  overview), `papers/Noesis/02_market_design.tex` (who the two sides are),
  `papers/Noesis/07_state_of_the_art.tex` (competitive landscape)

## The Cold-start Problem

- \Noesis{} is a two-sided market: a buyer only shows up if there is liquidity on the
  ask side, and a seller only shows up if there is order flow on the bid side
  - Neither side has a reason to be first
- The plan below breaks the deadlock the way most two-sided marketplaces do: fake the
  missing side first (`NoesisServer` as its own initial seller), then peel off real
  supply and demand once there is something to point at
  - E.g., Uber seeding driver supply before rider demand existed at scale, OpenRouter
    aggregating existing provider price lists instead of waiting for providers to
    integrate first

## Demand Side: Getting Buyers

### Remove the Reason to Wait for Real Sellers

- `NoesisServer` PR1's passthrough proxy already wraps 2-3 real providers directly
  ([[plan.Noesis]] Milestone 2)
- Until Milestone 9 has enough external sellers, let `NoesisServer` itself post the
  asks (at cost plus a small spread) that clear `NoesisMarket`'s auction, so a buyer
  gets a real fulfilled contract from day one
  - This is the standard "be the supply until supply shows up" move, not a permanent
    design choice: real seller asks should always be preferred once they exist, per
    `NoesisMarket`'s uniform-price clearing rule

### Who to Target First

- Agent framework developers (LangChain, CrewAI, AutoGen-style projects): low
  switching cost if \Noesis{} exposes an OpenAI-compatible endpoint, and they are the
  buyers Section~\ref{sec:market_participants} of
  `papers/Noesis/02_market_design.tex` describes as multi-agent systems delegating
  sub-tasks
- Cost-sensitive, high-volume batch workloads: document-summarization and eval
  pipelines, which the paper's own `Difficulty-aware routing` and `Answer fusion`
  contributions (`papers/Noesis/01_introduction.tex`) are built to save money on
- A campus pilot cohort: DATA605 and MSML610 students, who already build LLM-backed
  class projects in this repo, are a free, willing, technically literate first batch
  of buyers
  - This also produces the cost/quality frontier data Milestone 4 needs

### Channels

- Publish the routing/fusion cost-quality results (Milestones 4 and 6) as a short
  writeup targeting r/LocalLLaMA, r/MachineLearning, and Hacker News: a measured
  number ("N percent cheaper at the same quality") travels further than the protocol
  description alone
- Post the paper itself (`papers/Noesis/paper.tex`) to arXiv and submit to an
  LLM-systems or market-design workshop
- Ship a drop-in OpenAI-compatible SDK/adapter so integrating \Noesis{} costs one
  config change, matching the "single-API ease of integration" strength
  `papers/Noesis/01_introduction.tex`'s comparison table already grants gateways

## Supply Side: Getting Sellers

### Fold in Prospective Sellers Automatically

- `papers/Noesis/07_state_of_the_art.tex` already names serverless inference
  providers (Together AI, Fireworks AI, Groq, Baseten, Fal.ai) as prospective
  sellers, not competitors: each posts a per-token price and meters its own latency
  today
- Write an adapter that turns a provider's posted price list into a standing ask
  automatically, so onboarding a seller does not require that provider to build
  anything against `NoesisMarket`'s bid/ask schema
  - Lowers the integration bar from "adopt our contract schema" to "we already scrape
    your price page"

### Who to Target First

- The serverless inference providers above: easiest technical integration, already
  competing on price and already publish an SLA
- Idle-capacity holders: open-weight model hosts and, for the campus pilot, students
  or labs with spare GPU time who can post an ask for course-project reputation or
  platform credit instead of cash
- Bid/ask compute marketplaces (Vast.ai, Akash) surveyed in
  `papers/Noesis/07_state_of_the_art.tex`: they already have sellers used to bidding,
  just not on a (capability, latency, reliability) bundle yet

### Lower the Reputation Cold-start Penalty

- `NoesisMarket` PR3's reputation loop ([[plan.Noesis]]) has nothing to go on for a
  brand-new seller, which discourages the first sellers from joining
- Seed a new seller's initial reputation from a public benchmark (e.g.,
  artificialanalysis.ai, already named as a capability-measurement source in
  `papers/Noesis/01_introduction.tex`) instead of starting every seller at zero trust

## Sequencing

- **Phase 0 (self-liquidity)**: `NoesisServer` is its own seller; `NoesisMarket`
  clears against it; buyers get real fulfilled contracts with zero external sellers
  (depends on [[plan.Noesis]] Milestone 3)
- **Phase 1 (campus pilot)**: recruit DATA605/MSML610 students as buyers and a
  handful of idle-capacity holders as the first real sellers, running alongside Phase
  0's self-liquidity as a backstop
- **Phase 2 (design partners)**: onboard 2-3 serverless inference providers through
  the price-list adapter and 5-10 external buyers (agent-framework projects); publish
  the pricing-dissemination feed (`NoesisMarket` PR4) publicly as a transparency
  signal
- **Phase 3 (open beta)**: publish the cost/quality and reputation results; open
  signups; retire Phase 0's self-liquidity as real ask volume covers demand

## Positioning

- One-line pitch: \Noesis{} is the only LLM market that prices, verifies, and holds
  sellers accountable to capability, latency, and reliability, not just a token count
- This claim is exactly the empty cell `papers/Noesis/07_state_of_the_art.tex`
  identifies: every ingredient (auction pricing, bundled quality, metered
  fulfillment, reputation feedback) already exists somewhere in the industry
  landscape, but no surveyed system combines all four
- Lead marketing content with that comparison table (`tab:sota_comparison`), not with
  the protocol's internals

## Metrics

- Design-partner buyers and sellers onboarded (Phase 2 target)
- Contracts cleared per round and matched volume, not just bids submitted
- Fraction of cleared volume filled by real sellers vs. `NoesisServer` self-liquidity
  (should trend toward zero self-liquidity over Phases 1-3)
- Pricing-dissemination feed subscriber count (`NoesisMarket` PR4)

## Risks

- Thin liquidity widens the bid/ask spread in early rounds, which is the same open
  question 5 in [[plan.Noesis]]'s `NoesisMarket` section (batch vs. hybrid spot
  market for latency-sensitive buyers)
- Real, non-synthetic traffic through the campus pilot runs into `NoesisServer`'s
  open question 4 (PII/data-handling safeguards): keep the pilot on synthetic/test
  traffic until that is resolved
- Serverless providers may see the price-list adapter as unwanted disintermediation
  rather than a free sales channel; frame outreach around incremental demand, not
  price competition with their own storefront

## References

- [[plan.Noesis]]: implementation roadmap and PR list this plan assumes
- `papers/Noesis/01_introduction.tex`: protocol overview and comparison table
- `papers/Noesis/02_market_design.tex`: demand-side and supply-side participant
  dimensions
- `papers/Noesis/07_state_of_the_art.tex`: competitive landscape and the gap
  \Noesis{} occupies
- 2021, Chen, "The Cold Start Problem: How to Start and Scale Network Effects"
- 2003, Rochet et al., "Platform Competition in Two-Sided Markets"