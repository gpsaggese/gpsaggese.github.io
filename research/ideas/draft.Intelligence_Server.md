# Intelligence Server

## Status
- **Status:**: draft
- **Complete Specs:**: 15%

## Core Idea
- Build a lightweight LLM API gateway, modeled on OpenRouter, that proxies
  requests to multiple providers and models behind a single API, but with
  the explicit goal of logging every prompt/response pair to build a
  proprietary dataset for fine-tuning or evaluation
- This is the fulfillment and monitoring layer for
  [[draft.Intelligence_Market]]: when the market matches a contract, this
  server is what actually executes the requests and measures whether the
  delivered capability, latency, and reliability met the contract terms
- Beyond pure logging, add routing logic: pick which underlying model to use
  per request based on cost, latency, or task type, and measure whether
  routing decisions can be learned from the collected data itself
  (self-improving router)

## Formalization

### Difficulty-Aware Routing and Model Fusion
- The interesting research question inside the gateway is not "which model
  is best" but "how much intelligence does *this* request actually need"
- Train a difficulty estimator `q(prompt) -> required capability level`,
  then route: cheap or small model below a threshold, frontier model above
  it
- Fusion variant: send the request to several models and combine their
  answers (majority vote, verifier model, or a learned combiner) instead of
  picking one: trading cost for accuracy in the opposite direction from
  routing
- The estimator has to be *cheaper than the saving*, otherwise the router
  costs more than always routing to the strong model, which is the trap
  this design must measure explicitly
- Training signal comes for free from the logged data: for prompts answered
  by several models, whether the cheap model's answer matched the strong
  model's is a label for "this prompt was easy"
- Related: an arena backend ([[draft.Build_an_LLM_Arena]]) gives the
  pairwise quality judgments needed to score routing decisions

### Contract Fulfillment Monitoring
- For each request served under a [[draft.Intelligence_Market]] contract,
  record measured latency and whether the response met the contracted
  capability tier, then aggregate into `measured_reliability` over the
  contract window
- Flag a contract as violated when `measured_reliability < R_min` or
  `measured_latency > L_max` from the contract, and report the violation
  back to the market for pricing and reputation feedback

## Key Examples
- **Passive logging mode**: every request/response pair (with provider,
  model, latency, cost, and any user feedback signal) is stored, without
  changing behavior: just for future dataset construction
- **Routing experiment**: given a task classifier, route "easy" requests to
  a cheap or fast model and "hard" ones to a stronger model; measure cost
  savings versus quality degradation
- **Dataset reuse**: use the logged prompt/response pairs to fine-tune a
  smaller model to approximate a stronger model's outputs on the observed
  task distribution (distillation)
- **Fulfillment reporting**: a contract promising 99.9% reliability at
  "frontier-class" capability is served by three providers; the server
  detects that one provider's error rate breaches the threshold and reports
  the violation back to [[draft.Intelligence_Market]]

## Questions
1. How much of OpenRouter's value is the unified API versus the routing
   intelligence: is a simple pass-through logger already useful for
   dataset building, or does routing quality matter for data diversity?
2. Can request difficulty be estimated cheaply enough that difficulty-aware
   routing nets a real cost saving, and what fraction of traffic can be
   served by a small model with no measurable quality loss?
3. Fusion versus routing: for a fixed budget, is it better to pick one
   model well or to query several cheap models and combine their answers?
4. What privacy and data-handling safeguards are needed before logging real
   prompt/response pairs (PII scrubbing, opt-in or opt-out)?
5. Can a distilled small model trained on the logged data match the routed
   "best model per task" quality at a fraction of the cost?
6. How reliably can the server attribute a quality or latency shortfall to
   a specific provider, versus noise, so that fulfillment reporting to
   [[draft.Intelligence_Market]] is trustworthy enough to affect pricing?

## Research Topics
- LLM API gateway and proxy architecture (streaming passthrough, provider
  abstraction)
- Cost and latency-aware routing policies (bandit-style online learning)
- Prompt difficulty estimation and cascade or deferral policies (RouteLLM,
  FrugalGPT-style cascades)
- Answer fusion: majority vote, verifier-based selection, learned combiners
- Knowledge distillation from logged prompt/response pairs, with corpora
  from [[draft.Datasets_for_Training_and_Distilling_LLMs]]
- SLA monitoring and attribution: distinguishing genuine provider
  under-delivery from noise, for reliable fulfillment reporting

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan
- Milestone 1: build a minimal passthrough proxy with logging
  - Support 2-3 providers behind one API, define the storage schema for
    prompt, response, and metadata (provider, model, latency, cost)
  - This is the result: every request/response pair is logged and
    queryable

- Milestone 2: add routing and measure the cost/quality tradeoff
  - Add a simple routing policy (e.g., a task classifier) and compare cost
    and quality against always calling the strong model
  - This is the result: a measured cost/quality frontier for at least one
    routing policy

- Milestone 3: train a difficulty estimator and explore distillation
  - Train `q(prompt) -> required capability level` from logged agreement
    between cheap and strong models, and measure the cost/quality frontier
    it produces
  - Explore distilling a small model from the collected dataset
  - This is the result: a difficulty-aware router with a measured saving,
    and a first distillation experiment

- Milestone 4: wire fulfillment monitoring into the market
  - Track measured reliability and latency per contract, flag violations,
    and report them back to [[draft.Intelligence_Market]]
  - This is the result: a closed loop between a matched contract and its
    measured fulfillment

## References
- OpenRouter: unified API for multiple LLM providers (for reference, not
  reuse of proprietary code)
- Chen, L., et al., _FrugalGPT: How to Use Large Language Models While
  Reducing Cost and Improving Performance_. (2023)
- Ong, I., et al., _RouteLLM: Learning to Route LLMs with Preference Data_.
  (2024)
