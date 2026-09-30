# Build an LLM Arena for Pairwise Model Comparison

## Status
- **Status**: draft
- **Complete Specs**: 10%

## Core Idea
- Build a Chatbot-Arena-style system: a prompt is sent to two anonymized models,
  a judge (human or LLM) picks the better response, and the votes are aggregated
  into a ranking via a Bradley-Terry / Elo-style model
- The point is not to reproduce the public leaderboard but to have a
  self-hosted evaluation harness that works on _our_ task distribution, where
  absolute scoring is unreliable but pairwise preference is cheap
- This gives the agent-comparison ideas in this directory
  ([[in_progress.Comparison_of_Coding_Agents]],
  [[in_progress.Comparison_of_Data_Science_Agents]],
  [[draft.Benchmarking_Data_Science_Agents]]) a shared measurement backend
  instead of one ad-hoc scoring script each
- The research question underneath the engineering: how many pairwise votes are
  needed for a rank ordering to be statistically stable, and how much does an
  LLM judge's ranking diverge from a human one?

## Formalization
- Bradley-Terry: `P(i beats j) = sigma(r_i - r_j)`, fit by maximum likelihood
  over the vote set; report confidence intervals on `r_i`, not just point ranks
- Sampling matters: uniform random pairing wastes votes on already-separated
  models, so use an active pairing rule that samples pairs with high rank
  uncertainty
- Report the number of votes needed for the top-k ordering to stop changing

## Key Examples
- **Model arena**: two chat models, same prompt, blind pairwise vote
- **Agent arena**: two coding agents on the same repo task, judged on the diff
  they produce (needs a sandbox per side, not just a text response)
- **Judge calibration**: a held-out subset judged by both humans and an LLM, to
  measure agreement and detect judge bias (length, formatting, self-preference)
- **Failure mode**: with too few votes, rank instability sets in: the
  leaderboard reshuffles between runs and any conclusion drawn from it is
  noise, the same skill-vs-luck concern as
  [[draft.Skill_vs_Luck_in_Agent_Benchmarks]]

## Questions
1. How many votes per model pair are needed before the ranking is stable, and
   does active pair selection materially reduce that number?
2. Does an LLM judge preserve the human ranking, or does it systematically
   reward length/formatting over correctness?
3. Is a single scalar rating even well-defined across heterogeneous prompts, or
   does the ranking flip by task category (i.e., non-transitive preferences)?

## Research Topics
- Bradley-Terry/Elo estimation with confidence intervals and active sampling
- LLM-as-judge bias: position bias, verbosity bias, self-preference
- System design: blind two-way routing, vote storage, per-category leaderboards
- Reuse of the gateway from [[draft.Clone_openrouter]] as the model-routing
  layer so both ideas share one backend

## Next steps
- [ ] Look for related research (Chatbot Arena, LMSYS methodology, MT-Bench)
- [ ] Build a minimal two-model blind voting UI plus vote storage
- [ ] Implement Bradley-Terry fitting with confidence intervals
- [ ] Run a human-vs-LLM-judge agreement study on a small task set

## References
- Chiang, W.-L., et al., _Chatbot Arena: An Open Platform for Evaluating LLMs by
  Human Preference_. (2024)
- Zheng, L., et al., _Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena_.
  (2023)
- Bradley, R. A., & Terry, M. E., _Rank Analysis of Incomplete Block Designs_.
  (1952)
