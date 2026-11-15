# Tukey: An RL Agent for Automated EDA

- MSML610 Fall 2026 research project, issue
  [#628](https://github.com/gpsaggese/gpsaggese.github.io/issues/628)
- Team: Rayhan Patel (@Rayhanpatel), Ronak Vimal (@Kytamnh), Sai Katupilla
  (@Anirudh-Ka)
- [Spec](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/in_progress.RL_for_auto_EDA.md)

## What Tukey Does

- An agent gets a data table and must recover the causal graph that generated it
- It calls statistical tools on the training rows (independence tests,
  regressions, direction tests, mutual information), then emits a graph
- Training data come from random causal graphs, so every answer is scored
  exactly: structural distance to the true graph plus prediction on held-out rows
- The tool chooser and the graph decoder are trained with REINFORCE, starting
  from a copy of the prototype's scripted rule

## Status

- A first version of all four milestones is built locally; its last full check
  was on Oct 16 (fresh clone: 371 tests passed, both notebooks ran)
- We submit it one milestone per PR; each code PR reruns the full tests and the
  Docker build. This replaces
  [#629](https://github.com/gpsaggese/gpsaggese.github.io/pull/629) (one 93-file PR)
- The research is not finished: see the open questions below
- Plan: [`docs/04-plan.md`](docs/04-plan.md)

## Preliminary Results

- From local runs; each result file arrives with its milestone's PR, see
  [`docs/07-progress.md`](docs/07-progress.md)
- On 600 sealed synthetic graphs (5 seeds) the trained policy beats the scripted
  rule: score -0.275 vs -0.413 (+0.138, 95% CI [0.125, 0.151]); structural
  distance 4.25 vs 5.86
- GES (-0.211) and DirectLiNGAM (-0.184) beat the policy; PC is about level
- On real protein data (Sachs et al. 2005) every method scores low: skeleton F1
  0.53 for the policy, 0.52 for GES, DirectLiNGAM and the rule, 0.58 for PC

## Open Questions

- Does learned test selection help, or only the trained decoder? The policy
  rarely calls the direction test (0.0007 calls per graph)
- Why do GES and DirectLiNGAM stay ahead?
- Messy data (missing values, drift, mixed types; spec Question 2): not tested
- Human-level discovery without heuristics (spec Question 3): not shown
- Scoring Gaussian data against its equivalence class and hidden confounders
  against a PAG need the reviewers' agreement
  ([`D1`, `D2`](docs/03-decisions.md))

## Related Work

- In this repo: [#506](https://github.com/gpsaggese/gpsaggese.github.io/issues/506),
  the prototype [#527](https://github.com/gpsaggese/gpsaggese.github.io/pull/527),
  the PC/GES benchmark [#587](https://github.com/gpsaggese/gpsaggese.github.io/pull/587)
  and the [paper draft](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/RL_for_Automated_EDA/paper.md)
- RL for causal discovery: [Zhu et al. 2019](https://arxiv.org/abs/1906.04477)
- Amortized causal discovery: [Lorch et al. 2022 (AVICI)](https://arxiv.org/abs/2205.12934)
- RL for causal experiment design: [Annadani et al. 2024 (CAASL)](https://arxiv.org/abs/2405.16718)
- Shortcuts in simulated benchmarks: [Reisach et al. 2021](https://arxiv.org/abs/2102.13647)
- Agentic causal discovery: [Ma et al. 2026](https://arxiv.org/abs/2610.22846)

## Docs

- [Index](docs/README.md), [plan](docs/04-plan.md),
  [decisions](docs/03-decisions.md), [progress](docs/07-progress.md),
  [how we work](docs/08-how_we_work.md)
