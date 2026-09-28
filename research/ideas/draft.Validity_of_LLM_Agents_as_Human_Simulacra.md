# Validity of LLM Agents as Human Simulacra

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- Several projects in this directory simulate human organizations with LLM
  agents: committee vs single authority, batch vs iterative startup
  decisions, planning vs execution
  - All of them assume that an LLM agent is a usable stand-in for a human
    decision maker, and none of them test that assumption
- Proposal: run classic behavioral and experimental economics protocols with
  LLM agent populations and compare the full outcome distribution, not just
  the mean, against published human data
- The interesting failure mode is not bias in the mean but variance collapse:
  agent populations are far more homogeneous than human ones, which
  systematically understates disagreement and therefore biases every
  conclusion about committees, consensus, and pivots
- Second non-obvious issue is contamination: models have read the papers
  describing these experiments, so reproducing a known result may be recall
  rather than behavior
  - This forces the design of novel isomorphic variants as a control
- Deliverable: a validity envelope stating which classes of simulation
  conclusions are trustworthy, under which model, persona, and prompt
  conditions

## Formalization

- Let $P$ be a protocol with a published human outcome distribution $H_P$
- Let $A_P(m, \psi, \pi)$ be the agent outcome distribution for model $m$,
  persona distribution $\psi$, and prompt template $\pi$
- Distribution-level validity uses a distance such as the Wasserstein or total
  variation distance:
  $$
  d_P = W_1\big(A_P(m, \psi, \pi), \, H_P\big)
  $$
- Divergence is decomposed into three parts:
  - **Level bias**: difference in means
  - **Dispersion ratio**: $\mathrm{sd}(A_P) / \mathrm{sd}(H_P)$, where a value
    far below 1 is variance collapse
  - **Treatment distortion**: difference in the effect of an experimental
    manipulation, which is what simulations actually rely on
- The headline metric is the fraction of published treatment effects
  reproduced with the same sign and a comparable magnitude:
  $$
  \mathrm{Validity} =
  \frac{1}{|P|} \sum_{p} \mathbb{1}\big[
  \mathrm{sign}(\hat{\tau}^{A}_{p}) = \mathrm{sign}(\tau^{H}_{p})
  \ \wedge\ |\hat{\tau}^{A}_{p} - \tau^{H}_{p}| \leq \delta \big]
  $$

## Key Examples

- **Ultimatum game**: agents converge on near-even splits with tiny variance,
  while human offers and rejection thresholds are widely dispersed
  - A committee simulation built on such agents will find consensus far too
    easily, which is exactly the quantity
    `draft.Agentic_Analysis_of_Committee_vs_Authority.md` tries to measure
- **Keynesian beauty contest**: humans display level-k reasoning with a
  characteristic distribution of guesses, while agents may jump to the
  equilibrium answer
  - Any market or committee simulation inherits this distortion
- **Anchoring and sunk cost**: whether agents show human-like anchoring
  determines whether a simulated startup pivots at human-like rates, which is
  the core outcome in
  `draft.Causal_Analysis_Planning_vs_execution.md`
- **Contamination control**: rerun the ultimatum game with renamed roles,
  altered payoff units, and an unfamiliar cover story
  - If behavior changes sharply, the original match was recall of the
    literature, not simulated decision making

## Questions

1. Is variance collapse the dominant failure mode, and can persona sampling
   restore human-like dispersion without shifting the mean into a new bias?
2. Do treatment effects replicate even when levels do not, i.e., is the
   simulation valid for comparisons while invalid for absolute predictions?
3. How stable is validity across model updates?
   - If conclusions flip when the underlying model is replaced, agent-based
     social simulation results have a shelf life that must be reported
4. Once a persona distribution is calibrated on human data for a set of
   protocols, does it extrapolate to protocols it was not fit on, or is
   calibration just overfitting to the fitted experiments?
5. Are there protocol families (incentive design, coordination, information
   aggregation) where agents are systematically valid or invalid?

## Research Topics

- **Protocol suite**: curate experiments with public human distributions:
  ultimatum, dictator, trust, public goods, beauty contest, Asch conformity,
  anchoring, base-rate neglect, sunk cost
- **Persona sampling**: condition agents on sampled demographic and trait
  profiles and measure the effect on dispersion and level bias
- **Contamination testing**: build isomorphic variants with renamed entities
  and altered surface form, and compare behavior against the canonical form
- **Prompt sensitivity**: measure how much conclusions move under paraphrases
  of the instructions, linking to
  `draft.Measuring_Quality_of_Skills_and_Prompts.md`
- **Calibration and reweighting**: fit persona mixtures to human data and test
  out-of-protocol extrapolation
- **Retrofit**: re-run the existing agentic simulations in this directory
  under the calibrated setup and report which of their conclusions survive

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: assemble protocols and human baselines
  - Implement 6-10 protocols as scripted environments with a common interface
  - Digitize the published human outcome distributions, not only the means
  - This is the result: a reusable benchmark of human-anchored decision
    experiments

- Milestone 2: measure raw agent behavior
  - Run agent populations across models and temperatures, collecting full
    outcome distributions
  - Compute level bias, dispersion ratio, and treatment distortion per
    protocol
  - This is the result: the first validity map, with variance collapse
    quantified

- Milestone 3: contamination and persona experiments
  - Run isomorphic variants to separate recall from behavior
  - Run persona-sampled populations and measure dispersion recovery
  - This is the result: an answer on whether apparent validity is memorization
    and whether persona sampling fixes dispersion

- Milestone 4: retrofit existing simulations
  - Re-run the committee vs authority and lean startup simulations under
    calibrated and uncalibrated agents
  - This is the result: a list of which prior conclusions are robust and which
    were artifacts of homogeneous agents

## References

- Argyle et al., _Out of One, Many: Using Language Models to Simulate Human
  Samples_ (2022)
- Aher et al., _Using Large Language Models to Simulate Multiple Humans_
  (2023)
- Horton, _Large Language Models as Simulated Economic Agents_ (2023)
- Park et al., _Generative Agents: Interactive Simulacra of Human Behavior_
  (2023)
- `draft.Agentic_Analysis_of_Committee_vs_Authority.md`
- `draft.Causal_Analysis_Planning_vs_execution.md`
