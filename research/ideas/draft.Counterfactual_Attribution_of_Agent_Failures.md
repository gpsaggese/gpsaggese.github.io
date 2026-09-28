# Counterfactual Attribution of Agent Failures

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- When a long agent run fails, the only observable is the final outcome, so
  post-mortems are guesswork: the agent read the wrong file at step 3, but the
  visible symptom is a broken test at step 60
- Treat an agent trajectory as a sequence of interventions and estimate the causal
  effect of each step on the final outcome by replaying the run from a checkpoint
  with a counterfactual action
  - This applies the causal machinery already used in this directory for economic
    outcomes to the internals of an agent run
- The output is a blame profile over steps: which decision, which tool call, which
  context event actually caused the failure
- Non-obvious consequences:
  - If blame is concentrated in one pivotal step, agent reliability is a search
    problem (retry the right step), not a model capability problem
  - If blame is diffuse, incremental scaffold tweaks cannot help
  - If blame is near zero everywhere, the task itself is underspecified, which is a
    benchmark quality signal

## Formalization

- A trajectory is $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ with binary outcome $Y$
- The per-step causal effect of the taken action against an alternative policy
  $\pi_{alt}$ is:
  $$
  \Delta_k = E[Y \mid \tau_{<k}, a_k] -
             E[Y \mid \tau_{<k}, a_k \sim \pi_{alt}]
  $$
- Both terms are estimated by $M$ independent rollouts resumed from the checkpoint at
  step $k$, which requires a snapshot of both the environment and the agent context
- Interactions between steps are handled with a Shapley value over subsets
  $S \subseteq \{1, \dots, T\}$ of resampled steps:
  $$
  \phi_k = \sum_{S \subseteq \{1..T\} \setminus \{k\}}
  w(|S|) \, \big[ v(S \cup \{k\}) - v(S) \big]
  $$
- Cost is $O(T M)$ rollouts for the marginal profile and much more for full Shapley,
  so sampling and variance reduction (common random numbers across arms) are required

## Key Examples

- **Pivotal early mistake**: the agent opens the wrong module at step 3 and every
  later step is doomed
  - Attribution gives a large $\Delta_3$ and near-zero effects afterwards, even
    though the visible failure is at step 60
- **Context compaction event**: the harness truncates the context at step 40 and the
  agent drops a constraint stated in the task
  - Attribution isolates a harness event rather than a reasoning error, which a human
    post-mortem almost never identifies
- **Diffuse degradation**: many small deviations each with $\Delta_k \approx
  0.02$,
  indicating no single fix will move the outcome
- **Null profile**: all $\Delta_k \approx 0$ and the run fails from any starting
  point, so the task is ambiguous or the environment is broken, and the task should
  be flagged in the benchmark

## Questions

1. Empirically, is agent failure sparse (one pivotal step) or diffuse, and does the
   answer differ by task family (bug fix vs feature vs refactor)?
2. Do blame profiles transfer across models, i.e., does a step that is pivotal for
   one model stay pivotal for another?
   - Transfer implies intrinsic task difficulty, no transfer implies a model-specific
     weakness
3. Can a cheap surrogate (token log-probabilities, self-reported confidence, a judge
   model reading the trace) predict $\Delta_k$ well enough to skip the expensive
   rollouts?
4. Can the blame profile be used as a training or prompt-optimization signal, turning
   a diagnostic into step-level rewards?
5. How much of the measured $\Delta_k$ is causal and how much is the variance of
   resuming a stochastic agent, i.e., what is the noise floor?

## Research Topics

- **Replay infrastructure**: deterministic checkpoint and resume for agent runs,
  including container filesystem snapshots, tool state, and the exact context window
- **Estimator design**: marginal effects vs Shapley, sampling schedules, and variance
  reduction with common random numbers
- **Noise floor calibration**: resume the same step with the same action many times
  to measure irreducible outcome variance before interpreting any $\Delta_k$
- **Surrogate models**: train a predictor of $\Delta_k$ from trace features so
  attribution can be run at scale
- **Failure taxonomy**: cluster blame profiles into recurring modes (wrong file, lost
  constraint, premature commit, tool misuse) across benchmarks
- **Feedback into scaffolds**: use recurring blame modes to auto-generate skill and
  prompt fixes, connecting to `draft.Measuring_Quality_of_Skills_and_Prompts.md`

## Next Steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation Plan

- Milestone 1: build checkpoint and resume
  - Instrument an open agent harness to snapshot environment plus context at every
    step and to resume from any snapshot
  - Validate that a resumed run with the same seed reproduces the original trajectory
  - This is the result: a replayable agent runner and a measured noise floor for
    resumption
- Milestone 2: marginal blame profiles
  - For a set of failed runs, resample each step $M$ times under an alternative
    policy and estimate $\Delta_k$ with confidence intervals
  - This is the result: blame profiles for real failures, plus the first evidence on
    sparse vs diffuse failure
- Milestone 3: transfer and taxonomy
  - Repeat the analysis across models and task families, cluster the profiles, and
    label recurring failure modes
  - This is the result: a failure taxonomy with frequencies and an answer on
    cross-model transfer
- Milestone 4: cheap surrogate and feedback loop
  - Train a predictor of $\Delta_k$ from trace features and validate against the
    expensive estimate
  - Feed the top failure modes back into scaffold changes and measure the resulting
    success rate change
  - This is the result: a low-cost attribution tool and a measured improvement from
    acting on it

## References

- Shapley, _A Value for n-Person Games_ (1953)
- Pearl, _Causality: Models, Reasoning, and Inference_ (2009)
- `draft.Comparison_of_Debugging_Agents.md`
- `draft.Measuring_Quality_of_Skills_and_Prompts.md`
- `in_progress.Comparison_of_Coding_Agents.md`
- `draft.Skill_vs_Luck_in_Agent_Benchmarks.md`