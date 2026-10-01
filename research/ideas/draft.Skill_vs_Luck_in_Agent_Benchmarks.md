# Skill vs Luck in Agent Benchmarks

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- Agent leaderboards (e.g., SWE-bench style suites) are reported as a single number
  from one run over a few hundred tasks
  - Small score gaps are read as real capability differences
- The same decomposition applied to hedge fund managers in
  `draft.Causal_Analysis_of_Hedge_Fund_Performance.md` applies here: an observed
  benchmark score is skill plus luck
  - Luck enters through task sampling, decoding stochasticity, environment flakiness,
    and harness/scaffold configuration
- The non-obvious part is the selection effect: a lab tries many scaffolds, prompts,
  and checkpoints, then reports the best
  - This is exactly the backtest overfitting of `draft.Backtesting_Complexity.md`, so
    the reported score must be charged a penalty that grows with the size of the
    search
- Deliverable: a statistical protocol that reports agent scores with error bars,
  states the minimum number of tasks needed to resolve a given gap, and corrects for
  the winner's curse in scaffold search

## Formalization

- Let $s_{a,i,r} \in \{0,1\}$ be the success of agent $a$ on task $i$ in run $r$,
  with $N$ tasks and $R$ runs
- The reported score is the mean
  $$
  \hat{\theta}_a = \frac{1}{NR} \sum_{i,r} s_{a,i,r}
  $$
- Variance decomposes into a task-sampling term and a run-to-run term:
  $$
  \mathrm{Var}(\hat{\theta}_a) =
  \frac{\sigma^2_{task}}{N} + \frac{\sigma^2_{run}}{NR}
  $$
- To resolve a gap $\Delta$ between two agents with the paired estimator, the
  required number of tasks scales as:
  $$
  N \gtrsim \frac{2 z^2_{1-\alpha/2} \, \sigma^2_{pair}}{\Delta^2}
  $$
- Selecting the best of $K$ noisy candidates inflates the reported score by roughly
  $\sigma \sqrt{2 \ln K}$, so the corrected score is:
  $$
  \theta_{eff} = \hat{\theta}_{max} - \sigma \sqrt{2 \ln K}
  $$
  which mirrors $VC_{eff} = VC(H) + \log(N_{strategies})$

## Key Examples

- **Indistinguishable leaderboard neighbors**: two agents scoring 0.42 and 0.45 on
  500 tasks differ by 15 tasks
  - If the paired disagreement rate is high, the gap sits inside the confidence
    interval and the ranking is noise
- **Flaky environments**: tasks whose containers fail intermittently contribute
  variance but no signal, and inflate $\sigma^2_{run}$ without measuring capability
- **Persistence test**: split the benchmark into two disjoint halves and measure rank
  correlation across halves
  - Low correlation means the leaderboard measures the task sample, not the agent,
    the same conclusion drawn about fund manager persistence
- **Scaffold search**: a lab evaluating 200 prompt/scaffold variants and publishing
  the best is running a backtest with $K = 200$
  - Several points of the reported gain are winner's curse

## Questions

1. What fraction of published leaderboard rank changes survive a paired bootstrap
   over tasks and seeds?
2. Does agent ability rank persist across benchmark families, or is it
   benchmark-specific, i.e., are agents overfit to the benchmark that guided their
   development?
3. How much of the year-over-year benchmark improvement is scaffold search rather
   than model capability, and can the winner's curse correction be estimated from the
   number of reported ablations?
4. Under an item response theory (IRT) fit, how many benchmark tasks carry nonzero
   discrimination, i.e., what is the effective size of a benchmark?
5. If most tasks are uninformative, can an adaptive benchmark reach the same
   statistical power with an order of magnitude fewer evaluations?

## Research Topics

- **Variance decomposition**: estimate $\sigma^2_{task}$ and $\sigma^2_{run}$ on open
  benchmarks, and publish minimum-$N$ curves for a target gap
- **Paired inference**: paired bootstrap and McNemar-style tests on per-task outcome
  matrices instead of comparing marginal means
- **Item response theory**: fit a 2PL model to recover task difficulty and
  discrimination, and identify saturated or degenerate tasks
- **Adaptive benchmarking**: select the next task to maximize Fisher information
  about the ability gap between two agents
- **Winner's curse correction**: model the scaffold search as a selection process and
  estimate the shrinkage needed for reported scores
- **Contamination control**: compare pre- and post-cutoff task subsets to separate
  memorization from capability
- **Related ideas**: `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`,
  `draft.Backtesting_Complexity.md`, `draft.Measure_ability_to_predict_events.md`,
  `draft.Benchmarking_Data_Science_Agents.md`, and
  `in_progress.Comparison_of_Coding_Agents.md`

## Next steps

- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the per-task outcome matrix
  - Select 2-3 open agent benchmarks and 3-5 agents/scaffolds
  - Run each agent for $R \geq 5$ seeds, logging per-task binary outcomes, cost, and
    failure mode
  - This is the result: a reusable outcome tensor $s_{a,i,r}$ plus a harness that can
    replay any cell
- Milestone 2: quantify luck
  - Decompose variance into task and run components
  - Produce confidence intervals, paired tests, and minimum-$N$ curves
  - This is the result: a corrected leaderboard with error bars and an explicit
    statement of which ranks are indistinguishable
- Milestone 3: persistence, transfer, and IRT
  - Measure split-half rank correlation and cross-benchmark rank correlation
  - Fit a 2PL IRT model to estimate task difficulty and discrimination
  - This is the result: an estimate of the effective size of each benchmark and
    evidence on whether ability transfers across suites
- Milestone 4: selection effects
  - Simulate scaffold search with $K$ variants and measure realized winner's curse
    against a held-out task split
  - This is the result: a practical shrinkage rule for reported agent scores

## References

- Bailey et al., _The Probability of Backtest Overfitting_. (2014)
- Fama et al., _Luck versus Skill in the Cross-Section of Mutual Fund Returns_.
  (2010)
- Embretson et al., _Item Response Theory for Psychologists_. (2000)
