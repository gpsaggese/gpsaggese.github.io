# Pre-Registration of Milestones 3 and 4

- Written on 2026-11-26 (US Eastern time), before any analysis below was run
  and before any answer of the prompt variants was scored
  - The generation of the prompt variants started on 2026-11-26 at 19:36, but
    generating answers does not score them
  - The answers of Studies E and F were scored in phases 1 and 2, so the score
    of each model is known, but none of the analyses below has been run on them
  - The EvalPlus leaderboard file of Study E was downloaded on 2026-11-27 and
    only its format was checked: 89 models have both a HumanEval+ and an MBPP+
    score
- Same rules as `preregistration.md` and `preregistration_2.md`: every result
  gets reported, whether or not it matches the prediction, and any analysis not
  listed here is marked as an extra look
- Milestone 3 is Study E; Milestone 4 is Studies F, G, and H

## Study E: Do HumanEval+ and MBPP+ Rank Models the Same Way?

### Design

- Agents: the 18 models with a greedy answer to every task of both benchmarks
  - The 12 public EvalPlus greedy sets that cover both benchmarks: CodeGen 2B,
    6B, and 16B, CodeT5+ 2B, 6B, and 16B, Code Llama 7B, 13B, and 34B,
    StarCoder, Mistral 7B, and GPT-4 Turbo (`gpt-4-1106-preview`)
  - The 6 local models, one-shot: DeepSeek Coder 1.3B, CodeQwen 7B, and
    OpenCoder 8B (run 0 of phase 1), and CodeGemma 7B, Llama 3.1 8B, and
    DeepSeek Coder 6.7B (Study D)
- Outcome: 1 if the answer passes the plus tests; the tasks are those with an
  outcome for all 18 models
- E.1 Reliability of each benchmark: KR-20 (Cronbach's alpha for 0/1 tasks)
  over the 18 models, with the mean Spearman-Brown corrected split-half
  correlation over 1,000 random halves as a check
- E.2 Agreement: Pearson and Spearman correlation between the 18 HumanEval+
  scores and the 18 MBPP+ scores
- E.3 Agreement corrected for task luck: the Pearson correlation divided by
  `sqrt(KR-20 of HumanEval+ x KR-20 of MBPP+)`
  - 95% interval from 2,000 bootstrap samples that resample the models and the
    tasks of each benchmark
  - A value of 1 means the 2 benchmarks measure the same skill and differ only
    by task luck
- E.4 Benchmark-specific models: for each model, `d = logit(HumanEval+ score) -
  logit(MBPP+ score)`, with scores `(k + 0.5) / (n + 1)`, minus the median `d`
  of the 18 models
  - 95% interval from 2,000 bootstrap samples of the tasks, models fixed
  - A model is benchmark-specific when its interval excludes 0; reported in
    points
- E.5 Effective size: the 2PL fit of Milestone 3 (SVI) on each benchmark, the
  information of each task averaged over the 18 fitted abilities, and the
  effective size of D-3; also the number of tasks with mixed outcomes and the
  Spearman correlation of the fitted abilities between the 2 benchmarks
- E.6 Scale check: the Spearman correlation between the HumanEval+ and MBPP+
  scores of the 89 models of the EvalPlus leaderboard

### Predictions

- E1: KR-20 is at least 0.95 on both benchmarks
- E2: the Spearman correlation of the 18 models is at least 0.80
- E3: the corrected correlation is below 0.95, and the upper end of its 95%
  interval is below 1, so part of the skill is specific to each benchmark
- E4: at least 2 of the 18 models are benchmark-specific
- E5: the effective size is between 30% and 70% of the tasks on both
  benchmarks, and a smaller share of the tasks on MBPP+ than on HumanEval+
- E6: the Spearman correlation of the 89 leaderboard models is at least 0.85

## Study F: How Much Does Reporting the Best of K Runs Inflate a Score?

### Design

- A cell is 1 agent at 1 temperature on 1 benchmark with several runs on the
  same tasks
  - Public cells: the EvalPlus samples on HumanEval+ with 50 scored runs, for
    8 models at temperatures 0.2 and 0.8 and ChatGPT at 0.8
  - Local cells: the sampled runs of phases 1 and 2, each scaffold: 5 runs at
    temperature 0.8 in phase 1 (both benchmarks), and 3 runs in Study C (0.2,
    HumanEval+) and Study D (0.8, both benchmarks)
- For each cell and each `K` (2, 5, 10, and 20 runs for the public cells; 2 for
  every local cell and 3 for the phase-1 cells): 500 random draws of `K` runs
  - The best of the `K` runs is what a paper reporting its best seed shows
  - The truth is the mean score of the runs left out (1 to 48 runs), which
    the selection never saw
  - The inflation is the best score minus the truth
- Predicted inflation: `sigma_run x E[max of K standard normals]`, where
  `sigma_run^2 = sum_t p_t (1 - p_t) / n^2` and `p_t` is the pass rate of task
  `t` over all the runs of the cell
- The corrections of Study G are also applied to the best of the `K` runs,
  using only those `K` runs, and compared with the truth

### Predictions

- F1: for each `K`, the mean inflation over the public cells is within 20% of
  the mean predicted inflation, and the same holds over the local cells
- F2: on HumanEval+ at temperature 0.8, the best of 5 runs is at least 1.5
  points above the truth for each of the 9 public cells
- F3: the binomial correction C1 over-corrects the best of 5 runs: its mean
  error over the public cells is below -1 point

## Study G: How Much Does the Best of K Prompt Variants Inflate a Score?

### Design

- Answers: DeepSeek Coder 1.3B, greedy, one-shot, with each of the 20 prompt
  variants of `PROMPT_VARIANTS` (2 system prompts x 2 instructions x 5 lines of
  guidance), on every task of both benchmarks, made by `run_prompt_variants.sh`
- G.0 Check: the baseline variant is the prompt of phase 1, so its outcomes
  are compared with the greedy one-shot outcomes of phase 1
- G.1 Differences: Cochran's Q test of equal pass rates over the 20 variants,
  per benchmark, and the range of their scores
- G.2 Selection on held-out tasks: for each `K` (2, 5, 10, and 20) and 1,000
  trials, `K` variants are drawn at random (all 20 for `K = 20`) and the tasks
  are split at random into a selection half and an evaluation half
  - The winner is the variant with the best score on the selection half, ties
    broken at random
  - The drop is its selection score minus its evaluation score
- Corrections of the winner's selection score, each using only the selection
  half:
  - C0: none
  - C1: binomial rule, minus `sqrt(p (1 - p) / n) x E[max of K]`, with `p` the
    winner's score and `n` the number of tasks
  - C2: paired rule, minus `sqrt(v / 2) x E[max of K]`, with `v` the mean over
    pairs of variants of the variance of their score difference,
    `(share of tasks where they disagree - squared gap) / n`
  - C3: empirical Bayes, the mean of the `K` scores plus `B x (winner - mean)`,
    with `B = max(0, 1 - (v / 2) / variance of the K scores)`
  - C4: conditional inference on the winner (Andrews, Kitagawa, and McCloskey,
    "Inference on Winners"): the median-unbiased estimate of the winner's mean
    given that it won, with the `K` scores taken as jointly normal and their
    covariance estimated from the per-task outcomes
- The error of a correction is the corrected score minus the evaluation score;
  reported: the mean error and the root mean squared error

### Predictions

- G0: the baseline variant gives the same outcome as phase 1 on at least 95% of
  the tasks of each benchmark
- G1: Cochran's Q rejects equal pass rates (p below 0.05) on at least 1 of the
  2 benchmarks
- G2: at `K = 20`, the mean drop is at least 1 point on both benchmarks
- G3: at `K = 20`, C1 over-corrects: its mean error is below -1 point on both
  benchmarks
- G4: the mean error of C4 is within 1 point of 0 for every `K` on both
  benchmarks
- G5: at `K = 20`, C3 or C4 has the lowest root mean squared error on both
  benchmarks

## Study H: The Practical Shrinkage Rule

- The rule to recommend is chosen by this criterion, fixed now:
  - Among C2, C3, and C4, keep the corrections whose mean error is within 1
    point of 0 in every setting of Studies F and G: each `K`, each benchmark,
    the public cells pooled, and the local cells pooled
  - Recommend the one with the lowest root mean squared error averaged over
    these settings
  - If none passes, report that no rule passed
- Applications, as extra looks with no prediction:
  - The recommended rule applied to the leader of the SWE-bench Verified
    snapshot, picked among its 173 submissions on the same 500 tasks
  - A protocol function that reports, for each agent, the score with its 95%
    interval, its rank interval, the minimum number of tasks to resolve a given
    gap, and the corrected score when the agent was picked among `K` variants
