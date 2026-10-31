# Pre-Registration of the Local Coding-Agent Runs

- Written before the full runs, and committed before any of them was scored, so
  that the analysis and the predictions cannot be chosen after seeing the results
  - The generation was still running at the commit, but it never runs the hidden
    tests, so no score of the full runs was known
  - The public sampled zips of analysis 7 were not scored before the commit either
  - Picking the analysis that looks best is the same selection effect this project
    studies in Milestone 4
- Every result listed here gets reported, whether or not it matches the prediction

## Design

- Benchmarks, scored with `evalplus==0.3.1`:
  - HumanEval+ v0.1.10: 164 tasks
  - MBPP+ v0.2.0: 378 tasks
- Agents: 3 models x 2 scaffolds = 6 agents
  - Models, served by Ollama 0.40.8, all with Q4_K_M weights:
    - `deepseek-coder:1.3b-instruct-q4_K_M`: the weaker model
    - `codeqwen:7b-chat-v1.5-q4_K_M` and `opencoder:8b`: the close pair, 1.3
      points apart on HumanEval+ and 2.4 points apart on MBPP+ in the published
      EvalPlus scores, in opposite directions
  - Scaffolds:
    - `one_shot`: 1 answer
    - `repair`: 1 more attempt, with feedback from the examples shown in the
      prompt only (never the hidden tests)
      - In a pilot on 20 tasks (180 cells), the second attempt fixed 4 of the 13
        answers it got and the third fixed none of the 6 it got, so the full runs
        stop after 1 repair
    - The repair scaffold starts from the one-shot answer, so the 2 are paired
- Runs:
  - 5 runs per task at temperature 0.8, with `top_p` 0.95 and `top_k` 50, as in
    the EvalPlus Hugging Face backend
    - The sampler of Ollama (llama.cpp) can apply the `top_k` / `top_p` cuts and
      the temperature in a different order than Hugging Face, so the sampling is
      close to that of EvalPlus but not guaranteed to be identical
  - 1 greedy run per task
  - The seed of each cell depends only on the task and the run
    - A cell generated again with its seed often gives the same answer (8 of 9
      replays in the pilot), but not always: Ollama reuses its prompt cache between
      requests, which changes the arithmetic slightly
- Scoring: every answer is scored twice, to find flaky tests

## Analyses

1. Validation: greedy plus score of each model vs its published EvalPlus score
2. Variance decomposition of each agent at temperature 0.8: $\sigma^2_{task}$,
   $\sigma^2_{run}$, and the run share of the variance of 1 run
3. Paired comparisons, on each benchmark, with the per-task rates over the 5
   runs (paired bootstrap over tasks, paired t-test), with Holm over the family:
   - The close pair, for each scaffold: 4 tests
   - The weaker model against each model of the pair, for each scaffold: 8 tests
   - Repair against one-shot, for each model: 6 tests
4. Stability of a 1-run leaderboard of the 6 agents: chance that the order
   matches the order over 5 runs, Kendall tau, and the flip rate of each pair
5. Tasks needed to resolve the observed gap of the close pair, as a function of
   the runs per task
6. Share of flaky tests, and the failure modes of the runs that flip
7. The same variance decomposition on the public EvalPlus samples (8 models, 50
   samples per task, temperatures 0.2 and 0.8)

## Predictions

- Power check, with the median paired standard deviation of the SWE-bench
  neighbors (0.433), 80% power, and 1 run: 164 tasks resolve gaps of about 9.5
  points and 378 tasks about 6.2 points, while the published gaps of the close
  pair are 1.3 and 2.4 points
- P1: The close pair is not distinguishable on either benchmark, with either
  scaffold (Holm-adjusted p >= 0.05)
- P2: The weaker model is distinguishable from both models of the pair, on both
  benchmarks and with both scaffolds
- P3: At temperature 0.8, run-to-run luck is between 10% and 40% of the
  variance of 1 run, for every agent
- P4: Each greedy score is within 5 points of the published score (published
  scores use full-precision weights)
- P5: The repair scaffold raises the plus score of every model, by more on MBPP+
  (all tasks have an example in the prompt) than on HumanEval+ (122 of 164 do)
- P6: Less than 1% of the answers get a different result when scored twice
- P7: In the public samples, $\sigma^2_{run}$ at temperature 0.8 is at least 2
  times $\sigma^2_{run}$ at temperature 0.2, for every model

## How Each Prediction Is Checked

- Added on 2026-10-29, after the first commit of this file and before any of
  the full runs was scored, to fix how each prediction is read
  - The same rules are coded in `check_predictions()` in
    `skill_vs_luck_runs_utils.py`
- P1: none of the 4 comparisons of the close pair has a Holm-adjusted p below
  0.05
- P2: all 8 comparisons of the weaker model have a Holm-adjusted p below 0.05,
  with the weaker model behind
- P3: the run share is between 10% and 40% for each of the 6 agents on each
  benchmark, so 12 values
- P4: the greedy one-shot plus score of each model on each benchmark is within
  5 points of the published one, so 6 values
- P5: over the 5 runs at temperature 0.8, repair has a higher plus score than
  one-shot for each model on each benchmark, and each model gains more on MBPP+
  than on HumanEval+
- P6: less than 1% of the scored answers get a different result the second time
- P7: the ratio is at least 2 for each of the 8 public models
