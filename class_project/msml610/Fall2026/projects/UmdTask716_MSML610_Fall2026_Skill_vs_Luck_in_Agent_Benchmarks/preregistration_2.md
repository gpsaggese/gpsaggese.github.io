# Pre-Registration of Phase 2

- Written on 2026-10-29 and 2026-10-30, before any analysis below was run and
  before any phase-2 answer was scored
  - The phase-2 generation started on 2026-10-29 at 23:03, but generating
    answers does not score them
  - The SWE-bench Verified data of Study A was already used in Milestones 1 and
    2, so its scores and ranks are known, but the record analysis has not been
    run
- Same rules as `preregistration.md`: every result gets reported, whether or
  not it matches the prediction, and any analysis not listed here is marked as
  an extra look
- If Ollama stops an answer because the model keeps repeating the same token,
  the answer is saved with no code and scored as a failure
  - Ollama returns an error instead of the answer, and a retry with the same
    seed repeats it
  - With the repair scaffold, the model still gets its repair attempt, as for
    any answer with no code
  - Found on 2026-10-30 during the generation, before any scoring: CodeGemma
    7B at greedy decoding got stuck writing "db000..." on HumanEval/79

## Study A: How Many SWE-bench Verified Records Were Real?

### Design

- Data: the pinned snapshot of `SWE-bench/experiments` (commit `40f164d`), 173
  submissions x 500 tasks, each dated by its folder name
- A submission sets a record when its score is above every score dated before
  it
  - Submissions with the same date are compared with the record before that
    date, and only the best of them can set the record (if 2 tie, the one whose
    folder name sorts first)
  - The earliest submission starts the chain, so the records tested are the
    ones after it
- Each record is compared with the record it broke, on the same 500 tasks, with
  the exact McNemar test of Milestone 2, and Holm over all the records
- Reported:
  - The number of records, their gaps, and how many are distinguishable
  - How many records came from submissions that allow more than 1 attempt
  - The share of the total progress, from the first record to the last, that
    came in distinguishable steps

### Predictions

- A1: fewer than half of the records are distinguishable from the record they
  broke (Holm-adjusted p below 0.05)
- A2: the last record is distinguishable from the first one (exact McNemar p
  below 0.05)
- A3: more than half of the total progress, in points, came in distinguishable
  steps, so the progress is real even though most single records are not

## Study B: Can Run Luck Be Estimated From 1 Run?

### Design

- Leaderboards publish 1 run per agent, so their run luck is never measured
  - From 1 run, run luck cannot be told apart from an agent being good at a
    specific task without a model of the tasks, so this study tests how far a
    simple model gets
- The estimator, fixed here:
  1. Fit a Rasch model, logit P(pass) = ability of the agent minus difficulty
     of the task, to an agents x tasks matrix with 1 run per agent, by maximum
     likelihood with a small ridge penalty
     - Before any tuning, the penalty is 0.01 / 2 times the sum of the squared
       abilities and difficulties
  2. For each agent, estimate $\sigma^2_{run}$ as the average over the tasks of
     p (1 - p), with p the fitted chance of a pass
  3. Estimate the run share as that $\sigma^2_{run}$ divided by score x
     (1 - score), the whole variance of 1 outcome of 1 run
- Development set, where the estimator can be tuned (Rasch or 2PL, the penalty,
  a calibration factor, and which extra agents go into the matrix):
  - The public EvalPlus samples on HumanEval+: 8 models at 0.2 and 0.8, scored
    on the first 50 of the 200 released samples per task, with 1 sample drawn
    at random per task as their 1 run
  - The public greedy samples of the other models, as extra agents
- Test set, used once after the tuned estimator is committed:
  - Our runs, as 30 cases of 1 agent on 1 benchmark at 1 temperature:
    - Phase 1: the 6 old agents at 0.8 on both benchmarks, 12 cases
    - Phase 2: the 6 new agents (3 models x 2 scaffolds) at 0.8 on both
      benchmarks, 12 cases, and the 6 old agents at 0.2 on HumanEval+, 6 cases
  - The matrix of a case holds its agent and the one-shot agents of the other
    models with the same benchmark and temperature, each with its run 1, plus
    the extra agents of the tuned estimator
    - The one-shot and repair agents of a model are never in the same matrix,
      since the repair answer starts from the one-shot answer
  - The truth is the run share measured from all the runs of each agent
  - The run shares of the 12 phase-1 cases are already known (28% to 52%), so
    the predictions are decided on the 18 phase-2 cases, and the phase-1 cases
    are reported next to them
- Also reported: how many tasks need a second run, with 1 run on the other
  tasks, so that 95% of random picks of those tasks give a run share within 10
  points of the one from all the runs
- Extra look: the estimator applied to SWE-bench Verified, to estimate how much
  a rerun would reorder that leaderboard

### Predictions

- B1: the tuned estimate is within 10 points of the measured run share for at
  least 12 of the 18 phase-2 cases
- B2: the estimate before tuning is above the measured run share in more than
  half of the 18 phase-2 cases, since it counts agent-task strengths as luck

## Study C: Run Luck at Temperature 0.2 and at Temperature 0

### Design

- The 3 old models on HumanEval+, with both scaffolds:
  - 3 runs at temperature 0.2, with the same top-p and top-k as before
  - 1 more greedy run, so 2 greedy runs in all
- Same analyses as phase 1: the run share, the paired comparisons of the close
  pair (CodeQwen 7B and OpenCoder 8B, 1 test per scaffold, with Holm over the
  2), and 1-run leaderboards of the 3 models, for each scaffold
- A 1-run leaderboard at 0.8 uses only the first 3 runs, as at 0.2
  - With fewer runs, the drawn run is a bigger part of the mean it is checked
    against, so leaderboards with 3 runs and with 5 runs are not comparable

### Predictions

- C1: at temperature 0.2, the run share is between 5% and 30% for each of the 6
  agents
- C2: the run share at 0.2 is below the one at 0.8 (all 5 runs of phase 1), for
  each of the 6 agents
- C3: between the 2 greedy runs, fewer than 2% of the tasks change their
  one-shot result, for each model
- C4: at 0.2 the close pair still cannot be told apart, with either scaffold
  (Holm-adjusted p of 0.05 or more)
- C5: 1-run leaderboards of the 3 models keep the order more often at 0.2 than
  at 0.8, for each scaffold

## Study D: A Bigger Leaderboard With 3 More Models

### Design

- 3 more models, all in Ollama with 4-bit weights (Q4_K_M) and with published
  EvalPlus scores:

  | Model                        | Published HumanEval+ | Published MBPP+ |
  | :--------------------------- | :------------------- | :-------------- |
  | DeepSeek-Coder 6.7B Instruct | 71.3                 | 65.6            |
  | Llama 3.1 8B Instruct        | 62.8                 | 55.6            |
  | CodeGemma 7B Instruct (v1.0) | 51.8                 | 56.9            |

- Each one gets 1 greedy run and 3 runs at temperature 0.8, on both
  benchmarks, with both scaffolds
- On MBPP+, Llama 3.1 8B, CodeGemma 7B, and the DeepSeek-Coder 1.3B of phase 1
  (published 54.8) are within 2.1 points of each other, a cluster of 3 close
  models
- The 6-model leaderboard and the comparisons of the cluster use the one-shot
  scaffold at temperature 0.8, with the first 3 runs of every model, so all
  the models have the same number of runs

### Predictions

- D1: the greedy one-shot plus score of each new model is within 5 points of
  its published score (6 values)
- D2: no pair of the MBPP+ cluster can be told apart (Holm-adjusted p of 0.05
  or more for each of its 3 pairs)
- D3: on MBPP+, fewer than half of the random 1-run leaderboards keep the order
  of all 6 models
