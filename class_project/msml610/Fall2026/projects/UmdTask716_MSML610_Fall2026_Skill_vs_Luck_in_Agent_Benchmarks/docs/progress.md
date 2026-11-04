# Skill vs Luck in Agent Benchmarks Progress

- Issue: #716
- Research Google Doc:
  [Research Log](https://docs.google.com/document/d/1iJIWKuPMyGRoqac1RVM7up6cYdndUxApQUTU331Yom4/edit)
- Spec:
  [draft.Skill_vs_Luck_in_Agent_Benchmarks.md](/research/ideas/draft.Skill_vs_Luck_in_Agent_Benchmarks.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| M1: Outcome matrix (SWE-bench Verified) | Oct 28 | Done | #735 |
| M1: Outcome tensor (HumanEval+, MBPP+, 5 runs) | Oct 29 | In review | #808 |
| M2: Task luck, intervals, paired tests, minimum N | Oct 28 | Done | #735 |
| M2: Run-to-run luck | Oct 29 | In review | #808 |
| M2: Ranking under uncertainty | Oct 30 | In review | #855 |
| M3: Split-half rank correlation (SWE-bench) | Nov 15 | In review | #855 |
| M3: 2PL IRT and effective size (SWE-bench) | Nov 15 | In review | #855 |
| M3: Cross-benchmark rank correlation | Nov 15 | Not started | |
| M4: Winner's curse, simulated and held-out (SWE-bench) | Nov 15 | In review | #855 |
| M4: Measured drop on real prompt variants | Nov 15 | Not started | |
| Phase 2: Studies A to D, pre-registered | Nov 3 | Done on branch `_3`, PR after #808 | |
| Write-up and paper draft | Dec 7 | Not started | |

## Built

- `skill_vs_luck_utils.py`: loads the SWE-bench Verified matrix, computes the
  leaderboard, bootstrap score and rank intervals, paired tests with Holm,
  minimum-N, the Rasch simulations, split-half correlation, the winner's
  curse, and the 2PL IRT model
- `skill_vs_luck.example.ipynb`: the SWE-bench analysis, Parts 1 to 9
- `skill_vs_luck.API.ipynb`: a walk through each function
- `fetch_swebench_verified.py`: rebuilds `data/` from `SWE-bench/experiments`
  at commit `40f164d`
- Docker setup from `class_project/project_template`
- In PR #808: `run_coding_agents.py` (answers from local models through
  Ollama), `score_coding_runs.py` (EvalPlus tests in a sandbox),
  `skill_vs_luck_runs_utils.py`, and `skill_vs_luck.runs.ipynb`
- On [branch `_3`](https://github.com/prathambharati/gpsaggese.github.io/tree/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks_3/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks), for phase 2:
  - `preregistration_2.md`: the plan and the 13 predictions, committed before
    any of it was analyzed
  - `skill_vs_luck.records.ipynb`: Study A, the audit of the SWE-bench records
  - `skill_vs_luck.one_run.ipynb` and `skill_vs_luck.one_run_test_set.ipynb`:
    Study B, run luck from 1 run
  - `skill_vs_luck.temperatures.ipynb`: Study C, temperature 0.2 and a second
    greedy run
  - `skill_vs_luck.more_models.ipynb`: Study D, 3 more local models

## Findings

1. A single SWE-bench Verified score is known to about +/- 4 points: the
   median 95% bootstrap interval is 8.2 points wide
   - Data: 173 submissions x 500 tasks, `skill_vs_luck.example.ipynb` Cell 2.1
2. No pair of leaderboard neighbors differs after Holm (0 of 172, 2 before
   it), and 18 agents, down to rank 19, are tied with the leader
   - `skill_vs_luck.example.ipynb` Cells 3.1 and 3.2
3. 500 tasks resolve gaps of about 5.4 points at 80% power, while the median
   gap between neighbors is 0.2 points; a 1-point gap needs about 14,700 tasks
   - `skill_vs_luck.example.ipynb` Cells 4.1 and 4.2
4. Over 2,000 resamples of the tasks, the top 3 agents share 94% of the first
   place and none is above 34%; about 15 agents win at least 1 resample
   - Ties split; `skill_vs_luck.example.ipynb` Cell 5.2
5. A typical top-25 agent has a 95% rank interval about 18 ranks wide, and the
   2 co-leaders (396 tasks each) range from rank 1 to 6
   - `skill_vs_luck.example.ipynb` Cell 5.1
6. In a Rasch simulation with 10 agents and 500 tasks, the truly best agent
   comes out first about 50% of the time at a 1-point gap; above 90% needs
   10,000 tasks
   - 200 simulated benchmarks per setting, `skill_vs_luck.example.ipynb`
     Cell 5.3
7. At temperature 0.8, run-to-run luck is 28% to 52% of the variance of a
   1-run outcome on HumanEval+ and MBPP+, and 4 of 7 pre-registered
   predictions held
   - PR #808, `preregistration.md`
8. The ranking holds across random halves of the tasks for all 173 agents
   (median Kendall 0.90). Among the top 20 it is weak: Kendall 0.25, negative in
   1.4% of 1,000 splits. Picking on the total pushes the correlation down, since
   the 2 half scores add up to the total: on benchmarks simulated from the IRT
   fit, Kendall is 0.29 picked on the score and 0.40 picked on the true ability
   - `skill_vs_luck.example.ipynb` Cells 6.1 and 8.5
9. The leader of one half of the tasks scores 2.0 points less on the other
   half. The chance of leading both halves is not reported, since the halves
   add up to a fixed total
   - 1,000 splits, ties split; `skill_vs_luck.example.ipynb` Cell 6.2
10. The best of 20 equal variants on 500 tasks looks 3.8 points better than
    it is; the spec's `sigma * sqrt(2 ln K)` is 1.4 to 2.4 times the real
    inflation for K = 2 to 50. `sigma * E[max of K normals]` is about 10% too
    high, since `sigma` uses the binomial variance. With `sigma_tasks`, the
    noise given the drawn tasks, the correction matches the real inflation
    (ratio 1.00 on average, 0.97 to 1.05 across settings, the spread being
    trial noise)
    - 5,000 simulated benchmarks per setting, `skill_vs_luck.example.ipynb`
      Cell 7.1
11. The held-out drop of the best variant is about 1.4 times its inflation on
    all the tasks, close to sqrt(2)
    - `skill_vs_luck.example.ipynb` Cell 7.2
12. Under a 2PL IRT fit, the 500 tasks are worth about 107 equally
    informative tasks for the top 20 agents (297 for all 173), and 39 tasks
    hold half of the information; the top 20 all solve 254 tasks and all fail
    36
    - SVI fit, checked against NUTS on a 40 x 150 subset (ability correlation
      0.997); `skill_vs_luck.example.ipynb` Cells 8.1 and 8.4
13. The ranking by IRT ability matches the leaderboard overall (Kendall 0.97)
    but less in the top 20 (Kendall about 0.8), where ACoder drops from rank 9 to
    about 20
    - `skill_vs_luck.example.ipynb` Cell 8.2
14. The fitted difficulty rises with the human estimate of the time to fix
    (medians -1.6, 0.0, 2.9, 5.0 logits), with Spearman 0.45
    - `data/swebench_verified_tasks.csv`, `skill_vs_luck.example.ipynb`
      Cell 8.3
15. Phase 2: 8 of the 13 pre-registered predictions held (A1, A2, B2, C1 to
    C4, and D3), and A3, B1, C5, D1, and D2 did not
    - `preregistration_2.md` on branch `_3`
16. Only 2 of the 30 SWE-bench Verified records are distinguishable from the
    record they broke, and they carry 33% of the 74.8 points of progress;
    since June 2024, 27 records in a row were each within the noise of the
    record before, and together they still added 46.6 points
    - Study A, `skill_vs_luck.records.ipynb` on branch `_3`
17. On the public EvalPlus samples, a tuned 2PL fit estimates the run share
    from 1 run within 10 points in 86% of the draws, but its single test on
    our runs failed: only 7 of the 18 new cases are within 10 points, mostly
    because it is too high at temperature 0.8
    - Study B, `skill_vs_luck.one_run.ipynb` and
      `skill_vs_luck.one_run_test_set.ipynb` on branch `_3`
18. At temperature 0.2, runs are 15% to 21% of the variance of 1 run (37% to
    52% at 0.8), and 2 greedy runs give the same result on every task; a 1-run
    score is only a little more precise, though (standard error 3.1 to 3.9
    points against 3.4 to 3.9), and the close pair still cannot be told apart
    - Study C, `skill_vs_luck.temperatures.ipynb` on branch `_3`
19. With 6 models, only 44.4% of the 1-run leaderboards keep the order on
    MBPP+; the 3 models within 2.1 points of each other in the published MBPP+
    scores are 5 to 12 points apart in our runs, and Llama 3.1 8B scores 7.6
    points above its published greedy MBPP+ score
    - Study D, `skill_vs_luck.more_models.ipynb` on branch `_3`

## Next

1. M3: run the split-half and IRT analysis on the HumanEval+ and MBPP+
   results, and correlate the rankings across benchmarks
2. M4: compare the simulated held-out drop with the drop measured on real
   prompt variants
3. Phase 2: move the scores to Google Drive, as GP asked on #808, and open the
   PR of branch `_3` after #808 merges
4. Write-up: related work and a first draft of the paper sections

## History

| Date | Done |
| :--- | :--- |
| 2026-10-22 | Filed #716 |
| 2026-10-23 | GP asked for a results doc and simulations of ranking under uncertainty |
| 2026-10-28 | PR #735 merged: SWE-bench outcome matrix, intervals, paired tests, minimum N |
| 2026-10-29 | PR #808 opened: run-to-run luck on HumanEval+ and MBPP+ |
| 2026-10-30 | GP asked for the research Google Doc per #811; started it |
| 2026-10-30 | Task 1 committed: bootstrap rank intervals and ranking recovery simulation; ties split, see [decisions.md](decisions.md) D-1 |
| 2026-10-30 | Phase 2 pre-registered on branch `_3` (`preregistration_2.md`, `f9e4b00c`) |
| 2026-10-31 | Committed split-half rank correlation, held-out leader drop, and winner's curse simulation |
| 2026-10-31 | Phase 2 results of Studies A to D pushed to branch `_3` (`bc8141e1`): 8 of 13 predictions held |
| 2026-11-01 | Added the 2PL IRT model and the effective size; pinned `jax`, `jaxlib` 0.12.0.1 and `numpyro` 0.22.0; see D-2 and D-3 |
| 2026-11-03 | Review fixes: how the top 20 are picked (Cell 8.5), the correction given the tasks, 5,000 trials in Cell 7.1 |
| 2026-11-03 | PR #855 opened: ranking under uncertainty, split-half, winner's curse, and IRT |

## Open Work

- #808: run-to-run luck, waiting for review; its score files move to Google
  Drive, as GP asked there
- #855: ranking, split-half, winner's curse, and IRT, waiting for review
- #808 and #855 both edit `README.md`, so the second to merge has to resolve it
- Phase 2 is on branch `_3`, on top of #808, so its PR comes after #808 merges
