# Skill vs Luck in Agent Benchmarks Progress

- Issue: #716
- Research Google Doc:
  [Research Log](https://docs.google.com/document/d/1iJIWKuPMyGRoqac1RVM7up6cYdndUxApQUTU331Yom4/edit)
- Google Drive folder, with the scores of the coding runs:
  [MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks](https://drive.google.com/drive/folders/1rgI_x5rfX9HXiP0kZUHfyq2Zg8WWuV4y)
- Spec:
  [draft.Skill_vs_Luck_in_Agent_Benchmarks.md](/research/ideas/draft.Skill_vs_Luck_in_Agent_Benchmarks.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| M1: Outcome matrix (SWE-bench Verified) | Oct 28 | Done | #735 |
| M1: Outcome tensor (HumanEval+, MBPP+, 5 runs) | Oct 29 | Done | #808 |
| M2: Task luck, intervals, paired tests, minimum N | Oct 28 | Done | #735 |
| M2: Run-to-run luck | Oct 29 | Done | #808 |
| M2: Ranking under uncertainty | Oct 30 | Done | #855 |
| M3: Split-half rank correlation (SWE-bench) | Nov 15 | Done | #855 |
| M3: 2PL IRT and effective size (SWE-bench) | Nov 15 | Done | #855 |
| M3: Cross-benchmark rank correlation and effective size (HumanEval+, MBPP+) | Nov 15 | Done | #1014 |
| M4: Winner's curse, simulated and held-out (SWE-bench) | Nov 15 | Done | #855 |
| M4: Measured drop on real runs and prompt variants, shrinkage rule | Nov 15 | Done | #1014 |
| Phase 2: Studies A to D, pre-registered | Nov 3 | In review | #1013 |
| Write-up and paper draft | Dec 7 | Draft done, not built yet | #1014 |

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
  `skill_vs_luck_runs_utils.py`, `skill_vs_luck.runs.ipynb`, and
  `fetch_coding_scores.py` (downloads the scores from Google Drive)
- For phase 2 (the PR of branch `_3`):
  - `preregistration_2.md`: the plan and the 13 predictions, committed before
    any of it was analyzed
  - `skill_vs_luck.records.ipynb`: Study A, the audit of the SWE-bench records
  - `skill_vs_luck.one_run.ipynb` and `skill_vs_luck.one_run_test_set.ipynb`:
    Study B, run luck from 1 run
  - `skill_vs_luck.temperatures.ipynb`: Study C, temperature 0.2 and a second
    greedy run
  - `skill_vs_luck.more_models.ipynb`: Study D, 3 more local models
- For Milestones 3 and 4 (the PR of branch `_4`):
  - `preregistration_3.md`: the plan and the 15 predictions of Studies E to
    H, committed before any of them was run
  - `skill_vs_luck.transfer.ipynb`: Study E, transfer between HumanEval+ and
    MBPP+ and their effective size
  - `skill_vs_luck.selection.ipynb`: Studies F to H, the winner's curse on
    real runs and on 20 real prompt variants, the correction to recommend,
    and the reporting protocol on SWE-bench Verified
  - `run_prompt_variants.sh` and `PROMPT_VARIANTS`: the 20 prompt variants
  - `report_agent_scores()`: the reporting protocol of the project
  - `fetch_evalplus_data.py` also saves a snapshot of the EvalPlus
    leaderboard
- For the reproducibility rules of the research guide (PR #1014):
  - A `config` dict at the top of every experiment notebook, with all its
    parameters
  - `test/test_notebooks.py`: a smoke test per notebook on a smaller
    `config`, and superslow tests that freeze the results of Milestones 3
    and 4
  - The paper draft, `papers/Skill_vs_Luck_in_Agent_Benchmarks/paper.md`,
    with its figures copied from the notebooks by `figures/make_figures.py`

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
    - `preregistration_2.md`
16. Only 2 of the 30 SWE-bench Verified records are distinguishable from the
    record they broke, and they carry 33% of the 74.8 points of progress;
    since June 2024, 27 records in a row were each within the noise of the
    record before, and together they still added 46.6 points
    - Study A, `skill_vs_luck.records.ipynb`
17. On the public EvalPlus samples, a tuned 2PL fit estimates the run share
    from 1 run within 10 points in 86% of the draws, but its single test on
    our runs failed: only 7 of the 18 new cases are within 10 points, mostly
    because it is too high at temperature 0.8
    - Study B, `skill_vs_luck.one_run.ipynb` and
      `skill_vs_luck.one_run_test_set.ipynb`
18. At temperature 0.2, runs are 15% to 21% of the variance of 1 run (37% to
    52% at 0.8), and 2 greedy runs give the same result on every task; a 1-run
    score is only a little more precise, though (standard error 3.1 to 3.9
    points against 3.4 to 3.9), and the close pair still cannot be told apart
    - Study C, `skill_vs_luck.temperatures.ipynb`
19. With 6 models, only 44.4% of the 1-run leaderboards keep the order on
    MBPP+; the 3 models within 2.1 points of each other in the published MBPP+
    scores are 5 to 12 points apart in our runs, and Llama 3.1 8B scores 7.6
    points above its published greedy MBPP+ score
    - Study D, `skill_vs_luck.more_models.ipynb`
20. HumanEval+ and MBPP+ rank 18 models almost the same way (Spearman 0.98,
    and 0.99 once task luck is removed, interval up to 1.00), but not on the
    same scale: 6 of the 7 chat models score 14 to 21 HumanEval+ points more
    than a typical model with the same MBPP+ score, and 2 base models about 4
    to 5 points less (8 of 18 benchmark-specific); the effective size is
    about 119 tasks for HumanEval+ (72%) and 214 for MBPP+ (57%); 4 of the 6
    pre-registered predictions held
    - Study E, `skill_vs_luck.transfer.ipynb`, `preregistration_3.md`
21. On the public samples, the best of K runs is above the runs left out by
    just what `sigma_run x E[max of K]` predicts (1.2 points for K = 2 to 4.0
    for K = 20, ratios 1.00 to 1.04); on our runs, with 3 or 5 runs per cell,
    it is 1.41 times the prediction at K = 2, since `p_t (1 - p_t)` from few
    runs is too low (1.20 after correcting for that)
    - Study F, 47 cells with 500 draws per cell and K,
      `skill_vs_luck.selection.ipynb` Cells 1.3 and 1.4
22. The 20 prompt variants of DeepSeek Coder 1.3B score 50.0% to 60.4% on
    HumanEval+ and 41.8% to 54.0% on MBPP+ (Cochran's Q p 0.002 and about
    1e-33); a line asking to handle edge cases costs 7 to 12 points. The best
    of K variants, picked on half of the tasks, scores up to 5.1 points
    (HumanEval+) and 2.5 (MBPP+) less on the other half
    - Study G, 1,000 trials per setting, `skill_vs_luck.selection.ipynb`
      Cells 2.1 to 2.4
23. Only the paired correction is within 1 point of the held-out score in
    every setting (worst -0.97), with the lowest error, so rule H recommends
    it; the binomial rule corrects too much (up to -4.9), empirical Bayes too
    little (up to +2.0), and the conditional estimate is right on the median
    but misses by up to 7.7 points on average. The hybrid estimate, an extra
    look, cuts the error of the conditional one by 1.5 to 3.8 times but still
    misses by up to 1.5 points
    - Study H, `skill_vs_luck.selection.ipynb` Cells 3.1 and 3.2, D-7
24. The protocol takes the SWE-bench Verified leader from 79.2% to 75.1% with
    the paired rule; 6 of the 9 predictions of Studies F and G held, so 10 of
    the 15 of `preregistration_3.md`
    - `skill_vs_luck.selection.ipynb` Cells 4.1, 4.2, and 5.1
25. The correction of the spec, `sigma x sqrt(2 ln K)`, corrects even more than
    the binomial rule on the prompt variants: -8.0 (HumanEval+) and -6.4
    (MBPP+) points at K = 20, against -4.9 and -4.3
    - Extra look promised in #716, `skill_vs_luck.selection.ipynb` Cell 2.5
26. On a leaderboard the corrections fail: picked on half of the SWE-bench
    tasks, the leader drops by 1.88 points on the other half; the rules that
    assume equal candidates correct 3.8 to 6.3 points too much, empirical
    Bayes leaves most of the drop (+1.56, lowest RMSE 3.70), and the hybrid
    estimate comes closest on average (-1.01, RMSE 5.42), so none is within 1
    point; see D-8
    - Extra look, 1,000 splits, `skill_vs_luck.selection.ipynb` Cell 4.3

## Next

1. Upload the scores of the prompt variants to the Drive folder and download
   them in `fetch_coding_scores.py`
2. Build the PDF of the paper with `make`, which needs `pandoc` and `typst`
3. Split the research Gdoc into the tabs of the research guide, and move it to
   the `Research_Class_Projects` folder
4. Get #1013 and #1014 reviewed, then the video and the demo with GP

## History

| Date | Done |
| :--- | :--- |
| 2026-10-22 | Filed #716 |
| 2026-10-23 | GP asked for a results doc and simulations of ranking under uncertainty |
| 2026-10-28 | PR #735 merged: SWE-bench outcome matrix, intervals, paired tests, minimum N |
| 2026-10-29 | PR #808 opened: run-to-run luck on HumanEval+ and MBPP+ |
| 2026-10-30 | GP asked for the research Google Doc per #811; started it |
| 2026-10-30 | Task 1 committed: bootstrap rank intervals and ranking recovery simulation; ties split, see [decisions.md](decisions.md) D-1 |
| 2026-10-30 | Phase 2 pre-registered (`preregistration_2.md`, `f9e4b00c`) |
| 2026-10-31 | Committed split-half rank correlation, held-out leader drop, and winner's curse simulation |
| 2026-10-31 | Phase 2 results of Studies A to D pushed to branch `_3` (`bc8141e1`): 8 of 13 predictions held |
| 2026-11-01 | Added the 2PL IRT model and the effective size; pinned `jax`, `jaxlib` 0.12.0.1 and `numpyro` 0.22.0; see D-2 and D-3 |
| 2026-11-03 | Review fixes: how the top 20 are picked (Cell 8.5), the correction given the tasks, 5,000 trials in Cell 7.1 |
| 2026-11-03 | PR #855 opened: ranking under uncertainty, split-half, winner's curse, and IRT |
| 2026-11-09 | PR #855 merged |
| 2026-11-14 | GP on #808: it can merge once the score files are on Google Drive |
| 2026-11-16 | Merged `master` into #808 to resolve the `README.md` conflict with #855; moved the 88 score files of #808 and the phase-2 scores to the Google Drive folder, downloaded by `fetch_coding_scores.py` |
| 2026-11-21 | PR #808 merged; GP asked on #716 to move the research Gdoc to the `Research_Class_Projects` Drive folder |
| 2026-11-26 | Merged `master` into `_3`; `fetch_coding_scores.py` also downloads the phase-2 scores; opened the PR of `_3` |
| 2026-11-26 | Pre-registered Milestones 3 and 4 in `preregistration_3.md` (`a337ea14`), then started the 20 prompt variants |
| 2026-11-26 | Study E (Milestone 3): 4 of 6 predictions held; see D-6 for the ties in the conditional estimate |
| 2026-11-27 | Generated the 20 prompt variants in about 7.5 hours; 33 MBPP+ answers made by 2 extra streams running at the same time were deleted and made again one at a time, since parallel requests can change greedy answers; scored the 40 files twice |
| 2026-11-27 | Added the hybrid estimate as an extra look, before any analysis of the prompt variants; see D-7 |
| 2026-11-27 | Studies F to H (Milestone 4): 6 of 9 predictions held, and rule H recommends the paired correction; `report_agent_scores()` uses it by default; opened #1014 |
| 2026-11-27 | Extra looks: the correction of the spec on the prompt variants, and the corrections on held-out halves of SWE-bench (D-8); a `config` in every experiment notebook, smoke tests of the notebooks, and the paper draft |

## Open Work

- Phase 2 (#1013) is in review
- Move the research Gdoc to the `Research_Class_Projects` Drive folder, as GP
  asked on #716
