# Skill vs Luck in Agent Benchmarks Progress

- Issue: #716
- Research Google Doc: [Research Log](<LINK TO THE GOOGLE DOC>)
- Spec:
  [draft.Skill_vs_Luck_in_Agent_Benchmarks.md](/research/ideas/draft.Skill_vs_Luck_in_Agent_Benchmarks.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| M1: Outcome matrix (SWE-bench Verified) | Oct 28 | Done | #735 |
| M1: Outcome tensor (HumanEval+, MBPP+, 5 runs) | Oct 29 | In review | #808 |
| M2: Task luck, intervals, paired tests, minimum N | Oct 28 | Done | #735 |
| M2: Run-to-run luck | Oct 29 | In review | #808 |
| M2: Ranking under uncertainty | Oct 30 | In review | #___ |
| M3: Split-half rank correlation (SWE-bench) | Nov 15 | Done locally | |
| M3: 2PL IRT and effective size (SWE-bench) | Nov 15 | Done locally | |
| M3: Cross-benchmark rank correlation | Nov 15 | Not started | |
| M4: Winner's curse, simulated and held-out (SWE-bench) | Nov 15 | Done locally | |
| M4: Measured drop on real prompt variants | Nov 15 | Not started | |
| Write-up and paper draft | Dec 7 | Not started | |

## Built

- `skill_vs_luck_utils.py`: loads the SWE-bench Verified matrix, computes the
  leaderboard, bootstrap score and rank intervals, paired tests with Holm,
  minimum-N, the Rasch simulations, split-half correlation, the winner's
  curse, and the 2PL IRT model
- `skill_vs_luck.example.ipynb`: the SWE-bench analysis, Parts 1 to 9
- `data/swebench_verified_irt_agents.csv`, `data/swebench_verified_irt_tasks.csv`:
  the fitted IRT abilities, difficulties, and discriminations
- `skill_vs_luck.API.ipynb`: a walk through each function
- `fetch_swebench_verified.py`: rebuilds `data/` from `SWE-bench/experiments`
  at commit `40f164d`
- Docker setup from `class_project/project_template`

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
   place and none is above 34%; 14 agents win at least 1 resample
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
   (median Kendall 0.90), but not among the top 20 (median Kendall 0.25;
   negative in 1.4% of 1,000 splits)
   - `skill_vs_luck.example.ipynb` Cell 6.1
9. The leader of one half of the tasks scores 2.0 points less on the other
   half, and leads it only about 2% of the time
   - 1,000 splits, ties split; `skill_vs_luck.example.ipynb` Cell 6.2
10. The best of 20 equal variants on 500 tasks looks 3.7 points better than
    it is; the spec's `sigma * sqrt(2 ln K)` is 1.4 to 2.5 times the real
    inflation for K = 2 to 50, while `sigma * E[max of K normals]` is 6% to
    22% too high
    - 500 simulated benchmarks per setting, `skill_vs_luck.example.ipynb`
      Cell 7.1
11. The held-out drop of the best variant is about 1.5 times its inflation on
    all the tasks, close to sqrt(2)
    - `skill_vs_luck.example.ipynb` Cell 7.2
12. Under a 2PL IRT fit, the 500 tasks are worth about 107 equally
    informative tasks for the top 20 agents (297 for all 173), and 39 tasks
    hold half of the information; the top 20 all solve 254 tasks and all fail
    36
    - SVI fit, checked against NUTS on a 40 x 150 subset (ability correlation
      0.997); `skill_vs_luck.example.ipynb` Cells 8.1 and 8.4
13. The ranking by IRT ability matches the leaderboard overall (Kendall 0.97)
    but less in the top 20 (Kendall 0.81), where ACoder drops from rank 9 to 19
    - `skill_vs_luck.example.ipynb` Cell 8.2
14. The fitted difficulty rises with the human estimate of the time to fix
    (medians -1.6, 0.0, 2.9, 5.0 logits), with Spearman 0.45
    - `data/swebench_verified_tasks.csv`, `skill_vs_luck.example.ipynb`
      Cell 8.3

## Next

1. M3: run the split-half and IRT analysis on the HumanEval+ and MBPP+
   results, and correlate the rankings across benchmarks
2. M4: compare the simulated held-out drop with the drop measured on real
   prompt variants
3. Write-up: related work and a first draft of the paper sections

## History

| Date | Done |
| :--- | :--- |
| 2026-10-22 | Filed #716 |
| 2026-10-23 | GP asked for a results doc and simulations of ranking under uncertainty |
| 2026-10-28 | PR #735 merged: SWE-bench outcome matrix, intervals, paired tests, minimum N |
| 2026-10-29 | PR #808 opened: run-to-run luck on HumanEval+ and MBPP+ |
| 2026-10-30 | GP asked for the research Google Doc per #811; started it |
| 2026-10-30 | Task 1 committed: bootstrap rank intervals and ranking recovery simulation; ties split, see [decisions.md](decisions.md) D-1 |
| 2026-10-31 | Committed split-half rank correlation, held-out leader drop, and winner's curse simulation |
| 2026-11-01 | Added the 2PL IRT model and the effective size; pinned `jax`, `jaxlib` 0.12.0.1 and `numpyro` 0.22.0; see D-2 and D-3 |

## Open Work

- Task 1 and the work above are not on GitHub yet, waiting for Pratham's
  review
- #808: run-to-run luck, waiting for review; edits the same Takeaways cell of
  `skill_vs_luck.example.ipynb` as PR #___, so the second to merge needs a
  rebase
