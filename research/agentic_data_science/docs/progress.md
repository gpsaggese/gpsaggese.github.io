# Agentic Data Science Progress

## Status

| Milestone                                | Target date | Status  | PR                                    |
| :--------------------------------------- | :---------- | :------ | :------------------------------------ |
| `v0.1`: two harnesses run end to end     | -           | done    | #527, #587, `gpsaggese/tutorials#706` |
| `v0.2`: first real comparison            | TBD         | planned | -                                     |
| `v0.3`: insight quality is scored        | TBD         | planned | -                                     |
| `v0.4`: all in-house agents are entrants | TBD         | planned | -                                     |
| `v1.0`: results table and tutorial       | TBD         | planned | -                                     |

## Built

- Bake-off harness: `tutorials/agentic_eda/bakeoff/`, see its `README.md`
- RL-for-EDA environment, scoring, and agents: `../RL_for_auto_EDA_prototype.py`,
  design in `../RL_for_auto_EDA_prototype_design.md`
- RL-for-EDA benchmark against PC and GES: `../RL_for_auto_EDA_benchmark.py`
- Profiler agent, a future entrant: `../schema_agent/`, see its `README.md`

## Findings

1. On one 6-node graph with LiNGAM noise (seed 0), the Claude Code agent from
   `PR_R1` recovered the graph at SHD 2 (TPR 0.667, FDR 0.2, F1 0.727); the
   scripted agent got SHD 3 and a random agent about 9
   - This is a single live run, not a benchmark; its output is not committed
   - Command: `python RL_for_auto_EDA_prototype.py`, with `cc_lib` on the path
2. Measuring the direction of an edge needs a higher-order test: the Pearson
   correlation between a cause and its regression residual is zero by
   construction, while the correlation with the absolute residual is not
   - Fixing it lowered the mean SHD of the scripted agent from 4.4 to 1.8 and of
     a mock LLM agent from 9.0 to 6.2, on the prototype's test graphs
3. Mean CPDAG SHD over 10 random linear-Gaussian graphs per row (lower is better):

   | Nodes | Edges | Random | Scripted | PC   | GES  |
   | :---- | :---- | :----- | :------- | :--- | :--- |
   | 5     | 4     | 5.10   | 2.60     | 3.60 | 3.00 |
   | 6     | 6     | 7.30   | 4.90     | 4.50 | 3.50 |
   | 8     | 8     | 12.70  | 6.20     | 6.60 | 4.70 |
   | 10    | 10    | 15.90  | 6.80     | 6.10 | 7.20 |

   - The scripted agent is in the range of PC and GES, and all three beat random,
     so the reward separates good answers from bad ones
   - Command: `python research/agentic_data_science/RL_for_auto_EDA_benchmark.py`
     with `pgmpy 1.1.2` and `causal-learn`, rerun on 2026-11-07

## Next

1. `PR_B2`: load the real UCI Heart and NYC Taxi data
2. `PR_B3`: `schema_agent` as an entrant, after GP answers where the Bake-off
   lives (`plan.md`, Bake-off Harness, Open Question 1)
3. `PR_B4`: external baseline agent
4. Agree with the teams of #859 and #865 on who builds what

## History

| Date       | Done                                                                                              |
| :--------- | :------------------------------------------------------------------------------------------------ |
| 2026-08-14 | Merged #527: RL-for-EDA prototype with the Claude Code agent and SHD scoring                      |
| 2026-08-14 | Merged #531: SOTA review and benchmark scoping for issue #506                                     |
| 2026-10-14 | Merged #587: benchmark of the scripted agent against PC and GES                                   |
| 2026-10-14 | Merged #620: track plan with the Bake-off and Meta-Dataset projects                               |
| 2026-11-07 | Merged `gpsaggese/tutorials#706`: Bake-off harness skeleton; replaces the split PRs #704 and #705 |
| 2026-11-07 | Moved the plan to `docs/plan.md` in the planning-guide format (#880); decisions D-1 to D-7        |

## Open Work

- #879: align on the dev process (thin env, Docker, coding rules)
- #880: this plan, the progress log, and the decisions log
- #859: class project `Comparison_of_Data_Science_Agents`; overlaps with the Bake-off
- #865: Tukey class project; builds the trained RL policy on our prototype
