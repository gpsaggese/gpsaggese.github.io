# Modeling comparison: first run vs expanded dataset

First run: `02_modeling.ipynb` on the original founders sheet (1,494 rows).
New run: `02_modeling_v3.ipynb` on the merged sheet (2,047 rows, adds 553 Kaggle YC founders).

All results are descriptive associations, not causal estimates.

| Result | First run | New run | Change |
|---|---|---|---|
| Age sample | n=161 | n=160 | One duplicate row removed |
| Age bands (22 or under to 36+) | 37% rising to about 60-70% | 37%, 48%, 64%, 50%, 60%, 71% | Same pattern |
| Age, simple model | OR 1.10, p=0.033 | OR 1.10, p=0.030 | Same |
| Age, with industry and founding year | OR 1.17, p=0.003 | OR 1.13 (CI 1.03-1.24), p=0.012 | Still positive, a bit weaker |
| Age squared, with controls | p=0.045 | p=0.063 | No longer below 0.05 |
| Ivy League | 10/29 (34%) vs 139/261 (53%), p=0.077 | 10/28 (36%) vs 139/261 (53%), OR 0.49, p=0.110 | Same, not significant |
| Degree level | 52% / 55% / 56%, dropout 22%, p=0.335 | Same rates, p=0.333 | Same |
| YC batch, Winter vs Spring | 57% vs 53% | 57.0% (n=381) vs 52.8% (n=481), original rows only | Same |
| YC batch, pooled with Kaggle rows | not available | 61% vs 57% (n=998) | Biased upward, see below |

## Why age, Ivy and degree did not grow
The Kaggle founders have a founding year, batch and current status, but no age or education data yet. They do not enter the age, Ivy or degree sections. The small differences above come from notebook changes (how industries are grouped, a duplicate row removed, the Fisher odds ratio now labeled as the odds of success).

## Why the Kaggle rows are excluded from batch results
| Source | Success rate | Resolved rows |
|---|---|---|
| Original rows | 54.4% | 883 |
| Kaggle rows | 83.1% | 136 |

In 2014 to 2019 the Kaggle rows show 71-94% success, while the original rows in the same years show 30-63% (2017 is high in both). The Kaggle directory keeps surviving companies, and only 21 inactive companies were found among the 376, so failures are missing. 209 of the 376 statuses rest only on the YC badge. Pooling these rows inflates success rates, so batch results use the original rows only.

## Next steps
- Run the Profile Scraper on the 553 new founders to grow the age, Ivy and degree samples.
- Re-check the 209 low-confidence Kaggle statuses with a second source.
- Then propensity score matching with sensitivity analysis.
