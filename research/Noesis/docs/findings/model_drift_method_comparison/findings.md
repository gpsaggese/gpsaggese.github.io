# Capability versus Hamming-MMD comparison

## Scope

This is a frozen-evidence comparison. It makes no provider calls and reruns no permutation tests. It compares the completed capability and Hamming-MMD experiments at roughly the same per-pool evidence scale while preserving their different prompts, outputs, and trial designs.

## Live full-swap evidence

- Capability grading separated Cloudflare 70B (36/59) from Cloudflare 8B (19/57). Across 56 mutually gradable items, the exact paired McNemar p-value was 0.000855.
- Hamming-MMD separated a 60-response Cloudflare 70B pool from a 60-response Cloudflare 8B pool with empirical p < 0.001 (1000 permutations).
- Both methods therefore detected the large 70B-to-8B change, but they answer different questions: capability grading measures task-quality loss, while MMD measures an output-distribution change.

## Directional substitution comparison

| Scenario | Capability detection | Hamming-MMD detection |
| :-- | --: | --: |
| full swap | 100.0% (1000 trials) | 100.0% (1 trial) |
| partial substitution 10 | 4.4% (1000 trials) | 0.0% (100 trials) |
| partial substitution 25 | 72.7% (1000 trials) | 0.0% (100 trials) |
| partial substitution 50 | 100.0% (1000 trials) | 87.0% (100 trials) |

These rates are not a winner/loser comparison. The capability trials are independent synthetic accuracy windows; the MMD partial-substitution trials reuse fixed live response pools and are correlated.

## Healthy controls

- Capability control: 0 alerts across 1000 synthetic healthy trials.
- Hamming-MMD control: 0 raw and 0 Holm-adjusted alerts across 10 live same-model comparisons.

## Paper conclusion

The evidence supports a complementary two-lane monitor. Capability probes provide interpretable evidence that task performance worsened; Hamming-MMD provides grader-free evidence that behavior changed. The current experiments support large-change detection, not exact model-weight identification or a long-run production false-positive rate.

## Limitation

A strict head-to-head power ranking still requires a future unified collection in which both methods operate on the same prompts, model draws, intervention assignments, and total query budget. That larger campaign is not required for the Phase 0 feasibility paper.
