# Chess Milestone 2 — first validation stage

Completed branches: 300/300. Paid calls: 1482. Usage-rated spending: $1.527112; total accounted liability: $1.527112 / $3 approved.

| Original decision | Keep wins | Re-decide wins | Repair gain | 95% interval | Holm p |
| --- | ---: | ---: | ---: | --- | ---: |
| krk-m3-1_trial0_step002 | 0/50 | 0/50 | +0.00 | [-0.071, +0.071] | 1 |
| krk-m4-1_trial0_step001 | 33/50 | 17/50 | -0.32 | [-0.484, -0.124] | 0.007571 |
| krk-m8-1_trial0_step003 | 0/50 | 0/50 | +0.00 | [-0.071, +0.071] | 1 |

The original data is unchanged. Raw branch/checkpoint/tape evidence and per-request accounting are retained in the local paid-run store.

## Interpretation

- For krk-m4-1 trial 0 decision 1, keeping the original complete reply wins 66% versus 34% when re-deciding: repair gain -32 percentage points, unadjusted 95% interval [-48.4,-12.4], Holm-adjusted p approximately 0.00757. This is evidence that the original reply was helpful under this tested continuation policy.
- For the selected krk-m3-1 and krk-m8-1 decisions, both arms won 0/50. No recovery was observed. Each sampled arm's Wilson 95% upper win-rate bound is approximately 7.1%; these results do not prove recovery is impossible or that the task is underspecified.
- A theoretically losing chess move can have near-zero measured repair gain when the tested agent already has little chance of recovery immediately before it. Earlier checkpoints still need profiling; do not assign blame from the tablebase pivot alone.
- The direct-arm infrastructure and first selected contrasts are demonstrated. The checkpoint-value shortcut and full Milestone 2 are not yet validated/completed.

This completes only the first validation stage if all 300 branches finished. Milestone 2 and statistical validation of the shortcut remain incomplete. Primary profiles and stability repeats require subsequent release; they were not started.

Effect intervals are exploratory unadjusted 95% intervals. Holm p-values cover the three selected validation cases. Selection and the whole-reply intervention limit generalization.
