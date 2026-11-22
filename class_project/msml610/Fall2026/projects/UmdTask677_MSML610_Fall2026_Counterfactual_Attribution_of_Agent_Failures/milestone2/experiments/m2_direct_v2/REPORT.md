# Milestone 2 — direct chess blame profiles

Experimental deliverables are complete for the seven-run chess feasibility scope. This is not an official benchmark score or a claim of broad generalization.

All 3,900 continuations completed: 2,880 primary and 1,020 independent stability continuations. Additional usage-rated API cost: $14.075945 / $22 approved. Including the first validation stage: $15.603057.

Audited 16165 paid model calls across 3916 stored branch segments and replayed 18115 recorded decision steps. Parents, initial and subsequent API input hashes, compressed snapshot identities, tool outputs, legal games, referee outcomes and billing all reconcile.

Provider API credits ran out after 1414 completed samples. Completed outcomes were retained, and interrupted samples continued from their last saved states; no sampled prefixes were redrawn.

## Method

Every original decision in all seven failed runs has 30 independent keep-original-reply continuations and 30 re-decide continuations. Positive repair gain means retrying helps; negative gain means the original reply helped. The entire reply, including explanation and command, is the intervention. Original step limits, model policy and defender are preserved.

The initial shortcut equivalence design was underpowered. The prospectively versioned v2 protocol uses its documented direct-arm fallback, so labels do not depend on the unvalidated V-curve shortcut. Samples are fresh and independent of M1 and the first validation stage.

The state-value curves use re-decide wins for V(before) and keep-reply wins for V(after). Offline force-once checks verified that keeping the original reply restores its after-state conversation, board and remaining-call counters. Adjacent decisions provide independent estimates of shared checkpoint values; these are retained separately, not pooled or forced to telescope. See primary_state_values.csv and checkpoint_values.png/PDF.

## Profiles

| Task / trial | Decisions | Baseline wins | Resolved harmful steps | Resolved helpful steps | Top-two positive point share |
| --- | ---: | ---: | --- | --- | ---: |
| krk-m3-1 / 0 | 7 | 3/30 | [] | [] | 1.00 |
| krk-m3-1 / 1 | 7 | 5/30 | [] | [] | 1.00 |
| krk-m4-1 / 0 | 8 | 11/30 | [3, 5] | [] | 1.00 |
| krk-m4-1 / 1 | 1 | 11/30 | [1] | [] | 1.00 |
| krk-m6-1 / 0 | 10 | 1/30 | [] | [] | 0.94 |
| krk-m8-1 / 0 | 3 | 0/30 | [] | [] | n/a |
| krk-m8-1 / 1 | 12 | 1/30 | [] | [] | 1.00 |

Resolved means two-sided Fisher p < .05 after Holm correction across all 48 primary decisions. Point concentration is descriptive and can be distorted by sampling noise; it is not proof that all remaining effects are zero. Full gains and unadjusted 95% intervals are in primary_profiles.csv and primary_profiles.png/PDF.

A zero-win arm with 30 continuations has a Wilson 95% upper win-probability bound of approximately 11.4%. No observed recovery does not establish impossibility. Baseline recovery intervals are retained in pattern_summary.json.

## Independent stability repeats

- krk-m3-1 trial 0: rank agreement 0.9860132971832692; mean absolute gain difference 0.038. See stability_summary.json and stability_profiles.png/PDF for step-level intervals.
- krk-m6-1 trial 0: rank agreement 0.7981885710176262; mean absolute gain difference 0.067. See stability_summary.json and stability_profiles.png/PDF for step-level intervals.

## Milestone 4 handoff

m4_predecision_features.csv and m4_primary_labels.csv join on decision_id. The feature allowlist and group-split rules are in m4_dataset_schema.json. Oracle diagnostics and future outcomes are isolated in decision_diagnostics_NOT_FEATURES.csv. Independent repeat labels are separate. No surrogate has been trained.

The two krk-m4-1 first decisions have identical saved pre-generation context and board, but gains of -0.333 and +0.367 for different original replies. The state-only feature vectors are consequently identical. The timing audit is in m4_input_timing_audit_NOT_FEATURES.json; it contains labels and must not be used as predictor input. Define candidate-reply feature timing explicitly before attempting reply-specific Milestone 4 prediction.

Only four starting positions are represented. Expand the task population, hold out entire positions, and evaluate scaffold changes on unseen tasks before claiming Milestone 4 success.

## Limits

- Selected feasibility study: seven failed trajectories from only four positions.
- 30 samples per arm leave small effects unresolved; confidence intervals are unadjusted, p-values Holm-corrected across 48 primary decisions.
- Sparse/diffuse summaries are exploratory and policy-conditional, not population classifications.
- Independent repeats measure stability, but do not prove equality of all effects.
- The predeclared repeats cover krk-m3-1 and krk-m6-1; krk-m4-1 primary discoveries are not independently repeated by this stability phase.
- Fresh sampling uses no model seed; deterministic environment replay is distinct from model determinism.
- The requested model ID and sampling settings are pinned; no immutable provider model-weight digest is recorded.
- Pre-decision features cannot identify an unobserved candidate reply; action-specific surrogate inputs need an explicit timing definition in Milestone 4.
- Milestone 4 needs expanded task groups and held-out evaluation before a surrogate or feedback benefit can be claimed.
