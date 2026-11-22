# Milestone 2: Chess Blame Profiles

- **Goal**: Measure the effect of each original reply in the seven natural
  failures from Linu's merged [Milestone 1 PR](https://github.com/gpsaggese/gpsaggese.github.io/pull/917).
- **Scope**: The selected pilot failures, covering 48 decisions and four starting
  positions. This is a feasibility result for
  [the Milestone 2 plan](04-plan.md#milestone-2-blame-profiles), not completion of
  the remaining 15-task collection.
- **Research document**:
  [Counterfactual Attribution of Agent Failures](https://docs.google.com/document/d/1U7ZVjkG01Ba_PxDo_10wlvjIRB2eZwoguw2bH2Qn4as/edit).
- **Start here**: [Publication guide](../milestone2/PUBLICATION.md), then the
  [full results](../milestone2/experiments/m2_direct_v2/REPORT.md).

## Method

- For every original decision, restore the same before-step checkpoint in both
  arms, retaining the full public conversation, board and remaining limits:
  - **Keep**: Force the entire original reply once, then continue normally.
  - **Redecide**: Sample a new reply, then continue normally.
- Sample 30 independent continuations per arm. The measured repair gain is
  `P(win | redecide) - P(win | keep original reply)`.
- The original reply includes its explanation and command; this experiment does
  not isolate a move-only effect or a particular reasoning mechanism.
- Use Wilson intervals for arm probabilities, Newcombe intervals for differences,
  and two-sided Fisher tests with Holm correction over 48 primary decisions.
- The initial checkpoint-value shortcut was not validated at the proposed sample
  sizes. The prospectively versioned direct-arm fallback supplies the labels;
  [the protocol](../milestone2/experiments/m2_direct_v2/experiment_manifest.json)
  preserves this decision.

## Results

| Quantity | Result |
| :------- | :----- |
| Original failed runs / starting positions | 7 / 4 |
| Original decisions | 48 |
| Primary continuations | 2,880 |
| Independent stability continuations | 1,020 |
| Harmful primary replies resolved after Holm correction | 3 |
| Helpful primary replies resolved after Holm correction | 0 |
| Unresolved primary comparisons | 45 |
| Main-stage usage-rated API cost | $14.075944970 |
| Total M2 cost including initial validation | $15.603057070 |

- Resolved harmful replies are `krk-m4-1` trial 0 steps 3 and 5, and trial 1
  step 1. Their repair gains are approximately 0.667, 0.367 and 0.367.
- Independent repeats cover `krk-m3-1` and `krk-m6-1`, trial 0. Rank agreements
  are 0.986 and 0.798. These measure stability; they do not establish equality.
- The [primary table](../milestone2/experiments/m2_direct_v2/primary_profiles.csv)
  contains every estimate, interval, raw p-value and corrected p-value. The
  [state-value table](../milestone2/experiments/m2_direct_v2/primary_state_values.csv)
  retains independent estimates at adjacent decision boundaries.
- [Primary profiles](../milestone2/experiments/m2_direct_v2/primary_profiles.png),
  [checkpoint values](../milestone2/experiments/m2_direct_v2/checkpoint_values.png)
  and [stability profiles](../milestone2/experiments/m2_direct_v2/stability_profiles.png)
  provide the figures.

## Validation and Limits

- The complete-population audit checks 3,900 completed samples, 3,916 stored
  branch segments, 18,115 replayed decision steps and 16,165 paid model calls.
  Checkpoint identities, model input hashes, environment outputs, referee scores
  and the request ledger reconcile.
- A provider-credit interruption preserved all completed outcomes and resumed 16
  interrupted samples from their saved states. No sampled prefix was redrawn.
- On November 21, all 30 applicable M1/M2 tests passed again with network access
  blocked. The full audit and all 65 primary/repeat profile rows were checked
  again without new API calls. See the
  [publication verification](../milestone2/publication_verification.json).
- Four starting positions and 30 samples per arm limit precision and
  generalization. Zero wins in 30 continuations has a Wilson 95% upper bound of
  about 11.4%; it does not establish impossibility.
- Sparse/diffuse descriptions are exploratory. No failure taxonomy, cross-model
  execution or surrogate training is included in this PR.
- The numerical reports and sample summaries are in Git. Raw checkpoint stores,
  ledgers and credentials are excluded. Full replay requires the separately
  retained evidence archive described in the publication guide.
