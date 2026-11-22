# Chess Milestone 2 results

The completed report is [Milestone 2 results](experiments/m2_direct_v2/REPORT.md).
The publication includes the experiment protocols, source, all 3,900 completed
sample summaries, primary and repeated-profile tables, figures, audit results,
and dependency locks. A formatted report with all 48 decision estimates is
available in [HTML](reports/milestone2_results.html).

Raw checkpoint stores, local database backups, credentials, Google Docs exports,
and the 615 MB evidence archive are excluded from Git. The archive is retained
locally; its identity and verification are recorded in
[final_archive_verification.json](experiments/m2_direct_v2/final_archive_verification.json).
Full offline replay and the data-dependent tests require that evidence bundle;
the CSV/JSON results and figures can be inspected directly from this repository.

Prepared November 13, completed November 19, 2026. The completed remaining-phase experiment is `chess_m2_direct_v2`; `chess_m2_v1` preserves the preparation and first validation stage.

On November 18, the user approved **$3 for the first validation stage only**,
with results to be reported before releasing later phases. That stage finished
all 300 continuations (1,482 paid model calls) in 93.8 minutes for $1.5271121 of
usage-rated API spending, with zero infrastructure errors. Its authorization,
pinned execution manifest, ledger, results, audit and report are under
`experiments/m2_v1/validation_stage_20261118/`. The original preparation manifest is retained as the
historical proposal; the stage authorization records the subsequent approval.

On November 19, the user approved an **additional $22** to finish the remaining
Milestone 2 phases. The prospectively versioned direct-arm fallback in
`experiments/m2_direct_v2/experiment_manifest.json` specifies 2,880 primary
continuations and 1,020 independent stability continuations, 30 per arm at all
48 primary / 17 repeated decisions. One transactional reservation ledger caps
both phases together at $22. The original archive and first-stage evidence are
unchanged. **All 3,900 continuations and the complete-population audit passed
on November 19.** There were 16,165 paid calls in this stage, costing
$14.075944970 of the additional $22. Including the earlier validation stage,
total Milestone 2 usage-rated API cost is $15.603057070. The unused additional
allowance is $7.924055030; there are no outstanding or uncertain liabilities.

After an API account-credit interruption, all 1,414 completed samples were kept
and the 16 interrupted trajectories continued from their exact saved states.
No paid prefixes or completed outcomes were redrawn. The historical halt
snapshot remains under `experiments/m2_direct_v2/halt_credit_20261119/`.

Results are in `experiments/m2_direct_v2/REPORT.md` and `REPORT.json`, with the
seven primary profiles, checkpoint value curves, two independent stability
repeats, full CSV data, manual interpretation, and the Milestone 4 input-timing
audit. Three harmful original replies survive Holm correction across 48
primary tests. The repeated profiles have rank agreements 0.986 and 0.798;
these describe stability and do not prove equivalence. All 30 applicable tests
pass. The archive and its checksum are under
`evidence/chess_m2_completed_20261119.tar.gz`.

The v1 shortcut equivalence design was underpowered at 30/50 samples. V2 uses
direct keep-original-reply versus re-decide comparisons throughout. All 48
original decisions passed exact offline force-once state/context checks.
Compressed blobs preserve uncompressed identity hashes and all saved state.
The paid runners are retained for provenance; no paid collection remains.
Read-only status: `.venv/bin/python scripts/check_remaining_m2.py`.
Recompute the report without model calls:
`.venv/bin/python scripts/report_remaining_m2.py`.

These are completed experimental deliverables for seven selected failed runs
from four starting positions. They do not complete the broader 20-task batch,
Milestone 3, or Milestone 4. Identical pre-generation states can precede
opposite reply-specific gain estimates: preserve whole-task splits and define
candidate-reply feature timing before training a surrogate. The original v1
proposal below is historical.

## Inputs and provenance

The source is PR [917](https://github.com/gpsaggese/gpsaggese.github.io/pull/917),
commit `8799acfc484febd0c9156ccca0bfa00fb0724aeb`. All 50 project files were
verified against their Git blob IDs and are preserved in `vendor/pr917/`.
`source_provenance.json` records their SHA-256 values.

The original run archive was copied to `inputs/archive/m1_runs.tar.gz`, checked
against SHA-256 `d63ef2d94a695280cffa1896bbe201e0f2d3f79def9b90751801c6f09e3a7172`,
then freshly extracted to `inputs/m1_original/`. This is separate from the
temporary review store, which had accumulated additional replay branches.
Inputs are excluded from git and kept locally. The original archive remains
available in the [shared Drive folder](https://drive.google.com/drive/folders/1cI_sX1Okc0YeLyVycOty-NRNa0xeiiRm).

Historical checkpoints identify a dirty source checkout. That limitation remains
visible in the inventory; pinning the PR source does not retroactively prove
the exact original executable source. Earlier review reproduced 62/62 replay
checks with this PR implementation. New paid execution must pin its own code.

## Inventory

| Quantity | Count |
| --- | ---: |
| Thinking-on pilot runs | 10 |
| Starting positions | 5 |
| Successful / failed runs | 3 / 7 |
| Distinct positions represented by failures | 4 |
| Pilot checkpoints, including initial and terminal | 72 |
| Pilot decision boundaries | 62 |
| Decision boundaries in failed runs | 48 |
| Terminal failed-run endpoints, scored without API calls | 7 |

`experiments/m2_v1/checkpoint_inventory.csv` includes all 72 pilot checkpoints;
`decision_inventory.csv` links each original reply to its before/after pair;
`run_inventory.csv` lists the ten runs. Oracle fields are diagnostic metadata,
never permitted predictor inputs. The original store includes 2,904 snapshots
across all 634 branches, including M1 replay, screening, and A/A branches; those
descendants are historical evidence, not additional independent pilot runs.

Integrity checks verified 4,040 content-addressed blobs, all 634 tape hash
chains, database integrity, tablebase hashes, source hashes, and unchanged
original database and ledger. See `integrity.json`.

## Proposed experiment and budget

| Phase | Design | Continuations | Expected USD | Proposed ceiling USD |
| --- | --- | ---: | ---: | ---: |
| Estimator validation | 3 decisions × 2 arms × 50 | 300 | 1.45 | 3 |
| Primary profiles | All 48 failed-run boundaries × 30 | 1,440 | 6.98 | 16 |
| Stability repeats | Two complete profiles, 17 boundaries × 30 | 510 | 2.47 | 6 |
| Total | | 2,250 | 10.90 | 25 |

This is a historical-usage forecast, not a quote or guaranteed bill. The
planning range is $8.18–$16.35 (0.75–1.5 times the historical-mean forecast),
not a statistical confidence interval. The estimate uses 500 original fresh
forks and 2,489 actual model calls, repriced at current standard rates. Their
mean is $0.004845 per fork and their empirical 95th percentile is $0.008925.
The source selected still-winnable checkpoints; M2 also includes flat and lost
states, so its cost distribution can differ. A forced terminal reply may end
a branch without a new paid call; the mean-based forecast does not discount it.

Pricing checked November 13, 2026 against
[Anthropic's official pricing](https://platform.claude.com/docs/en/about-claude/pricing):
Haiku 5.5 standard input/output costs $0.10/$0.50 per million tokens up to
100K input tokens; five-minute cache writes are $0.125 and cache reads $0.01.
Above 100K input tokens the base input/output rates become $0.50/$2.50.
No batch discount is assumed. Account-specific charges and taxes are excluded.
`budget.json` records assumptions and stage ceilings.

Recommended authorization sequence: first approve only the $3 validation
stage, review execution quality and spending, then release the remaining
phases within the proposed $25 total. No amount is approved by this document.

The vendor's pre-call ledger check is not a strict concurrent spending cap.
`m2_runtime/budget.py` now replaces it in the stage executor with transactional
SQLite reservations. Each call reserves $0.665: a maximum 1M-token input entirely
at the expensive five-minute cache-write rate, plus the full 16K-token output
allowance at long-context pricing. Requests use standard global inference.
Actual usage, including thinking and cache tokens, reconciles the reservation;
unknown transport failures retain its full liability and stop the stage.
Outstanding reservations survive a crash. Hidden model retries are disabled.
These controls passed seven budget tests and three transport integration tests,
alongside all eleven original chess tests. The first validation stage used the approved $3 ceiling; the subsequent remaining phases used one separately approved shared $22 ceiling.

## Estimand and decisions selected for validation

Repair gain is `P(win | re-decide) - P(win | keep original reply)`; the original
proposal's Delta has the opposite sign. The intervention unit is the complete
assistant reply, including explanation and command, not the move alone.
Both arms restore the same before-checkpoint, same remaining limits, and same
model policy. A forced reply consumes its original step.

| Case | Original decision | Reason selected |
| --- | --- | --- |
| A | `krk-m3-1`, trial 0, step 2 | Nonterminal loss of the forced win |
| B | `krk-m4-1`, trial 0, step 1 | Position remains winnable |
| C | `krk-m8-1`, trial 0, step 3 | Terminal rook loss |

Selection is deterministic from the original trajectories and tablebase labels,
without using M1 screening win rates or new M2 outcomes. These are diagnostic
cases and not a representative causal-effect sample. `force_once_preflight.json`
records offline matching of environment, API-visible conversation, and step /
termination counters for all three cases. Actual spending metadata intentionally
differs for a forced reply.

Primary sampling uses fresh independent outcomes, 30 per nonterminal boundary,
without reusing M1 outcomes or direct-arm validation samples. Terminal values
come from actual sealed-board outcomes, without paid calls. Repeat the full
trial-0 profiles of `krk-m3-1` and `krk-m6-1` independently for stability.

Compare direct gains with `V(before) - V(after)` only after primary profiles
provide independent checkpoint estimates. Use the frozen equivalence criterion
in the manifest; unresolved comparisons stay unresolved. With 30/50 samples,
the criterion can be underpowered. Do not enlarge the tolerance after seeing
results or present unvalidated checkpoint-value changes as causal blame.

Use Wilson intervals for win rates and Newcombe intervals for differences.
Apply Holm correction for primary decision-level tests. Retain shared-checkpoint
dependence when summarizing whole profiles. No optional stopping or automatic
extra sampling. Additional confirmatory experiments require another manifest.
Sparse/diffuse conclusions remain exploratory when uncertainty is large.

## Reproduce the preparation without paid runs

Use Python 3.12. `requirements-preparation.lock` records the exact environment
used on this Mac; package availability may differ on another platform.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-preparation.lock
.venv/bin/python scripts/prepare_m2.py
.venv/bin/python scripts/verify_force_once_v2.py
```

The inventory script imports only board/store modules and opens the original
database read-only. Force-once verification copies the store to a temporary
directory, makes only the scratch database writable, blocks socket connections and live model queries, uses a local
LiteLLM cost map, and avoids the user's model configuration directory. It never
launches the job list. The separately authorized paid stage command is
`scripts/run_validation_stage.py`, which refuses to initialize a second run
over an existing execution manifest.

## Milestone 4 compatibility

All trials and descendants of one starting position must stay together in
future train/evaluation splits. Keep pre-decision features in a separate table
from oracle diagnostics, outcomes, and gain labels. These seven failures cover
only four starting positions; expand the task population before making claims
about generalization. No retail jobs or Milestone 4 training are part of this
preparation.
