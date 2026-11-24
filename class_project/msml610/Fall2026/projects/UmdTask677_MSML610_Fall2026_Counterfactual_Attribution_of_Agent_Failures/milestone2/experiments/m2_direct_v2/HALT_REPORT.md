# Milestone 2: paused for API account credits

November 19, 2026. **Milestone 2 is incomplete.** The provider rejected new
requests because the API account has insufficient credits. A separate free
token-count request confirmed HTTP 400 `invalid_request_error` with that reason.
This is an account-credit prerequisite, not exhaustion of the approved
experiment budget. No account purchases or plan changes have been made.

## Collection and spending

| Item | Status |
| --- | ---: |
| Fixed remaining-phase design | 3,900 continuations |
| Completed primary continuations | 1,414 |
| Interrupted, with restorable saved prefixes | 16 |
| Primary continuations not started | 1,450 |
| Independent stability continuations pending | 1,020 |
| Settled paid requests in this stage | 5,964 |
| Additional-stage usage-rated cost | $5.182657765 |
| Additional approved hard ceiling | $22.00 |
| Remaining approved allowance | $16.817342235 |
| Outstanding or uncertain request liabilities | $0 |
| Earlier completed validation-stage cost | $1.527112100 |
| Total Milestone 2 usage-rated cost to date | $6.709769865 |

Ten provider rejections occurred before inference and carry zero usage cost.
Six other active continuations stopped safely after the shared stop flag.
The interrupted prefixes retain 82 paid calls, costing $0.059942140; these are
included in the stage total. Prices are usage-rated estimates, excluding any
account-specific taxes or billing adjustments.

The first validation stage previously completed all 300 continuations. Its
results and limitations remain in
`../m2_v1/validation_stage_20261118/REPORT.md`. The new collection uses the
prospectively declared direct keep/redecide fallback: no checkpoint-value
shortcut is used or claimed validated. Partial, unbalanced collection is not
used to declare primary scientific results.

## Preserved evidence and verification

The local snapshot directory `halt_credit_20261119/` preserves the execution
manifest, progress, results, provider rejection, source audit, and consistent
SQLite backups of the jobs and spending ledgers.

The independent audit of all 1,414 completed continuations passed:

- 6,590 recorded decision steps replayed with matching environments and tool outputs.
- 8,266 content-addressed blobs verified, along with snapshot identities and model-input hashes.
- All final referee scores and per-job paid-call ledgers reconciled exactly.

The original post-run source assertion included a mutable pytest cache file.
Only that non-executable cache changed; every pinned executable and input hash
still matches. Resume source capture excludes runtime/test caches. This is
documented in `halt_source_audit.json`, rather than silently rewriting the
original execution manifest.

Offline preflight restored all 16 interrupted contexts and environments exactly,
with network and model calls blocked. The resume implementation was tested on
segmented trajectories, including exact replay and billing reconciliation.
All 30 applicable original/runtime tests passed. The original M1 archive,
database and ledger remain unchanged. No retail, Milestone 3 or Milestone 4
paid work was started.

## Resume prerequisite and procedure

The API account owner must replenish API credits. No additional experiment
budget approval is needed while continuing within the remaining $16.82.
The existing credential stays outside the repository and is never included in
evidence archives.

```bash
.venv/bin/python scripts/resume_remaining_m2.py --check-only
.venv/bin/python -u scripts/resume_remaining_m2.py --resume
```

The first command performs a free availability check. The second refuses to
dispatch paid inference unless that check succeeds. It preserves all completed
jobs, continues each interrupted prefix from its last saved checkpoint, and
uses the original shared $22 ledger without resetting spending. It does not
redraw interrupted samples. Collection retains the same model policy and fixed
sample counts. Runtime source hashes are captured for the resumed epoch.

After all 3,900 continuations finish, run `scripts/report_remaining_m2.py` for
the complete replay/billing audit, corrected decision tests, uncertainty
intervals, stability analysis, and the isolated predecision feature export.
Only then assess completion of these Milestone 2 deliverables. Four starting
positions cannot establish generalization; no official benchmark score or
Milestone 1 certification is implied.
