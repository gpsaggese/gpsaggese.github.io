# Milestone 2 Publication Guide

- **Goal**: Review and reproduce the completed chess blame-profile feasibility
  study, building on the existing Milestone 1 project.
- **Input**: Seven original failed pilot dialogues, their 48 decisions, and fresh
  keep/redecide continuations under the same model policy.
- **Output**: Primary profiles, checkpoint-value estimates, independent repeats,
  figures, sample summaries and audit records.
- **Users**: Project reviewers and teammates continuing the research.

## Review Order

| File | Purpose |
| :--- | :------ |
| [`../docs/milestone2.md`](../docs/milestone2.md) | Short method, results and limitations |
| [`experiments/m2_direct_v2/REPORT.md`](experiments/m2_direct_v2/REPORT.md) | Complete scientific report |
| [`experiments/m2_direct_v2/experiment_manifest.json`](experiments/m2_direct_v2/experiment_manifest.json) | Prospectively recorded direct-arm protocol |
| [`experiments/m2_direct_v2/primary_profiles.csv`](experiments/m2_direct_v2/primary_profiles.csv) | All 48 primary reply-effect estimates |
| [`experiments/m2_direct_v2/stability_profiles.csv`](experiments/m2_direct_v2/stability_profiles.csv) | All 17 independent repeat estimates |
| [`experiments/m2_direct_v2/results.json`](experiments/m2_direct_v2/results.json) | All 3,900 completed sample summaries |
| [`experiments/m2_direct_v2/REPRODUCIBILITY.md`](experiments/m2_direct_v2/REPRODUCIBILITY.md) | Full archived-data reproduction |
| [`publication_verification.json`](publication_verification.json) | Fresh publication tests and complete-population audit |

- The initial 300-continuation estimator-validation stage is preserved under
  `experiments/m2_v1/validation_stage_20261118/`. Its results are separate from
  the subsequent 3,900 direct-arm continuations.
- The existing HTML report is `reports/milestone2_results.html`. Figures are
  available as PNG and PDF alongside the numerical tables.
- Figures and HTML follow the professor repository's Git LFS configuration.
  Install Git LFS and run `git lfs pull` if a checkout contains pointer files.
- The feature/label separation and timing checks were already part of the M2
  deliverable. Files prefixed `m4_` are compatibility exports, not surrogate
  training or a claim that a later milestone is complete.

## Relationship to M1

- The publication exports the committed M2 result tree, identified in
  [publication_source.json](publication_source.json). Uncommitted later-milestone
  work is excluded by construction.
- The existing M1 runner, configuration, tests and results are unchanged.
- `vendor/pr917` is a relative compatibility link to the parent project. It lets
  the frozen M2 scripts use their original import paths without duplicating M1
  source or changing the experiment code.
- All 50 parent-project files recorded in `source_provenance.json` match the M1
  source pin at publication time. Historical M1 dirty-checkout and hosted-model
  digest limitations remain documented; publication does not resolve them.
- [M2 decisions](docs/03-decisions.md) explain the direct-arm fallback, force-once
  intervention and separation of the frozen study from later work.

## Verification

- Thirty existing tests pass with network access blocked: the 11 M1 chess tests
  and 19 M2 budget, storage, replay, resumption and export checks.
- The entire 3,900-sample replay audit matches the published audit exactly.
- All 48 primary and 17 repeat rows were recomputed from sample summaries,
  including arm counts, gains, intervals, Fisher tests and Holm correction.
- The publication includes no new experiment calls or spending. Reported costs
  describe the completed November 18-19 collection.
- Original evidence databases and the separate local M3 work remain unchanged.

## Reproduction and Raw Evidence

- Reading the tables and reports requires only this checkout.
- Replaying every environment transition requires the full completed-run archive:
  - File: `chess_m2_completed_20261119.tar.gz`
  - Size: 615,108,765 bytes
  - SHA-256:
    `95e4352caf451a5771623638149c9cc3ac9e58584888b43dd7f628649bf0e8af`
  - The archive is retained locally and is not uploaded or publicly hosted by this
    PR. Reviewers need a separate copy to perform the complete raw-data audit.
  - Its verified contents include original M1 inputs, both M2 stores, ledgers,
    source and tablebases, with credentials and the virtual environment excluded.
- Extract that archive into a separate working directory, never over the parent
  M1 project or this compatibility link. Follow
  [the archived reproduction instructions](experiments/m2_direct_v2/REPRODUCIBILITY.md)
  there to regenerate the analysis and run all 30 tests.
- Collection is complete. Paid runners are historical source evidence; do not
  restart them under the completed manifests. No API key is needed for analysis,
  recorded replay or tests.

## Scope

- This PR publishes the selected seven-failure M2 feasibility study. It does not
  claim completion of the remaining 15-task batch, broad generalization or an
  official benchmark score.
- Failure taxonomy, cross-model runtime changes and all M3 files are excluded.
- Independent repeats did not repeat the `krk-m4-1` discoveries; their evidence is
  the primary corrected comparison, with the stated uncertainty.
- Forty-five primary comparisons are unresolved. Nonsignificance is not proof
  that the corresponding original reply had no effect.
