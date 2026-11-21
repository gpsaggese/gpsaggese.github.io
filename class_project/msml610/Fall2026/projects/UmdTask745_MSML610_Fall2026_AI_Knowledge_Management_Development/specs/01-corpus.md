# 01. Corpus

**Status:** draft for review. Every number here comes from `tools/measure_corpus.py` (saved in `corpus_measurements.json`).

## Source

| | |
|---|---|
| Repository | `eclipse-edc/Connector`, Apache-2.0 |
| Pinned commit | `a66db8963acfecec5a7f3316115f42eaf54c0374` (2026-11-04) |
| History | 3,404 commits, 71 MB |
| Records available | 113 under `docs/developer/decision-records/` |
| Code at this commit | 1,751 non-test Java files in 166 modules, about 1.3M tokens |

## Getting the data

```bash
git clone --no-checkout https://github.com/eclipse-edc/Connector.git prototype/.cache/connector
```

Always address content by the pinned SHA. **Never check out:** about 100 paths exceed the Windows path limit, and a checkout drops them with only a warning.

## The 10 records

Your brief: 8 whose references find code, 2 that do not, so that churn is exercised both ways.

| Record | Refs found | Files | Why included |
|---|---|---|---|
| `2025-05-28-dataspace-profile-context` | 3 | 3 | Root of the family |
| `2025-07-25-multiple-participant-identifiers` | 13 | 16 | Builds on 05-28 |
| `2025-08-26-participant-context` | 15 | 15 | Partly superseded by 09-22 |
| `2025-10-15-participant-context-config` | 6 | 6 | Same family |
| `2025-10-29-participant-identifiers` | 4 | 4 | Quietly replaces part of 07-25 |
| `2026-05-11-multi-profile-virtual-connector` | 12 | 49 | References three others; contains real drift |
| `2026-09-22-per-participant-resource-ids` | 21 | 21 | The only explicit supersede in the set |
| `2026-09-24-remove-atomikos` | 6 | 14 | Unrelated decision, for contrast |
| `2025-02-07-http-proxy-data-plane-deprecation` | **0** | **0** | Names only modules that no longer exist |
| `2024-10-10-daps-deprecation` | **0** | **0** | Names only a module that no longer exists |

Left out on purpose: `2026-06-18-spi-module-consolidation` (723 files) and `2026-06-10-remove-dispatcher-registry` (306 files). Links that broad would swamp retrieval and churn.

## Churn: the formula needed fixing

The project spec defines:

`stale(d) = churn(C(d)) / (1 + churn(d))`, "measured over the commits since d was last edited".

Taken literally, `churn(d)` counts edits to the record since its last edit, which is **always 0**, so the denominator does nothing. Measuring also showed that the naive way of counting code churn is wrong in two ways.

**Decision for this prototype:**

1. **Window:** from the record's first commit to the pinned SHA. `churn(d)` is the number of later edits to the record.
2. **Follow renames.** Files were moved in 2026, and without following renames, history stops at the move.
3. **Exclude shared config from churn** (`gradle/libs.versions.toml`, root `settings.gradle.kts`, root `build.gradle.kts`, `gradle.properties`). They stay linked for retrieval. Commits touching them are nearly always unrelated.

**Please check this fix.** The effect of each step:

| Record | First commit | Edits | Naive | + renames | + no shared config | **stale** |
|---|---|---|---|---|---|---|
| `2025-08-26-participant-context` | 2025-08-29 | 1 | 30 | 63 | 63 | **31.5** |
| `2025-07-25-multiple-participant-identifiers` | 2025-07-30 | 0 | 21 | 27 | 27 | **27.0** |
| `2025-05-28-dataspace-profile-context` | 2025-06-12 | 0 | 5 | 16 | 16 | **16.0** |
| `2026-09-22-per-participant-resource-ids` | 2026-09-23 | 0 | 11 | 11 | 11 | **11.0** |
| `2025-10-15-participant-context-config` | 2025-10-17 | 0 | 6 | 10 | 10 | **10.0** |
| `2025-10-29-participant-identifiers` | 2025-10-30 | 0 | 4 | 9 | 9 | **9.0** |
| `2026-05-11-multi-profile-virtual-connector` | 2026-05-11 | 1 | 16 | 17 | 17 | **8.5** |
| `2026-09-24-remove-atomikos` | 2026-09-24 | 0 | 28 | 28 | 0 | **0.0** |
| `2025-02-07-http-proxy-data-plane-deprecation` | 2025-02-10 | 0 | n/a | n/a | n/a | **null** |
| `2024-10-10-daps-deprecation` | 2024-10-10 | 0 | n/a | n/a | n/a | **null** |

Result: **8 scored and 2 null, as you asked.** Two rows show why the rules matter:

- `remove-atomikos`: all 28 commits came from the shared build files. Its real code had none, so it drops from the top of the list to the bottom.
- `participant-context`: following renames doubles the count.

`0.0` and `null` mean different things. 0.0 means linked code that has not changed; null means there is no linked code to measure.

**Limits of the score:**

- It is a ranking, not proof of staleness. Commits soon after a record may be its implementation.
- It favours older records and records with more linked files.
- The warning threshold (15, see `03-offline-pipeline.md`) is a starting value, not a validated one.

## Dead identifiers

A dead identifier appeared in the repo's history and appears nowhere outside `docs/` at the pinned SHA. This is stricter than "not found", which would also catch external types such as `JsonObject`.

| Record | Dead identifiers | What it means |
|---|---|---|
| `2026-05-11-multi-profile-virtual-connector` | `ParticipantProfileResolver`, `DspHttpDispatcherV2025Extension` | **Real drift.** The record introduces this SPI, it was deleted on 2026-05-19 in favour of `ParticipantProfileService`, and the record's last edit on 2026-05-20 still names it |
| `2025-08-26-participant-context` | `RemoteMessageDispatcher` | Later removed, so the record's description is out of date |
| `2025-07-25-multiple-participant-identifiers` | `DefaultDcpParticipantAgentServiceExtension` | Expected: the record says it will be replaced |
| `2026-09-24-remove-atomikos` | `transaction-atomikos` | Expected: the decision was carried out |
| `2025-02-07-http-proxy-data-plane-deprecation` | `data-plane-api`, `data-plane-public-api-v2` | Expected: deprecation followed through |
| `2024-10-10-daps-deprecation` | `oauth2-daps` | Expected: deprecation followed through |

So dead identifiers cannot be read as "stale" automatically, because for a removal decision they are a sign of success. The warning states the fact, and the reader judges. QA item T1 tests the real-drift case.

## Linker signals

| Signal | Records with live links | Finding |
|---|---|---|
| References resolved to code | 8 of 10 | The main signal |
| Files in the commit that added the record | 0 of 10 | 9 are docs-only. The 10th (`daps-deprecation`) touched 3 code files, all since deleted |

This is a finding for RQ2 and not a bug: in this repo, records are merged in docs-only pull requests, separately from the implementation.
