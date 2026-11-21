# Offline half: results

What was run: the offline build on 10 decision records of `eclipse-edc/Connector`, pinned at commit `a66db89`, in a fresh environment with no API key (rule-based replacement detection only). All 21 tests pass. The data is identical to the prototype's build, except that the prototype's `records.json` also carries the model extractor's evidence for the one replacement edge.

**Read these as a first check, not a finding.** The answer keys in `specs/gold/` were drafted by one person and have not been reviewed by the rest of the team. They cover 10 records, and all 10 use the same template.

## Reading the records

| What | Correct |
|---|---|
| Title, date | 10 of 10 |
| Decision, Rationale, Approach sections | 10 of 10 each |
| Links to other records | 7 of 7 |
| Replacements | 1 of 1 |
| Status | none present (EDC records have no status line), so nothing to score |

This says little about messier repositories, because every record follows one template.

## Finding replacements

The one replacement in these records (2026-09-22 replaces part of 2025-08-26, quoting the statement that changed) is found by the rules, and also by the model extractor when a key is set.

The rule parser matches EDC's own wording and link style. On 6 other ADR repositories (117 records) it found 0 of 7 replacements, and the model extractor found all 7 with 0 extra edges. That reference set was built by us from our own detectors, so treat it as tentative.

## Linking records to code

128 links were produced. Against the draft key (75 files):

| Signal | Precision | Recall |
|---|---|---|
| Names in the text (class, module, path, file) | 0.56 (72 of 128) | 0.96 (72 of 75) |
| Files in the commit that added the record | no links | 0.00 |
| Combined | 0.56 | 0.96 |

It finds nearly all the right files, but only about half of what it picks is right. Sentences like "this will remain unchanged" and names mentioned in passing cause wrong links. The commit signal found nothing usable here, because those commits touch only the record itself.

## Staleness

Score = later commits to the linked code ÷ (1 + later edits to the record). 8 records are scored and 2 are null, because no related code was found for them (deliberately, to see how the system behaves).

| Record | Later commits | Record edits | Score |
|---|---|---|---|
| 2025-08-26 participant-context | 63 | 1 | 31.5 |
| 2025-07-25 multiple-participant-identifiers | 27 | 0 | 27.0 |
| 2025-05-28 dataspace-profile-context | 16 | 0 | 16.0 |
| 2026-09-22 per-participant-resource-ids | 11 | 0 | 11.0 |
| 2025-10-15 participant-context-config | 10 | 0 | 10.0 |
| 2025-10-29 participant-identifiers | 9 | 0 | 9.0 |
| 2026-05-11 multi-profile-virtual-connector | 17 | 1 | 8.5 |
| 2026-09-24 remove-atomikos | 0 | 0 | 0.0 |
| 2025-02-07 http-proxy-data-plane-deprecation | none | 0 | null |
| 2024-10-10 daps-deprecation | none | 0 | null |

With the warning threshold at 15, three records are flagged. The threshold is untested. The count includes commits that carried the decision out, so a high score is a hint, not proof, that a record is out of date.

## Gaps

Gap = complexity × (1 − coverage), over the repository's main (non-test) Java modules. Coverage counts only links from the 10 records used, out of 113 records in the repository. Top 5 modules:

| Module | Complexity | Coverage | Gap |
|---|---|---|---|
| spi/control-plane-spi | 1754 | 0.091 | 1594 |
| core/common/lib/core-lib | 1188 | 0.019 | 1165 |
| spi/core-spi | 1067 | 0.059 | 1004 |
| core/control-plane/lib/control-plane-lib | 366 | 0.0 | 366 |
| core/control-plane/control-plane-transform | 310 | 0.0 | 310 |

Coverage here only reflects the 10 records, so a module marked uncovered may be covered by one of the other 103 records.

## Reproducing

`lizard` is pinned to 1.25.0 in `requirements.txt`. Version 1.26.1 counts complexity slightly differently (for example 1755 instead of 1754 for the first module), which changes the order of two modules in the gap list.
