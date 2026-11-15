# Security analysis results comparison

Updated: October 31, 2026 (America/New_York).

The fresh Bandit and Semgrep runs reproduce the original benchmark scores.
Cisco quick mode has now completed on all 62 snippets; see
[QUICK_SCAN.md](QUICK_SCAN.md) for local model setup and detailed outputs.
Cisco AI Deep SAST was run with `openai/gpt-oss-120b` on Groq. The daily
free-tier quota stopped triage before completion. A conservative partial
evaluation covers 16 fully evaluated snippets; 46 incomplete snippets are
excluded. No complete 62-snippet Cisco score is available.

## Scope and scoring

All baseline runs cover the same 62 Python snippets: 32 vulnerable and 30 safe.
The benchmark includes 20 seeded vulnerabilities, 18 safe examples, and 12
vulnerable/fixed CVE pairs. Tools receive anonymized code with labels removed.

The table uses snippet-level scoring: any finding flags a snippet. TP and FN
are counts among the 32 vulnerable snippets; FP and TN are counts among the
30 safe snippets. FPR is FP / (FP + TN). These metrics assess detection, not
whether a proposed fix works or an explanation is accurate.

## Existing and fresh baseline results

| Tool | Run | Version | TP | FP | FN | TN | Precision | Recall | F1 | FPR | MCC |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bandit | Original `r1` | 1.9.4 | 12 | 7 | 20 | 23 | 63.16% | 37.50% | 0.471 | 23.33% | 0.154 |
| Bandit | Fresh `fresh` | 1.9.4 | 12 | 7 | 20 | 23 | 63.16% | 37.50% | 0.471 | 23.33% | 0.154 |
| Semgrep | Original `r1` | 1.182.1 | 9 | 2 | 23 | 28 | 81.82% | 28.12% | 0.419 | 6.67% | 0.281 |
| Semgrep | Fresh `fresh` | 1.183.0 | 9 | 2 | 23 | 28 | 81.82% | 28.12% | 0.419 | 6.67% | 0.281 |
| Cisco quick | `quick_r1` | `741a9e025ce0` | 15 | 6 | 17 | 24 | 71.43% | 46.88% | 0.566 | 20.00% | 0.284 |
| Semgrep, Cisco rules | `quick_r1` | 1.183.0 | 15 | 6 | 17 | 24 | 71.43% | 46.88% | 0.566 | 20.00% | 0.284 |

Bandit catches three more vulnerable snippets and flags five more safe
snippets than Semgrep. Both miss most of the benchmark's vulnerabilities.
Semgrep's higher MCC reflects its lower false-alarm count. Identical aggregate
scores across baseline runs do not establish that every finding is identical.

Original runs were recorded on October 28, 2026; fresh runs were recorded on
October 31. Semgrep used `p/python` and `p/owasp-top-ten`; Bandit used its
default checks. Semgrep registry rules can change independently of the tool
version. These are single runs; no confidence intervals or repeated LLM
evaluations are available.

Machine-readable scores: [scores.csv](scores.csv). Findings and metadata:
[Bandit](normalized/bandit/) and [Semgrep](normalized/semgrep/). Original `r1`
artifacts are retained; `fresh` contains the runs previously saved under `/tmp`.

Cisco quick's higher recall than the original Semgrep baseline comes from its
broader rules (`p/default`, `p/secrets`, and Cisco custom rules). Its detection
records and metrics exactly match the control with those same rules. The local
model adds explanations and remediation suggestions, without removing or adding
detections. Quick mode uses local Foundation-Sec-8B-Instruct Q8_0; deep mode used
Groq `openai/gpt-oss-120b`, so this is also a comparison of different models.

The quick run completed all 41 required local model analyses, supplied three
additional deterministic-rule analyses, and recorded 38,351 native input/output
tokens in 2,041.05 seconds. It generated 41 remediation sections containing code;
these suggestions have not been applied or tested. See the
[quick report](ai_deep_sast_quick/quick_r1.report.md).

## Cisco partial results: comparison on the same 16 snippets

The completed subset contains **8 vulnerable and 8 safe snippets**. All
tools below are scored on these exact same snippets, including when
re-scoring the existing baseline findings. Do not compare Cisco's subset
metrics directly with the 62-snippet baseline table above.

| Tool | TP | FP | FN | TN | Precision | Recall | F1 | FPR | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bandit, fresh | 1 | 3 | 7 | 5 | 25.00% | 12.50% | 0.167 | 37.50% | -0.289 |
| Semgrep, fresh | 1 | 2 | 7 | 6 | 33.33% | 12.50% | 0.182 | 25.00% | -0.160 |
| Cisco, `r2_partial` | 3 | 1 | 5 | 7 | 75.00% | 37.50% | 0.500 | 12.50% | 0.289 |
| Cisco quick, restricted to this subset | 3 | 4 | 5 | 4 | 42.86% | 37.50% | 0.400 | 50.00% | -0.126 |

The partial deep scan flagged three vulnerable CVE snippets (`CVE-2018-18074_vuln`,
`CVE-2020-7471_vuln`, and `CVE-2022-29217_vuln`) and one safe example
(`SAFE-015`). It missed the other five vulnerable snippets in this subset.
On these same snippets, quick mode caught three vulnerabilities and flagged four
safe snippets. This is an incomplete, nonrandom subset, so these numbers do
**not** establish deep scan's full-benchmark performance or a general ranking.

Selection uses execution state, without consulting vulnerability labels:
all indexed functions in the file must have been processed, the exploratory
phase must be complete, and every candidate in that file must have a triage
verdict. A file with an unresolved candidate or failed function is excluded.
The subset contains seven vulnerable CVE versions, three fixed CVE versions,
five secure examples, and one seeded vulnerability. Its exact snippet IDs
are recorded in the normalized metadata.

Artifacts:

- [Partial Cisco findings](normalized/ai_deep_sast/r2_partial.findings.jsonl)
  and [metadata](normalized/ai_deep_sast/r2_partial.meta.json): four findings
  on the 16 complete snippets. The main scorer uses only these snippet IDs.
- [Matched-subset scores](ai_deep_sast/r2.partial-comparison.csv): all four
  tools scored on those same 16 snippets.
- [All 42 published findings observed so far](ai_deep_sast/r2.observed-findings.jsonl):
  includes findings from incomplete snippets; **not** used to score all 62.
- [Attempt status and phase summary](ai_deep_sast/r2.status.json).

Run `r2` indexed 62 files and 113 functions, successfully analyzed 112
functions, and completed all 64 exploratory batches. One function response
contained Markdown bold markers inside its JSON and was rejected. Detection
created 161 candidates (69 per-function and 92 exploratory). Triage completed
78: 42 confirmed/published, 22 rejected, 12 needs-review, and 2 not-applicable.
The remaining 83 candidates hit Groq's daily quota. Upstream nevertheless
reported success; the adapter correctly rejected the full run because its
phase counters contain errors.

The attempt lasted **1,371.68 seconds (22.86 minutes)**, with 259 saved model
responses totaling 161,218 input and 84,056 output tokens (245,274 combined,
including the rejected formatting response). These are recorded token counts,
not billable or rate-limit-adjusted counts. Actual dollar cost is unknown;
upstream's generic $8.7225 estimate is **not** a Groq billing measurement.
The interrupted default-transport attempt `r1` is retained separately and
is not included in these usage figures or scores.

Groq documents per-minute and per-day limits, and excludes cached prompt
tokens from rate limits; those details explain why total recorded tokens can
exceed the nominal daily allowance before the API refuses more requests.
See [Groq rate limits](https://console.groq.com/docs/rate-limits).

## Cisco installation and configuration

- Source: [cisco-open/ai-deep-sast](https://github.com/cisco-open/ai-deep-sast).
- Exact revision: `741a9e025ce0810cf06764f5869e490d1b80278c`.
- Installation: ignored `third_party/ai-deep-sast/`, with dependencies in
  the project's Python 3.11 `.venv`.
- Mode: default deep scan, with per-function detection, exploratory detection,
  and subsequent evidence-based triage; no upstream code was modified.
- Dry run: 62 files parsed, zero skipped, zero indexing errors, 113 functions,
  615 calls, and 62 coverage items. Every snippet has at least one indexed
  function. No LLM calls were made.
- Model: `openai/gpt-oss-120b` via Groq, reasoning effort `low`, temperature
  zero, and a 4,096-token completion limit in attempt `r2`.
- Integration: cached, validated JSON transport around Cisco's unmodified
  prompts and analysis pipeline. Per-minute rate limits are retried. A future
  attempt stops immediately if the daily quota is exhausted.
- Detection status: **partial, daily quota exhausted**. Only the 16 completely
  evaluated snippets contribute a normalized Cisco score.

Setup evidence: [status JSON](ai_deep_sast/setup_20261031.status.json).
Installed package versions: [environment.txt](ai_deep_sast/environment.txt).
Raw cases, index, and logs live under
`results/raw/ai_deep_sast/setup_20261031/` (git-ignored).

## Resume after Groq quota is replenished

A Groq key is configured in the local git-ignored `.env`.
The runner loads credentials locally and passes them through the subprocess
environment; the key is not included in command arguments or metadata.
No paid-provider fallback is selected automatically.

```bash
export HELPERS_ROOT="$PWD/third_party/helpers_root"
source setenv.sh
python runners/run_ai_deep_sast.py --run_id r2 --resume
python score_results.py
```

`third_party/helpers_root` is a local copy of the course helper dependency.
A normal checkout can instead use the sibling course repo described in the
README. To recreate Cisco's installation:

```bash
git clone https://github.com/cisco-open/ai-deep-sast.git third_party/ai-deep-sast
git -C third_party/ai-deep-sast checkout 741a9e025ce0810cf06764f5869e490d1b80278c
uv pip install -r requirements-ai-deep-sast.txt
```

A completed run writes normalized findings and metadata under
`results/normalized/ai_deep_sast/`, plus a status file under
`results/ai_deep_sast/`. Raw reports, the SQLite store, summary, and logs remain
in `results/raw/ai_deep_sast/<run_id>/`. Use a unique run ID for each evaluation.
`--resume` retains Cisco's store so an interrupted attempt can skip processed
functions and completed exploratory batches. Response caches also prevent
repeat calls. Prior logs and summaries are archived on resume. Failed attempts
do not automatically create scoreable normalized results; this partial export
was explicitly restricted to files with complete analysis and triage.

## Interpretation limits for the Cisco comparison

Only Cisco's **published `true-positive` verdicts** count as findings;
`needs-review` and rejected candidates are excluded. Cisco's own verdict is
an agent judgment, not the benchmark ground truth. The repo's scorer decides
whether each reported finding is a benchmark true or false positive.

Cisco's report does not retain an exact vulnerable-statement line. The adapter
uses the indexed function start when it resolves uniquely, otherwise original
line zero means unknown. Therefore Cisco's line-localization and strict
CWE-plus-line scores must be interpreted as approximate; the primary fair
comparison here is snippet-level detection. Locations are never inferred
from ground-truth labels. Missing fix/confidence fields remain empty.

The runner rejects reported indexing, detection, triage, and JSON parsing
errors rather than counting failed analyses as clean snippets. Upstream can
still turn some unexpected model responses into empty finding lists; inspect
raw logs and reports before making research claims. Full function coverage
also does not guarantee analysis of module-level behavior.

Actual token counts and requests are recorded for completed LLM runs. Monetary
cost is left unknown in normalized metadata because upstream uses a generic
pricing fallback for unfamiliar models; that estimate is not a Groq bill.
The upstream Markdown report also hardcodes a different model label, so use
the normalized metadata to identify the actual configured model.

## Validation

All 20 adapter and transport tests passed, including cache reuse, malformed
and truncated response rejection, formatting-only Markdown cleanup, and
immediate stopping on a daily quota error. The full project suite produced
66 passes and one localhost-ping exploit test timeout in the sandbox. Rerunning that
test with localhost network access passed (1/1). The Cisco index-only run
passed on the complete benchmark. A detection invocation without credentials
was also checked: it exits before scanning and creates no scoreable results.
The partial export was checked against the SQLite execution state; its
normalized metadata restricts scoring to 16 complete snippets.

When Groq quota is available again, resume `r2`, retry the formatting failure,
finish triage, and regenerate `scores.csv`. Keep the partial snapshot labeled
as partial when adding a completed 62-snippet evaluation. The transport
removes Markdown bold markers only outside quoted JSON strings, preserving
response values; malformed or truncated responses still fail validation.

The numeric-only CWE export bug was corrected after inspecting the native
reports. Re-exporting the existing deep-scan findings restored their reported
CWE IDs without changing locations, snippet verdicts, or snippet-level metrics.
All 49 library unit tests passed after the changes. The quick scan and its
matched Semgrep control passed full coverage and native report validation; its setup and outputs are documented in QUICK_SCAN.md.
