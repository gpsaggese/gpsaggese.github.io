# Progress Log: Hawkish-Dovish Models for FOMC Text

- **Issue**: [#657](https://github.com/gpsaggese/gpsaggese.github.io/issues/657)
- **Authors**: Mitali Purwar (@MitaliPurwar21), Rachel Bubeck (@rbubeck)
- **Format**: follows `docs/planning/all.document_research.how_to_guide.md`.
  The notebooks hold the details, this log holds the experiments and findings,
  and the Google Doc holds the visual evidence.
- **Google Doc**: to be linked here once created
- **Dates**: UTC

## Status

| Milestone | Scope | Status | PR |
| :-------- | :---- | :----- | :- |
| 1 | Container and data | Done | [#710](https://github.com/gpsaggese/gpsaggese.github.io/pull/710), [#732](https://github.com/gpsaggese/gpsaggese.github.io/pull/732), [#736](https://github.com/gpsaggese/gpsaggese.github.io/pull/736) |
| 2 | `weave.API.ipynb` | Done, not yet merged | Branch `..._Project1_2`, PR to be opened |
| 3 | `weave.example.ipynb` end to end | In progress: tone and yield analysis and the FOMC-RoBERTa evaluation are open | |

| Task | Status |
| :--- | :----- |
| Ingest the Data | Done |
| Define the Problem | Done |
| Trace the Models | Code done; FOMC-RoBERTa blocked by gated access |
| Evaluate with Weave | Baseline and zero-shot done; FOMC-RoBERTa blocked by gated access |
| Link Tone to Yields | Not started |
| Visualize and Report | Not started |

## Built

- `weave.example.ipynb` / `.py`: ingestion, problem definition, tracing, and
  evaluation sections
- `weave.API.ipynb` / `.py`: Weave API tutorial on a toy keyword classifier
- `weave_utils.py`: `TfidfLogRegModel`, `FomcRobertaModel`, `ZeroShotModel`,
  `accuracy_scorer`, `MacroF1Scorer`, and `run_evaluations()`
- `test/test_docker_weave.py`: runs both notebooks inside Docker (marked `slow`)
- `README.md`: setup, run instructions, and Docker changes

## Findings

All model metrics are on `test.csv` (496 sentences: 135 dovish, 114 hawkish,
247 neutral) unless noted otherwise. Code: `weave.example.ipynb`, section
"Evaluating the Models with Weave". Weave project: `msml610-weave-fomc`.

| Model | Macro-F1 | Accuracy | F1 dovish | F1 hawkish | F1 neutral | Latency / sentence |
| :---- | -------: | -------: | --------: | ---------: | --------: | ----------------: |
| TF-IDF + logistic regression | 0.518 | 0.601 | 0.439 | 0.410 | 0.705 | 0.001 s |
| Zero-shot (`facebook/bart-large-mnli`) | 0.344 | 0.375 | 0.478 | 0.393 | 0.161 | 1.06 s |
| FOMC-RoBERTa | not evaluated | | | | | |

1. The baseline reaches macro-F1 0.518 and accuracy 0.601. Accuracy is flattered
   by the neutral class, which is half of the test set (F1 0.705, against 0.439
   dovish and 0.410 hawkish).
2. The `weave.Evaluation` macro-F1 for the baseline (0.5180) equals the direct
   sklearn result in the "Problem Definition" section (0.5180), which confirms
   that `MacroF1Scorer` and the Weave wrapper do not change the result.
3. The zero-shot model scores below the baseline (macro-F1 0.344). It beats the
   baseline on dovish sentences only (F1 0.478 against 0.439) and has F1 0.161
   on neutral. Its label descriptions were not tuned, since that would mean
   tuning on the test set.
4. FOMC-RoBERTa has no official result. Access to the gated repo
   `gtfintechlab/FOMC-RoBERTa` was requested on 2026-10-27 and was still
   awaiting review on 2026-11-02.
   - A preliminary code test used a public copy, `tim9510019/FOMC-RoBERTa`, whose
     config names the official repo as its source. On 120 random test
     sentences it gave macro-F1 0.70 and accuracy 0.71. This confirms the label
     mapping only. It is not comparable with the table above: the sample is
     different, and the copy has not been verified against the official weights.
5. Latency needs serial execution to be meaningful. With Weave's default of 20
   parallel rows, the zero-shot model reported 10.7 s per sentence on a 30-row
   test, because rows waited on the model lock. With `WEAVE_PARALLELISM=1` it
   reports 1.06 s on the full test set.
6. Data coverage: 85 statements were found for 2015–2024. One was excluded:
   2020-03-31, which announces the FIMA Repo Facility and is not a policy
   statement. That leaves 84.
7. 83 of the 84 announcements have a same-day yield. The exception is 2020-03-15,
   a Sunday, which uses the 2020-03-16 close (`same_day_yield = False`).
8. Over the 84 announcements, the 2-year yield change has mean -2.23 bp, standard
   deviation 7.33 bp, and range -27 bp to +11 bp. Evidence:
   `data/processed/fomc_treasury_aligned_2015_2024.csv`.
9. A statement ingestion bug only showed on a clean machine: all 85 downloads
   failed because a function was missing from the committed notebook, while a
   local cache hid the problem. After the fix, a run from an empty statement
   cache on 2026-11-02 loaded 85 statements with 0 failures.
10. The toy model in `weave.API.ipynb` goes from accuracy 0.833 (v1) to 1.000
    (v2). The v2 keywords were chosen by looking at the same 12 sentences, so
    this score shows how versions are compared and says nothing about new text.

### Reproducing the Results

- **Environment**: Python 3.12, `weave` 0.54.0, `transformers` 5.21.0, `torch`
  2.14.1 (CPU), `scikit-learn` 1.9.1; full list in `docker_build.version.log`
- **Baseline**: TF-IDF with unigrams and bigrams, `max_features=10000`; logistic
  regression with `C=1.0`, `max_iter=1000`, `random_state=42`
- **Zero-shot**: `facebook/bart-large-mnli`, with the label descriptions and
  hypothesis template in `ZeroShotModel`
- **Commands**: `./docker_build.sh`, `./docker_jupyter.sh`, then **Run All Cells**
  in `weave.example.ipynb`. FOMC-RoBERTa cells need approved Hugging Face access.

## Next

1. Compute statement tone and run the tone and yield analysis (Spearman
   correlation and bootstrap 95% CI, compared with the baseline)
2. Evaluate FOMC-RoBERTa once access is approved, and check its model card for
   the data it was trained on
3. Add the tone against yield change plot and the final report
4. Create the Google Doc with screenshots and link it from the issue, the PR,
   and this log

## History

| Date | Done |
| :--- | :--- |
| 2026-10-21 | Project skeleton and Weave dependencies. PR #710 merged. |
| 2026-10-23 | Data ingestion and caching. PR #732 merged. |
| 2026-10-27 | Fixed the merge conflict on PR #736. Found the statement ingestion bug. Traced the three models and evaluated baseline and zero-shot. Wrote `weave.API.ipynb` and the README. Removed the unused template files. Requested FOMC-RoBERTa access. |
| 2026-10-28 | Problem definition, baseline, tone definition, and the ingestion fix. PR #736 merged. |
| 2026-11-02 | Merged `master` into `..._Project1_2`. Verified the ingestion fix from an empty cache. Copied the fix to `weave.example.py`, which had only been applied to the `.ipynb`. Created this log. |
