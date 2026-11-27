# Controlled Warm-up Findings

## Scope

This was a 240-attempt, four-route warm-up over 60 original prompts: 12 each
from math, expert science, code tracing, instruction following, and multi-step
reasoning. It is an implementation and pilot-data result, not proof of model
identity, numerical precision, or provider substitution.

The original run halted after two Cloudflare 8B timeouts at attempt 143. After
manual review, a separately recorded continuation attempted only the remaining
97 calls. The timeout calls were preserved and were not retried.

## Collection Integrity

- 240 planned calls were attempted exactly once.
- 237 calls returned provider status `ok`; three Cloudflare 8B calls timed out.
- Two `ok` Cloudflare calls returned null completion text: one 3B and one 70B.
  They are retained as ungradable response failures.
- Every successful response matched the requested provider and model.
- No fallback, attribution, usage-metadata, or persistence failure occurred.
- Total measured cost was **$0.0193483421**, below the $10 campaign ceiling.

## Capability Results

`Strict` is the frozen primary grader: the answer must be correct and follow
the requested exact output format. `Substantive` is a post-collection audit
added after raw-response review: for math, science, and multi-step questions it
accepts a final `ANSWER:` marker and leading multiple-choice letter even when
the model also emitted reasoning. It does not replace or overwrite the frozen
strict score.

| Route | Usable responses | Strict | Substantive audit | Mean successful latency | Cost |
| :---- | ---------------: | -----: | ----------------: | ----------------------: | ---: |
| Cloudflare Llama 3.1 8B | 57/60 | 16/57 (28.1%) | 19/57 (33.3%) | 5.25 s | $0.002076 |
| Cloudflare Llama 3.2 3B | 59/60 | 12/59 (20.3%) | 22/59 (37.3%) | 0.82 s | $0.000981 |
| Cloudflare Llama 3.3 70B | 59/60 | 22/59 (37.3%) | 36/59 (61.0%) | 2.49 s | $0.010814 |
| CoreWeave Llama 3.3 70B | 60/60 | 23/60 (38.3%) | 35/60 (58.3%) | 1.01 s | $0.005478 |

The substantive 95% Wilson intervals are wide: 22.5–46.3% for Cloudflare 8B,
26.1–50.0% for Cloudflare 3B, 48.3–72.4% for Cloudflare 70B, and 45.7–69.9%
for CoreWeave 70B.

### Paired 70B comparison

The two 70B routes had 59 mutually gradable items. On the substantive audit:

- both were correct on 32 items;
- only Cloudflare was correct on 4;
- only CoreWeave was correct on 2;
- both were wrong on 21.

The exact paired McNemar p-value is 0.6875. This pilot therefore found no
statistically detectable capability difference between the two 70B routes.
That is not evidence that the routes are equivalent; the sample is small and
contains one stochastic completion per route/item.

### 70B family detail

| Family | Cloudflare 70B | CoreWeave 70B |
| :----- | -------------: | ------------: |
| Code tracing | 4/11 (36.4%) | 5/12 (41.7%) |
| Expert science | 11/12 (91.7%) | 11/12 (91.7%) |
| Instruction following | 6/12 (50.0%) | 6/12 (50.0%) |
| Math | 7/12 (58.3%) | 5/12 (41.7%) |
| Multi-step reasoning | 8/12 (66.7%) | 8/12 (66.7%) |

## Interpretation

The intended size signal appears in this pilot: both 70B routes scored well
above the smaller routes on the substantive audit. Cloudflare 70B did not score
below the independent CoreWeave 70B control; their aggregate and family-level
results were close.

Cloudflare 8B had the weakest operational behavior in this run: three timeouts,
95% usable-response availability, and the highest mean successful latency. The
3B route slightly outscored 8B on this small bank, which should be treated as a
pilot anomaly rather than a model-ranking conclusion.

This run cannot establish that Cloudflare served particular weights or fp8
precision, cannot rule out intermittent substitution, and cannot estimate a
false-positive rate. The next evidentiary step is repeated independent A/A
windows followed by the frozen fingerprint test, with the strict and
substantive metrics specified before collection.

## Instrumentation Findings

The run found two issues that are now preserved in the audit trail:

1. The frozen strict grader mixed answer correctness with exact-format
   compliance. The original results remain in `grades.jsonl` and
   `grade_summary.json`; the transparent substantive audit is stored separately
   in `grades_v2.jsonl` and `grade_summary_v2.json`.
2. The gateway initially accepted `content: null` as an `ok` completion. The
   two affected observations remain ungraded, and the gateway now classifies
   future missing or non-text completion content as an upstream error.
