# Cisco quick scan

The AI-enabled quick scan uses `aideepsast.py`: Semgrep detects candidate
locations and local Foundation-Sec-8B-Instruct adds a nine-part explanation,
including remediation code. It does not use Groq or another model API.

Run **`quick_r1` completed on all 62 anonymized Python snippets**. It produced
44 findings on 21 snippets: 41 successful local model analyses and three
analyses supplied by deterministic rule metadata. No required analysis failed.

## Results

| Metric | Cisco quick | Semgrep with identical Cisco rules |
| :--- | ---: | ---: |
| Vulnerable snippets detected | 15 / 32 | 15 / 32 |
| Safe snippets flagged | 6 / 30 | 6 / 30 |
| Precision | 71.43% | 71.43% |
| Recall | 46.88% | 46.88% |
| F1 | 0.566 | 0.566 |
| False positive rate | 20.00% | 20.00% |
| MCC | 0.284 | 0.284 |

By category, the detector caught 13/20 seeded bugs and 2/12 vulnerable CVE
versions. It flagged 4/18 secure examples and 2/12 fixed CVE versions.
The full-benchmark comparison and the comparison on the 16 completed deep-scan
snippets are in [COMPARISON.md](COMPARISON.md).

All 41 local model responses contain extracted remediation sections with code
blocks. No response reached the configured completion limit. This establishes
that suggestions were produced, not that the proposed fixes are correct.

Native counters recorded **17,835 input tokens and 20,516 generation runs**
(38,351 combined) across 41 invocations. These counters use this model's
vocabulary and native timing conventions; they are not directly interchangeable
with another provider's tokenizer or billing records. There were no model API
requests or model API charges.

The runner's monotonic elapsed timer recorded **2,041.05 seconds (34.02
minutes)**, excluding initial download/build setup. The mode name “quick”
describes the Semgrep-first workflow; local generation was not faster than the
unfinished 22.86-minute remote deep-scan attempt on this hardware. The two runs
also differ in model, coverage completed, context, and output requirements.

The upstream CI quality gate returned **1**, meaning findings met its severity
threshold. This was not an execution failure; all required analyses and reports
completed successfully.

## Configuration

- Cisco revision: `741a9e025ce0810cf06764f5869e490d1b80278c`.
- Model: `fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF`.
- Model revision: `16deb1934d4b1d28769f7020c15ae018aa537d1f`.
- Model SHA-256: `d0072df70235e92bd996d4efd2347b38c9db530ed83df21f6be3f8958a9832d0`.
- Model size: 8,541,888,288 bytes; downloaded once and checksum-verified.
- llama.cpp revision: `0ec4027a2a0183de87c007069adddc06e665f7de`.
- Native Metal backend; six CPU threads; 8,192-token context.
- Temperature 0.1; maximum 1,024 generated tokens per finding.
- Rules: `p/default`, `p/secrets`, Cisco `custom-secrets.yaml`, and Cisco
  `custom-zipslip.yaml`.
- AI threshold: WARNING or ERROR. Rules starting with `config.` use their
  deterministic metadata instead of invoking the model, as upstream does.
- The upstream prompt requests OWASP **2021**, while the benchmark labels use
  OWASP **2025**. This run does not evaluate OWASP category accuracy.

## Detection and remediation semantics

Quick mode retains all Semgrep findings, including ones whose AI explanation
questions exploitability. The AI adds analysis rather than automatically
filtering detections. Therefore the same-rule `semgrep_cisco_rules` control
should have identical detection metrics. Improvements over the original
Semgrep baseline can come from different rules; they must not be attributed
to the local model finding additional vulnerabilities.

Normalized locations and CWE/severity metadata come from Semgrep. The
`explanation` includes the model's analysis; `fix` contains the remediation
section when supplied. Suggested fixes have not been applied or tested.

The adapter retains upstream prompts and scanner code. Its local transport
uses the checksum-verified model file instead of re-downloading from Hugging
Face on every invocation, caches each model response, and records native
llama.cpp token counters and timing. Errors do not become clean results.

The source copies are scanned outside the Git working tree because Cisco's
CLI does not expose Semgrep's `--no-git-ignore` option and `results/raw/` is
ignored. Anonymized copies and their line mapping remain archived in the run
directory. The agents receive no benchmark labels or bug-line markers.

## Artifacts

- [Normalized findings](normalized/ai_deep_sast_quick/quick_r1.findings.jsonl)
  and [run metadata](normalized/ai_deep_sast_quick/quick_r1.meta.json).
- [Readable report with remediation code](ai_deep_sast_quick/quick_r1.report.md).
- [Native JSON report](ai_deep_sast_quick/quick_r1.report.json).
- [JUnit XML report](ai_deep_sast_quick/quick_r1.junit.xml).
- [Rule-matched Semgrep control](normalized/semgrep_cisco_rules/quick_r1.findings.jsonl).
- [Aggregate scores](scores.csv).

Raw source copies, line mapping, scanner reports, console logs, and cached
stdout/stderr for each local model invocation are under
`results/raw/ai_deep_sast_quick/quick_r1/` (git-ignored).

## Run again

The local installation is in ignored `third_party/` and `.venv/` directories.
Use the course helper setup described in the main README; the installed local
helper copy can be selected as follows:

```bash
export HELPERS_ROOT="$PWD/third_party/helpers_root"
source setenv.sh
python runners/run_ai_deep_sast_quick.py --run_id quick_r2
python score_results.py
```

To resume an interrupted attempt that has no completed normalized run, pass
the original run ID and `--resume`. Response caches are reused. A completed
run requires a new run ID to avoid overwriting results.

For a small smoke test, select a snippet and a separate output directory:

```bash
python runners/run_ai_deep_sast_quick.py \
  --run_id smoke --snippet_ids SEED-001 --max_tokens 128 \
  --raw_dir /tmp/cisco-smoke/raw --results_dir /tmp/cisco-smoke/normalized
```

The smoke test's short response budget checks integration, not explanation
or fix quality. It is excluded from the main benchmark comparison.

## Recreate the native dependencies

After installing the project requirements and Cisco checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout 0ec4027a2a0183de87c007069adddc06e665f7de
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build --target llama-completion -j 6
mkdir -p third_party/models
curl -L --fail --retry 3 -C - \
  -o third_party/models/foundation-sec-8b-instruct-q8_0.gguf \
  https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF/resolve/16deb1934d4b1d28769f7020c15ae018aa537d1f/foundation-sec-8b-instruct-q8_0.gguf
```

The runner independently checks the model checksum before scanning. Model
weights, native build outputs, and raw caches are not added to Git.

## Validation

The one-snippet integration smoke test passed before the full run. The full
run scanned all 62 source copies with no Semgrep parsing errors, completed all
41 required LLM analyses successfully, and generated all three native report
formats. All 20 adapter and transport tests passed, and the complete library
unit-test directory passed (49 tests). The normalized detection records match the same-rule Semgrep control exactly.

The existing deep-scan adapter's numeric-only CWE parsing bug was also fixed.
Its 42 observed findings and four scored partial findings were re-exported
with corrected CWE values. Snippet-level detection results are unchanged.
