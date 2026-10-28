# Frozen diffusion generators with fixed autoregressive repair

Phase 1 implementation and a preregisterable research protocol. Prepared 2026-09-22.
No generated-model results are included. Phases 2–5 are planned, not implemented.

## Research question

Does one frozen, large visual critic plus one frozen AR image editor improve
compositional correctness across four pretrained generators, and are the gains worth
measured end-to-end inference costs? This is a cross-architecture transfer study,
not a causal parameter-scaling law: datasets, encoders, objectives, and resolutions
also differ. Compare against NEP's own refinement, SLD, and xLARD where reproducible.
No backbone training in Phases 1–4. Optional controller training comes later.

## Generator ladder

| Key | Hugging Face repository | Denoiser size (approx.) | Resolution | Steps / guidance |
|---|---|---:|---:|---:|
| sd15 | stable-diffusion-v1-5/stable-diffusion-v1-5 | 0.86B | 512 | 30 / 7.5 |
| sdxl | stabilityai/stable-diffusion-xl-base-1.0 | 2.6B | 1024 | 30 / 5.0 |
| sd35 | stabilityai/stable-diffusion-3.5-large | 8B | 1024 | 28 / 3.5 |
| flux | black-forest-labs/FLUX.1-dev | 12B | 1024 | 50 / 3.5 |

These are established current/former leading open-weight families, not a claim that
all four are SOTA today or on every benchmark. SD1.5 is the historical low-cost anchor;
SDXL uses its base alone, without a refiner. Counts exclude text encoders and VAE;
the runner records actual component parameter counts. Increasing denoiser size does
not guarantee increasing runtime. Measure the ordering rather than assuming it.
The latter two models extend the study from diffusion to flow matching.

**Licenses:** SD1.5: CreativeML OpenRAIL-M; SDXL: Open RAIL++-M;
SD3.5: Stability AI Community; FLUX-dev: FLUX.1-dev Non-Commercial. The latter two
have gated access. Accept applicable terms yourself and authenticate before loading.
These are not four uniformly permissive OSI-open-source releases. If strictly
Apache/MIT weights are mandatory, revise the ladder. Do not silently replace
FLUX-dev with four-step schnell: that changes the compute regime.

**Hardware:** one BF16-capable 80GB CUDA GPU is the target for the primary study;
128GB host RAM and substantial model-cache space are practical planning targets,
not measured minima. Run models separately. CPU offload is available for smaller
GPUs, but transfers count toward latency; use the same offload mode for comparisons.
No quantization in the primary study. SD1.5/SDXL use FP16; SD3.5/FLUX use BF16.
This precision difference is documented, not treated as an isolated size effect.

## Fixed AR correction method: Phase 2 onward

**Visual error detector:** `Qwen/Qwen2.5-VL-32B-Instruct`, frozen, BF16, greedy
text decoding, one 512x512 image, max 256 new tokens. This preserves the original
hypothesis of a detector larger than every generator denoiser. A 7B critic is a
separate budget ablation, not a substitute pooled into the primary results.

**AR image editor:** `hmwhwm/NEP`, file `NEP_512.pt`. HF API verified revision
`95f1071fdb13410c001af984d64f05b70006b144` contains NEP_256.pt, NEP_512.pt and
RLlamaGen.pt. Some README checkpoint links are placeholders; use the actual files.
Code: https://github.com/Huiimin5/NEP . Pair it with the upstream VQ checkpoint
`peizesun/llamagen_t2i/vq_ds16_t2i.pt` and upstream T5 conditioning. Freeze source
revision, weights and sampling settings before testing. Available weights do not
mean the integration has already been GPU-tested.

NEP cannot directly consume all four generators' native continuous latents. The
shared interface is RGB -> NEP visual codes -> repaired visual codes -> RGB:

1. Generate at native resolution and retain the generator-native final latent.
2. Resize output to 512x512, fixed Lanczos, no cropping.
3. Qwen returns JSON with `needs_edit`, `error_type`, `instruction`, normalized
   `bbox=[x0,y0,x1,y1]`, and `preserve`. Give it only image and prompt, never GenEval
   answers, annotation metadata, or evaluator error messages.
4. Convert the rectangle to a binary NEP token-grid mask. Include source/destination
   regions for moves. Invalid boxes or masks larger than 25% of image area cause an
   abstention. Record reason and area. This initial policy is deliberately conservative.
5. NEP encodes the image and autoregressively replaces selected VQ codes conditioned
   on its edit instruction and unmasked context. Start with upstream sampling defaults;
   use CFG 7.5, temperature 1.0, top-k 1000, top-p 1.0 (upstream defaults);
   verify and freeze on disjoint development prompts. One proposal, one candidate,
   one edit round. No benchmark-dependent resampling.
6. Decode the repaired representation. Initially apply no extra acceptance judge:
   directly measure repairs and harm. Later acceptance passes must count their cost.

This is **shared-token latent repair after decoding**, not direct native-latent
editing. Native-latent repair across architectures would require compatible
pretrained bridges or learned adapters. Matching latent channel counts is not enough.

Required controls:

- A: native generator image.
- B: resized-only 512x512 image.
- C: NEP VQ encode/decode with unchanged tokens.
- D: NEP-edited image.

D–C isolates editing benefit; D–B measures net benefit including codec degradation.
For abstentions, scientific condition D uses the same codec round trip as C.
Separately report a deployment variant bypassing the codec on abstention. Also
report outside-mask RGB compositing as a separate ablation, never a hidden change.

## Phase-wise protocol

### Phase 1 — Frozen baselines (implemented)

Use official GenEval (NeurIPS 2023), 553 prompts and four seeds per prompt:
2,212 images per model, 8,848 total. Preserve prompt order and seed assignment.
Same numerical seeds do not imply equal noise tensors across architectures.
Capture images, optional final native latents, model revisions, scheduler settings,
software versions, component parameter counts, synchronized latency, and peak memory.

Use the official evaluator's six categories and their macro-average. Do not replace
it with CLIPScore or a self-grading critic. The implementation validates coverage.
A two-prompt smoke job verifies loading/I/O only; it is not a benchmark result.
Prepare separate development prompts using the upstream prompt generator with a
new seed, explicitly removing overlap with the official test prompts. Do not tune
on the 553 test prompts.

Gate: complete images/checksums and evaluator records, manual output spot-check,
stable hardware/offload setup. Phase 1 measures the actual compute ordering.

### Phase 2 — Editor compatibility and oracle repair

Implement the frozen NEP bridge and controls A/B/C. Create 120 independent development
prompts (20 per category), two seeds each per model. Use human diagnoses and masks
on a subset to test whether the editor can repair known errors before studying the
critic. Measure codec loss C–B, editing benefit D–C, and net benefit D–B.

Gate: positive net correction without unacceptable visual damage. If the codec
loses more quality than editing gains, change representation/editor before training
RL. This is a real risk at 512 resolution for otherwise strong 1024 generators.

### Phase 3 — Fixed critic and AR editor

Freeze Qwen32B + NEP-512, all prompts, mask rules and decoding settings on development
data. Apply the same one-round pipeline to all 8,848 GenEval images. Report category
changes, wrong-to-right repair rate, right-to-wrong damage rate, abstentions, edited
tokens, and 95% paired bootstrap intervals resampling prompts with all seeds retained.
Evaluate all images, including already-correct images, abstentions and failed edits.

Add ConceptMix (NeurIPS 2024) for held-out transfer: freeze 100 prompts per k=1,...,7,
two seeds each, giving 1,400 images/model. This is our declared subset, not necessarily
the paper's full sampling protocol. Preserve concept metadata and questions. Use
the official grading procedure and record the evaluator model/version; the original
uses GPT-4o. If its exact endpoint/version is unavailable, disclose the replacement
and do not treat scores as directly identical to published ones. Qwen32B must not
be the sole judge of its own corrections. Audit disagreement cases with humans.

### Phase 4 — Compute-matched comparisons

Compare raw generation, codec-only, fixed AR repair, best-of-N selected by the same
critic, critic-directed full regeneration, and NEP's original critic localization.
Compare quality/latency curves including critique, encoding, editing and verification.
Add a generator step sweep separately. Extra calls alone do not establish superiority.

Measure outside-mask LPIPS/SSIM against resized originals as preservation measures,
including the codec-only controls. They do not measure semantic correctness. Use at
least 200 blinded paired human judgments stratified across models/error types, two
raters with adjudication. Do not claim FID improvements from a tiny compositional
set without appropriate matching real data.

### Phase 5 — Capacity allocation, then optional learning

Swap Qwen7B vs Qwen32B with NEP fixed. Compare small-generator/large-critic against
large-generator/small-critic at matched total costs. This tests the original idea.
Only if useful, train a small action controller on separate editing rollouts; compare
against a supervised controller and fixed rules. Include rollout GPU-hours in its
training cost. No RL training is needed for Phases 1–4.

## Phase 1 setup

Use Python 3.10/3.11. Create an environment and install a CUDA-compatible PyTorch
build following https://pytorch.org/get-started/locally/ before the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the appropriate CUDA PyTorch wheel here.
python -m pip install -r requirements.txt
hf auth login
```

The package includes official prompts and a checkpoint revision lock. To recreate
these in fresh locations (existing locks are deliberately never overwritten):

```bash
python prepare.py geneval --out data/new_geneval \
  --revision af4902f24d3ca90ebbb446dd9891a59e0f82725f
python prepare.py models --out new_model_lock.json
```

Gated weights still require your authorized access. HF tokens are never recorded
in output manifests. No weights are bundled.

### Dry run and smoke generation

```bash
python generate.py --model sd15 --metadata data/geneval/evaluation_metadata.jsonl \
  --out runs/smoke/sd15 --samples 1 --limit 2 --save-latents --dry-run

python generate.py --model sd15 --metadata data/geneval/evaluation_metadata.jsonl \
  --out runs/smoke/sd15 --samples 1 --limit 2 --save-latents
```

### Full generation

```bash
for model in sd15 sdxl sd35 flux; do
  python generate.py --model "$model" --metadata data/geneval/evaluation_metadata.jsonl \
    --out "runs/baseline/$model" --samples 4 --save-latents
  python audit.py --run "runs/baseline/$model" \
    --metadata data/geneval/evaluation_metadata.jsonl
done
```

For offload add `--offload model` for every model and use a distinct experiment
folder. Do not interpret resident-versus-offloaded timings as a pure size effect.
Resume with the same command: only complete checksummed samples are skipped.
Configuration, code or environment changes require a new run directory. Never use
concurrent writers in one run. Full-step warmup is excluded; load time is separate.

Timing includes pipeline text encoding, denoising, VAE decode, transfers and output
postprocessing, excluding PNG/latent disk writes. Reported memory includes resident
weights. Native latent files are diagnostic, not NEP inputs. FLUX saves packed BNC
latents; decode only with the matching unpacking and VAE scaling/shift conventions.
Safety filtering, if supplied by the pipeline, stays enabled and flags are logged.
A flagged saved image may therefore differ from its pre-filter latent representation.

### Official GenEval evaluation in a separate environment

GenEval uses older MMDetection dependencies; isolate them from Diffusers:

```bash
git clone https://github.com/djghosh13/geneval.git external/geneval
git -C external/geneval checkout af4902f24d3ca90ebbb446dd9891a59e0f82725f
conda env create -f external/geneval/environment.yml
```

Follow upstream README to install MMDetection 2.x in that environment and run
`evaluation/download_models.sh` to obtain its detector. Use the resulting absolute
Python and detector paths below (replace placeholders):

```bash
python evaluate.py --run runs/baseline/sd15 --geneval-repo external/geneval \
  --geneval-python /absolute/path/to/geneval/bin/python \
  --detector-dir /absolute/path/to/detectors --out results/sd15.jsonl
python scores.py --results results/sd15.jsonl \
  --metadata data/geneval/evaluation_metadata.jsonl --samples 4 \
  --out results/sd15_summary.json
```

Repeat for all models. The wrapper preserves upstream evaluator behavior, pins its
revision, and records package versions and detector hashes. The summary refuses
missing samples or categories. It reports both the official task-macro mean and the
separately labelled image-micro mean. GenEval requires no paid judge API.

Later compare codec-only and edited results with matching IDs:

```bash
python scores.py --results results/sd15_codec.jsonl --after results/sd15_edited.jsonl \
  --metadata data/geneval/evaluation_metadata.jsonl --samples 4 \
  --out results/sd15_paired.json
```

Keep edit provenance to establish that after-images derive from the same initial
samples; IDs and metadata alone cannot prove that relationship.

## Implementation and validation

- `prepare.py`: pinned official prompts and model revisions.
- `generate.py`: four frozen pipelines, native latents, timing/memory, safe resume.
- `audit.py`: image completeness, integrity, dimensions and measured costs.
- `evaluate.py`: official evaluator wrapper and provenance.
- `scores.py`: category macro scores and paired prompt bootstrap.
- `tests/test_protocol.py`: 10 integrity/aggregation/pairing tests.
- `configs/models.json`: explicit baseline settings.

```bash
python -m unittest discover -s tests -v
```

The authoring workspace has no CUDA GPU or PyTorch. CPU tests and dry runs were
performed; heavyweight loading, GPU callbacks and actual GenEval detection were not.
The pinned Python dependencies are proposed versions, not a fully GPU-validated
container. Start with the smoke job on your hardware before allocating the full run.

## Sources

- GenEval (NeurIPS 2023): https://github.com/djghosh13/geneval
- NEP (NeurIPS 2025): https://nep-bigai.github.io/ ; https://github.com/Huiimin5/NEP
- Editor weights: https://huggingface.co/hmwhwm/NEP/tree/main
- Critic: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- ConceptMix (NeurIPS 2024): https://github.com/princetonvisualai/conceptmix
- SD1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- SD3.5: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- FLUX: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Callbacks: https://huggingface.co/docs/diffusers/using-diffusers/callback
- xLARD: https://arxiv.org/abs/2603.24965

Bundled benchmark data retain the upstream MIT license. Third-party weights and
model implementation source are not redistributed. Original toolkit code: MIT;
see LICENSE. Findings and gains remain hypotheses until the experiments are run.
