# Canary candidate bank

This directory contains versioned, deterministic questions used by the Noesis
checker.

The checked-in `v1` bank contains 300 candidates: 50 each from arithmetic,
code tracing, modular arithmetic, list ordering, string transformation, and
table aggregation. Each JSONL record stores:

- a stable ID and bank version
- a task family
- the prompt and known answer
- the deterministic answer normalizer
- the generator seed

These records are **candidates**, not automatically trusted canaries. The live
calibration runtime must run each candidate against the configured healthy and
weak targets. Only candidates that satisfy the calibration policy are persisted
as calibrated and made available to the prober.

Regenerate the bank from the repository root with:

```bash
python3 research/Noesis/scripts/generate_canary_bank.py \
    --output-dir research/Noesis/data/canaries
```

Changing the generator or any answer requires a new bank version after `v1` is
used by a calibration run. Do not silently rewrite a version that has produced
evidence.
