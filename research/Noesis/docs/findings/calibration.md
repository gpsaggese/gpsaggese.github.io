# Canary calibration findings

- Status: **pending reviewed live execution**
- Candidate bank: `v1` (300 deterministic questions)
- Required repetitions: 3 per target
- Healthy-target requirement: 100% correct on every attempt
- Weak-model requirement: at most 60% correct
- Activation requirement: at least 150 surviving canaries

This file intentionally contains no fabricated measurements. Validate the plan
without provider calls:

```bash
> python3 -m research.Noesis.prober.prober_cli calibrate \
    --healthy groq=meta-llama/llama-3.3-70b-instruct \
    --healthy together=meta-llama/llama-3.3-70b-instruct \
    --weak deepinfra=meta-llama/llama-3.1-8b-instruct
```

After the targets and printed call count are reviewed, repeat with `--execute`.
That command writes measured per-target accuracy, p95 latency, cost, survivor
count, and activation status over this placeholder. It requires an OpenRouter
key and PostgreSQL and makes paid provider calls.
