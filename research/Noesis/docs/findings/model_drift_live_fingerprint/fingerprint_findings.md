# Live Behavioral-fingerprint Pilot Findings

## Scope

- Live pools: two Cloudflare Llama 3.3 70B pools and one Cloudflare Llama 3.1 8B pool
- Pool size: `60` responses (`5` prompts x `12` samples)
- Detector: `mmd_hamming`, `1000` permutations, alpha `0.05`
- The A/A and full-swap rows are single live tests.
- Partial-mixture rates are conditional sensitivity checks on these fixed saved pools; they are not independent estimates of statistical power.

## Results

| Scenario | Swap | Count | Trials | Detection | Median p | p range |
| :------- | ---: | ----: | -----: | --------: | -------: | :------ |
| `healthy_a_a` | 0% | 0 | 1 | 0.000 | 0.4340 | 0.4340–0.4340 |
| `full_model_swap` | 100% | 60 | 1 | 1.000 | 0.0000 | 0.0000–0.0000 |
| `partial_substitution_10` | 10% | 6 | 100 | 0.000 | 0.4490 | 0.2530–0.6670 |
| `partial_substitution_25` | 25% | 15 | 100 | 0.000 | 0.3295 | 0.1390–0.5200 |
| `partial_substitution_50` | 50% | 30 | 100 | 0.870 | 0.0325 | 0.0120–0.1060 |

## Pool diagnostics

| Pool | Responses | Median characters | At least 1,000 characters |
| :--- | --------: | ----------------: | ------------------------: |
| `reference-a` | 60 | 1224.5 | 59 |
| `reference-b` | 60 | 1240.0 | 60 |
| `alternative` | 60 | 1504.0 | 59 |

## Interpretation guardrails

- A non-alerting A/A result is necessary but does not estimate the long-run false-positive rate from one comparison.
- A detected full swap shows separation for 70B versus 8B on this prompt distribution; it does not prove weight identity.
- Longitudinal drift claims require additional independent windows collected later under the same frozen plan.
