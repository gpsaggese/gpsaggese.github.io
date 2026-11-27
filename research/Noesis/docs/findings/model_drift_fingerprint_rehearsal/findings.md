# Offline Behavioral-fingerprint Control Findings

## Scope

- Validation type: `synthetic implementation control`
- Provider calls: `0`
- Live-provider conclusions supported: `none`
- Overall acceptance: `PASS`
- Statistic: `mmd_hamming`
- P-value: permutation test with `200` permutations

## Results

| Scenario | Swap | Count | Detect | False alarm | Median p | Gate |
| :------- | ---: | ----: | -----: | ----------: | -------: | :--- |
| `healthy_a_a` | 0% | 0 | 0.050 | 0.050 | 0.3550 | `PASS` |
| `full_model_swap` | 100% | 60 | 1.000 | 0.000 | 0.0000 | `PASS` |
| `partial_substitution_10` | 10% | 6 | 0.050 | 0.000 | 0.4300 | `MEASURE` |
| `partial_substitution_25` | 25% | 15 | 0.100 | 0.000 | 0.3175 | `MEASURE` |
| `partial_substitution_50` | 50% | 30 | 0.950 | 0.000 | 0.0050 | `PASS` |

## Interpretation

- A/A uses independent draws from the same synthetic generator
- Partial swaps use exact seeded counts, not Bernoulli sampling
- The evaluator owns swap labels; the detector sees only two pools
- Passing proves the experiment wiring can separate these controls
- It does not validate power on archived or live model responses
