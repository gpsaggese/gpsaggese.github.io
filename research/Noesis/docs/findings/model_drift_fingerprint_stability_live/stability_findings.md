# Compressed Fingerprint Stability Findings

Five same-day Cloudflare 70B pools were compared pairwise using the frozen Hamming-MMD detector (1,000 permutations).

| Pools | p value | Holm-adjusted p | Raw alert | Holm alert | Peak worker RAM |
| :---- | ------: | --------------: | :-------: | :--------: | --------------: |
| `reference-a` / `reference-b` | 0.4340 | 1.0000 | no | no | 276.7 MB |
| `reference-a` / `reference-c` | 0.3430 | 1.0000 | no | no | 300.8 MB |
| `reference-a` / `reference-d` | 0.6500 | 1.0000 | no | no | 305.8 MB |
| `reference-a` / `reference-e` | 0.3560 | 1.0000 | no | no | 293.3 MB |
| `reference-b` / `reference-c` | 0.4730 | 1.0000 | no | no | 284.8 MB |
| `reference-b` / `reference-d` | 0.5050 | 1.0000 | no | no | 309.9 MB |
| `reference-b` / `reference-e` | 0.3230 | 1.0000 | no | no | 285.0 MB |
| `reference-c` / `reference-d` | 0.8040 | 1.0000 | no | no | 230.5 MB |
| `reference-c` / `reference-e` | 0.7440 | 1.0000 | no | no | 276.0 MB |
| `reference-d` / `reference-e` | 0.4450 | 1.0000 | no | no | 299.3 MB |

Raw alerts: `0/10`; Holm-familywise alerts: `0/10`.
Peak observed worker RSS: `309.9 MB` (hard cutoff: `2048 MB`).

These pools were collected in one compressed session, so this measures same-session stochastic stability rather than multi-day provider drift.
