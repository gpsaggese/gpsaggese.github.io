# Tukey Progress

## Status

| Part                         | Status                                   | PR               |
| :--------------------------- | :--------------------------------------- | :--------------- |
| Plan and docs                | Drafted                                  | `PR_P1`          |
| M1-M2: environment, scoring  | First version built; to revalidate       | `PR_B1`          |
| M3: RLVR training            | First version built; to revalidate       | `PR_R1`          |
| M4: curriculum, sim-to-real  | First version built; to revalidate       | `PR_G1`, `PR_G2` |
| Open research questions      | Open, see [`04-plan.md`](04-plan.md)     | `PR_X1`, `PR_X2` |
| Final submission (Dec 7)     | Not started                              | `PR_F1`          |

## Built

- 13 generator families, a train/test split, training-only standardization, an
  environment that hides the true graph and the test rows
- Tools with a call budget; the prototype's rule; PC, GES, DirectLiNGAM and
  sorting baselines; scoring per [`D1`](03-decisions.md#d1-score-against-the-graph-the-data-can-identify)
- A tool policy and graph decoder trained per
  [`D8`](03-decisions.md#d8-behaviour-cloning-then-reinforce); the curriculum,
  noise and real-data studies
- Last full check: Oct 16, fresh clone, 371 tests passed, notebooks ran

## Findings

- Preliminary; each data file arrives with the PR in brackets

1. Raw simulated data leak the causal order through variance (var-sortability
   0.73-1.00; iSCM 0.49); training-only standardization brings it to 0.50
   - `results/m1/sortability.json` (`PR_B1`)
2. The scripted rule reproduces the prototype within 0.1 SHD
   - `results/m2/proto_check.json` (`PR_B1`)
3. On 600 sealed graphs (5 seeds) the policy beats the rule: -0.275 vs -0.413,
   +0.138 [0.125, 0.151]; SHD 4.25 vs 5.86. The policy is capped per
   [`D6`](03-decisions.md#d6-the-policy-runs-under-a-call-budget); the rule is not
   - `results/m3/confirmatory.json` (`PR_R1`)
4. GES (-0.211, SHD 3.76) and DirectLiNGAM (-0.184, SHD 3.28) beat the policy;
   PC is about level (-0.296, SHD 4.22)
   - `results/m3/confirmatory.json` (`PR_R1`)
5. The policy mostly calls independence tests (53.3 per graph) and regressions
   (15.4), almost never the direction test (0.0007); its gain over the rule is
   +0.194 and +0.202 on Gaussian and uniform noise but +0.027 on exponential
   - `results/m3/confirmatory.json` (`PR_R1`)
6. An RL decoder fed random tool calls already gains +0.075 over the rule on
   development graphs (+0.119 with the cloned policy's calls), so the role of
   test selection is unresolved; the random-tools interval is too narrow (one
   tool sequence reused across seeds)
   - `results/m3/dev_random_tools.json`, `dev_bc_tools_rl_decoder.json` (`PR_R1`)
7. Training on prediction alone (`lam = 0`) collapses to dense graphs: -0.537
   - `results/m3/dev_lam0.json` (`PR_R1`)
8. Staged training vs training on the last stage only: +0.024 [0.015, 0.032];
   a recipe comparison, not proof that stage order helps
   - `results/m4/confirmatory_report.json` (`PR_G1`)
9. Noise ordered low to high vs shuffled: -0.025 [-0.039, -0.013], all 5 seeds
   negative
   - `results/noise_v1/report.json` (`PR_G1`)
10. Sachs (853 cells, one condition): the policy's skeleton F1 falls from 0.74 on
    matched simulated graphs to 0.53; GES, DirectLiNGAM and the rule 0.52, PC 0.58
    - `results/m4/real_final.json`, `results/post_audit/sachs_agreement.json` (`PR_G2`)

## Next

1. `PR_B1`, with the full tests and the Docker build rerun
2. Reviewers' answer on `D1` and `D2`; team roles confirmed
3. Research Google Doc, linked from #628

## History

| Date (UTC) | Done                                                       |
| :--------- | :--------------------------------------------------------- |
| 2026-10-13 | Opened #628; opened and closed #629 (93 files, too large)  |
| 2026-10-16 | First version of M1-M4 runs locally; 371 tests passed      |
| 2026-11-04 | Drafted `PR_P1`; decisions `D1`-`D9`                       |

- Note: the docs were drafted with AI assistance; the authors check and explain
  every line
