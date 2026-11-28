- [.] Make changes to paper1 according to the notes
- [ ] Make paper2 match paper1 (same nomenclature, same concepts)
- [ ] Make paper2 clear
- [ ] Merge evaluation from papers/Noesis_Server_Evaluation into papers/Noesis_Probing_Model_Quality/

- [ ] Propagate changes to business/competitors.md and vc_pitch.slides.md

- [ ] Write paper3 based on paper1 (with the structure of paper2)
- [ ] Merge evaluation from papers/Noesis... into 

## Plan: Merge the Server Evaluation into the Probing Paper

- Source (evaluation):
  `papers/Noesis_Server_Evaluation/Noesis_capability-drift_research_results.md`
- Target (paper): `papers/Noesis_Probing_Model_Quality/`
- Evidence: `research/Noesis/docs/findings/`

### Context
- The two documents share the core idea (known-answer canaries, exact grading,
  black-box monitoring), but they describe different systems
- Do not paste one into the other

| Topic       | Paper                                        | Evaluation                                          |
| :---------- | :------------------------------------------- | :-------------------------------------------------- |
| Code        | `checker/`, `prober/`                        | `monitor/`                                          |
| Question    | Did one contract meet `Q_min`?               | Did a series change from its baseline?              |
| Bank        | 300 candidates, 6 families, not calibrated   | Other 60 items, 5 families, not calibrated          |
| Rule        | Upper confidence bound below 0.85, n >= 8    | Fisher p<0.01, >=10-point drop, 2 windows of 60     |
| Timeout     | Quality failure, charged to seller           | Excluded from the accuracy count                    |
| Empty reply | Grade 0                                      | Ungradable                                          |
| Consequence | `FAILED` verdict, meant for reputation       | No automatic penalty                                |
| Live data   | None                                         | Capability warm-up and MMD pools                    |
| "Tier"      | Capability level `c_level`                   | Monitoring layer 0/1/2                              |

- The live numbers of the evaluation do not measure detection with the paper's
  bank

### Decisions to Take First
- [ ] Scope: full integration, including MMD (recommended), or only an
      evaluation section
- [ ] Calibration: run the planned 2,700 paid calls before the merge
      (recommended), or merge with the pilot data only
- [ ] Timeouts: keep both rules and explain why (recommended), or make them the
      same in the code

### Step 1: Fix the Evaluation
- [ ] Report the frozen strict grader for 70B vs 8B: 22/59 vs 16/57, paired
      McNemar p=0.27 on 56 items
    - The reported p=0.00086 comes only from the grader added after reading the
      responses
- [ ] State that the live capability comparison used McNemar, and that Fisher and
      CUSUM ran only on synthetic data
- [ ] Correct the claim "both signals activated" in Section 5.5
- [ ] Add the `small_immaterial_drop` row (2.7% false alarms) to the results table
- [ ] State that the warm-up halted at call 143 and continued manually
- [ ] Remove "stochastic" for the capability completions
    - The plans set no temperature, and the code defaults to 0
- [ ] Add a reference list and use one year for Gao et al. (the paper uses 2025)

### Step 2: Update Stale Parts of the Paper
- [ ] Update the code commit, now `e5f730c3`
- [ ] Remove the claim that the window-closing worker is missing
      (see `checker/checker_runtime.py`)
- [ ] Update the test counts: 96 passed, 5 skipped for `checker/` and `prober/`
      at commit `57a7ec97` (the paper says 51 passed, 4 skipped)
- [ ] Recount modules and lines (`checker/` now has 18 modules)
- [ ] Check the grading and failure tables against the current gateway, which now
      classifies a missing completion as an upstream error

### Step 3: Integrate
- [ ] Abstract and introduction: add the monitoring and pilot contributions,
      replace "no measured model accuracies", update the scope
- [ ] Background: add the four evidence classes and partial substitution to the
      threat model
- [ ] Architecture: add `monitor/` and the layered figure, rename "Tier" to
      "Layer"
- [ ] Canary bank: add the pilot bank, clearly separate from `v1`
- [ ] New section after the verdict section: Fisher, CUSUM, MMD, and how they
      differ from the verdict rule
- [ ] Rename the validation section to "Evaluation" and add:
    - Offline checker replay (`checker_replay/`, 30,000 simulated contracts)
    - Synthetic Fisher results (`model_drift_capability/`)
    - CUSUM pilot (`model_drift_cusum_pilot/`)
    - Live warm-up with both graders (`model_drift_controlled_warmup_live/`)
    - MMD pools and stability (`model_drift_live_fingerprint/`,
      `model_drift_fingerprint_stability_live/`)
- [ ] Related work, limitations, and conclusion: merge without repeats
    - Add limitations: grader defined after collection, correlated MMD trials,
      one collection session, CUSUM given the true healthy rate
- [ ] Add references for Fisher, McNemar, Holm, and CUSUM

### Step 4: Figures and Checks
- [ ] Regenerate the figures from the findings data in the style of the paper,
      instead of copying the PNGs
- [ ] Check every number in the new text against the findings files
    - The synthetic and MMD experiments were not rerun during the comparison
- [ ] Rebuild the PDF and check the layout
