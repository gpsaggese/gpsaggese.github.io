# Noesis: Minimum-Detectable-Shortfall Bound for the Reputation Loop

## Status
- **Status**: draft
- **Complete Specs**: 15%
- **Assignee**: TBD

## Core Idea
- The Noesis paper's mechanism-design-risks section
  (`papers/Noesis/04_noesis_market.tex`, `sec:mechanism_design_risks`) lists
  capability misrepresentation, bid shading, and collusion, cites the
  general auction-theory literature (Myerson, McAfee, McAfee and McMillan),
  and concludes: "we do not claim to resolve these problems in this paper"
- This is a citation-only treatment with no result specific to Noesis. The
  paper already has the ingredients for one: the one-sided reliability test
  of Eq. `reliability_lower_bound` is a statistical hypothesis test, and
  hypothesis tests have a standard notion of statistical power, i.e., how
  large a true effect (here, a reliability shortfall) has to be before the
  test reliably detects it
- This idea is to derive that bound: given a window size `|W|` and
  confidence level `delta`, how large a gap `R_min - r_true` can a seller
  sustain while still passing the test with high probability. This turns
  the disclaimed "we do not resolve this" into one concrete, checkable
  result

## Formalization
- Reuses `papers/Noesis/04_noesis_market.tex` and
  `05_noesis_server.tex` notation: measured reliability
  `r_hat(kappa, W) = (1/|W|) * sum_{x in W} s(x)`, and the one-sided lower
  confidence bound
  ```text
  r_lower(kappa, W) = r_hat(kappa, W)
                     - z_{1-delta} * sqrt(r_hat*(1-r_hat) / |W|)
  ```
  with a violation flagged when `r_lower(kappa, W) < R_min(kappa)`
- Minimum-detectable-shortfall bound to derive: for a seller with true
  reliability `r_true = R_min - epsilon`, find the smallest `epsilon` such
  that the test rejects `H_0` (compliance) with probability at least
  `1 - power` (a standard power-analysis calculation for a one-sided
  proportion test):
  ```text
  min_detectable_shortfall(|W|, delta, power) = ...
  ```
  Expected shape: `epsilon` shrinks roughly as `1/sqrt(|W|)`, so the bound
  should be reportable as a simple table of `|W|` vs. detectable `epsilon`
  at fixed `delta` and `power`
- Extend to a multi-round setting: since Eq. `reputation_update` applies an
  exponential-decay update with rate `lambda` across rounds, derive how
  many consecutive rounds a seller shading its reliability by `epsilon` can
  sustain before `rho_alpha(t)` crosses the exclusion threshold `rho_min`

## Key Examples
- **Single-window detection**: a seller with `R_min = 0.999` and true
  reliability `0.997` (a 0.2 percentage-point shortfall); compute the
  window size `|W|` needed to detect this at 95% confidence and 80% power
- **Multi-round evasion**: a seller shades its reliability by exactly the
  minimum-detectable amount every round; using Eq. `reputation_update` with
  a given `lambda`, compute how many rounds it takes before `rho_alpha(t)`
  falls below `rho_min` and the seller is excluded
- **Validation against the simulator**: run the misbehaving-seller scenario
  from [[draft.Noesis_Prototype_Validation]] and check that the number of
  rounds to exclusion observed in simulation matches the number predicted
  by the derived bound

## Questions
1. What sample size `|W|` does the bound say is needed to catch a seller
   shading its true reliability by, say, one percentage point, within a
   single reputation-decay cycle?
2. Does the bound change qualitatively once a seller can adapt its shading
   strategy round to round in response to observing whether it was
   flagged, rather than shading by a fixed `epsilon`?
3. How does verifier error in `c_hat(x)` and `l_hat(x)` (the paper's own
   open question on attribution reliability, `sec:open_questions_server`)
   propagate into the derived bound, i.e., does noisy capability
   measurement widen the undetectable-shortfall region?
4. Is a normal approximation (as used in the paper) tight enough for this
   bound, or does the bound need to be re-derived with an exact interval
   (e.g., Clopper-Pearson) for the small-`|W|` regime where it matters most?

## Research Topics
- Statistical detection theory: minimum-detectable-effect and power
  analysis for one-sided proportion tests, applied to
  Eq. `reliability_lower_bound`
- Sequential/multi-round extensions: how a per-round detection bound
  compounds through an exponential-decay reputation update
  (Eq. `reputation_update`)
- Adaptive-adversary analysis: whether a seller that observes past
  flagging outcomes can systematically stay just under the bound, and
  what that implies for choosing `lambda` and `rho_min`

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan
- Milestone 1: derive the single-window bound
  - Work out `min_detectable_shortfall(|W|, delta, power)` analytically
    from the normal-approximation test in Eq. `reliability_lower_bound`
  - This is the result: a closed-form (or numerically tabulated) bound,
    with a worked table for representative `|W|`, `delta`, `power` values

- Milestone 2: extend to the multi-round reputation loop
  - Combine the single-window bound with Eq. `reputation_update` to derive
    rounds-to-exclusion for a seller shading at the detectable margin
  - This is the result: a second table, rounds-to-exclusion as a function
    of `|W|` and `lambda`

- Milestone 3: validate against the simulator
  - Compare the derived bound's predictions against the misbehaving-seller
    runs from [[draft.Noesis_Prototype_Validation]]
  - This is the result: empirical confirmation (or a documented gap) that
    the analytical bound matches simulated behavior

- Milestone 4: update the paper
  - Replace the "we do not claim to resolve these problems" disclaimer in
    `papers/Noesis/04_noesis_market.tex` (`sec:mechanism_design_risks`)
    with the derived bound and its two tables
  - This is the result: a formal result in place of a citation-only
    discussion

## References
- Noesis paper: `papers/Noesis/04_noesis_market.tex`
  (`TODO(gp)` comment at the end of `sec:mechanism_design_risks`)
- Related ideas: [[draft.Noesis_Prototype_Validation]],
  [[draft.Intelligence_Server]] (attribution reliability, `sec:routing`)
- Myerson, R., _Optimal Auction Design_. (1981)
- McAfee, R. P., _A Dominant Strategy Double Auction_. (1992)
