# Closed-Form Formula Discovery from Causal/Skill Analysis

## Status
**Status:**: draft
**Complete Specs:**: 10%
**Assignee:**: TBD

## Core Idea

- After running a causal or skill-vs-luck analysis (e.g., the skill/luck
  decomposition used in `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`),
  the fitted relationship is typically a black-box model
- Instead, fit a closed-form symbolic formula that approximates the same
  relationship, and test whether that formula continues to hold
  out-of-sample
  - This is a far stronger test than in-sample fit, since a simple formula
    that survives a new time period or population is much better evidence
    of a real effect than one that only fits the data it was derived from

## Formalization

- Given a fitted relationship $\hat{g}(x)$ from a causal/skill analysis
  (e.g., a tree ensemble or NN), use symbolic regression to find a
  closed-form $g_{\text{sym}}(x)$ minimizing a combination of fit error and
  formula complexity:
  $$
  g_{\text{sym}} = \arg\min_{g \in \mathcal{G}} \;
  \mathrm{error}(g, \hat{g}) + \lambda \cdot \mathrm{complexity}(g)
  $$
- Then evaluate $g_{\text{sym}}$ on a held-out period/population disjoint
  from the one it was fit on

## Key Examples

- **Hedge fund skill/luck distillation**: distilling a skill-vs-luck causal
  model for hedge fund performance (see
  `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`) into an
  interpretable formula, then checking whether it still predicts
  performance in a later market regime
- **Cross-population transfer check**: checking whether a formula
  discovered on one population of agents/funds transfers to a disjoint
  population, as a test of whether it captures a real mechanism rather than
  in-sample noise

## Questions

1. Does forcing a closed form (lower complexity than the original black-box
   model) sacrifice too much in-sample accuracy, or does it actually
   generalize better out-of-sample, as Occam's razor would predict?
2. How should the complexity penalty and basis functions be chosen so the
   discovered formula isn't simply overfitting in a different way?
3. Can existing symbolic regression tools (see
   `draft.Symbolic_regression.md`) be applied directly to the outputs of a
   causal skill/luck framework, or do they need to be adapted?

## Research Topics

- **Symbolic regression tools**: apply symbolic regression (PySR, AI
  Feynman, gplearn, see `draft.Symbolic_regression.md`) to distill causal
  skill/luck models
- **Out-of-sample stability**: measure the stability of the discovered
  formula across time periods and populations
- **Direct regularization baseline**: compare against directly regularizing
  the original black-box model (e.g., via a sparsity penalty) instead of a
  separate distillation step

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Saggese et al., _Causal Analysis of Agent Skill and Luck_
- `draft.Symbolic_regression.md`
- `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`
- Derived from `draft.Misc_ML_ideas.md` (Section: Closed Formulization)
