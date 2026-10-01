# Predicting Sports Outcomes from Player-Level Features

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Predict the outcome of football and basketball games using the same
  approach as the paper referenced below
- Download the data and find the per-player features available in Madden NFL
- Investigate how these player-level features can be used to predict game
  outcomes

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- [Topic 1]: [What to investigate]
- [Topic 2]: [What to investigate]
- [Topic 3]: [What to investigate]

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the player-level feature dataset
  - Read the reference paper (Google Drive folder) to confirm its modeling
    approach and target metric, so replication stays comparable
  - Scrape or download Madden NFL player ratings per season, and pull
    historical NFL game results and rosters (e.g., via `nfl_data_py`)
  - Build a player-to-team roster mapping per game week, then aggregate
    player-level ratings into team-level feature vectors (mean, weighted by
    snap count, and top-N starters)
  - This is the result: a joined dataset of team-level aggregated player
    features and game outcomes (win/loss, point differential) across
    multiple NFL seasons

- Milestone 2: train and benchmark outcome-prediction models
  - Build a baseline model using only team-level aggregate stats (e.g., Elo,
    prior point differential), without player-level features
  - Reimplement the reference paper's modeling approach on the Madden
    player-level features (logistic regression, gradient boosting, and a
    small NN) to predict game outcome
  - This is the result: a comparison table of predictive accuracy (accuracy,
    log loss) for the baseline vs. the player-level-feature models

- Milestone 3: validate robustness of the player-level signal
  - Ablate the aggregation method (mean vs. weighted vs. top-N starters) and
    compare feature importances across models
  - Evaluate on a temporal holdout (train on past seasons, test on a future
    season) to check the signal is not just in-sample overfitting
  - This is the result: an ablation and holdout-evaluation report showing
    which aggregation choices and features drive predictive power

- Milestone 4: extend the pipeline to basketball
  - Identify an equivalent per-player rating source for basketball (e.g.,
    NBA 2K ratings or advanced box-score metrics) and a game-outcome dataset
  - Repeat the feature-aggregation and modeling steps from Milestones 1-2 on
    the basketball data
  - This is the result: a basketball-specific dataset and model, letting the
    football findings be checked for cross-sport generalization

## References

- [Reference paper (Google Drive
  folder)](https://drive.google.com/drive/folders/1GgW3PVQeMYRP3Bvqfk1AF78T6YlXINew)
