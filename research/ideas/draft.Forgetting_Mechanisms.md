# Forgetting Mechanisms in Non-Stationary Environments

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- In drifting environments, classical learning theory's assumption that
  "more data is always better" breaks down
- Old data may hurt performance because it reflects an obsolete distribution
- The rate at which a model forgets (or discounts) old observations is a
  critical design parameter that determines the adaptation-speed vs.
  stability tradeoff

## Formalization

- Exponential decay weighting of observations:
  $$
  w_{t} = e^{-t/\tau}
  $$
- $\tau$ is the forgetting time constant
- Short $\tau$ means fast adaptation but high variance
- Long $\tau$ means stability but slow adaptation

## Key Examples

- **Reinforcement learning in changing environments**: An RL agent playing a
  game where rules gradually change must weight recent experience more
  heavily. Too much memory and it keeps using obsolete strategies; too little
  and it can't learn patterns.
- **Online advertising**: Click-through rate models must balance learning user
  preferences (requires memory) with adapting to changing tastes (requires
  forgetting). A user who clicked sports ads in January might prefer fashion
  ads by March.
- **Robotics in aging systems**: A robot learning to compensate for motor
  degradation must forget its old calibration while retaining general
  manipulation skills. Which memories should persist and which should decay?

## Questions

1. If we forget too fast ($\tau$ too small), we lose generalization. If we
   forget too slow, we overfit to obsolete data. Is there a "no free lunch"
   theorem for memory decay?
2. Should forgetting rate $\tau$ itself be learned from data? Or does this
   create a "meta-overfitting" problem where we overfit to recent drift
   patterns?
3. Can we design "content-aware" forgetting (forgetting irrelevant noise
   while retaining stable patterns)? Is this possible without knowing the
   future?
4. Does human memory's selective forgetting (we remember important events but
   forget mundane details) suggest an optimal forgetting strategy for ML? Can
   we formalize "importance-weighted forgetting"?
5. Is there a fundamental trade-off between adaptation speed and stability?
   Can we prove lower bounds on regret in terms of drift rate and memory
   window?

## Research Topics

- Optimal $\tau$ under bounded drift
- Bias-variance tradeoff in non-stationary environments
- Regret bounds with exponential forgetting
- Content-aware forgetting mechanisms
- Relationship between forgetting and regularization

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build non-stationary benchmarks with a known drift rate
  - Implement a synthetic drifting task (a non-stationary bandit or
    regression target) with a controllable drift rate
  - Implement the two Key Examples testbeds: an RL environment with
    gradually changing rules, and a CTR-style environment with drifting
    user preference
  - Instrument each environment to report the ground-truth drift rate and
    a reference optimal-policy baseline
  - This is the result: two tested environments with configurable drift
    rate and a reference optimal baseline to measure regret against

- Milestone 2: implement exponential-decay forgetting and sweep $\tau$
  - Add exponential decay weighting $w_t = e^{-t/\tau}$ to a baseline
    online learner (online regression or Q-learning)
  - Sweep $\tau$ over a wide range and measure regret/error against drift
    rate on both testbeds
  - Fit the empirical relationship between the best-performing $\tau^*$
    and the environment's drift rate
  - This is the result: adaptation-speed vs. stability curves for fixed
    $\tau$, and an estimated $\tau^*$-vs-drift-rate relationship

- Milestone 3: test a learned $\tau$ against the best fixed $\tau$
  - Implement an online or meta-learned procedure that adapts $\tau$
    during training instead of fixing it
  - Compare it against the best fixed $\tau$ from Milestone 2 on the drift
    pattern used for tuning
  - Re-evaluate on drift patterns not seen during tuning to check for
    meta-overfitting to the tuning pattern
  - This is the result: a comparison of learned vs. fixed $\tau$,
    including whether the learned rule generalizes to unseen drift
    patterns

- Milestone 4: test content-aware forgetting against uniform decay
  - Design a content-aware forgetting rule that weights observations by
    estimated relevance or surprise, not only by age
  - Build a task with mixed stationary and drifting components (some
    features stable, others changing) to separate the two forgetting
    styles
  - Compare content-aware and exponential-decay forgetting on this task
    and relate the results to the bias-variance tradeoff from the Research
    Topics
  - This is the result: quantitative evidence on whether content-aware
    forgetting beats exponential decay when structure is mixed
    stationary/non-stationary

## References

- Derived from `Research_plan/paper.tex` (Section: Quasi-Stationary Learning /
  Forgetting Mechanisms)
