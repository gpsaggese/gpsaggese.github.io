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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Derived from `Research_plan/paper.tex` (Section: Quasi-Stationary Learning /
  Forgetting Mechanisms)
