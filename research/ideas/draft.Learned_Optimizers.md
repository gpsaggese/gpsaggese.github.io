# Gradient Descent as a Neural Network (Learned Optimizers)

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- Replace a hand-crafted parameter-update rule (SGD, Adam, RMSprop, ...)
  with a neural network that consumes the gradient (and its history) and
  outputs the update itself
- Meta-train the optimizer across many optimization tasks so that it
  _learns_ the update rule, rather than having it hand-designed (cf.
  _Learning to Learn by Gradient Descent by Gradient Descent_, Andrychowicz
  et al., 2016)

## Formalization

Classical update rule (SGD):

\[
\theta_{t+1} = \theta_{t} - \eta \nabla L(\theta_{t})
\]

Learned update rule, where \(g_{\phi}\) is a small RNN/LSTM with hidden state
\(h_{t}\), meta-trained by minimizing the cumulative loss along the
optimization trajectory:

\[
(\Delta\theta_{t}, h_{t+1}) = g_{\phi}\big(\nabla L(\theta_{t}), h_{t}\big),
\qquad \theta_{t+1} = \theta_{t} + \Delta\theta_{t}
\]

\[
\phi^{*} = \arg\min_{\phi} \; \mathbb{E}_{\text{task}}
\left[ \sum_{t=1}^{T} L(\theta_{t}) \right]
\]

## Key Examples

- **Meta-training on small networks**: train the LSTM optimizer on small
  MLPs/CNNs (e.g., MNIST), then test whether it transfers to unseen
  architectures or larger models
- **Learned learning-rate schedules**: a restricted special case where
  \(g_{\phi}\) only outputs a scalar step size rather than a full update
  direction
- **Inner-loop optimizers for meta-learning**: e.g., a learned update rule
  used as the inner-loop optimizer in a MAML-style few-shot learning setup

## Questions

1. Does a learned optimizer generalize to loss landscapes/architectures far
   outside its meta-training distribution, or does it overfit to that
   distribution of tasks?
2. Is the compute overhead of running the optimizer network at every step
   justified by faster convergence, compared to well-tuned Adam/SGD?
3. Can a learned optimizer be distilled back into a simple, interpretable
   closed-form update rule?

## Research Topics

- Benchmark learned optimizers against Adam/SGD/RMSprop on convergence speed
  and final loss across a range of architectures
- Stability and generalization of learned optimizers outside the
  meta-training task distribution
- Compute/memory overhead of the optimizer network itself

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

- Andrychowicz et al., _Learning to Learn by Gradient Descent by Gradient
  Descent_ (2016)
- Metz et al., _Tasks, Stability, Architecture, and Compute: Training More
  Effective Learned Optimizers_ (2020)
- Derived from `draft.Misc_ML_ideas.md` (Section: Gradient Descent as NN)
