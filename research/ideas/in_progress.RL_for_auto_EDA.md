# Reinforcement Learning for Automated Exploratory Data Analysis

## Status

- **Status:**: in_progress
- **Complete Specs:**: 15%

## Core Idea

- The final goal is to build the equivalent of Claude Code, but for data science
  - An agent dropped in front of an unknown dataset
  - The agent autonomously runs the exploratory data analysis (EDA) loop:
    - Forms hypotheses about the data-generating process
    - Tests the hypotheses with statistical tools
    - Converges on a correct model
- The central insight is that this loop can be trained with reinforcement learning
  with verifiable rewards (RLVR) if the training data comes from randomly generated
  graphical models rather than real datasets
- The ground-truth generating process (the graph) is known at training time
  - The agent's discovered structure can be scored exactly against it
  - This gives a clean, verifiable reward signal
  - This is the same recipe that makes RLVR work for math and code

## Formalization

- Sample a random graph $G^*$ representing the process to discover, using existing
  generators
  - E.g., `pgmpy` random Bayesian networks
  - E.g., `sklearn` `make_regression` / `make_classification` with latent structure
- Generate a dataset $D$ from $G^*$ and split into $D_{train}$ and $D_{test}$
  (in-sample vs. out-of-sample)
- Give the agent a toolbox $T$ of statistical operations
  - E.g., $y$-$x$ regression
  - E.g., hypothesis tests
  - E.g., correlation and independence tests
- The agent's policy $\pi_\theta$ chooses a sequence of tool calls over $D_{train}$
  and emits a discovered graph $\hat{G}$
- Reward is verifiable because $G^*$ is known at training time:

  ```text
  r = score(G_hat, G*, D_test)
  ```

  - `score` combines structural agreement (e.g., structural Hamming distance between
    $\hat{G}$ and $G^*$) with predictive performance on $D_{test}$
- Train $\pi_\theta$ with policy gradient methods (REINFORCE / PPO), as described in
  https://rlhfbook.com/c/06-policy-gradients

## Key Examples

- **Linear-Gaussian recovery**: A random DAG generated with `pgmpy` produces a
  linear-Gaussian dataset
  - The agent uses the `y`-`x` regression tool repeatedly to recover both edge
    existence and direction
  - The agent is rewarded for matching $G^*$
- **Confounded features**: A dataset generated with `sklearn`'s `make_regression`
  includes a hidden confounder that creates a spurious correlation between two
  otherwise unrelated variables
  - The agent must use conditional independence tests to avoid adding a false edge
- **Overfitting to tool-call patterns**: The agent learns a shortcut policy that
  always runs the same fixed sequence of tests regardless of the data
  - The policy scores well on training graphs but fails to generalize to graphs with
    a different topology
  - This failure mode is worth explicitly evaluating

## Questions

1. What is the right verifiable reward for structure discovery, given that ground
   truth is only available on synthetic data and never at deployment time on real
   datasets?
2. Does a policy trained on randomly generated synthetic graphs transfer to real,
   messy datasets with mixed types, missing values, and non-stationary distributions?
3. If RLVR training on synthetic causal graphs is sufficient, could this produce an
   EDA agent that discovers structure comparable to a human data scientist, without
   hand-crafted heuristics of the kind used in existing AutoEDA tools?

## Research Topics

- **Graph generators**: Evaluate `pgmpy` and `sklearn` random model generators for
  producing training environments of controlled complexity
- **Tool design**: Define the toolbox available to the agent (regression, statistical
  tests, mutual information, causal discovery tests) and how tool outputs are fed
  back into the policy
- **Reward design**: Design a verifiable reward that combines structural correctness
  against $G^*$ with out-of-sample predictive performance
- **Policy training**: Apply policy gradient methods (REINFORCE, PPO) per
  https://rlhfbook.com/c/06-policy-gradients to train the tool-use policy
- **Curriculum learning**: Increase graph size, edge density, and noise level over
  training to improve generalization
- **Sim-to-real evaluation**: Benchmark the trained agent on real datasets where the
  ground-truth graph is unknown

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: Synthetic environment
  - Build a random graph generator (`pgmpy` Bayesian networks or `sklearn`
    `make_regression` / `make_classification`), sample $D$, and split into
    $D_{train}$ / $D_{test}$
  - This is the result: a Gym-style environment that exposes $D_{train}$, $D_{test}$,
    and holds out the ground-truth graph $G^*$ for scoring
- Milestone 2: Tool-use baseline
  - Implement the statistical toolbox ($y$-$x$ regression, correlation and
    independence tests) and a scripted / heuristic agent that calls the tools and
    emits a discovered graph $\hat{G}$
  - This is the result: a working scoring pipeline comparing $\hat{G}$ against $G^*$
    (structural Hamming distance plus out-of-sample predictive performance)
- Milestone 3: RLVR training loop
  - Wire up policy gradient training (REINFORCE / PPO) over tool-call sequences using
    the verifiable reward from Milestone 2
  - This is the result: a trained policy $\pi_\theta$ that outperforms the scripted
    baseline on held-out synthetic graphs
- Milestone 4: Generalization and sim-to-real
  - Add a curriculum over graph size, edge density, and noise, then evaluate transfer
    to real datasets where the ground-truth graph is unknown
  - This is the result: a report on the sim-to-real gap and the observed failure
    modes

## References

- _RLHF Book_, Chapter 6: Policy Gradients
  https://rlhfbook.com/c/06-policy-gradients
- `pgmpy` documentation (random Bayesian network generation)
- `scikit-learn` documentation (`make_regression`, `make_classification`)
