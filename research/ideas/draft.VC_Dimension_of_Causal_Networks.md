# VC Dimension of Causal / Bayesian Networks

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Causal structure provides inductive bias that can dramatically reduce
  sample complexity
- The VC dimension of a causal or Bayesian network depends not only on the
  number of parameters but on the structural properties of the DAG: number
  of nodes, maximum indegree, and the parametric family of conditional
  distributions
- Understanding this complexity is essential for characterizing the sample
  complexity of causal discovery and the generalization properties of
  causally-structured models

## Formalization

- The VC dimension of a causal/Bayesian network depends on:
  - **Number of nodes**: more nodes -> larger hypothesis space
  - **Maximum indegree**: higher indegree -> more complex conditional
    distributions
  - **Parametric family of conditionals**: e.g., linear Gaussian vs.
    non-linear vs. non-parametric

## Key Examples

- **10-node DAG vs. fully-connected network**: a 10-node causal DAG with max
  indegree 3 and linear Gaussian conditionals has far lower VC dimension than
  a fully-connected 10-node network
  - This suggests causal structure provides inductive bias that improves
    generalization, but only if the assumed structure is correct

## Questions

1. Does learning the wrong causal structure hurt generalization more than
   ignoring causality entirely and using pure correlation?
2. Can we define a "causal VC dimension" that captures not just the
   complexity of the model but the complexity of interventions it can
   represent?
3. If two causal graphs are Markov equivalent (indistinguishable from
   observational data), do they have the same VC dimension?
4. Is there a fundamental trade-off between causal interpretability and
   predictive accuracy?

## Research Topics

- Structural VC dimension of DAGs
- Sample complexity of causal discovery
- MDL penalties for structure learning
- Causal VC dimension for interventional distributions
- Bounds on generalization error for causal vs. purely correlational models

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: derive VC dimension bounds for parametrized DAG families
  - Formalize the hypothesis class $\mathcal{H}_G$ induced by a DAG $G$
    with maximum indegree $k$ and a fixed conditional family, starting with
    linear Gaussian conditionals
  - Derive a closed-form or bounded VC dimension as a function of the
    number of nodes $n$, indegree $k$, and per-node parameter count
  - Check that the bound reduces to the known linear-classifier VC bound in
    the degenerate single-node case
  - This is the result: a derived VC dimension formula or bound for the
    linear-Gaussian DAG family, with a proof sketch

- Milestone 2: build a synthetic DAG benchmark
  - Generate synthetic DAGs at varying $n$, maximum indegree, and
    conditional family (linear Gaussian, then non-linear), replicating the
    10-node DAG vs. fully-connected comparison from Key Examples across
    multiple configurations
  - Estimate empirical sample complexity via simulation (minimum samples to
    reach a target generalization error) and compare it to the Milestone 1
    bound
  - This is the result: a benchmark suite and plots of empirical vs.
    derived sample complexity across DAG configurations

- Milestone 3: test structural misspecification
  - Generate data from a true DAG $G^*$, then fit models under a
    misspecified structure $G'$ (varying degree of misspecification) and
    under a structure-free correlational model
  - Compare generalization error across the correct-structure,
    misspecified-structure, and structure-free models to test whether
    wrong structure hurts more than ignoring causality (Question 1)
  - This is the result: a quantitative comparison of generalization error
    across the three model classes

- Milestone 4: extend to Markov equivalence and interventions
  - Test whether Markov-equivalent DAGs (same skeleton and v-structures)
    yield the same VC dimension under the Milestone 1 formula and
    empirically (Question 3)
  - Propose and work out an example of an "interventional VC dimension"
    that counts the complexity of representable interventions, not just
    observational fit (Research Topic)
  - This is the result: an answer, proof or counterexample, to whether
    Markov-equivalent graphs share VC dimension, plus a worked example of
    the interventional VC dimension definition

## References

- Derived from _Research_plan/paper.tex_ (Section: Quasi-Stationary Learning
  / VC Dimension of Causal / Bayesian Networks)
