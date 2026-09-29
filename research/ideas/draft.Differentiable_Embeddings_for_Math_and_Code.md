# Differentiable Embeddings for Math and Code

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Study differentiable embeddings in math and coding
- Can the correctness of a problem or a proof in math be determined by
  gradient descent?

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

- Milestone 1: build a minimal testbed and correctness-labeled dataset
  - Choose a narrow domain: short algebraic identities (polynomial
    equalities) and short Python functions paired with unit tests
  - Generate a labeled dataset: one correct proof or solution per problem,
    plus several perturbed incorrect variants (sign flips, off-by-one errors,
    swapped operators)
  - Encode each proof or program as a parse tree/AST, and pick a
    differentiable embedding architecture (tree-LSTM, graph neural net, or
    transformer over the token/AST sequence)
  - This is the result: a labeled correct/incorrect dataset and an embedding
    model that maps proofs and code to vectors

- Milestone 2: train a differentiable correctness scorer
  - Train an energy/score function $E_\theta$ on the embeddings with a
    contrastive or margin loss, so correct proofs score low and perturbed
    incorrect ones score high
  - Evaluate the scorer as a binary correctness classifier (accuracy,
    precision, recall) on held-out perturbations
  - This is the result: a trained scorer with a measured accuracy at
    distinguishing correct from incorrect proofs and code

- Milestone 3: test gradient descent as a correctness-repair mechanism
  - Given an incorrect proof or program, run gradient descent on a continuous
    relaxation (Gumbel-softmax token distribution, or a VAE latent code) to
    minimize $E_\theta$
  - Decode the optimized representation back to a discrete proof or program
    and check validity with a symbolic verifier (unit tests or a proof
    checker)
  - This is the result: a measured repair success rate, the fraction of
    perturbed incorrect examples gradient descent turns into
    verifier-confirmed correct ones

- Milestone 4: compare against baselines and test cross-domain transfer
  - Compare gradient-based repair against random mutation search and greedy
    symbolic search under the same compute budget
  - Test whether a scorer trained on one domain (math) transfers to the other
    (code) without retraining
  - This is the result: a comparison table of repair success rate and compute
    cost across methods and domains, showing whether gradient descent is
    competitive as a correctness-search mechanism

## References
- Author(s), _Title_. (Year)
