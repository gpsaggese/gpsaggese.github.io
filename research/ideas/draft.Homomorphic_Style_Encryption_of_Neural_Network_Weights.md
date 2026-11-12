# Homomorphic-Style Encryption of Neural Network Weights

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- Transform a neural network's weights with a private key, then apply the
  same key to the inputs and outputs to remove the transform (like
  private-key encoding)

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

- Milestone 1: prototype exact functional equivalence on a linear network
  - Pick a small linear (no-activation) MLP and define a private key as an
    invertible matrix $K$ per layer boundary
  - Transform weights as $W' = K_{out} W K_{in}^{-1}$ and apply $K_{in}$ to
    inputs, $K_{out}^{-1}$ to outputs
  - Verify numerically that the transformed network reproduces the original
    outputs exactly, up to floating-point error
  - This is the result: a script proving the encode/decode round trip
    preserves the network's function on linear layers

- Milestone 2: extend to nonlinear networks
  - Restrict the key group to transforms that commute with the network's
    nonlinearities (e.g., signed permutation matrices commute with ReLU)
  - Implement the restricted-key transform for an MLP/CNN with ReLU and
    measure output equivalence before and after encryption
  - This is the result: a nonlinear network whose encrypted weights produce
    identical predictions to the plaintext network

- Milestone 3: measure encryption strength versus accuracy trade-off
  - Attempt to recover the original weights from the encrypted weights
    alone, with no key, using naive inversion and statistical fingerprinting
    attacks
  - Sweep the key space size (permutation-only vs permutation plus scaling)
    and record how attack success rate changes
  - This is the result: a table of key-space size vs empirical
    weight-recovery difficulty, with no accuracy loss in any configuration

- Milestone 4: validate on a pretrained model and study the threat model
  - Apply the key-based transform to a small pretrained transformer's
    weights and measure the runtime/memory overhead of the extra matrix
    multiplications
  - Document what an attacker holding only the encrypted weight file, with
    no key and no architecture-specific assumptions, can and cannot infer
  - This is the result: an overhead and security write-up for a realistic
    pretrained model, not just toy networks

## References
- Author(s), _Title_. (Year)
