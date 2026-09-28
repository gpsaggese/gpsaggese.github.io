# Straight-Through Estimators for Differentiable Discrete Operations

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea

- The Straight-Through Estimator (STE) makes a discrete operation trainable by
  gradient descent through a trick applied at a single point in the
  computation graph: run the true discrete op (rounding, thresholding, sign,
  argmax) in the forward pass, but during backpropagation substitute an
  identity (or clipped identity) for that op's local Jacobian, as if it had
  been the identity function all along. This is a *variable-level* patch: it
  targets one discrete node embedded inside an otherwise differentiable
  pipeline (e.g., a weight, an activation, or a gate), leaving the rest of
  the network and the loss function untouched. This is the key contrast with
  relaxation-style approaches (softmax/Gumbel-softmax, Sinkhorn, surrogate
  losses), which instead replace the *objective or the variable's domain*
  with a continuous approximation before optimizing
- Hypothesis under test: for a chosen discrete-op-in-a-pipeline problem, STE
  lets standard SGD/Adam optimize end-to-end despite the local gradient being
  wrong almost everywhere, and produces solutions competitive with an exact
  or greedy discrete search when the discrete op is shallow (few stacked
  instances), but the bias introduced at each STE node compounds through the
  chain rule as more such nodes are stacked, degrading solution quality and
  training stability once depth crosses some problem-dependent threshold
- Why non-obvious: an identity backward pass is *provably* the wrong gradient
  almost everywhere the discrete op is not the identity, and yet STE-trained
  binary/quantized networks are standard, competitive practice at production
  scale. That gap between the naive theoretical objection and the empirical
  success is exactly the open question worth mapping in a small, fully
  understood setting rather than taking on faith

## Formalization

### General STE definition

- Forward pipeline: $\mathcal L(\theta) = g(f(h(\theta)))$, where
  $h: \Theta \to \mathbb{R}^n$ is a differentiable map producing a continuous
  pre-activation $z = h(\theta)$, $f: \mathbb{R}^n \to \mathcal D$ is a
  discrete forward operation (rounding, sign, threshold, argmax) mapping into
  a discrete codomain $\mathcal D$, and $g$ is a differentiable function of
  the discrete output (e.g., a downstream layer plus loss)
- $f$ has zero or undefined true Jacobian almost everywhere ($\partial f /
  \partial z = 0$ a.e. for piecewise-constant $f$, or a Dirac delta at
  discontinuities), so plain backprop gives no usable signal for $\theta$
- STE substitutes a surrogate local Jacobian at the $f$ node only:

```latex
\widehat{\frac{\partial f}{\partial z}} :=
\begin{cases}
  1 & \text{(plain STE)} \\
  \mathbb{1}[\, |z| \le 1 \,] & \text{(clipped STE, Hubara et al. 2016)}
\end{cases}
```

- The backward pass then computes a biased gradient estimate by chain rule
  with the true Jacobian replaced at that one node:

$$
\widehat{\nabla_\theta \mathcal L} =
  \left(\frac{\partial g}{\partial f}\Big|_{f(z)}\right)^{\!\top}
  \widehat{\frac{\partial f}{\partial z}}
  \frac{\partial h}{\partial \theta}
$$

- In general $\mathbb E[\widehat{\nabla_\theta \mathcal L}] \neq \nabla_\theta
  \mathcal L$: STE is a biased estimator with no formal descent guarantee.
  When $L$ such nodes are stacked (one per layer or gate), the substitution
  is applied at each node along the backward path, and the per-node biases
  compose multiplicatively through the chain rule, which is the mechanism
  this idea sets out to measure empirically

### Concrete example: stacked binary-weight classifier (BinaryConnect-style)

- Task: binary classification on a synthetic dataset $\{(x_i, y_i)\}_{i=1}^N$,
  $x_i \in \mathbb{R}^d$, $y_i \in \{-1, +1\}$
- Network with $L$ layers, each holding a real-valued "shadow" weight matrix
  $\theta^{(\ell)}$ that is binarized in the forward pass:

$$
z^{(0)} = x, \qquad
W^{(\ell)} = \mathrm{sign}\!\left(\theta^{(\ell)}\right), \qquad
z^{(\ell)} = \phi\!\left(W^{(\ell)} z^{(\ell-1)}\right), \quad \ell = 1,\dots,L
$$

  where $\phi$ is $\tanh$ for hidden layers and the identity for the last
  layer, and $\hat y = z^{(L)}$ is the scalar output
- Loss (fully differentiable given a fixed set of binary weight matrices):
  $\mathcal L(\theta) = \frac1N \sum_i \max(0,\, 1 - y_i \hat y_i)$ (hinge)
- Forward discrete op at each layer: $W^{(\ell)} = \mathrm{sign}(\theta^{(\ell)})$
- Backward STE surrogate at each layer:
  $\widehat{\partial W^{(\ell)} / \partial \theta^{(\ell)}} :=
  \mathbb{1}[\,|\theta^{(\ell)}_{jk}| \le 1\,]$ (clipped, per Hubara et al.
  2016), applied independently at each of the $L$ sign() nodes; the shadow
  weights $\theta^{(\ell)}$, not the binarized $W^{(\ell)}$, are what SGD
  actually updates
- This is deliberately a variable-level transform: only the weight tensors
  are discretized, the loss itself ($\max(0, 1-y\hat y)$) stays exactly the
  hinge loss, unlike surrogate-loss approaches that relax the objective

## Key Examples

- **Five known STE deployments** (the survey set for this idea):
  - **Binarized Neural Networks**: weights and/or activations constrained to
    $\{-1,+1\}$ via $\mathrm{sign}(\cdot)$, gradients passed through with STE
    (Courbariaux, Bengio & David 2015 "BinaryConnect"; Hubara et al. 2016)
  - **Ternary Weight Networks**: weights quantized to $\{-1, 0, +1\}$ via a
    learned threshold, STE used for the piecewise-constant quantizer (Li,
    Liu & Wang 2016)
  - **VQ-VAE discrete latents**: encoder output snapped to the nearest
    codebook vector via $\mathrm{argmin}_k \lVert z_e - e_k \rVert$; STE
    copies the decoder's gradient straight back to the encoder output,
    bypassing the non-differentiable nearest-neighbor lookup (van den Oord,
    Vinyals & Kavukcuoglu 2017)
  - **Hard/discrete gating and conditional computation**: binary gates that
    switch sub-networks or experts on/off (e.g., top-1 discrete routing,
    hard attention windows) trained with an identity backward pass through
    the gate (Bengio, Leonard & Courville 2013)
  - **Learned discrete pruning masks**: a binary mask $m = \mathbb{1}[\sigma(
    \alpha) > 0.5]$ zeroes out weights/channels during training; STE lets
    gradients flow to the mask logits $\alpha$ despite the hard threshold
- **Concrete worked example (this idea's testbed)**: the stacked binary-weight
  hinge-loss classifier defined above, $L \in \{1, 2, 4, 8\}$ layers,
  compared against a non-differentiable discrete solver on the same weight
  space
- **Edge case / expected failure mode**: as $L$ grows, shadow weights in deep
  layers receive gradient signal that has been passed through several
  identity substitutions plus several true $\tanh$ nonlinearities; the
  hypothesis is that beyond some $L^*$, STE training either stalls (shadow
  weights drift into the clipped region, $|\theta^{(\ell)}_{jk}| > 1$, and
  stop receiving gradient) or oscillates, while a greedy discrete search
  restricted to the same small $L=1$ case still finds a usable solution,
  isolating whether the failure is inherent to STE bias compounding or is
  just an optimization-landscape effect shared with the discrete search

## Questions

1. How does the STE-vs-exact accuracy gap scale with the number of stacked
   discrete nodes $L$, and is there a depth $L^*$ beyond which STE training
   fails to converge at all while greedy discrete local search (restricted
   to a comparably small weight space) still finds a competitive solution?
2. Does clipped STE (zero gradient once $|\theta^{(\ell)}_{jk}| > 1$)
   meaningfully reduce bias and improve stability relative to plain identity
   STE in the stacked setting, or does clipping just trade instability for
   dead (saturated, non-updating) weights?
3. Provocative implication: if brute-force or greedy discrete search matches
   or beats STE+SGD whenever the discrete space is small enough to search
   directly, is STE's real justification gradient quality at all, or purely
   combinatorial scalability, i.e., is STE only "worth it" once the discrete
   space is too large for exact/greedy methods to touch?

## Research Topics

- **Bias measurement**: quantify the STE gradient's deviation from a
  ground-truth gradient proxy (finite differences on small instances, or
  exhaustive local neighborhood evaluation on the discrete cube) as a
  function of layer depth and training progress
- **STE variants**: compare plain identity STE, clipped STE (Hubara et al.
  2016), and a temperature-annealed soft-sign hybrid ($\tanh(z/T)$ with
  $T \to 0$) on the same stacked-binarization testbed
- **Depth scaling and mitigations**: characterize the maximum stable depth
  $L^*$ for plain vs. clipped STE, and test whether standard BNN
  stabilizers (batch normalization between binarized layers, learning-rate
  scaling per layer) push $L^*$ higher
- **Scalability crossover**: find the problem size (input dimension $d$,
  depth $L$) at which greedy/brute-force discrete search becomes
  intractable while STE+SGD remains practical, to test whether STE's value
  is really about scale rather than gradient fidelity

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: survey known STE use cases, pick the concrete testbed, and
  formalize it
  - Confirm the 5 known-usage examples with primary-source references and a
    one-line summary of the exact discrete op and STE substitution used in
    each
  - Lock down the stacked binary-weight classifier formalization (dataset,
    architecture, loss, clipped-STE rule) as in the Formalization section
  - This is the result: a written formal spec of the testbed problem and its
    STE transform, ready to implement, plus a short annotated bibliography
    of the 5 known deployments

- Milestone 2: implement the discrete baseline in Python
  - Implement the $L=1$ binary-weight linear classifier and a brute-force
    solver over $\{-1,+1\}^d$ for small $d$ (e.g., $d \le 20$)
  - Implement a greedy bit-flip local search (hill-climbing with random
    restarts) usable for larger $d$ and for $L>1$ (flipping one weight of
    one layer at a time), as a solver that scales past brute force
  - This is the result: a non-differentiable discrete solver returning the
    best training-loss binary weight configuration found, with wall-clock
    and iteration counts logged, for $L \in \{1,2,4,8\}$

- Milestone 3: implement STE + gradient descent in Python
  - Implement the forward binarization ($\mathrm{sign}$) and both plain and
    clipped STE backward rules via a custom autograd function
  - Train the same $L \in \{1,2,4,8\}$ architectures with SGD/Adam on the
    hinge loss, across multiple random seeds
  - This is the result: trained shadow weights $\theta^{(\ell)}$ and their
    binarized $W^{(\ell)} = \mathrm{sign}(\theta^{(\ell)})$ for each $L$,
    with training curves (loss, accuracy) logged per seed

- Milestone 4: compare quality/stability and characterize bias accumulation
  - Compare final train/test accuracy, convergence speed, and cross-seed
    variance between the discrete baseline (Milestone 2) and STE+SGD
    (Milestone 3) at each depth $L$
  - Plot accuracy gap and training-stability metrics (loss variance across
    seeds, fraction of saturated/dead weights) as a function of $L$, for
    both plain and clipped STE
  - This is the result: an empirical answer to whether and where STE bias
    compounding causes a measurable breakdown (the depth $L^*$ from Question
    1), and whether clipped STE delays that breakdown relative to plain STE

## References
- Bengio, Y., Léonard, N., & Courville, A., _Estimating or Propagating
  Gradients Through Stochastic Neurons for Conditional Computation_. (2013)
- Courbariaux, M., Bengio, Y., & David, J.-P., _BinaryConnect: Training Deep
  Neural Networks with Binary Weights during Propagations_. (2015)
- Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y.,
  _Binarized Neural Networks_. (2016)
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K., _Neural Discrete
  Representation Learning_. (2017)
