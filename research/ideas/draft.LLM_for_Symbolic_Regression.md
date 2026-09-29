# Training an LLM from Scratch for Symbolic Regression

## Status

- **Status**: draft
- **Complete Specs**: 30%
- **Assignee**: TBD

## Core Idea

- Goal: train an LLM from scratch that, given a set of data points, proposes a
  closed-form expression fitting them
- Training data can be generated without limit by inverting the problem
  - Sample a random expression from a grammar
  - Evaluate it at random points to get a dataset
  - The (`data`, `expression`) pair is a supervised example with exact ground
    truth
- The key design claim: this problem cannot be solved in one shot, it requires
  search
  - A single forward pass has to guess the whole expression from the numbers
  - The realistic formulation is a proposer inside a hill-climbing loop, where
    the model proposes candidates, a fitter scores them, and residuals feed
    back into the next proposal
- The interesting question is what the pretraining corpus should be
  - Pure synthetic expressions teach the grammar and the numeric-to-symbolic
    mapping
  - Math and physics corpora teach which expressions are plausible, i.e., a
    prior over formulas that genetic programming does not have

## Formalization

- Sequence-to-sequence formulation: encode the point set, decode the
  expression tokens
  ```text
  p_theta(g | {(x_i, y_i)}) = prod_t p_theta(g_t | g_<t, encode(D))
  ```
  - The encoder must be permutation-invariant over points, e.g., a set encoder
    rather than a plain sequence encoder
- Constants are handled by a hybrid scheme: the model emits a skeleton with
  placeholders, and a numeric optimizer fits the constants
  ```text
  skeleton:  c1 * sin(c2 * x) + c3
  fit:       argmin_c MSE(skeleton(c), D)
  ```
- Search loop, since one shot is insufficient:
  ```text
  pool = {}
  repeat:
      cand = sample K skeletons from p_theta(. | D, pool_feedback)
      fit constants, score each by MSE + lambda * complexity
      keep the Pareto front, feed residuals and best-so-far back into the prompt
  until budget exhausted
  ```
- Curriculum over difficulty, with two controlled axes:
  - Expression complexity: node count, operator set, nesting depth
  - Number of data points $n$ and noise level $\sigma$
- Evaluation is recovery of the ground-truth expression, checked symbolically,
  not string match, using the harness from
  `draft.Symbolic_regression.md` so results are comparable to genetic
  programming and neural-guided baselines

## Key Examples

- **Synthetic generator**: sample `c1 * exp(-c2 * x) + c3`, evaluate at 50
  random points, and the training target is the skeleton with placeholders
- **Hill-climbing recovery**: the model first proposes a polynomial, the fit
  leaves structured residuals, and the residual pattern drives the next
  proposal toward a periodic term
- **Prior from physics corpora**: a model pretrained on physics text prefers
  dimensionally sensible forms (products of powers, exponentials of
  dimensionless groups) over arbitrary operator soup
- **Failure mode**: the model memorizes the generator's distribution and fails
  on expressions from a different grammar, so the held-out set must use a
  generator the model never saw
- **Failure mode**: numeric precision, where a model reading raw floats as
  tokens cannot distinguish $1.01$ from $1.10$, which argues for an explicit
  numeric encoding rather than plain text digits

## Questions

1. How should numbers be encoded so the model can actually use magnitude and
   precision? Digit tokens, scientific-notation tokens, or a learned numeric
   embedding
2. How much does pretraining on math and physics corpora help, relative to
   training only on the synthetic generator? This isolates prior over
   formulas from raw pattern fitting
3. What is the smallest model that works, given that the task is narrow and
   the data is unlimited?
4. Does the LLM proposer beat genetic programming at equal compute, or is its
   value only as an initializer that GP then refines?
5. If a small model trained on generated data can propose good candidates,
   the bottleneck in scientific formula discovery moves from search to the
   quality of the prior over expressions

## Research Topics

- **Dataset generation**: expression grammar, sampling that avoids degenerate
  or numerically unstable expressions, controlled complexity levels
- **Math corpora** for pretraining:
  - Open-WebMath: https://huggingface.co/datasets/open-web-math/open-web-math
  - MathPile: https://huggingface.co/datasets/GAIR/MathPile (gated, requires
    accepting terms), commercial-use version:
    https://huggingface.co/datasets/GAIR/MathPile_Commercial
  - InfiMM-WebMath-40B:
    https://huggingface.co/datasets/Infi-MM/InfiMM-WebMath-40B
  - MegaMath: https://huggingface.co/datasets/LLM360/MegaMath
  - FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
  - NuminaMath-CoT: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
  - MetaMathQA: https://huggingface.co/datasets/meta-math/MetaMathQA
  - MathInstruct: https://huggingface.co/datasets/TIGER-Lab/MathInstruct
- **Physics corpus**:
  - PHYSICS (Zheng et al., 2025), repo: https://github.com/Zhengsh123/PHYSICS
    (paper: https://arxiv.org/abs/2506.00022), with download links and
    instructions for the data files
- **Architecture**: set encoder for the point cloud, numeric tokenization,
  skeleton plus constant-fitting decode
- **Search integration**: how residual feedback is presented to the model, and
  whether the loop should be beam search, evolutionary, or MCTS-style, which
  connects to `draft.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.md`
- **Baselines**: the library comparison in `draft.Symbolic_regression.md`, and
  the related framing in `draft.Closed_Form_Formula_Discovery.md`

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: data generator
  - Implement the expression grammar, the sampler, and the evaluator that
    produces (`points`, `skeleton`, `constants`) triples at controlled
    complexity
  - Filter degenerate and numerically unstable expressions
  - This is the result: an unlimited synthetic dataset with a documented
    difficulty scale

- Milestone 2: one-shot baseline model
  - Train a small seq2seq model with a set encoder and skeleton decoding,
    plus a numeric optimizer for constants
  - Measure one-shot recovery vs expression complexity
  - This is the result: the recovery-vs-complexity curve that quantifies how
    far one shot gets, and where it breaks

- Milestone 3: search loop
  - Wrap the model in the propose, fit, score, feed-back-residuals loop
  - Compare against the one-shot baseline at equal sample budget
  - This is the result: evidence on how much search adds over one shot

- Milestone 4: priors and baselines
  - Pretrain on a math or physics corpus, then fine-tune on the generator,
    and compare with the generator-only model
  - Benchmark against PySR and gplearn at equal compute on the shared suite
  - This is the result: an attribution of performance to prior vs search, and
    a placement of the LLM proposer against established methods

## References

- Biggio et al., _Neural Symbolic Regression that Scales_. (2021)
- Kamienny et al., _End-to-End Symbolic Regression with Transformers_. (2022)
- Udrescu and Tegmark, _AI Feynman: A Physics-Inspired Method for Symbolic
  Regression_. (2020)
- La Cava et al., _Contemporary Symbolic Regression Methods and their
  Relative Performance (SRBench)_. (2021)
- Romera-Paredes et al., _Mathematical Discoveries from Program Search with
  Large Language Models (FunSearch)_. (2024)
