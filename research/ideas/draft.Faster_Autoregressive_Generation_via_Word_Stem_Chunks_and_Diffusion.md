# Faster Autoregressive Generation Via Word/Stem Chunks and Diffusion

## Status

- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

// TODO(ai_gp): Improve this idea

- Problem: how to make autoregressive NN generation faster?
- Approach 1: generate words and stems (caveman style)
- Approach 2: mix diffusion and autoregression
  - E.g., diffusion as first step, then patch up with autoregression

## Formalization

- Baseline autoregressive (AR) generation of $T$ tokens needs $T$ sequential
  forward passes:
  $p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$, with latency
  $L_{AR} = T \cdot t_{fwd}$
- Approach 1 (word/stem chunks): generate a shorter chunk sequence $c_{1:K}$
  and expand it with a cheap decoder $g$
  - Compression ratio $\rho = T / K$ (tokens per chunk)
  - $L_1 = K \cdot t_{fwd} + t_{expand}$, so the speedup is
    $S_1 = L_{AR} / L_1 \approx \rho$ when $t_{expand}$ is small
  - Quality constraint: $\mathrm{Loss}(g(c_{1:K}), x_{1:T}) \leq \epsilon$
- Approach 2 (diffusion draft + AR patch): a diffusion model produces a
  draft $\hat{x}_{1:T}$ in $D$ parallel denoising steps, then the AR model
  rewrites only the $m$ positions where it disagrees with the draft
  - $L_2 = D \cdot t_{diff} + m \cdot t_{fwd}$, so
    $S_2 = L_{AR} / L_2$ is large only when $m \ll T$
  - Patch rule: position $t$ is rewritten if
    $p_{AR}(\hat{x}_t \mid \hat{x}_{<t}) < \tau$
- Both approaches are compared at equal quality: same perplexity or task score
  as the AR baseline, reported as a speed-quality curve

## Key Examples

- **Caveman-style generation**: the model emits `function return sum two input
  list` (6 chunks) and a small expander restores "The function returns the sum
  of the two input lists" (10 tokens), for $\rho \approx 1.7$
- **Diffusion draft plus patch**: a 64-token paragraph is drafted in 8
  denoising steps, then the AR model rewrites the 5 positions whose likelihood
  falls below $\tau$, instead of running 64 sequential passes
- **Failure mode**: a stem such as `run` does not fix tense, and a wrong
  token in the draft can change the meaning of everything after it, so the
  AR patch must repair a cascade, not a single token; numbers, identifiers,
  and code cannot be compressed at all

## Questions

1. What compression ratio $\rho$ can stem/word chunking reach before the
   expansion step loses information that the task needs?
2. Is there a break-even point where the cost of the expander or the AR patch
   pass cancels the saving from fewer sequential steps?
3. If a diffusion draft plus a few AR patches matches AR quality, is strict
   left-to-right generation needed for only a small fraction of tokens?

## Research Topics

- **Chunk vocabulary design**: compare stems, whole words, and frequent
  phrases as the generation unit, and measure the information lost by each
- **Draft-and-patch hybrid**: how to choose the threshold $\tau$ and the
  number of denoising steps $D$, and how to batch the AR patch pass
- **Baselines**: speculative decoding and non-autoregressive decoding
  (mask-predict) as the speed-quality reference points

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: measure the baseline and the chunking limit
  - Pick a small open LM (e.g., GPT-2 small) and a text corpus, and measure
    tokens/second and perplexity as the AR baseline
  - Convert the corpus to stem-only text and measure $\rho$
  - Train a small expander and measure reconstruction accuracy
  - This is the result: a table of $\rho$ vs reconstruction quality

- Milestone 2: chunked generation end to end
  - Fine-tune the LM on chunked text and add the expander
  - Measure latency and quality at equal quality against the baseline
  - This is the result: a measured speedup $S_1$ and the quality it costs

- Milestone 3: diffusion draft plus AR patch
  - Build a small discrete-diffusion drafter and the threshold-based patch
    pass
  - Sweep $D$ and $\tau$ and compare against speculative decoding
  - This is the result: a speed-quality curve for $S_2$ and a recommendation
    on which approach to pursue

## References

- Vaswani et al., _Attention Is All You Need_. (2017)
- Sennrich et al., _Neural Machine Translation of Rare Words with Subword
  Units_. (2016)
- Gu et al., _Non-Autoregressive Neural Machine Translation_. (2018)
- Ghazvininejad et al., _Mask-Predict: Parallel Decoding of Conditional Masked
  Language Models_. (2019)
- Austin et al., _Structured Denoising Diffusion Models in Discrete
  State-Spaces_. (2021)
- Li et al., _Diffusion-LM Improves Controllable Text Generation_. (2022)
- Leviathan et al., _Fast Inference from Transformers via Speculative
  Decoding_. (2023)
