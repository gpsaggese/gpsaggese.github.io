# Detecting LLM-Generated Text and Humanizing It

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: TBD

## Core Idea
- Train a classifier (e.g., a fine-tuned encoder, or a small NN on top of
  perplexity/stylometric features) that separates LLM-generated text from
  human-written text on the same topic and in the same register
- Then close the loop: train or prompt a "humanizer" that rewrites LLM text to
  defeat the detector, while preserving meaning, and re-train the detector on
  the humanized output
- The non-obvious part is that detector and humanizer form an adversarial pair,
  so the interesting quantity is not detector accuracy at a point in time but
  where the game converges: does the detector keep winning as the humanizer
  improves, or does the equilibrium land at chance level once meaning-preserving
  rewrites can span the human style distribution?
- The repo already has a `.claude/skills/blog.humanize` skill, so the humanizer
  side has a concrete starting baseline

## Formalization
- Detector $d_\theta(x) \to P(\text{LLM})$, humanizer $g_\phi: x \to x'$ with a
  semantic constraint $\text{sim}(g_\phi(x), x) \ge \tau$
- Adversarial objective:
  - $g_\phi$ minimizes $d_\theta(g_\phi(x))$ subject to the semantic constraint
  - $d_\theta$ is retrained on
    $\{\text{human}\} \cup \{\text{LLM}\} \cup \{g_\phi(\text{LLM})\}$
- Report detector AUC at a fixed false-positive rate (false accusations are the
  costly error), not raw accuracy

## Key Examples
- **In-domain detection**: student essays vs. LLM essays on the same prompt,
  where topic and length are controlled
- **Distribution shift**: a detector trained on one model family's output tested
  on another (e.g., trained on GPT output, tested on Claude/Llama output)
- **Failure mode**: the detector is really a topic/formatting classifier (em
  dashes, bullet lists, "delve") and collapses once those surface cues are
  stripped by the humanizer
- **Failure mode**: humanizing degrades content (the text passes as human but
  loses facts or coherence), so semantic fidelity must be scored, not assumed

## Questions
1. Does detection accuracy survive an adaptive adversary, or is any fixed
   detector defeated by a meaning-preserving rewrite?
2. How much of the signal is model-specific fingerprint vs. a general "machine
   register", i.e., does a detector transfer across model families?
3. What is the false-positive rate on non-native-English human writing, which is
   the known fairness failure of existing detectors?

## Research Topics
- **Zero-shot vs supervised detection**: DetectGPT-style curvature and
  log-likelihood/entropy features vs. supervised fine-tuned detectors
- **Watermarking**: an alternative to post-hoc detection, and its robustness
  to paraphrase
- **Style transfer with semantic constraints**: connects to
  [[draft.Compression_as_Proxy_for_Understanding]] for the fidelity metric

## Next steps
- [ ] Look for related research (DetectGPT, GPTZero, watermarking literature)
- [ ] Assemble a paired human/LLM corpus with topic and length controlled
- [ ] Train a baseline detector and report AUC at a fixed false-positive rate
- [ ] Add the humanizer loop and measure the adversarial equilibrium

## Implementation plan

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References
- Mitchell, E., et al., _DetectGPT: Zero-Shot Machine-Generated Text Detection
  Using Probability Curvature_. (2023)
- Kirchenbauer, J., et al., _A Watermark for Large Language Models_. (2023)
- Sadasivan, V. S., et al., _Can AI-Generated Text Be Reliably Detected?_ (2023)
