# Measuring Quality of LLM Skills and Prompts

## Status

- **Status:**: draft
- **Complete Specs:**: 20%
- **Assignee:**: TBD

## Core Idea

- Develop quantitative metrics for prompt/skill quality beyond simple
  accuracy: robustness to paraphrasing, consistency across runs,
  sensitivity to irrelevant context, and instruction-following fidelity
- Goal: build a quality measurement system for LLM skills and prompts that
  produces interpretable, reproducible scores across multiple quality
  dimensions
  - This enables data-driven prompt engineering and skill maintenance at
    scale

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```text
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. What is the ground truth for prompt quality when task outputs are
   subjective?
2. How many evaluation samples are needed for a statistically reliable
   quality estimate?
3. Can quality scores generalize across models, or are they
   model-specific?
4. How do you avoid reward hacking when optimizing prompts against the
   quality metric?

## Research Topics

- **Automated evaluation pipeline**: run a skill against a held-out
  evaluation set and report a quality score, enabling A/B comparison of
  prompt versions
- **LLM-as-judge approaches**: use a stronger model to rate skill outputs
  on correctness, completeness, conciseness, and style
- **Quality decomposition**: break skill quality into sub-dimensions, e.g.,
  for a coding skill, whether it produces runnable code, follows style
  conventions, and handles edge cases
- **Quality drift detection**: how prompt quality degrades as the
  underlying model is updated or replaced
- **Automated prompt optimization**: use quality metrics as a reward
  signal to iteratively improve prompt wording via DSPy-style compilation
  or evolutionary search
- **Data sources**:
  - MT-Bench (LMSYS): multi-turn conversation benchmark with GPT-4 as
    judge, useful for studying LLM-as-judge methodology
    (https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
  - AlpacaEval (Tatsu Lab, Stanford): automated evaluation of
    instruction-following quality using win-rate against a reference model
    (https://github.com/tatsu-lab/alpaca_eval)
  - DSPy optimization traces (Stanford NLP): prompt optimization traces
    showing how prompt rewrites affect downstream task performance
    (https://github.com/stanfordnlp/dspy)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the evaluation harness and held-out task sets
  - Select 2-3 target skills/prompts to study (e.g., a coding skill and a
    summarization skill), and construct held-out task sets with reference
    outputs or grading rubrics for each
  - Implement a harness that runs a skill/prompt $N$ times per task and
    collects the raw outputs, latencies, and token counts
  - This is the result: a working eval harness producing raw output logs
    for a chosen skill across its held-out task set

- Milestone 2: implement the quality metric suite
  - Implement consistency (variance of scores across repeated runs at
    fixed settings), robustness (score delta under paraphrased
    instructions/inputs), and context sensitivity (score delta after
    injecting irrelevant context) scorers
  - Implement an instruction-following fidelity scorer via an
    LLM-as-judge, following an MT-Bench/AlpacaEval-style rubric
  - This is the result: per-skill scorecards reporting each quality
    sub-dimension as a reproducible number

- Milestone 3: validate the metrics against known-good/known-bad prompts
  - Construct prompt variants with deliberately injected defects
    (ambiguous instructions, missing constraints, excessive verbosity) as
    known-bad controls
  - Check whether the metric suite ranks the known-good and known-bad
    variants in the expected order, and run a bootstrap/variance analysis
    to determine how many eval samples are needed for a stable score
  - This is the result: a validation report showing metric sensitivity to
    known defects and the minimum sample size needed for a reliable score

- Milestone 4: close the loop with automated prompt optimization
  - Use the quality metrics as a reward signal in a DSPy-style iterative
    prompt-rewrite loop applied to one target skill from Milestone 1
  - Check the optimized prompt for reward hacking (outputs that game the
    LLM-judge score without improving actual task success)
  - This is the result: before/after quality scorecards for the optimized
    prompt, plus a documented check for reward-hacking artifacts

## References

- Khattab et al., _DSPy: Compiling Declarative Language Model Calls into
  Self-Improving Pipelines_. (2023)
- Promptfoo: open-source prompt testing with custom evaluators
- RAGAS: evaluation framework for RAG pipeline quality
- Liu et al., _G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment_. (2023)
