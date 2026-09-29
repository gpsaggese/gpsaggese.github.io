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

- Milestone 1
  - Do this and that
  - This is the result

- Milestone 2
  - Do this and that
  - This is the result

## References

- Khattab et al., _DSPy: Compiling Declarative Language Model Calls into
  Self-Improving Pipelines_. (2023)
- Promptfoo: open-source prompt testing with custom evaluators
- RAGAS: evaluation framework for RAG pipeline quality
- Liu et al., _G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment_. (2023)
