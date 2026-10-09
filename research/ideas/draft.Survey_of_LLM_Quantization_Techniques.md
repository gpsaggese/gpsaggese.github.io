# Survey of LLM Quantization Techniques

## Status
- **Status:**: draft
- **Complete Specs:**: 10%

## Core Idea

- What are the Python tools?
- What are the tradeoffs?
- What are the different approaches?

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

- Milestone 1: map the technique and tool landscape
  - Enumerate the main quantization approaches (post-training quantization
    vs. quantization-aware training, GPTQ, AWQ, bitsandbytes, GGUF/
    `llama.cpp`, SmoothQuant) and the Python tool implementing each
  - Read each technique's source paper and extract its algorithmic
    difference (calibration data needs, weight-only vs. weight+activation,
    supported bit-widths)
  - This is the result: a reference table mapping techniques to tools,
    supported bit-widths, and supported model families

- Milestone 2: build a common benchmarking harness
  - Pick 1-2 open-weight LLMs and quantize each with every selected tool at
    matching bit-widths (4-bit and 8-bit)
  - Measure perplexity/accuracy on a standard benchmark, inference latency,
    and memory footprint, all on the same hardware
  - This is the result: a benchmark harness runnable across quantization
    tools on the chosen models, producing comparable metrics

- Milestone 3: run the cross-tool tradeoff analysis
  - Execute the harness across all selected tools and bit-widths
  - Plot quality degradation against compression ratio and against
    inference speedup for each technique
  - This is the result: a comparison table/plot of accuracy retention vs.
    compression ratio vs. inference speed across techniques

- Milestone 4: write up the survey
  - Synthesize the benchmark results and the tool/technique landscape into
    a written survey
  - Include a recommendation matrix for choosing a technique given
    deployment constraints (edge vs. server, latency vs. quality)
  - This is the result: a survey document with a technique-selection
    recommendation matrix, backed by the Milestone 3 benchmark data

## References

- Author(s), _Title_. (Year)
