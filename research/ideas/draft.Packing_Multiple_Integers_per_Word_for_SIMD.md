# Packing Multiple Integers per Word for SIMD

## Status
- **Status:**: draft
- **Complete Specs:**: 10%
- **Assignee:**: ...

## Core Idea

- Pack more than one integer into a single word and perform SIMD operations
  on the packed word
  - Open question: is this already a known, standard technique?
  - E.g., apply as a streaming computation on an FPGA over very large VLIW
    batches

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

- Milestone 1: survey prior art and define the target spec
  - Survey existing packed-arithmetic techniques (SIMD-within-a-register/
    SWAR, bit-slicing, packed arithmetic in cryptography and DSP) to
    answer the open question of whether this is already a standard
    technique, and pin down what would be novel
  - Define a concrete spec: word width, number and width of packed
    lanes (e.g. four 16-bit or eight 8-bit integers per 64-bit word), and
    the target operations (add, subtract, compare, and optionally
    multiply)
  - This is the result: a written comparison against existing SWAR/SIMD
    techniques, plus a concrete packing-layout and operation-set spec

- Milestone 2: implement and verify the software reference
  - Implement packed-word add, subtract, and compare routines in a
    systems language (C or Rust), with guard bits or masking to prevent
    overflow bleeding between adjacent lanes
  - Unit-test the packed routines against an unpacked scalar-loop
    reference on randomized inputs, including boundary and overflow
    cases
  - This is the result: a packed-word arithmetic library that matches the
    scalar reference on randomized correctness tests

- Milestone 3: benchmark against native SIMD and scalar baselines
  - Benchmark the packed-word implementation, native SIMD intrinsics
    (SSE/AVX or NEON), and a plain scalar loop on a representative
    streaming workload (e.g. large-batch counting or accumulation)
  - Measure throughput and instructions-per-element for each approach
  - This is the result: a throughput comparison table showing where
    packed-word SWAR wins, loses, or ties against native SIMD and scalar
    baselines

- Milestone 4: prototype the FPGA/VLIW streaming variant
  - Prototype the streaming computation described in Core Idea, either as
    an FPGA design (via an HLS tool or Verilog, in simulation) or as a
    cycle-accurate simulation of a VLIW pipeline processing packed words
  - Measure achievable throughput per unit resource (LUTs, or simulated
    issue width) against an unpacked baseline on the same pipeline
  - This is the result: an FPGA/VLIW simulation or synthesis result
    quantifying the throughput and resource-efficiency gain of packed-word
    streaming over the unpacked baseline

## References

- Author(s), _Title_. (Year)
