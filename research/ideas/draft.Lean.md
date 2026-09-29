# Lean

## Status
- **Status:**: draft
- **Complete Specs:**: TBD
- **Assignee:**: TBD

## Core Idea

- Explore using LLMs for automated theorem proving in Lean, building on
  existing formal-proof environments, benchmarks, and libraries
- Investigate combining LLMs with both Lean and Python for proof search
  and formalization workflows

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

1. [Open question 1: what remains unknown?]
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **LeanDojo** (https://leandojo.org): reinforcement learning environment
  for Lean proofs
- **MiniF2F** (https://github.com/openai/miniF2F): math problem benchmark
  for Lean and LLMs
- **Mathlib4** (https://github.com/leanprover-community/mathlib4):
  comprehensive Lean 4 mathematics library
- **ProofNet** (https://github.com/openai/proofnet): dataset for neural
  theorem proving
- **Lean Copilot**: experimental VSCode extension for LLM-assisted Lean
  proofs (community prototype on GitHub)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: stand up the Lean/LeanDojo/Mathlib4 baseline
  - Install Lean 4, Mathlib4, and LeanDojo, and extract a subset of MiniF2F
    problems into LeanDojo's interaction format
  - Run an off-the-shelf LLM (prompted via API, no fine-tuning) as a
    tactic-proposal model inside LeanDojo's proof-search loop
  - This is the result: a reproducible pass rate on the MiniF2F subset for
    the prompted-LLM baseline, with logged proof traces for every attempt

- Milestone 2: build the Python-Lean proof-search bridge
  - Use LeanDojo's interaction API to let the LLM propose tactics
    iteratively, feeding the resulting proof state (or Lean error message)
    back into the next prompt
  - Add multi-sample proposal (best-of-n tactics per step) and a retry
    budget per proof
  - This is the result: an end-to-end, Python-driven proof-search loop that
    attempts the MiniF2F subset and reports success or failure with retries

- Milestone 3: improve search with retrieval and error-conditioned repair
  - Add premise retrieval from Mathlib4 (e.g. embedding-based lemma search)
    to narrow the tactic search space
  - Condition retries on the Lean error message rather than resampling
    blind, and compare against the ProofNet dataset for informal-to-formal
    statement translation
  - This is the result: a measured pass-rate improvement over the Milestone
    1 baseline, with an ablation showing which addition (retrieval vs
    error-conditioned repair) contributed most

- Milestone 4: distill solved traces into a lighter model
  - Fine-tune (or train a lightweight adapter for) a smaller model on the
    proof traces collected in Milestones 2 and 3
  - This is the result: a comparison of the fine-tuned model against the
    prompted-only baseline on a held-out MiniF2F split

## References

- Lean and LLM background papers and notes:
  - https://arxiv.org/pdf/2202.01344
  - https://arxiv.org/abs/2102.11107
  - https://projectnumina.ai/
  - https://huggingface.co/blog/AI-MO/kimina-prover
  - https://ista.ac.at/en/research/locatello-group/
  - ChatGPT session, "LLM and Lean":
    https://chatgpt.com/share/6906bdd6-e4b8-8013-8a77-a7194618d7bb
  - ChatGPT session:
    https://chatgpt.com/share/6906bf0b-b5c0-8013-bdee-26e9d7e75175
  - ChatGPT session, "LLM and Lean":
    https://chatgpt.com/share/68e1c12d-5210-8013-a8d2-d198ff3d4a1d
  - ChatGPT session, "Python-Lean":
    https://chatgpt.com/share/68e1c5e6-d3a8-8013-b038-46e2f4082092

- Uncategorized notes and links captured in this file, not yet organized
  into their own idea:
  - Product roadmap and ROI notes (no detail captured)
  - Class tutorials: write notes, then record videos
    - Livestream: https://youtube.com/live/2Auc57lxgeU
    - Repo: https://github.com/pymc-labs/ai_decision_workshop
  - Automatically generate images with the OpenAI API
  - Get a notification for weather through an API
  - Generate research for understanding the relationship between people
    (note incomplete in source)
  - Add papers (note incomplete in source)
  - `tmux.py` traceback and fix:
    ```text
    > ./dev_scripts_umd_msml610/thin_client/tmux.py
    Traceback (most recent call last):
      File "dev_scripts_umd_msml610/thin_client/tmux.py", line 12, in <module>
        assert os.path.exists(os.path.join(dir_name, "thin_client_utils.py"))
    AssertionError: Can't find thin_client_utils.py
    ```
    Fix:
    ```bash
    > ln -sf ../../helpers_root/dev_scripts_helpers/thin_client/tmux.py \
        ./dev_scripts_umd_msml610/thin_client/tmux.py
    ```
  - Python call-graph and complexity-analysis tools:
    - PyCG: practical call graph generator
    - flake8-function-order
    - pyan3: lightweight call graph generator using static parsing
    - SnakeViz: runtime profiling, gives function timing and hierarchy
    - pycallgraph2: creates runtime call graphs (requires code execution)
    - code2flow: turns structured Python code into flowcharts, less
      accurate for dynamic code
    - pyreverse:
      https://pylint.readthedocs.io/en/latest/user_guide/usage/run.html#cmdoption-pyreverse
    - radon: static analysis for complexity, can complement class
      structure graphs
    - Pydeps
    - Xenon: enforces complexity thresholds based on radon, CI-friendly
      enforcement
    - Lizard: measures cyclomatic complexity for many languages,
      lightweight and fast
    - Vulture: finds dead (unused) code
  - Causal AI and other business or research links:
    - https://hbr.org/2025/09/ai-generated-workslop-is-destroying-productivity
    - https://www.youtube.com/watch?v=PA7js-mSU3Q
    - https://www.youtube.com/watch?v=LQY3CzUfJgA
    - WSJ, on AI use in finance teams:
      https://www.wsj.com/articles/amazons-finance-teams-are-relying-more-on-ai
    - https://distyl.ai/
    - PR Newswire, Pecan AI DemandForecastAI launch:
      https://www.prnewswire.co.uk/news-releases/pecan-ai-launches-demandforecastai-to-fix-the-gap-with-genai-powered-supply-chain-insights-302540307.html
    - https://www.youtube.com/watch?v=Szlz3JE-L5M
    - "AI Assisted Causal Inference", with Sam Witty:
      https://samwitty.github.io/papers/Witty_Dissertation.pdf
    - "How to speak":
      https://www.youtube.com/watch?v=Unzc731iCUY&list=PLOe2Tlpw8fRABuFqrQg9tqcCJ0R0_Eubi
    - https://parabole.ai/
    - Forbes, causal AI at Georgia-Pacific:
      https://www.forbes.com/sites/stevebanker/2024/04/15/what-georgia-pacific-is-doing-with-causal-ai-is-remarkable/
    - LinkedIn post, AI companies disrupting Wall Street:
      https://www.linkedin.com/posts/davewangmia_i-mapped-81-ai-companies-disrupting-wall-activity-7358518819625000960-578I/
    - https://www.res-group.com/resources/blog/data-as-the-path-to-lower-operating-costs-and-higher-performance/
    - https://www.evolver.ai/
  - Papers and talks:
    - https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4706629_code2969338.pdf
    - https://m.youtube.com/watch?v=WWCWsub3YkE
    - https://m.youtube.com/playlist?list=PLJePd8QU_LYKZwJnByZ8FHDg5l1rXtcIq
