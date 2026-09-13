# Summary

## Title
- From LLMs to Agents: Foundations and Frontiers of Agentic AI

- Building Agentic AI: From Language Models to Autonomous Systems

## Target Audience
- Graduate students and ML engineers who already know deep learning and want to
  build and evaluate LLM agents
- Working knowledge of transformers, Python, and basic reinforcement learning
  assumed
- No prior experience with agent frameworks, RLHF, or web automation required

## Approach of the Book
- Focus on:
  - The minimal mathematics to understand the problem and the solutions
  - Intuition first, formalism only where it changes a design decision
  - Toy examples that expose the mechanism (a single ReAct trace, one DPO
    gradient step, one retrieval failure)
  - How to make the theory operational
    - Referring to packages and frameworks in the Python ecosystem (AutoGen,
      DSPy, LangGraph, vLLM, TRL)
  - Jupyter notebooks to back up the intuition with runnable agents and
    benchmarks

- Anchor every chapter on the primary papers and benchmarks, so the reader can
  go one level deep
  - The lecture decks in `book_Agentic_AI/lectures_source`
  - References to papers, benchmark suites, and open training recipes

## Short TOC
- The sequence of the parts in the book is:
  - Foundations
    - 01, What Is an Agentic AI?
    - 02, LLM Building Blocks
    - 03, A Brief History of LLM Agents
    - 04, LLM Reasoning
  - Core Agent Capabilities
    - 05, Reasoning, Memory, and Planning
    - 06, Inference-Time Techniques
    - 07, Tool Use and Retrieval
    - 08, Learning to Reason
  - Training Agentic Models
    - 09, Post-Training and Verifiable Agents
    - 10, Open Training Recipes for Reasoning
    - 11, Lessons from Training Agentic Models
    - 12, Neural-Symbolic Decision Making
  - Multi-Agent and Multimodal Systems
    - 13, Multi-Agent AI
    - 14, Agent Frameworks
    - 15, Multimodal Autonomous Agents
    - 16, From Perception to Action
  - Applications
    - 17, Coding Agents
    - 18, Enterprise Workflows
    - 19, Agents for Scientific Discovery
    - 20, Mathematical Reasoning and Theorem Proving
    - 21, Embodied Agents and Robotics
  - Evaluation, Safety, and Systems
    - 22, Evaluating Agents
    - 23, System Design for Agents
    - 24, Safety and Security of Agentic AI
    - 25, Trust, Capabilities, and Policy
  - Outlook
    - 26, Open Problems and the Road Ahead

## All Lesson Materials
// From ./generate_all_tocs.sh

- `book_Agentic_AI/all_tocs.md`
- `book_Agentic_AI/lectures_source/*.smd`

- `msml610/all_tocs.md`
- `msml610/lectures_source/*.smd`

- `data605/all_tocs.md`
- `data605/lectures_source/*.smd`

- `book_AI_Software_Engineering/lectures_source/*.smd`

- `book_cs_refreshers/lectures_source/*.smd`

## Roadmap

| Chap                                           | Slides % | Criticize | Tutorial | Book |
| ---------------------------------------------- | -------- | --------- | -------- | ---- |
|                                                |          |           |          |      |
| **Foundations**                                |          |           |          |      |
| 01. What Is an Agentic AI?                     | 100%     |           |          |      |
| 02. LLM Building Blocks                        | 100%     |           |          |      |
| 03. A Brief History of LLM Agents              | 100%     |           |          |      |
| 04. LLM Reasoning                              | 100%     |           |          |      |
| **Core Agent Capabilities**                    |          |           |          |      |
| 05. Reasoning, Memory, and Planning            | 100%     |           |          |      |
| 06. Inference-Time Techniques                  | 100%     |           |          |      |
| 07. Tool Use and Retrieval                     | 100%     |           |          |      |
| 08. Learning to Reason                         | 100%     |           |          |      |
| **Training Agentic Models**                    |          |           |          |      |
| 09. Post-Training and Verifiable Agents        | 100%     |           |          |      |
| 10. Open Training Recipes for Reasoning        | 100%     |           |          |      |
| 11. Lessons from Training Agentic Models       | 100%     |           |          |      |
| 12. Neural-Symbolic Decision Making            |          |           |          |      |
| **Multi-Agent and Multimodal Systems**         |          |           |          |      |
| 13. Multi-Agent AI                             |          |           |          |      |
| 14. Agent Frameworks                           |          |           |          |      |
| 15. Multimodal Autonomous Agents               |          |           |          |      |
| 16. From Perception to Action                  |          |           |          |      |
| **Applications**                               |          |           |          |      |
| 17. Coding Agents                              |          |           |          |      |
| 18. Enterprise Workflows                       |          |           |          |      |
| 19. Agents for Scientific Discovery            |          |           |          |      |
| 20. Mathematical Reasoning and Theorem Proving |          |           |          |      |
| 21. Embodied Agents and Robotics               |          |           |          |      |
| **Evaluation, Safety, and Systems**            |          |           |          |      |
| 22. Evaluating Agents                          |          |           |          |      |
| 23. System Design for Agents                   |          |           |          |      |
| 24. Safety and Security of Agentic AI          |          |           |          |      |
| 25. Trust, Capabilities, and Policy            | 40%      |           |          |      |
| **Outlook**                                    |          |           |          |      | 
| 26. Open Problems and the Road Ahead           |          |           |          |      |

## Tutorials

```
> find book_Agentic_AI -name *.ipynb
```

- No notebooks exist yet for this book

## TODOs
- Add benchmarks to `book_Agentic_AI/agentic_ai_toc.md`, per its trailing
  `TODO(ai_gp)`
- Write the decks for chapters 12-26, starting with the chapters that already
  have partial coverage from other courses (13, 17, 21, 22, 23, 25)
- Decide whether `Lesson15.1-Causal_Reasoning_Agents.smd` becomes its own
  chapter or is split between chapters 12 and 25
- Renumber `book_Agentic_AI/lectures_source/Lesson01.*` so the lesson number
  matches the chapter number (currently all decks sit under Lesson01)
- Add the tutorial notebooks (ReAct loop, RAG, DPO, SWE-bench harness)
- Extract the reference lists from `book_Agentic_AI/agentic_ai_toc.md` into a
  per-chapter `### References` subsection, once the chapters are drafted

# Detailed TOC

# Part I: Foundations

## 01: What Is an Agentic AI?

### Goals
- Separate agents from chatbots along a spectrum of autonomy and control
- Introduce the perceive-plan-act loop as the core abstraction of the book
- Build a taxonomy of agentic systems from tools, environments, observations

### Topics
- Motivation: From Generation to Action
  - The limits of pure text generation
  - What makes a system "agentic"
- Agents vs. Chatbots
  - Definitions and key differences
  - The spectrum of autonomy, with worked examples
  - Human-in-the-loop vs. full autonomy
- The Perceive-Plan-Act Loop
  - Perceive: observations and state
  - Plan: reasoning about the next action
  - Act: executing in the environment and closing the loop
- Tools, Environments, and Observations
  - Tools: from function calls to APIs and tool schemas
  - A taxonomy of environments
  - What the agent sees, and what it does not
- Grounding Language in Action
  - The grounding problem and action spaces
  - Grounding via tool schemas, with the WebShop example
- A Taxonomy of Agentic Systems
  - Reactive vs. deliberative agents
  - Single-agent vs. multi-agent systems

### Slides
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [25%]: ReAct thought-action-observation trace as the loop in practice,
    WebShop environment and action space as the grounding example
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [15%]: Retrieval as a tool call, which extends the tool definition beyond
    function calls and APIs
- _Not covered_
  - [5%]: Quantitative comparison of autonomy levels across deployed products,
    and the cost/latency implications of each level

## 02: LLM Building Blocks

### Goals
- Explain the transformer stack that every agent runs on
- Connect pretraining and scaling to the abilities agents rely on
- Show what transformers can and cannot compute in a single forward pass

### Topics
- Motivation: The Engine Inside the Agent
  - What a language model is, and tokenization
- The Transformer Architecture
  - Embeddings, positional encoding, the transformer block
  - Residual connections, layer normalization, feed-forward networks
- Attention as the Core Mechanism
  - Query, key, value and scaled dot-product attention
  - Multi-head, self- vs. cross-, and causal (masked) attention
  - Complexity and cost of attention
- Pretraining Objectives and Scaling
  - Masked vs. autoregressive objectives, the pretraining pipeline
  - Scaling laws and emergent abilities
- Inference, Decoding, and Context Windows
  - Autoregressive generation, greedy vs. sampling decoding
  - Top-k and top-p sampling
  - The context window and the KV cache
- Expressivity: What Transformers Can and Cannot Compute
  - Depth limits of a single forward pass
  - Chain-of-thought as serial computation

### Slides
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [95%]: Chapter's own deck: tokenization, transformer block, attention
    variants and cost, pretraining objectives, scaling laws and emergence,
    decoding strategies, context window and KV cache, expressivity limits
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [20%]: Depth limit and chain-of-thought as serial computation, one token
    per gate intuition, trading tokens for depth
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: Neural-network background (layers, training objectives) that the
    transformer section assumes
- _Not covered_
  - [5%]: Mixture-of-experts routing and modern attention kernels (FlashAttention,
    paged attention), which chapter 23 covers from the systems side

## 03: A Brief History of LLM Agents

### Goals
- Trace the path from prompting to acting, and why ReAct was the turning point
- Show what early web agents revealed about the human-machine gap
- Extract the recurring failure modes that later chapters try to fix

### Topics
- Motivation: Why the History Matters
  - A timeline of LLM agents
- From Prompting to Acting
  - Language models as text predictors, in-context learning
  - Chain-of-thought as reasoning in text
  - Acting on the world
- ReAct and the Reasoning-Acting Synergy
  - The thought-action-observation trace, with a worked example
  - ReAct vs. chain-of-thought vs. act-only
  - Why ReAct worked
- Grounded Web Agents
  - WebShop: environment, action space, and findings
  - The human-machine gap in web agents
- Early Failures and Lessons
  - Compounding errors and hallucinated tool calls
  - Getting stuck in loops
  - Brittle parsing and grounding
- The Road to General-Purpose Agents
  - From narrow to general-purpose agents
  - The rise of tool-use standards
  - Scaling meets agents

### Slides
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [95%]: Chapter's own deck: timeline, prompting to acting, ReAct trace and
    ablations, WebShop, the four early failure modes, tool-use standards and
    the road to general-purpose agents
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [25%]: Preview of ReAct and the loop, WebShop grounding example, agents
    vs. chatbots framing that the history builds on
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.smd`
  - [15%]: MCP as one protocol for many tools, which continues the tool-use
    standards thread into today's products
- _Not covered_
  - [5%]: Pre-LLM agent research (BDI architectures, classical planning agents)
    that would put the timeline in a longer historical context

## 04: LLM Reasoning

### Goals
- Explain why chain-of-thought helps, and when it does not
- Show that reasoning can be elicited without any prompt at all
- Expose the brittleness of LLM reasoning: premise order and self-correction

### Topics
- Motivation: What "Reasoning" Means for an LLM
  - Reasoning as serial computation
- Chain-of-Thought and Its Variants
  - Two hypotheses for why CoT helps
  - Zero-shot vs. few-shot CoT, self-consistency
  - Search over reasoning: tree and graph of thought
- Reasoning Without Explicit Prompting
  - Greedy decoding hides reasoning
  - CoT-decoding and the confidence signal
- Premise Order and Brittleness
  - Reordering premises, and the effect on math (R-GSM)
  - An autoregressive bias explanation
  - Brittleness beyond order
- Serial Computation and Depth
  - The depth limit and how CoT lifts it
  - Each token simulates one gate
  - Trading tokens for depth
- Limits of Self-Correction
  - When models cannot fix their own mistakes
  - When self-correction does help

### Slides
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [95%]: Chapter's own deck: CoT variants and self-consistency, tree/graph
    of thought, CoT-decoding and the confidence signal, premise order and
    R-GSM, serial computation and depth, limits of self-correction
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [30%]: Grounded vs. self-judged feedback, which explains when iterative
    refinement and self-correction help
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [20%]: Expressivity section, decoding strategies (greedy, top-k, top-p)
    that CoT-decoding builds on
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.smd`
  - [15%]: Structured chain-of-thought, self-planning, least-to-most
    decomposition, sample-and-vote applied to code
- _Not covered_
  - [5%]: Reasoning models trained with long CoT (o-series, R1-style) and their
    test-time scaling behavior, which chapter 09 touches only from the RL side

# Part II: Core Agent Capabilities

## 05: Reasoning, Memory, and Planning

### Goals
- Distinguish working memory from long-term memory in an agent loop
- Show how graph-structured memory fixes multi-hop retrieval failures
- Use an LLM as a world model to plan before acting

### Topics
- Working Memory vs. Long-Term Memory
  - Definitions and comparison
  - The core problem is retrieval, not storage
- Neurobiologically Inspired Memory (HippoRAG)
  - Why standard RAG struggles with multi-hop facts
  - Hippocampal indexing theory as the inspiration
  - Offline indexing, knowledge-graph construction, retrieval via personalized
    PageRank
  - Worked example, results, and efficiency
- Implicit Reasoning and Grokking
  - Can reasoning live inside the weights
  - Composition vs. comparison experiments
  - Generalizing vs. memorizing circuits
  - Weights as implicit long-term memory
- World Models for Planning
  - Reactive agents and the cost of search
  - An LLM as a world model that predicts state transitions
  - WebDreamer: simulate before you act
  - Reasoning, memory, and planning as one loop

### Slides
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [95%]: Chapter's own deck: working vs. long-term memory, HippoRAG pipeline
    and results, grokking and implicit reasoning, LLM world models and
    WebDreamer, the full loop
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [35%]: RAG loop, embeddings and nearest-neighbor search, single-hop
    retrieval, which HippoRAG is contrasted against
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [25%]: Model-based vs. model-free planning, MDP/POMDP belief states, value
    iteration, which formalize "simulate before you act"
- `msml610/lectures_source/Lesson09.1-Reasoning_over_time.smd`
  - [15%]: State, observation, and transition models over time, the formal
    background for a world model
- _Not covered_
  - [10%]: Episodic-memory engineering in deployed agents (summarization,
    compaction, memory eviction policies) and memory-write consistency

## 06: Inference-Time Techniques

### Goals
- Use the LLM itself as an optimizer over prompts and solutions
- Show why grounded feedback makes iterative refinement work
- Allocate inference compute where it buys the most accuracy

### Topics
- Motivation: Beyond a Single Reasoning Chain
  - Three levers: optimize, debug, allocate
- LLMs as Optimizers (OPRO)
  - Optimization without gradients, the meta-prompt
  - Algorithm and worked example on GSM8K
  - OPRO vs. gradient-based optimization
  - Why grounded scores matter
- Self-Debugging Code
  - Why generate-once code is often wrong
  - Two feedback channels: unit tests vs. code explanation
  - Algorithm, worked example on SQL, results
- When Does Iterative Refinement Help?
  - Grounded code vs. ungrounded reasoning
  - A spectrum of groundedness
  - Design the verifier, not just the loop
- Compute-Optimal Inference Strategies
  - How much inference compute an agent should spend
  - Easy vs. hard prompts and adaptive allocation
  - Best-of-N vs. compute-optimal search
  - Inference-time vs. training-time compute

### Slides
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [95%]: Chapter's own deck: OPRO, self-debug with two feedback channels,
    groundedness spectrum, compute-optimal scaling and adaptive allocation
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [30%]: Self-consistency, tree/graph of thought search, and the limits of
    self-correction that motivate grounded verifiers
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.smd`
  - [30%]: Self-refine and self-debug, test-first prompting, flow engineering,
    sample and vote, the refinement loop
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-verification, an inference-time technique reframed as a
    training-time signal
- _Not covered_
  - [10%]: Speculative decoding and other latency-oriented inference tricks,
    plus the dollar-cost accounting of best-of-N in production

## 07: Tool Use and Retrieval

### Goals
- Frame retrieval as a special case of tool use inside the agent loop
- Compare long-context stuffing with retrieval on cost and failure modes
- Show what needle-in-a-haystack evaluations do and do not measure

### Topics
- Motivation: Two Ways an Answer Can Be Wrong
  - Grounding facts, not just actions
- Retrieval as a Tool Call
  - The RAG loop, and RAG as a special case of tool use
  - Deciding whether to retrieve
  - Single-hop retrieval: when similarity is enough
- Vector Databases and Search
  - Embeddings, exact vs. approximate nearest-neighbor search
  - ANN indexes and hybrid semantic-plus-keyword search
  - A worked vector-database pipeline
- Grounding on Enterprise Knowledge
  - Three enterprise grounding strategies
  - Dynamic retrieval vs. parametric knowledge
  - High-fidelity grounding for regulated domains, fact-checking answers
- Long-Context vs. Retrieval
  - The cost of long-context stuffing and of retrieval failures
  - A side-by-side comparison
- The Needle-in-a-Haystack Evaluation
  - Test design, results, and what the test misses
  - Multiple needles: where recall breaks down

### Slides
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [95%]: Chapter's own deck: RAG as tool use, embeddings and ANN indexes,
    hybrid search, enterprise grounding strategies, long-context vs. retrieval
    trade-off, needle-in-a-haystack design and limits
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [30%]: Multi-hop retrieval failure of standard RAG, HippoRAG graph
    retrieval, retrieval vs. storage framing
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [20%]: Context window and KV-cache cost, the mechanism behind the
    long-context side of the trade-off
- `data605/lectures_source/Lesson12.2-Neo4j.smd`
  - [15%]: Graph storage and query engines, the infrastructure under
    graph-structured retrieval
- `data605/lectures_source/Lesson04.3-Data_Storage.smd`
  - [10%]: Indexing and storage layouts that vector databases specialize
- _Not covered_
  - [5%]: Agentic and iterative RAG (query planning, re-ranking cascades) and
    the eval harnesses for enterprise grounding quality

## 08: Learning to Reason

### Goals
- Derive DPO and explain why it removes the separate reward model
- Extend preference optimization to multi-step reasoning chains
- Reduce hallucination by factoring verification into separate questions

### Topics
- Motivation: From Inference-Time Fixes to Learned Reasoning
  - Two routes to better reasoning
- Reward Models and Preference Data
  - Preference data and the Bradley-Terry model
  - The classical reward-model plus PPO pipeline
- Direct Preference Optimization
  - The optimal policy in closed form
  - Reparameterizing the reward via the policy
  - The DPO loss, algorithm, and what its gradient rewards
  - DPO vs. RLHF: results and comparison
- Iterative Preference Optimization
  - Why DPO struggles at multi-step reasoning
  - From human preferences to correctness-based preferences
  - Preference pairs from self-generated chains of thought
  - The missing NLL term, and IRPO results
- Reducing Hallucination via Verification
  - Why holistic self-review failed
  - Chain-of-verification: algorithm and results
  - Why factored verification escapes anchoring bias
- Aligning Reasoning with Feedback
  - Training-time vs. inference-time grounding

### Slides
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [95%]: Chapter's own deck: preference data and Bradley-Terry, DPO
    derivation and gradient, IRPO with the NLL term, chain-of-verification,
    training- vs. inference-time grounding
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [40%]: DPO vs. PPO in practice, when to use each, preference hacking and
    mode collapse, preference-data collection at scale
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [25%]: From DPO to verifiable RL, why preference feedback alone is not
    enough
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [15%]: Policy search, reward shaping, and the RL vocabulary that PPO
    assumes
- _Not covered_
  - [5%]: GRPO and other critic-free RL variants used by current reasoning
    models, and the compute cost of each recipe

# Part III: Training Agentic Models

## 09: Post-Training and Verifiable Agents

### Goals
- Explain why verifiable rewards beat preference feedback for agent tasks
- Walk through SWE-bench Verified and BrowseComp as training signals
- Expose reward hacking and the gap between what we measure and what we want

### Topics
- Motivation: The Verification Problem
  - Why preference feedback alone is not enough
  - The verification-utility trade-off
- Verifiable Rewards from Real Tasks
  - What makes a task verifiable
  - From verification to training signals, and from DPO to verifiable RL
- SWE-Bench Verified: Software Tasks with Deterministic Outcomes
  - The SWE-bench family and why the Verified split matters
  - Training on SWE-bench trajectories, and results before and after
  - The setup burden, distribution shift, and wrong tests as an edge case
- BrowseComp: Benchmarking Browsing Agents
  - Browsing as a verifiable task, and the benchmark structure
  - Training on BrowseComp, and why it matters
- Comparing Verifiable Benchmarks
  - Task structure, solving rate vs. generalization
- Reward Hacking and Verification Gaps
  - What reward hacking is and why it happens
  - Testing against reward hacking
  - What we measure vs. what we care about

### Slides
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [95%]: Chapter's own deck: verifiability criteria, SWE-bench Verified
    training and pitfalls, BrowseComp, benchmark comparison, reward hacking
    and the verification gap
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [35%]: Reward hacking as pitfall 1, cost of verification at scale,
    trajectory quality and filtering
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [25%]: Preference-based training that verifiable RL replaces or augments
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [15%]: Train/validation/test discipline and metric choice, which the
    "training on a benchmark" discussion depends on
- _Not covered_
  - [5%]: Contamination auditing of agent benchmarks and the licensing/
    infrastructure cost of running SWE-bench-scale harnesses

## 10: Open Training Recipes for Reasoning

### Goals
- Compare DPO and PPO on setup cost, stability, and final quality
- Walk through Tulu 3 as a fully open post-training pipeline
- Show what reproducibility costs, and what it buys

### Topics
- Motivation: From Proprietary to Open Post-Training
  - Why open recipes matter, and the reproducibility challenge
- DPO vs. PPO: Theory, Practice, and Pitfalls
  - PPO in practice: setup and challenges
  - DPO in practice: simplicity and trade-offs
  - A decision framework for choosing between them
  - Preference hacking and mode collapse
- Tulu 3: An Open Post-Training Pipeline
  - Goals, data pipeline, training configuration
  - Evaluation, results, and ablations
- Preference Feedback: Collection and Best Practices
  - Human annotation vs. automatic scoring
  - Mixing multiple preference sources
  - Handling edge cases in preference data
- OpenScholar: Reasoning Training for Scientific Synthesis
  - The RAG plus reasoning pipeline, training data, results
- Reproducibility and Open-Source Lessons
  - Anatomy of a reproducible pipeline
  - The reproducibility-performance trade-off
  - Common pitfalls in open release

### Slides
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [95%]: Chapter's own deck: DPO vs. PPO decision framework, Tulu 3 data and
    training pipeline with ablations, preference-feedback collection,
    OpenScholar, reproducibility checklist and pitfalls
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [40%]: DPO derivation, reward models, PPO pipeline, iterative preference
    optimization, all of which this chapter turns into a recipe
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [30%]: Data curation, synthetic vs. real data, curriculum learning, stable
    training loops
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [15%]: RAG pipeline that OpenScholar extends to scientific synthesis
- _Not covered_
  - [5%]: License and data-governance questions in open releases, and the
    hardware budget needed to reproduce a Tulu-scale run

## 11: Lessons from Training Agentic Models

### Goals
- Show what makes training data "agentic" and how to filter trajectories
- Extract the practical recipe behind KIMI K2 and DeepSeek-V3
- Name the failure modes that break large-scale RL runs

### Topics
- Data Curation for Agentic Behavior
  - What makes data agentic, synthetic vs. real sources
  - Trajectory quality and filtering, preference data for agentic tasks
  - Curriculum learning for agent training
- Case Study: KIMI K2
  - Data curation at quantity and quality
  - The multi-stage training pipeline
  - Tool grounding in-context vs. in weights
  - Cost and efficiency
- Case Study: DeepSeek-V3
  - Mixture-of-experts at scale, and expert specialization
  - Training data, training challenges and solutions
  - Cost breakdown
- Stability and Cost of Large-Scale RL
  - The stability challenge, and techniques for stable RL
  - Cost of verification at scale, cost vs. capability trade-off
- Practical Pitfalls and Recipes
  - Reward hacking and domain generalization
  - The stable training loop
  - Monitoring and metrics that matter

### Slides
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [95%]: Chapter's own deck: agentic data curation, KIMI K2 and DeepSeek-V3
    case studies, stability and cost of large-scale RL, pitfalls, monitoring
    and recommendations
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [35%]: Verifiable rewards, reward hacking, cost of verification, which the
    case studies apply at scale
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [30%]: Open pipeline anatomy, preference-data practices, reproducibility
    lessons
- `data605/lectures_source/Lesson08.1-Cluster_Architecture.smd`
  - [10%]: Cluster and network-bandwidth constraints behind the training-cost
    discussion
- _Not covered_
  - [10%]: Vendor-specific training-infrastructure detail (parallelism plans,
    checkpointing, failure recovery), which chapter 23 addresses from the
    serving side

## 12: Neural-Symbolic Decision Making

### Goals
- Show how transformers can learn search and planning dynamics
- Contrast fast (intuitive) and slow (deliberate) inference in one model
- Use learned surrogates to attack combinatorial optimization

### Topics
- Search and Planning in Transformers
  - Planning as sequence prediction
  - Search dynamics bootstrapping: Beyond A*
  - What the model learns beyond the final plan
- Fast and Slow Thinking
  - Dualformer: randomized reasoning traces
  - Controlling the fast/slow mode at inference
  - Accuracy vs. token-cost trade-off
- Algebraic Structure of Reasoning
  - Composing global optimizers in neural nets
  - What the structure of a solved task looks like inside the weights
- Surrogates for Combinatorial Optimization
  - SurCo: learning linear surrogates for nonlinear problems
  - Where a learned surrogate beats a solver, and where it does not
- Symbolic Knowledge in an Agent
  - Logic and knowledge bases as an agent memory
  - Neuro-symbolic architectures and their trade-offs

### TODO
- [ ] Write the deck for this chapter
- [ ] Decide whether the causal-planning material in
  `Lesson15.1-Causal_Reasoning_Agents.smd` belongs here or in chapter 25

### Slides
- TODO: no deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [35%]: MDPs, value and policy iteration, dynamic decision networks, the
    search-and-planning formalism that "Beyond A*" learns to imitate
- `msml610/lectures_source/Lesson03.1-Knowledge_representation.smd`
  - [30%]: Symbolic vs. sub-symbolic representation, neuro-symbolic conceptual
    spaces, logical agents and knowledge bases
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [25%]: Connecting LLMs to formal causal models, planning under causal
    uncertainty, causal MDPs, worst-case policy robustness
- `msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd`
  - [20%]: Inference in propositional and first-order logic, the symbolic half
    of neural-symbolic systems
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [15%]: Tree and graph of thought as search over reasoning, serial-depth
    argument behind fast vs. slow thinking
- _Not covered_
  - [60%]: Search dynamics bootstrapping (Beyond A*), Dualformer's randomized
    traces, algebraic objects in neural nets, and SurCo have no deck in any
    course yet

# Part IV: Multi-Agent and Multimodal Systems

## 13: Multi-Agent AI

### Goals
- Motivate when multiple agents beat one agent with more tools
- Compare conversation-driven and state-driven orchestration
- Analyze competition and negotiation with game-theoretic tools

### Topics
- Why Multiple Agents
  - Specialization, parallelism, and separation of privilege
  - When a single agent with more tools is the better answer
- Conversation-Driven Collaboration
  - AutoGen: agents as conversable entities
  - Roles, group chat, and termination conditions
- State-Driven Workflows
  - StateFlow: task solving as a state machine
  - Comparing conversation-driven and state-driven control
- Competition, Negotiation, and Game Theory
  - Normal-form and extensive-form games, Nash equilibrium
  - Repeated games, mechanism design, auctions
  - Multi-agent reinforcement learning
- Emergent Coordination and Failure Modes
  - Deadlock, echo chambers, and runaway loops
  - Cost blow-up and error propagation across agents

### TODO
- [ ] Write the deck for this chapter
- [ ] Add a running multi-agent example reused in chapters 14 and 18

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_cs_refreshers/lectures_source/Lesson95.Refresher_game_theory.smd`
  - [45%]: Normal- and extensive-form games, Nash equilibrium, zero-sum and
    repeated games, mechanism design, auctions, MARL, learning in games
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [20%]: Single-agent vs. multi-agent taxonomy, reactive vs. deliberative
    agents
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`
  - [20%]: Division of labor, shared conventions, multiple engineers with one
    or more agents, escalation paths
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [10%]: Compounding errors and loop failures, which reappear as multi-agent
    failure modes
- _Not covered_
  - [55%]: AutoGen's conversable-agent API, StateFlow's state-machine
    formulation, and empirical results on emergent coordination

## 14: Agent Frameworks

### Goals
- Frame an agent as a compound AI system, not a single prompt
- Show how DSPy compiles programs into optimized prompts and weights
- Compare framework designs on control, debuggability, and portability

### Topics
- Compound AI Systems
  - Modules, control flow, and the interfaces between them
  - Why the system, not the model, is the unit of design
- Programming, Not Prompting
  - DSPy signatures, modules, and teleprompters
  - Declaring behavior instead of writing prompt strings
- Optimizing Instructions and Demonstrations
  - Multi-stage LM programs and credit assignment
  - Bootstrapping demonstrations, instruction search
- Joint Fine-Tuning and Prompt Optimization
  - Two steps that work better together
  - When to move capability from the prompt into the weights
- Framework Design Trade-offs
  - Graph-based vs. conversation-based vs. compiler-based frameworks
  - Tool protocols (MCP) and portability across models
  - Observability, tracing, and evaluation hooks

### TODO
- [ ] Write the deck for this chapter
- [ ] Choose the frameworks to cover hands-on (DSPy, LangGraph, AutoGen)

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.smd`
  - [30%]: MCP as one protocol for many tools, agent surfaces, CI integration,
    background sessions, the framework layer seen from the product side
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [25%]: OPRO as prompt optimization without gradients, the mechanism DSPy
    teleprompters generalize
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.smd`
  - [25%]: Prompt patterns as reusable templates, prompts as versioned
    artifacts, decomposition into scoped steps
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [15%]: Tool schemas and the agent-tool-environment decomposition that a
    framework materializes
- _Not covered_
  - [50%]: DSPy's concrete API, the MIPRO instruction/demonstration optimizer,
    and joint fine-tuning plus prompt-optimization results

## 15: Multimodal Autonomous Agents

### Goals
- Survey the benchmark ladder from Mind2Web to VisualWebArena
- Show what a realistic web environment adds over a static trace dataset
- Explain how tree search improves web agents at inference time

### Topics
- Generalist Web Agents
  - Mind2Web: task diversity and the generalization splits
  - Element grounding on real HTML
- Realistic Web Environments
  - WebArena: self-hosted sites and functional correctness
  - Why reproducible environments changed the metrics
- Visual Web Tasks
  - VisualWebArena: when the screenshot carries the information
  - Text-only vs. multimodal observation spaces
- Tree Search for Agents
  - Search over action sequences at inference time
  - Backtracking and state restoration in a browser
  - Cost and latency of search
- Perception-Reasoning Integration
  - Accessibility tree vs. DOM vs. pixels
  - Where multimodal agents still fail

### TODO
- [ ] Write the deck for this chapter
- [ ] Add a WebArena-style local environment for the tutorial

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [30%]: WebShop environment and action space, the human-machine gap in web
    agents, brittle parsing and grounding
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [30%]: WebDreamer model-based planning for web agents, simulate-before-act,
    efficiency and real-world results
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [20%]: Observation spaces, environment taxonomy, and action grounding
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [15%]: Compute-optimal search and best-of-N, the inference-time budget that
    tree search spends
- _Not covered_
  - [55%]: Mind2Web, WebArena, and VisualWebArena benchmark designs and
    results, plus multimodal observation encoders

## 16: From Perception to Action

### Goals
- Define GUI agents and the computer-use action space
- Show what open-ended computer environments measure
- Compare accessibility-tree grounding with pure-vision grounding

### Topics
- GUI Agents and Computer Use
  - The desktop as an environment: windows, files, applications
  - Action primitives: click, type, scroll, keyboard shortcuts
- Open-Ended Computer Environments
  - OSWorld: task setup, execution-based evaluation
  - Why open-ended tasks resist string-match scoring
- Pure-Vision GUI Interaction
  - AGUVIS: unified vision-only agents
  - Coordinate prediction and element grounding from pixels
- Action Grounding and Execution
  - Mapping intent to a concrete UI action
  - Recovery from mis-clicks and stale screens
- Evaluating Real-Computer Tasks
  - Sandboxing, resets, and reproducibility
  - Human baselines and the current gap

### TODO
- [ ] Write the deck for this chapter
- [ ] Decide how much of chapter 15 to merge here, since both cover grounding

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [25%]: Grounding problem, action spaces, environment taxonomy, observation
    definition
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [15%]: Brittle parsing and grounding, compounding errors, which dominate
    GUI agent failures
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`
  - [15%]: Confirmation points for hard-to-reverse actions, three-tier
    oversight, which computer-use agents need
- _Not covered_
  - [70%]: OSWorld and AGUVIS themselves, vision-based coordinate grounding,
    and sandboxed desktop evaluation infrastructure

# Part V: Applications

## 17: Coding Agents

### Goals
- Show why the agent-computer interface, not the model, gates coding agents
- Compare open platforms for software agents on architecture and results
- Apply coding agents to vulnerability discovery in real code

### Topics
- Agent-Computer Interfaces
  - SWE-agent: designing commands an LM can use reliably
  - Feedback, error messages, and guardrails as interface design
- Open Platforms for Software Agents
  - OpenHands: sandbox, event stream, and agent skills
  - Reproducing results across models
- Coding-Agent Workflows
  - Autocomplete, chat, and agentic assistance
  - Prompt engineering for code: specs, examples, and reasoning prompts
  - The refinement loop, restart vs. refine
- AI for Vulnerability Detection
  - Interactive tools that assist LM agents in finding vulnerabilities
  - From Naptime to Big Sleep: real-world bug finding
- Limits of Coding Agents
  - Hallucination types in code, knowledge conflicts
  - Context budget on large codebases, long files, ambiguous specs

### TODO
- [ ] Write the deck for this chapter, or fold in the
  `book_AI_Software_Engineering` decks
- [ ] Add a SWE-bench-style tutorial harness

### Slides
- TODO: no deck yet; see `book_AI_Software_Engineering/lectures_source/*.smd`

### Lesson Materials
- `book_AI_Software_Engineering/lectures_source/Lesson01.1_LLM_Assisted_Code_Generation.smd`
  - [55%]: Code generation mechanics, prompt anatomy, hallucination taxonomy
    and its root causes, context budget and degradation on large codebases
- `book_AI_Software_Engineering/lectures_source/Lesson02.1_AI_Pair_Programming_Workflows.smd`
  - [50%]: Autocomplete vs. chat vs. agentic assistance, Claude Code / Copilot
    / Cursor, CLI and IDE surfaces, MCP, CI integration, automated PR review
- `book_AI_Software_Engineering/lectures_source/Lesson03.1_Prompt_Engineering_for_Code.smd`
  - [45%]: Specs and acceptance criteria, few-shot patterns, test-first
    prompting, self-refine and self-debug, decomposition and the refinement
    loop
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [30%]: Self-debug with unit-test feedback, grounded verifiers, which are
    the core loop of a coding agent
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [25%]: SWE-bench Verified as both benchmark and training signal, setup
    burden, wrong-test edge cases
- _Not covered_
  - [35%]: SWE-agent and OpenHands architectures in detail, and the security
    work (interactive vulnerability tools, Naptime/Big Sleep)

## 18: Enterprise Workflows

### Goals
- Frame knowledge work as a benchmarkable agent task
- Show what compositional planning adds over single-step workflows
- Explain how to test conversational agents before they meet customers

### Topics
- Knowledge Work as Agent Tasks
  - WorkArena: enterprise UI tasks on a real platform
  - Task taxonomy and difficulty
- Compositional Planning
  - WorkArena++: composing atomic tasks into workflows
  - Planning, memory, and long-horizon reasoning in the enterprise
- Holistic Agent Development
  - TapeAgents: a tape as the unit of state and debugging
  - Optimization and replay of agent runs
- Dual-Control Conversational Agents
  - Tau2-bench: when both agent and user can act
  - Policy compliance and tool use under a rulebook
- Testing Before Deployment
  - Voice sims: simulating customers before production
  - Regression suites, escalation paths, and human handoff

### TODO
- [ ] Write the deck for this chapter
- [ ] Pick one enterprise scenario to carry through the chapter end to end

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`
  - [35%]: Three collaboration modes, the handoff problem, task-risk matching,
    three-tier oversight, escalation when the agent is uncertain,
    machine-readable contracts, team-level impact measurement
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [30%]: Enterprise grounding strategies, high-fidelity grounding for
    regulated domains, fact-checking grounded answers
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [15%]: Verifiable task design, which enterprise benchmarks copy
- `data605/lectures_source/Lesson07.1-Airflow.smd`
  - [10%]: Workflow orchestration and DAG scheduling, the classical baseline
    that agentic workflows are compared against
- _Not covered_
  - [55%]: WorkArena and WorkArena++ task suites, TapeAgents, tau2-bench
    dual-control setup, and voice-sim testing methodology

## 19: Agents for Scientific Discovery

### Goals
- Show agents acting as collaborators across a full research cycle
- Walk through the Virtual Lab nanobody design result end to end
- Turn a paper into a reliable, callable agent

### Topics
- Agents as Collaborators in Research
  - Literature triage, hypothesis generation, experiment design
  - Where the human stays in the loop
- The Virtual Lab
  - A team of specialized agents designing SARS-CoV-2 nanobodies
  - Meetings, critiques, and the wet-lab validation step
- Papers as Interactive Agents
  - Paper2Agent: extracting tools and workflows from a paper
  - Reliability of the extracted agent
- Retrieval-Augmented Scientific Synthesis
  - OpenScholar-style pipelines over the literature
  - Citation grounding and fabricated-reference detection
- Reliability and Reproducibility in Science
  - Verification, replication, and provenance
  - What can go wrong when the reviewer is also an agent

### TODO
- [ ] Write the deck for this chapter
- [ ] Add a small literature-synthesis notebook as the tutorial

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [30%]: OpenScholar motivation, RAG plus reasoning pipeline, training data,
    results, which cover the scientific-synthesis topic
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [25%]: Retrieval quality, fact-checking grounded answers, long-context vs.
    retrieval trade-off for literature-scale corpora
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [25%]: How to do research, hypothesis framing, and experiment design, the
    human process agents are asked to imitate
- `book_Agentic_AI/lectures_source/Lesson08.1-Learning_to_reason.smd`
  - [15%]: Chain-of-verification for reducing fabricated claims
- _Not covered_
  - [55%]: The Virtual Lab and Paper2Agent case studies, wet-lab validation
    loops, and domain-specific scientific tooling

## 20: Mathematical Reasoning and Theorem Proving

### Goals
- Connect self-play RL to competition-level formal mathematics
- Explain retrieval-augmented proving and premise selection
- Bridge informal sketches and formal proofs via autoformalization

### Topics
- From Self-Play to Formal Math
  - AlphaZero-style self-play as the precedent
  - AlphaProof and IMO-level results
- Retrieval-Augmented Theorem Proving
  - LeanDojo: environment, premise selection, retrieval
  - miniCTX: long-context proving
- Autoformalization
  - Translating natural-language statements into Lean
  - Autoformalizing Euclidean geometry as a case study
- Bridging Informal and Formal Proofs
  - Draft, sketch, and prove
  - Lean-STaR: interleaving thinking and proving
- Abstraction, Discovery, and Proof Optimization
  - ImProver: automated proof optimization
  - Symbolic regression with a learned concept library

### TODO
- [ ] Write the deck for this chapter
- [ ] Decide the Lean vs. Coq scope for the tutorial

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [30%]: Chain-of-thought and its variants, tree/graph search over reasoning,
    self-consistency, the substrate proof search runs on
- `msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd`
  - [25%]: Syntax, semantics, and inference in propositional and first-order
    logic, the formal-proof background
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [20%]: Compute-optimal search and best-of-N, the search budget a prover
    spends per goal
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [20%]: Verifiable rewards, which a proof checker supplies exactly
- _Not covered_
  - [55%]: AlphaProof, LeanDojo, autoformalization, Lean-STaR, ImProver, and
    the proof-assistant tooling itself

## 21: Embodied Agents and Robotics

### Goals
- Show open-ended skill acquisition driven by an LLM curriculum
- Use code-writing LLMs to design reward functions
- Explain what sim-to-real transfer costs and how to reduce it

### Topics
- Open-Ended Embodied Agents
  - Voyager: automatic curriculum, skill library, iterative prompting
  - Lifelong learning without gradient updates
- Reward Design via Code
  - Eureka: LLM-written reward functions, evolutionary search
  - Human-level reward design and its evaluation
- Sim-to-Real Transfer
  - DrEureka: LLM-guided domain randomization
  - The reality gap and safety margins
- Superhuman Control with Deep RL
  - Gran Turismo: outracing champion drivers
  - Reward shaping and sportsmanship constraints
- Whole-Body Real-World RL
  - SLAC: simulation-pretrained latent action spaces
  - Sample efficiency and safety on real hardware

### TODO
- [ ] Write the deck for this chapter
- [ ] Reuse the `msml610` gridworld notebooks as the RL warm-up

### Slides
- TODO: no deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [45%]: MDPs and POMDPs, utilities over time, Bellman equation, value and
    policy iteration, model-based vs. model-free, active vs. passive RL, safe
    exploration, policy search, generalization in RL
- `book_Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [25%]: World models for planning, simulate-before-act, long-term memory as
    a skill library
- `book_Agentic_AI/lectures_source/Lesson06.1-Inference_time_techniques.smd`
  - [20%]: LLMs as optimizers and self-debug, the mechanism behind LLM-written
    reward functions
- `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.smd`
  - [10%]: Exploration-exploitation trade-off underlying curriculum design
- _Not covered_
  - [55%]: Voyager, Eureka, DrEureka, Gran Turismo, and SLAC themselves, plus
    robot hardware, control stacks, and simulator tooling

# Part VI: Evaluation, Safety, and Systems

## 22: Evaluating Agents

### Goals
- Explain why agent evaluation is harder than model evaluation
- Put error bars on evals and separate signal from benchmark noise
- Give a checklist for designing a trustworthy agent benchmark

### Topics
- What Makes Agent Evaluation Hard
  - Multi-step trajectories, partial credit, and non-determinism
  - Environment resets, statefulness, and cost per run
- A Survey of LLM-Agent Evaluation
  - Capability, application, and generalist benchmarks
  - Trajectory-level vs. outcome-level metrics
- Statistical Rigor
  - Error bars on evals: variance from sampling and from the benchmark
  - Paired comparisons and clustered standard errors
- Predictable Noise in Benchmarks
  - Item difficulty, contamination, and saturation
  - Why small benchmark deltas mean nothing
- Designing Trustworthy Benchmarks
  - Verifiability, held-out splits, and anti-gaming design
  - Reporting cost, latency, and tool calls next to accuracy

### TODO
- [ ] Write the deck for this chapter
- [ ] Add an eval notebook that computes error bars on a small agent benchmark

### Slides
- TODO: no deck yet

### Lesson Materials
- `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`
  - [40%]: Train/validation/test splits, in- vs. out-of-sample error, metric
    choice, precision/recall and AUC, model selection as learning
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [40%]: SWE-bench and BrowseComp structure, benchmark comparison, solving
    rate vs. generalization, reward hacking and the verification gap
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [20%]: Experimental discipline and how to claim progress honestly
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [15%]: Monitoring and metrics that matter, cost of verification at scale
- _Not covered_
  - [40%]: The agent-evaluation survey taxonomy and the statistical machinery
    for error bars on evals (variance decomposition, paired tests)

## 23: System Design for Agents

### Goals
- Lay out the agent stack from hardware to product surface
- Explain what dominates serving cost and latency for agent workloads
- Show the architectural choices that make large models cheap to run

### Topics
- The AI Engineer's View of the Stack
  - Model, inference server, orchestration, tools, product surface
  - Where state lives across the stack
- Training and Inference Infrastructure
  - Clusters, accelerators, and network bandwidth
  - Batching, KV-cache management, and continuous batching
- Serving Agents at Scale
  - KIMI K2 serving lessons
  - Multi-turn, tool-calling traffic vs. single-shot chat traffic
- Efficient Large Models
  - DeepSeek-V3 mixture-of-experts and expert specialization
  - Quantization and distillation for agent workloads
- Latency, Cost, and Reliability
  - Budgeting tokens per task, caching, and retries
  - Failure isolation, timeouts, and graceful degradation

### TODO
- [ ] Write the deck for this chapter
- [ ] Add a cost model that students can fill in for their own agent

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [40%]: KIMI K2 pipeline, cost and efficiency, DeepSeek-V3 mixture-of-
    experts, expert specialization, cost breakdown, cost vs. capability
- `data605/lectures_source/Lesson11.2-Cloud_Computing_Enablers.smd`
  - [30%]: Data-center capex/opex, virtualization, Docker, programming
    frameworks, cloud benefits and limitations
- `data605/lectures_source/Lesson08.1-Cluster_Architecture.smd`
  - [25%]: Cluster architecture, network bandwidth, storage infrastructure,
    distributed file systems
- `book_Agentic_AI/lectures_source/Lesson02.1-LLM_Building_Blocks.smd`
  - [20%]: Context window, KV cache, attention complexity, the mechanisms that
    set serving cost
- `data605/lectures_source/Lesson11.1-Cloud_Computing.smd`
  - [15%]: IaaS/PaaS/SaaS layering and deployment models
- _Not covered_
  - [30%]: Agent-specific serving concerns (tool-call routing, sandbox pools,
    long-running session state) and current inference-server internals

## 24: Safety and Security of Agentic AI

### Goals
- Map the attack surface that tools, memory, and autonomy open up
- Explain prompt injection and memory poisoning with concrete attacks
- Apply privilege separation and programmable policy to contain agents

### Topics
- The Expanded Attack Surface of Agents
  - Untrusted content as instructions
  - Tools, credentials, and irreversible actions
- Prompt Injection and Its Detection
  - Direct and indirect injection
  - DataSentinel: game-theoretic detection
- Memory and Knowledge-Base Poisoning
  - AgentPoison: red-teaming memory and RAG stores
  - Persistence and blast radius of a poisoned memory
- Privilege Separation and Control
  - Privtrans: partitioning programs for privilege separation
  - Progent: programmable privilege control for LLM agents
  - Least privilege, confirmation points, and sandboxing
- Adversarial Attacks on Aligned Models
  - Universal and transferable adversarial suffixes
  - Why alignment training does not close the hole

### TODO
- [ ] Write the deck for this chapter
- [ ] Add a hands-on injection lab against a sandboxed agent

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`
  - [25%]: Confirmation points for hard-to-reverse actions, what to delegate
    vs. what needs a human, three-tier oversight, risk matrix
- `book_Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [20%]: Retrieval stores and grounding pipelines, the assets that poisoning
    attacks target
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [15%]: Reward hacking, the training-time analogue of adversarial gaming
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [10%]: Tool schemas, action spaces, and the autonomy spectrum that sets the
    blast radius
- _Not covered_
  - [70%]: DataSentinel, AgentPoison, Privtrans, Progent, and transferable
    adversarial attacks, plus agent-specific threat modeling

## 25: Trust, Capabilities, and Policy

### Goals
- Measure dangerous capabilities before deployment, not after
- Explain responsible scaling policies and capability thresholds
- Assess trustworthiness across explainability, robustness, and fairness

### Topics
- Measuring Dangerous Capabilities
  - Capability evaluations and elicitation
  - Cybench: cybersecurity capability and risk evaluation
- Responsible Scaling Policies
  - Capability thresholds and required safeguards
  - Computer-use models and their specific risks
- Trustworthiness Assessment
  - DecodingTrust: dimensions of trustworthiness
  - Robustness, privacy, fairness, and toxicity
- Explainability for Trust
  - Local vs. global explanations, feature attribution (SHAP, LIME)
  - Counterfactual explanations, faithfulness and stability
- Causal Reasoning for Trustworthy Agents
  - Causal explanations for decisions
  - Causal constraints for robustness, fairness, and safety
  - Causal monitoring and adaptation
- Evidence-Based AI Policy
  - What regulators can and cannot measure

### TODO
- [ ] Split `Lesson15.1-Causal_Reasoning_Agents.smd` between this chapter and
  chapter 12, or promote it to its own chapter

### Slides
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - Covers the causal-trust portion only; the capability-evaluation and policy
    portions still need a deck

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [45%]: Transparency and interpretability, causal explanations for
    decisions, robustness through causal constraints, causal fairness
    definitions, safety constraints on harmful outcomes, causal monitoring
- `msml610/lectures_source/Lesson13.1-Explainability.smd`
  - [40%]: Accuracy vs. interpretability, taxonomy of explainability methods,
    SHAP and LIME, permutation importance, counterfactual explanations,
    faithfulness and stability of explanations
- `book_Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [15%]: Verification gaps and reward hacking, which capability evaluations
    must anticipate
- `book_AI_Software_Engineering/lectures_source/Lesson14.1_Human_AI_Collaboration_Patterns.smd`
  - [15%]: Calibrating trust vs. reliance, evidence and provenance for team
    trust
- _Not covered_
  - [40%]: Responsible scaling policies, computer-use risk assessment,
    DecodingTrust, Cybench, and evidence-based policy frameworks

# Part VII: Outlook

## 26: Open Problems and the Road Ahead

### Goals
- Contrast open and closed foundation models on progress and accountability
- Name the grand challenges that current agents do not solve
- Give the reader a research agenda, not just a summary

### Topics
- Open-Source vs. Closed Foundation Models
  - What openness buys: reproducibility, auditing, derivative research
  - What it costs: safety review, misuse, and funding
- Science in the Era of Foundation Models
  - Benchmarks as scientific instruments
  - Reproducibility under moving models
- Cybersecurity Capabilities and Risks
  - Dual-use capability growth
  - Cybench-style measurement over time
- Generalist Agents and Open-Endedness
  - Continual learning and skill accumulation
  - Self-improvement loops and their limits
- Grand Challenges for the Next Decade
  - Reliable long-horizon autonomy
  - Verification that scales with capability
  - Economics and governance of deployed agents

### TODO
- [ ] Write the deck for this chapter
- [ ] Keep the open-problem list in sync as later chapters are drafted

### Slides
- TODO: no deck yet

### Lesson Materials
- `book_Agentic_AI/lectures_source/Lesson10.1_Open_training_recipes_for_reasoning.smd`
  - [35%]: Why open training recipes matter, the reproducibility challenge and
    trade-off, pitfalls in open release, open problems and future directions
- `book_Agentic_AI/lectures_source/Lesson11.1_Lessons_from_training_agentic_models.smd`
  - [25%]: Emerging patterns, recommendations for practitioners, open problems
- `book_Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [20%]: Open challenges on the road to general-purpose agents
- `book_Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [10%]: Summary and open questions on what "agentic" will mean next
- _Not covered_
  - [50%]: The open-vs-closed policy argument, Cybench results, and the
    open-endedness research agenda

# Appendix

## Refreshers
- `book_cs_refreshers/lectures_source/Lesson95.Refresher_game_theory.smd`:
  background for chapter 13
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`: MDP and RL
  background for chapters 08, 09, 12, and 21
- `msml610/lectures_source/Lesson03.1-Knowledge_representation.smd` and
  `msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd`:
  symbolic background for chapters 12 and 20

## Resources
- `book_Agentic_AI/lectures_source/L01.08_resources.md`: reading list that backs
  the per-chapter reference lists
- `book_Agentic_AI/berkeley_classes.smd` and
  `book_Agentic_AI/berkeley_classes2.smd`: transcripts of the Berkeley LLM
  agents course, used as a cross-check on chapter coverage
- `book_Agentic_AI/agentic_ai_toc.md`: the source table of contents, including
  the per-chapter reference URLs
