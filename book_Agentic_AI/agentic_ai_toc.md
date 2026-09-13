# From LLMs to Agents: Foundations and Frontiers of Agentic AI

## Part I — Foundations

### 1. What Is an Agentic AI?
- Agents vs. chatbots: autonomy and the spectrum of control
- The perceive–plan–act loop
- Tools, environments, and observations
- Grounding language in action
- A taxonomy of agentic systems
- **References**
  - ReAct: Synergizing Reasoning and Acting in Language Models: https://arxiv.org/abs/2210.03629
  - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents: https://arxiv.org/abs/2207.01206

### 2. LLM Building Blocks
- The transformer architecture
- Attention as the core mechanism
- Pretraining objectives and scaling
- Inference, decoding, and context windows
- Expressivity: what transformers can and cannot compute
- **References**
  - Attention Is All You Need: https://arxiv.org/abs/1706.03762
  - Chain-of-Thought Prompting: https://arxiv.org/abs/2201.11903
  - (Optional) Neural GPUs: https://arxiv.org/abs/1511.08228
  - (Optional) GPT becoming a Turing machine: https://arxiv.org/abs/2303.14310

### 3. A Brief History of LLM Agents
- From prompting to acting
- ReAct and the reasoning–acting synergy
- Grounded web agents
- Early failures and lessons
- The road to general-purpose agents
- **References**
  - ReAct: Synergizing Reasoning and Acting in Language Models: https://arxiv.org/abs/2210.03629
  - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents: https://arxiv.org/abs/2207.01206

### 4. LLM Reasoning
- Chain-of-thought and its variants
- Reasoning without explicit prompting
- Premise order and brittleness
- Serial computation and depth
- Limits of self-correction
- **References**
  - Chain-of-Thought Reasoning Without Prompting: https://arxiv.org/abs/2402.10200
  - Large Language Models Cannot Self-Correct Reasoning Yet: https://arxiv.org/abs/2310.01798
  - Premise Order Matters in Reasoning with Large Language Models: https://arxiv.org/abs/2402.08939
  - Chain-of-Thought Empowers Transformers to Solve Inherently Serial Problems: https://arxiv.org/abs/2402.12875

## Part II — Core Agent Capabilities

### 5. Reasoning, Memory, and Planning
- Working memory vs. long-term memory
- Neurobiologically inspired memory (HippoRAG)
- Implicit reasoning and grokking
- World models for planning
- Model-based planning for web agents
- **References**
  - Grokked Transformers are Implicit Reasoners: https://arxiv.org/abs/2405.15071
  - HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models: https://arxiv.org/abs/2405.14831
  - Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents: https://arxiv.org/abs/2411.06559

### 6. Inference-Time Techniques
- Sampling, search, and self-consistency
- LLMs as optimizers
- Self-debugging code
- The limits of self-correction
- Compute-optimal inference strategies
- **References**
  - Large Language Models as Optimizers: https://arxiv.org/abs/2309.03409
  - Large Language Models Cannot Self-Correct Reasoning Yet: https://arxiv.org/abs/2310.01798
  - Teaching Large Language Models to Self-Debug: https://arxiv.org/abs/2304.05128

### 7. Tool Use and Retrieval
- Retrieval-augmented generation
- Vector databases and search
- Grounding on enterprise knowledge
- Long-context vs. retrieval
- The needle-in-a-haystack evaluation
- **References**
  - Google Cloud expands grounding capabilities on Vertex AI: https://cloud.google.com/blog/products/ai-machine-learning/rag-and-grounding-on-vertex-ai
  - The Needle In a Haystack Test: Evaluating the performance of RAG systems: https://towardsdatascience.com/the-needle-in-a-haystack-test-a94974c1ad38
  - The AI detective: The Needle in a Haystack test and how Gemini 1.5 Pro solves it: https://cloud.google.com/blog/products/ai-machine-learning/the-needle-in-the-haystack-test-and-how-gemini-pro-solves-it

### 8. Learning to Reason
- Reward models and preference data
- Direct Preference Optimization
- Iterative preference optimization
- Reducing hallucination via verification
- Aligning reasoning with feedback
- **References**
  - Direct Preference Optimization: Your Language Model is Secretly a Reward Model: https://arxiv.org/abs/2305.18290
  - Iterative Reasoning Preference Optimization: https://arxiv.org/abs/2404.19733
  - Chain-of-Verification Reduces Hallucination in Large Language Models: https://arxiv.org/abs/2309.11495

## Part III — Training Agentic Models

### 9. Post-Training and Verifiable Agents
- Why verifiability matters
- Reinforcement learning from verifiable rewards
- Benchmarking real software tasks (SWE-bench Verified)
- Benchmarking browsing agents (BrowseComp)
- Reward hacking and verification gaps
- **References**
  - Introducing SWE-bench Verified: https://openai.com/index/introducing-swe-bench-verified/
  - BrowseComp: a benchmark for browsing agents: https://openai.com/index/browsecomp/

### 10. Open Training Recipes for Reasoning
- Open post-training pipelines (Tulu 3)
- DPO vs. PPO in practice
- Preference feedback best practices
- Retrieval-augmented scientific synthesis (OpenScholar)
- Reproducibility and open models
- **References**
  - Tulu 3: Pushing Frontiers in Open Language Model Post-Training: https://arxiv.org/abs/2411.15124
  - Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback: https://arxiv.org/abs/2406.09279
  - OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs: https://arxiv.org/abs/2411.14199

### 11. Lessons from Training Agentic Models
- Data curation for agentic behavior
- Open agentic intelligence (KIMI K2)
- Scaling mixture-of-experts (DeepSeek-V3)
- Stability and cost of large-scale RL
- Practical pitfalls and recipes
- **References**
  - KIMI K2: Open Agentic Intelligence: https://agenticai-learning.org/slides/d11.pdf
  - DeepSeek-V3 Technical Report: https://agenticai-learning.org/slides/d22.pdf

### 12. Neural-Symbolic Decision Making
- Search and planning in transformers
- Beyond A*: learned search dynamics
- Fast and slow thinking (Dualformer)
- Algebraic structure of reasoning
- Surrogates for combinatorial optimization
- **References**
  - Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping: https://arxiv.org/abs/2402.14083
  - Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces: https://arxiv.org/abs/2410.09918v1
  - Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets: https://arxiv.org/abs/2410.01779
  - SurCo: Learning Linear Surrogates For Combinatorial Nonlinear Optimization Problems: https://arxiv.org/abs/2210.12547

## Part IV — Multi-Agent and Multimodal Systems

### 13. Multi-Agent AI
- Why multiple agents
- Conversation-driven collaboration (AutoGen)
- State-driven workflows (StateFlow)
- Competition, negotiation, and game theory
- Emergent coordination and failure modes
- **References**
  - AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation: https://arxiv.org/abs/2308.08155
  - StateFlow: Enhancing LLM Task-Solving through State-Driven Workflows: https://arxiv.org/abs/2403.11322

### 14. Agent Frameworks
- Compound AI systems
- Programming, not prompting (DSPy)
- Optimizing instructions and demonstrations
- Joint fine-tuning and prompt optimization
- Framework design trade-offs
- **References**
  - Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs: https://arxiv.org/abs/2406.11695
  - Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together: https://arxiv.org/abs/2407.10930

### 15. Multimodal Autonomous Agents
- Generalist web agents (Mind2Web)
- Realistic web environments (WebArena)
- Visual web tasks (VisualWebArena)
- Tree search for agents
- Perception–reasoning integration
- **References**
  - Mind2Web: Towards a Generalist Agent for the Web: https://arxiv.org/abs/2306.06070
  - WebArena: A Realistic Web Environment for Building Autonomous Agents: https://arxiv.org/abs/2307.13854
  - VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks: https://jykoh.com/vwa
  - Tree Search for Language Model Agents: https://jykoh.com/search-agents

### 16. From Perception to Action
- GUI agents and computer use
- Open-ended computer environments (OSWorld)
- Pure-vision GUI interaction (AGUVIS)
- Action grounding and execution
- Evaluating real-computer tasks
- **References**
  - OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments: https://arxiv.org/pdf/2404.07972
  - AGUVIS: Unified Pure Vision Agents For Autonomous GUI Interaction: https://arxiv.org/pdf/2412.04454

## Part V — Applications

### 17. Coding Agents
- Agent–computer interfaces (SWE-agent)
- Open platforms for software agents (OpenHands)
- AI for vulnerability detection
- Interactive tools for security analysis
- From Naptime to Big Sleep: real-world bug finding
- **References**
  - SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering: https://arxiv.org/abs/2405.15793
  - OpenHands: An Open Platform for AI Software Developers as Generalist Agents: https://arxiv.org/abs/2407.16741
  - Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities: https://arxiv.org/abs/2409.16165
  - From Naptime to Big Sleep: Using Large Language Models To Catch Vulnerabilities In Real-World Code: https://googleprojectzero.blogspot.com/2024/10/from-naptime-to-big-sleep.html

### 18. Enterprise Workflows
- Knowledge work as agent tasks (WorkArena)
- Compositional planning (WorkArena++)
- Holistic agent development (TapeAgents)
- Dual-control conversational agents (τ2-Bench)
- Testing before deployment (Voice Sims)
- **References**
  - WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?: https://arxiv.org/abs/2403.07718
  - WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks: https://arxiv.org/abs/2407.05291
  - TapeAgents: a Holistic Framework for Agent Development and Optimization: https://rdi.berkeley.edu/llm-agents-mooc/assets/tapeagents.pdf
  - τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment: https://arxiv.org/pdf/2506.07982
  - Voice Sims: test agents in real world conditions before they talk to customers: https://sierra.ai/blog/voice-sims-test-agents-in-real-world-conditions-before-they-talk-to-your-customers

### 19. Agents for Scientific Discovery
- Agents as collaborators in research
- The Virtual Lab: designing nanobodies
- Papers as interactive agents (Paper2Agent)
- Hypothesis generation and experiment design
- Reliability and reproducibility in science
- **References**
  - The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies: https://www.nature.com/articles/s41586-025-09442-9
  - Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents: https://arxiv.org/abs/2509.06917

### 20. Mathematical Reasoning and Theorem Proving
- From self-play to formal math (AlphaProof)
- Retrieval-augmented theorem proving (LeanDojo)
- Autoformalization
- Bridging informal and formal proofs
- Abstraction, discovery, and proof optimization
- **References**
  - AI achieves silver-medal standard solving International Mathematical Olympiad problems: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/
  - Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm: https://arxiv.org/pdf/1712.01815
  - LeanDojo: Theorem Proving with Retrieval-Augmented Language Models: https://arxiv.org/abs/2306.15626
  - Autoformalization with Large Language Models: https://arxiv.org/abs/2205.12615
  - Autoformalizing Euclidean Geometry: https://arxiv.org/abs/2405.17216
  - Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs: https://arxiv.org/abs/2210.12283
  - miniCTX: Neural Theorem Proving with Long-Contexts: https://www.arxiv.org/pdf/2408.03350
  - Lean-STaR: Learning to Interleave Thinking and Proving: https://arxiv.org/abs/2407.10040
  - ImProver: Agent-Based Automated Proof Optimization: https://arxiv.org/abs/2410.04753
  - An In-Context Learning Agent for Formal Theorem-Proving: https://arxiv.org/abs/2310.04353
  - Symbolic Regression with a Learned Concept Library: https://arxiv.org/abs/2409.09359

### 21. Embodied Agents and Robotics
- Open-ended embodied agents (Voyager)
- Reward design via code (Eureka)
- Sim-to-real transfer (DrEureka)
- Superhuman control with deep RL (Gran Turismo)
- Whole-body real-world RL (SLAC)
- **References**
  - Voyager: An Open-Ended Embodied Agent with Large Language Models: https://voyager.minedojo.org/
  - Eureka: Human-Level Reward Design via Coding Large Language Models: https://eureka-research.github.io/
  - DrEureka: Language Model Guided Sim-To-Real Transfer: https://eureka-research.github.io/dr-eureka/
  - Outracing Champion Gran Turismo Drivers with Deep Reinforcement Learning: https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-nature22.html
  - SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL: https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-jiaheng_hu_2025.html

## Part VI — Evaluation, Safety, and Systems

### 22. Evaluating Agents
- What makes agent evaluation hard
- A survey of LLM-agent evaluation
- Statistical rigor: error bars on evals
- Predictable noise in benchmarks
- Designing trustworthy benchmarks
- **References**
  - Survey on Evaluation of LLM-based Agents: https://arxiv.org/pdf/2503.16416
  - Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations: https://arxiv.org/pdf/2411.00640

### 23. System Design for Agents
- The AI-engineer's view of the stack
- Training and inference infrastructure
- Serving agents at scale (KIMI K2)
- Efficient large models (DeepSeek-V3)
- Latency, cost, and reliability
- **References**
  - KIMI K2: Open Agentic Intelligence: https://agenticai-learning.org/slides/d11.pdf
  - DeepSeek-V3 Technical Report: https://agenticai-learning.org/slides/d22.pdf

### 24. Safety and Security of Agentic AI
- The expanded attack surface of agents
- Prompt injection and its detection (DataSentinel)
- Memory and knowledge-base poisoning (AgentPoison)
- Privilege separation and control (Privtrans, Progent)
- Adversarial attacks on aligned models
- **References**
  - Privtrans: Automatically Partitioning Programs for Privilege Separation: https://dawnsong.io/papers/privtrans.pdf
  - DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks: https://arxiv.org/abs/2504.11358
  - AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases: https://arxiv.org/abs/2407.12784
  - Progent: Programmable Privilege Control for LLM Agents: https://arxiv.org/html/2504.11703v1
  - Universal and Transferable Adversarial Attacks on Aligned Language Models: https://arxiv.org/abs/2307.15043

### 25. Trust, Capabilities, and Policy
- Measuring dangerous capabilities
- Responsible scaling policies
- Computer-use models and risk
- Trustworthiness assessment (DecodingTrust)
- Evidence-based AI policy
- **References**
  - Announcing our updated Responsible Scaling Policy: https://www.anthropic.com/news/announcing-our-updated-responsible-scaling-policy
  - Developing a computer use model: https://www.anthropic.com/news/developing-computer-use
  - A Path for Science‑ and Evidence‑based AI Policy: https://understanding-ai-safety.org/
  - DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models: https://arxiv.org/abs/2306.11698
  - Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models: https://arxiv.org/abs/2408.08926

## Part VII — Outlook

### 26. Open Problems and the Road Ahead
- Open-source vs. closed foundation models
- Science in the era of foundation models
- Cybersecurity capabilities and risks
- Generalist agents and open-endedness
- Grand challenges for the next decade
- **References**
  - Open-Source and Science in the Era of Foundation Models (Percy Liang): https://rdi.berkeley.edu/llm-agents/assets/percyliang.pdf
  - Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models: https://arxiv.org/abs/2408.08926

# TODO(ai_gp): Add benchmarks
