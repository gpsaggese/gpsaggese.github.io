The goal is to build a system that converts financial papers into theory / models,
by:

- Creating a causal DAG from the paper explaining the theory  
- Converting the “theory” into Python model  
- Optimizing the causal DAG (where the optimization criteria are specified later)

This idea sits where four bodies of work.

## **Relevant academic work**

**1\. Causal inference foundations (the formal language for your DAG)**

* Pearl, *Causality* (2009), and Pearl & Mackenzie, *The Book of Why*  
  * These cover structural causal models (SCMs), do-calculus, and backdoor/frontdoor
    identification.  
* Peters, Janzing & Schölkopf, *Elements of Causal Inference* (2017)  
  * It's free online and the most code-friendly treatment.  
* Imbens (2020), "Potential Outcome and Directed Acyclic Graph Approaches to
  Causality" (*JEL*)  
  * This bridges DAGs and the econometrics tradition finance papers actually use.  
* Hünermund & Bareinboim, "Causal Inference and Data Fusion in Econometrics."  
  * It maps IV, diff-in-diff and similar designs onto graphical identification.

**2\. Causality in finance specifically (the motivation and the "spec")**

* López de Prado, *Causal Factor Investing* (Cambridge Elements, 2023, open access)  
  * He argues that virtually all factor-investing articles make associational claims:
    authors don't identify a causal graph, justify specifications by correlations,
    and propose no falsification experiments. That is exactly the gap your system
    fills.  
* López de Prado & Zoonekynd, "Causality and Factor Investing: A Primer" (2025) and
  "A Protocol for Causal Factor Investing."  
  * They introduce the "factor mirage," a model that looks statistically valid but is
    causally misspecified, plus a seven-step protocol. The protocol starts with
    causal discovery that combines algorithms such as PC or LiNGAM with domain
    knowledge. Treat it as a ready-made checklist for your validator.  
* López de Prado, Lipton & Zoonekynd, "The Case for Causal Factor Investing" (2024)  
  * It argues that choosing the correct specification requires knowledge of the
    causal graph behind the data-generating process.  
* Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns."  
* Hou, Xue & Zhang (2020), "Replicating Anomalies."  
* Jensen, Kelly & Pedersen (2023), "Is There a Replication Crisis in Finance?"  
  * This is the replication literature your outputs will be judged against.

**3\. Extracting causal structure from text**

* Yang, Han & Poon (2022), "A Survey on Extraction of Causal Relations from Natural
  Language Text." This is the pre-LLM baseline.  
* FinCausal shared tasks (FNP workshop, 2020–2023). These are finance-specific
  cause/effect span detection datasets.  
* Kıcıman et al. (2023), "Causal Reasoning and Large Language Models: Opening a New
  Frontier." Also Jin et al. on Corr2Cause/CLadder, and Zečević et al., "Causal
  Parrots." Read the last one for the failure modes.  
* "Zero-shot Causal Graph Extrapolation from Text via LLMs" (arXiv 2312.14670). It
  builds causal graphs from iterated pairwise LLM queries over text.  
* LACR (arXiv 2402.15301). It uses RAG over aggregated literature, has the LLM label
  associational relations, and applies self-consistency to reduce uncertainty in
  graph recovery.  
* **ReCast (2025)** is the most directly relevant benchmark. It contains 292
  expert-annotated causal graphs from peer-reviewed economics and public-policy
  articles and scores extraction with normalized Structural Hamming Distance plus
  per-node/per-edge LLM judging.

**4\. Paper-to-code and LLM quant research agents**

* Paper2Code / PaperCoder (Seo et al., ICLR 2026). It runs three stages, planning,
  analysis and generation, each with specialized agents. Generated repos needed only
  minor modifications to run, averaging 0.48% of code lines. Their
  planning→analysis→coding split is worth copying.  
* PaperBench (OpenAI, 2025). Its rubric-based replication grading is a good template
  for your evaluation.  
* RD-Agent(Q) (Microsoft, NeurIPS 2025). A Research stage forms hypotheses from
  domain priors, and a Development stage uses a code agent to implement them and run
  real-market backtests. It's open source and the closest finance analogue.  
* FactorEngine. It turns unstructured financial reports into executable Python
  factors through a two-stage reflect-and-validate workflow that outputs structured
  JSON and pseudo-code.  
* Chen & Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing." It provides
  code and data for 300+ published signals, which makes it your best ground truth for
  the code-generation half.

## **How I'd structure the system**

The core design decision is to **not go directly from DAG to code.** A bare DAG is
too thin to generate a model from. Build a typed intermediate representation (a
"Theory Spec") that the DAG is one view of, and make every stage read and write that
object.

PDF ─► 1\. Parse ─► 2\. Claim extraction ─► 3\. Theory Spec (IR) ─► 4\. Causal
validation  │                        │  ▼                        ▼  
                                  5\. Code generation ◄────── identification result
                                  │  ▼  
                                  6\. Execute & verify against paper's tables ──►
                                  feedback to 2/3/5

**1\. Parse.** Use PDF→structured text with equations and tables kept intact (Nougat,
Marker or GROBID). Tables matter because they become your verification targets later.

**2\. Claim extraction (LLM, multi-pass).** Pull out four distinct things:

* *Variables*, each with an operational definition (e.g. "B/M \= book equity from
  Compustat at fiscal year-end t−1 / market cap in Dec t−1").  
* *Causal claims*, each with sign, mechanism text, lag and functional form if stated.  
* *Empirical specification*: regressions, sorts, controls, fixed effects, sample.  
* *Testable implications* the authors state.

Every item must carry a **provenance pointer** (page and span). This is your main
defense against hallucinated edges.

**3\. Theory Spec (the IR).** A Pydantic/JSON schema along these lines:

* `nodes`: id, canonical name, observed or latent, data recipe, unit, time index  
* `edges`: source, target, sign, lag, functional-form hint, confidence, provenance,
  and `stated` vs `inferred` status  
* `assumptions`: exogeneity, no unobserved confounding, and so on  
* `estimand`: e.g. ATE of characteristic X on next-month return  
* `empirical_spec`: what the paper actually ran

Canonicalize variable names against a small finance ontology (value, size, momentum,
accruals, leverage, and so on). This lets graphs from different papers merge into a
literature-level graph later.

**4\. Causal validation.** This is the step that makes the system more than a code
generator. Using DoWhy, pgmpy or y0:

* enforce acyclicity, and handle feedback by time-indexing (X\_t → Y\_{t+1})  
* run identification: is the paper's estimand identifiable from its own DAG?  
* **compare the theory DAG to the empirical spec.** Flag controls that are colliders
  or mediators and confounders that were omitted. This is precisely López de Prado's
  "factor mirage" diagnostic, automated.  
* list implied conditional independencies, which are testable against data

**5\. Code generation (PaperCoder-style plan → analyze → code).** Generate three
separate artifacts from the IR:

* **A simulator.** An SCM that produces synthetic data with known effects, used to
  check that the paper's estimator recovers the truth under the stated graph. This
  needs no proprietary data, so it always runs.  
* **A replication module.** Data construction plus the paper's actual estimator
  (Fama-MacBeth, portfolio sorts, panel regressions via
  `linearmodels`/`statsmodels`).  
* **A causal estimator.** The DoWhy-identified estimate, for comparison with the
  paper's.

Template the boilerplate (CRSP/Compustat merges, portfolio sorts, Newey-West) as a
tested library. The LLM should compose those functions, not rewrite them each time.

**6\. Execute and verify.** Run the code in a sandbox and compare outputs with the
paper's reported tables within tolerances. Route mismatches back to the right stage:
a wrong data recipe goes to 2, a missing confounder to 3, a bug to 5\.

## **Things that will bite you**

* **Theory papers ≠ empirical papers.** Equilibrium models (simultaneous equations,
  fixed points) don't map cleanly onto a DAG. Route those to a "structural equations
  → solver" path instead of forcing a graph.  
* **Most finance papers never state a DAG.** Much of the graph is implicit. Keep
  `stated` and `inferred` edges separate and never let the code silently depend on an
  inferred edge.  
* **Look-ahead bias in data recipes.** Reporting lags and fiscal-year alignment are
  where generated code goes wrong most often. Encode availability timing explicitly
  in the IR.  
* **Evaluate the two halves separately.** Use ReCast-style SHD and edge-level
  precision/recall for extraction. Use Chen-Zimmermann signal replication
  (correlation with their published signal, matching t-stats) for code.

A good first milestone is to restrict scope to cross-sectional anomaly papers. You
get hundreds of ground-truth implementations from Chen & Zimmermann, the DAGs are
small, and the "factor mirage" check produces something novel right away. If you'd
like, I can draft the Theory Spec schema as a Pydantic model next.
