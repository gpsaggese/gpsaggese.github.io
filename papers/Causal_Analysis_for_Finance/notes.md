The goal is to build a system that converts financial papers into theory / models,
by:

- Creating a causal DAG from the paper explaining the theory  
- Converting the “theory” into Python model  
- Optimizing the causal DAG (where the optimization criteria are specified later)

This idea sits where four bodies of work meet.

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
  * It reviews graph-based methods for confounding bias, sample selection bias,
    surrogate experiments, and transportability. It stresses that the
    identification criteria are algorithmic, so they can be automated. Published in
    *The Econometrics Journal* 28(1), 2025.

**2\. Causality in finance specifically (the motivation and the "spec")**

* López de Prado, *Causal Factor Investing* (Cambridge Elements, 2023, open access)  
  * He argues that virtually all factor-investing articles make associational claims:
    authors don't identify a causal graph, justify specifications by correlations,
    and propose no falsification experiments. That is exactly the gap your system
    fills.  
* López de Prado & Zoonekynd, "Correcting the Factor Mirage: A Research Protocol for
  Causal Factor Investing" (2024, SSRN 4697929, forthcoming in the *Journal of
  Portfolio Management*).  
  * It describes the "factor mirage," a model that looks statistically valid but is
    causally misspecified, and proposes a seven-step protocol: variable selection,
    causal discovery, causal adjustment, estimation of causal effects, portfolio
    construction, backtesting, and multiple testing adjustment. Step 2 builds a DAG
    from domain knowledge and discovery algorithms such as PC or LiNGAM. Treat the
    protocol as a ready-made checklist for your validator.  
* López de Prado & Zoonekynd, "Causality and Factor Investing: A Primer" (2025, SSRN
  5277078, ADIA Lab Research Paper Series No. 16).  
  * It circulated in May 2025 under the title "A Protocol for Causal Factor
    Investing." It is the same paper, not a second one.  
  * It shows how confounder bias and collider bias distort regression-based factor
    models. It applies the PC algorithm to the daily returns of the risk factors of
    85 Barra risk models, and lists 26 models where adding a collider changes the sign
    of the estimated coefficient. It also turns the protocol into a checklist of
    questions for due diligence.  
* López de Prado, Lipton & Zoonekynd, "The Case for Causal Factor Investing" (2024)  
  * It argues that choosing the correct specification requires knowledge of the
    causal graph behind the data-generating process.  
* Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns."  
* Hou, Xue & Zhang (2020), "Replicating Anomalies."  
* Jensen, Kelly & Pedersen (2023), "Is There a Replication Crisis in Finance?"  
  * This is the replication literature your outputs will be judged against. The
    three studies disagree: Harvey et al. argue that new factors need a t-statistic of
    at least 3.0, Hou et al. find that 65% of 452 anomalies fail the single-test
    hurdle (|t| = 1.96) once microcaps are mitigated, and Jensen et al. find that most
    factors replicate. None of them tests the causal specification.

**3\. Extracting causal structure from text**

* Yang, Han & Poon (2022), "A Survey on Extraction of Causal Relations from Natural
  Language Text." This is the pre-LLM baseline.  
* FinCausal shared tasks (Mariko et al. 2020, FNP workshop, with later editions each
  year). These are finance-specific tasks: classify causal sentences (Task 1) and
  extract cause/effect spans (Task 2). They label spans inside a segment and do not
  build a graph.  
* Kıcıman et al. (2023), "Causal Reasoning and Large Language Models: Opening a New
  Frontier." Also Jin et al. (ICLR 2024), "Can Large Language Models Infer Causation
  from Correlation?" (the Corr2Cause data set), and Zečević et al., "Causal Parrots."
  Read the last one for the failure modes.  
* Antonucci, Piqué & Zaffalon (2023), "Zero-shot Causal Graph Extrapolation from Text
  via LLMs" (arXiv 2312.14670). It builds causal graphs from iterated pairwise LLM
  queries over text, tested on biomedical abstracts.  
* Zhang et al. (2024), "Causal Graph Discovery with Retrieval-Augmented Generation
  based Large Language Models" (arXiv 2402.15301), which introduces the method LACR.
  It uses RAG over papers from Google Scholar and PubMed, has the LLM extract
  associations instead of causal claims, aggregates the decisions across documents by
  majority vote, and recovers the skeleton before the edge orientation. It is tested
  on three small medical graphs (ASIA, SACHS, CORONARY).  
* Garg & Fetzer (2025), "Causal Claims in Economics" (arXiv 2501.06873). It builds
  evidence-annotated claim graphs for 44,852 economics papers (1980 to 2023). Each
  edge is labeled by its evidentiary basis. It is a study of the literature and does
  not feed an identification or estimation step.  
* **ReCITE** (Saklad et al., arXiv 2505.18931, version 4, April 2026; named ReCAST in
  version 2) is the most directly relevant benchmark, but it is not specific to
  finance. It contains 292 causal graphs transcribed by annotators from the causal
  loop diagrams of open-access MDPI and PLOS articles. Only 5.5% of the articles fall
  under economics, econometrics, and finance. It scores extraction with an LLM judge
  (node and edge precision and recall) plus structural Hamming distance and its
  normalized version. The best model reaches an F1 score of 0.535.

**4\. Paper-to-code and LLM quant research agents**

* Paper2Code / PaperCoder (Seo et al., ICLR 2026). It runs three stages, planning,
  analysis and generation, each with specialized agents. Generated repos needed only
  minor modifications to run, averaging 0.81% of code lines in the latest version (v5;
  the 0.48% figure comes from an earlier version). It targets machine learning
  papers. Their planning→analysis→coding split is worth copying.  
* PaperBench (Starace et al., OpenAI, 2025). Agents replicate 20 ICML 2024 papers
  from scratch and are graded on rubrics with 8,316 tasks. Its rubric-based grading is
  a good template for your evaluation.  
* RD-Agent(Q) (Microsoft, NeurIPS 2025). A Research stage forms hypotheses from
  domain priors, and a Development stage uses a code agent to implement them and run
  real-market backtests. It's open source and the closest finance analogue.  
* Lin et al. (2026), "FactorEngine: A Program-level Knowledge-Infused Factor Mining
  Framework for Quantitative Investment" (arXiv 2603.16365). It turns unstructured
  financial reports into executable Python factors through a two-stage
  reflect-and-validate workflow that outputs structured JSON and LaTeX-style
  pseudo-code. It is evaluated on the CSI 300 and CSI 500 universes. The paper does
  not mention causal graphs.  
* Chen & Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing." It provides
  code and data for 319 characteristics, which makes it your best ground truth for
  the code-generation half. For the 161 characteristics that were clearly significant
  in the original papers, 98% of the reproduced long-short portfolios have t-statistics
  above 1.96.

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
* **Evaluate the two halves separately.** Use ReCITE-style SHD and edge-level
  precision/recall for extraction. ReCITE has few finance papers, so plan a finance
  gold set. Use Chen-Zimmermann signal replication (correlation with their published
  signal, matching t-stats) for code.

A good first milestone is to restrict scope to cross-sectional anomaly papers. You
get hundreds of ground-truth implementations from Chen & Zimmermann, the DAGs are
small, and the "factor mirage" check produces something novel right away. If you'd
like, I can draft the Theory Spec schema as a Pydantic model next.
