# Goal
- The Theory Store turns a corpus of quantitative economics and finance papers into
  testable, auditable hypotheses
  - Every claim traces to a page span in a paper
  - Every test is counted in a ledger, so backtests are deflated honestly
- The goal is not to remove humans from the loop
  - Automate the repetitive stages: parse, extract, normalize, map to data,
    bookkeeping
  - Humans choose papers, approve names and DAGs, and pass the final gate
- Target: automate 80-90% of the effort
  - Speed up = manual minutes / tool minutes, measured per stage
  - Human gates bound the speed up: 5x at 80%, 10x at 90%

- Non-goals
  - Do not discover causality from data alone
  - Do not treat agreement between papers as causal truth
  - Do not replace the researcher
  - Do not discover causality from data alone
  - Do not auto-generate trading strategies
  - Do not treat agreement between papers as causal truth (Publication bias exists)

## Applications

- The applications of the Theory Store are:
  - Literature meta-analysis
    - Find which papers agree with each other or disagree, by looking at the causal
      structures
  - Replication of papers
    - Create financial models descripted by papers / books
  - Ask to a corpus of papers questions like "what drives inflation expectations?"
  - Find gap: list variable pairs that no paper tests, giving the researchers a map
    of open questions
  - Economic prior for ML: use the reduced DAG to pick features, set lags, and
    regularize
    - A model that follows a causal story overfits less

# Workflow

## Build Paper Bank
- On-board papers / books / ...
  - Human guides the search
- Parse papers from PDF into markdown
- Filter based on:
  - Keywords (e.g., microeconomic, macroeconomic)
  - Regions (e.g., APAC, EMEA, US)
  - Citations
  - Quality of the researchers / institution
  - Prefer identification design over author prestige and citations
    - Citations lag and favor known authors
    - Label each paper by evidence: causal design vs other
- Merge the per-paper Theory Specs into one literature-level graph
- Human gate: approve the paper list

## Extract Causal Claims
- Extract causal claims
  - (Cause, effect, sign)
  - Context
  - Annotate with reference for human analysis
- Store each claim as one record
  - (Cause, effect, sign, lag, region, period, identification design)
  - Status: `stated` or `inferred`
  - Provenance: page and span
- Never let later stages depend silently on an `inferred` claim

## Normalize Variables
- Normalize variable names
  - E.g., monetary tightening, policy rate hike, increase in federal fund rate
  - Use specialized LLM (trained on finance literature)
  - Review by human
  - Create operators (increase, delay, ...)
- Map each variable to a canonical id in a small finance ontology
- Human gate: approve the canonical names and operators

## Create DAG
- Research questions
  - How to handle equilibrium?
    - Route equilibrium papers to a structural equations and solver path
    - Do not force them into a DAG
  - How to handle delays?
    - Delay operator = edge lag
- Create DAG with:
  - Nodes: variables
  - Edges: relationships
  - Enforce DAG-ness (e.g., remove feedback loops)
    - Prefer unrolling in time (`X_t -> Y_{t+1}`) over removing edges
    - Removing an edge is arbitrary and loses real feedback
- Handle conflicts between papers
  - Papers disagree on sign, region, and period
  - Turn a sign conflict into a conditional or regime edge
  - Do not drop the conflicting claim
- Human gate: approve the DAG

## Map Nodes to Data
- Map nodes to timeseries to data store (e.g., Atlas, ...)
  - Assess quality of each timeseries
    - E.g., coverage, time span, gaps
- Use point-in-time vintages
  - Macro series are revised
  - Use the value known on each date, to avoid look-ahead bias
- Encode the availability timing of each series

## Merge DAGs
- "Merge" (i.e., vote, union, weighted average) DAGs from different papers
  - This is like ensemble learning from the body of literature

## Reduce DAG
- DAG "reduction"
  - Not all the edges are meaningful
    - Test for causality / conditional independence / association / ...
  - Not all the nodes are predictive or have high-quality data
  - Propose missing nodes and edges from LLM or human knowledge
  - "Usual" causal flow
- Use the discovery split only (see "Hypothesis Ledger and Data Splits")
- Log every test and every proposed edge in the ledger
- Human gate: approve the reduced DAG

## Optimize Against Proxy
- Optimize against a proxy
  - E.g., use model to predict supply and demand, instead of prices
  - Regularize to control complexity vs predictive power
- Log each proxy choice in the ledger

## Formalize Hypotheses
- Formalize set of hypothesis to test
  - Create a Bayesian prior from the DAG
  - Fit
  - Estimate uncertainty
- Build the prior from the evidence per edge
  - Number of supporting papers
  - Quality of the identification design
  - Agreement on the sign
- Log each fit in the ledger

## Validation
// This is the official flow ADIA uses to validate a proposal
- Create a proposal

- Vote on the proposal
// TODO(gp): Check who votes and on which criteria

- Run backtesting
  - Use 
  - Deflate / adjust by tested hypothesis
    - Take the number of tested hypotheses from the ledger
  - Use the validation split

- Final gate
  - Run OOS
  - The final OOS split stays untouched until this gate

# Cross-Cutting Rules

## Theory Spec
- One typed object per paper, with the DAG as one view of it
  - `nodes`: id, canonical name, observed or latent, data recipe, unit, time index
  - `edges`: source, target, sign, lag, confidence, provenance, `stated` or `inferred`
  - `assumptions`, `estimand`, `empirical_spec`
- Every stage reads and writes this object
- Stage mapping to the pipeline of `paper.tex`
  - Build Paper Bank: Stage 1, Parse
  - Extract Causal Claims: Stage 2, Claim extraction
  - Normalize Variables and Create DAG: Stage 3, Theory Spec
  - Reduce DAG: Stage 4, Causal validation
  - Map Nodes to Data, Optimize Against Proxy, Formalize Hypotheses: Stage 5, Code generation
  - Validation: Stage 6, Execute and verify
// TODO(gp): Check the stage mapping

## Hypothesis Ledger and Data Splits
- Problem: DAG reduction tests edges, then backtesting reuses the same data
  - This is double dipping
  - "Deflate by tested hypothesis" needs a count
- Ledger: one record per test
  - Edge tests in DAG reduction
  - Edges proposed by the LLM or by a human
  - Proxy choices
  - Prior fits
- Three data splits
  - Discovery: DAG reduction and proxy choice
  - Validation: backtesting
  - Final OOS: touched once, at the final gate
- Deflate the backtest statistics by the size of the ledger

## Human Review Gates and Metrics
- Gates
  - After paper list, normalization, DAG, reduction, and before the proposal
  - At each gate the human sees a compact artifact with provenance
  - The human can accept, edit, or reject
- Log every human decision
- Metrics
  - Minutes per stage, tool vs manual baseline
  - Acceptance rate per stage
  - Edge-level precision and recall against the gold graph of the pilot
