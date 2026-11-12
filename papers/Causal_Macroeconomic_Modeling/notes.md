Extension of the paper-to-model framework in `papers/Causal_Analysis_for_Finance` to macroeconomic variables. The text below was moved from the notes of that paper.

Macro data changes the framework in four places: the shape of the graph, the time
model, identification, and the data layer. Here's how to handle each.

## **1\. Extend the graph: macro variables play three different roles**

Macro nodes aren't just "more variables." They sit at a different level (indexed by
time only, not firm × time), and a paper can use them in three distinct ways. The
Theory Spec should say which one each edge means.

* **Common cause (confounder).** Rates, credit spreads and the business cycle drive
  both firm characteristics and returns. For example, a rate rise lowers valuations,
  which raises B/M, and also lowers realized returns directly. If the DAG omits this,
  the value effect gets misattributed. This connects straight to the "factor mirage"
  check: macro confounders are often the omitted variables.  
* **Effect modifier (conditional theory).** Many papers claim X → R holds mainly in
  certain regimes, such as momentum crashing after market rebounds or value paying
  off in recoveries. A plain DAG can't express this. Add a `modulated_by` attribute
  on the edge, or model the interaction as its own node.  
* **Direct cause / priced risk.** Macro shocks as factors themselves, as in Chen,
  Roll & Ross (1986) or consumption- and investment-based asset pricing. Here the
  macro node points straight to returns, with exposure (beta) as the firm-level link.

In practice this becomes a **two-level (hierarchical) SCM**. A macro layer evolves
over time, and a cross-sectional layer of firms × time hangs beneath it. Edges can
run from macro to firm, but almost never from one firm to macro.

## **2\. Make time explicit**

Cross-sectional papers can get away with a static graph. Macro forces a
**time-unrolled DAG**:

* **Lags on every edge.** Separate contemporaneous edges (X\_t → Y\_t) from lagged
  ones (X\_{t−1} → Y\_t). Feedback loops, such as policy reacting to markets and
  markets reacting to policy, become acyclic once time-indexed.  
* **Mixed frequency.** GDP is quarterly, CPI monthly, returns daily or monthly. Store
  a native frequency per node and an aggregation rule (end-of-period, average, MIDAS
  weights) per edge.  
* **Stationarity transforms.** Record each series' transform (level, log-difference,
  gap) in the IR. FRED-MD's transformation codes (McCracken & Ng, 2016\) are a good
  standard. Feeding levels of trending series into a regression is the classic
  spurious-regression trap.

## **3\. Point-in-time data: the biggest practical risk**

Macro data is **revised**, and released with a lag. GDP for Q1 is first published
weeks later and then revised for years. Using today's revised numbers in a backtest
is look-ahead bias, and generated code will do it by default.

So every macro node in the IR needs:

* `release_lag`, the time from the period's end to first publication  
* `vintage_policy`, either "first-release," "real-time vintage," or "final-revised,"
  depending on what the paper's theory requires

Sources that support this are ALFRED (St. Louis Fed vintages), the Philadelphia Fed
Real-Time Data Set, and FRED-MD/FRED-QD for large monthly and quarterly panels. Your
data-layer adapter should take `(series, as_of_date)` and return only what was
knowable then.

## **4\. Identification is harder, so lean on identified shocks**

There's only one macro history, it is short (a few hundred monthly observations), and
nothing is randomized. Two consequences for the validation stage:

* **Use exogenous shock series as instrument nodes.** The literature has built many:
  high-frequency monetary policy surprises (Gertler & Karadi, 2015; Bauer & Swanson),
  narrative shocks (Romer & Romer), and oil supply shocks (Kilian). Add an
  `instrument` node type to the IR. When a paper claims "monetary policy affects X,"
  the validator should check whether the estimate uses an identified shock or just
  regresses on the policy rate, which is endogenous to the economy.  
* **Use time-series causal discovery for the implied-independence tests.** The best
  fits are PCMCI (Runge et al., 2019, the `tigramite` package), VAR-LiNGAM (Hyvärinen
  et al., 2010, the `lingam` package) and Granger tests as a weak baseline. Treat the
  output as a consistency check against the paper's stated graph, not as ground
  truth. With that little data, discovery algorithms are noisy.

Also add **structural break / regime tests** (Bai-Perron, or Hamilton
regime-switching). A macro edge that holds before 2008 and vanishes after is a
finding, not a bug.

## **5\. Code generation: new estimator templates**

Add these to the tested function library the LLM composes from:

| Paper's claim | Template | | ----- | ----- | | Dynamic effect of a macro shock |
Local projections (Jordà, 2005), SVAR (`statsmodels`) | | Macro-conditional factor
premium | Conditional Fama-MacBeth with macro interactions (Ferson & Harvey) | |
Regime-dependent effect | Markov-switching regression | | Macro risk priced in
cross-section | Two-pass beta estimation on macro shock series | | Mixed-frequency
prediction | MIDAS regression |

The **simulator** also changes. Generate the macro layer first as a VAR or
regime-switching process, then the firm layer conditional on it. This lets you test
whether the paper's estimator recovers known effects when macro confounding is
present, which is often where cross-sectional estimators break.

## **6\. Bonus: macro is the glue for a literature-level graph**

Once macro nodes have canonical names ("real policy rate," "term spread," "credit
spread," "industrial production growth"), graphs extracted from macro papers and from
cross-sectional finance papers connect through them. A monetary-transmission paper
(policy → credit spread) and an asset-pricing paper (credit spread → distress-stock
returns) chain into a pathway that neither paper tested alone. The system can then
generate that combined model and test it.

## **Updated IR fields (summary)**

* **Node:** `level` (macro | firm | industry), `frequency`, `transform`,
  `release_lag`, `vintage_policy`, `source_series_id`  
* **Edge:** `lag`, `contemporaneous` flag, `modulated_by`, `aggregation_rule`,
  `regime_scope`  
* **New node type:** `instrument` / `shock`, with the source study for its
  identification  
* **Validation:** time-series discovery check, endogeneity-of-macro-regressor check,
  structural-break check

A sensible order is to add the macro confounder role first, a handful of FRED-MD
series with point-in-time handling, applied to the anomaly papers you're already
starting with. That alone will show which published factors survive macro adjustment,
and it reuses everything else in the pipeline. I can sketch the point-in-time data
adapter or the extended Pydantic schema if useful.

