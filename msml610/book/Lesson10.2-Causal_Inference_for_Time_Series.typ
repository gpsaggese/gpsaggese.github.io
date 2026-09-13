// f69c795f 2026-06-23
// Import AIMA style formatting and macros
#import "../../helpers_root/dev_scripts_helpers/typst/aima_style.typ": aima-style, algorithm, chapter, glossary, wrap-content

// Document metadata
#set document(
  title: "Causal Inference for Time Series",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules)
#show: aima-style

#chapter(10, "Causal Inference for Time Series")

== Introduction and Core Concepts

=== Time Series vs. Cross-Sectional Causality

// Slide: Temporal Causal Structures

Causal inference in time series data differs fundamentally from cross-sectional
approaches. The distinguishing feature is #strong[temporal structure]: in time
series, we observe the same units repeatedly over time, creating opportunities
and challenges absent in cross-sectional designs.

#strong[Univariate time series causal inference] compares a unit to *itself*
over time. For example, economists might compare a city's crime rate *before*
and *after* a new policing policy. The unit serves as its own control, but this
creates a critical complication: observations are *not* independent. Each
observation depends on the previous one through autocorrelation, violating the
independence assumption of standard statistical tests.

By contrast, #strong[cross-sectional causal inference] compares different units
at a single point in time---for instance, comparing treatment outcomes between
patients in one cohort. Cross-sectional designs assume units are (conditionally)
exchangeable, which is straightforward but requires many comparison units.

Time series designs exploit what we might call the #strong[arrow of time]:
causes must precede effects. This principle eliminates backward arrows from
causal graphs a priori, pruning half the possible causal directions. However,
time series designs face distinct challenges. Observations are correlated across
time, treatments and confounders may drift or trend, and experiments are often
impossible. A canonical example illustrates the difficulty: did the Federal
Reserve rate cut in 2020 cause the stock market rebound? We cannot rerun 2020
with a different policy, so we must rely on temporal structure and domain
knowledge to reason counterfactually.

=== Temporal Causal Structures: Basic Setup

// Slide: Temporal Causal Graphs

In time series causal models, we typically observe sequences of variables over
time: a treatment process $T_t$ (continuous or binary), an outcome $Y_t$, and
covariates $X_t$. The fundamental principle is that *effects cannot precede
causes*. This means treatment at time $t$ can affect outcomes at times $t$,
$t+1$, $t+2$, and beyond, but never at times before $t$. Mathematically, $T_t$
can cause $Y_t$, $Y_{t+1}$, etc., but never $Y_{t-1}$.

This constraint is immensely powerful because it eliminates a large class of
incorrect causal directions without any data analysis. We can represent this
temporal structure in a #strong[temporal causal graph]. @fig:temporal-causal
illustrates a simple first-order graph showing how variables at consecutive time
points relate. The restriction that $T_t$ affects $Y_t$ but not prior outcomes
is encoded by the arrow structure.

// rendered_images:begin
// ```graphviz
// digraph TemporalCausal {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.6;
//     ranksep=0.6;
//     node [shape=ellipse, style="filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
//     T_tm1 [label=<T<SUB>t-1</SUB>>, fillcolor="#A6E7F4"];
//     T_t   [label=<T<SUB>t</SUB>>,     fillcolor="#A6E7F4"];
//     T_tp1 [label=<T<SUB>t+1</SUB>>, fillcolor="#A6E7F4"];
//     Y_tm1 [label=<Y<SUB>t-1</SUB>>, fillcolor="#A6C8F4"];
//     Y_t   [label=<Y<SUB>t</SUB>>,     fillcolor="#A6C8F4"];
//     Y_tp1 [label=<Y<SUB>t+1</SUB>>, fillcolor="#A6C8F4"];
//     T_tm1 -> T_t -> T_tp1;
//     Y_tm1 -> Y_t -> Y_tp1;
//     T_tm1 -> Y_t;
//     T_t   -> Y_tp1;
//     Y_tm1 -> T_t [style=dashed];
// }
// ```
// label=fig:temporal-causal
// caption=First-order temporal causal graph showing treatment and outcome
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson10.2-Causal_Inference_for_Time_Series.typ.figs/Lesson10.2-Causal_Inference_for_Time_Series.1.png"),
  caption: [First-order temporal causal graph showing treatment and outcome],
) <fig:temporal-causal>
// render_images:end

=== Temporal Causal Structures: Key Quantities

// Slide: Causal Effects in Time Series

The central causal quantity in time series is the effect of a *sustained
intervention*. Suppose we force the treatment sequence from some time t_start
onward to a fixed level. The #strong[sustained intervention causal effect] is
defined as:

The causal effect of a sustained intervention is the difference in expected
outcomes between forcing treatment on versus off from time t_start onward.

This measures the average outcome at time $t$ when we force the treatment to one
level versus zero from $t_0$ onward. The intuition is that we *force* the
treatment sequence, contrasting it with what would have happened without
intervention.

#strong[Dynamic treatment] occurs when treatment at time $t$ depends on the
unit's history: covariates $X_{1:t}$, past outcomes $Y_{1:t-1}$, and past
treatments $T_{1:t-1}$. This is common in medicine, where dosing depends on
patient response so far.

Two special cases are particularly important. *Contemporaneous effects* occur
when $T_t \to Y_t$ instantly. *Lagged effects* occur when for some lag $k > 0$.
In monetary policy, for example, a rate cut this quarter affects GDP over
several subsequent quarters (a lagged effect). Simultaneously, the Fed adjusts
rates based on past GDP, creating feedback loops that complicate causal
inference.

== Challenges Specific to Time Series

=== Autocorrelation and Standard Errors

// Slide: Autocorrelation

#strong[Autocorrelation] means that $Y_t$ is correlated with its own past
values: $Y_{t-1}$, $Y_{t-2}$, etc. This has profound consequences for
statistical inference. When we compute standard errors assuming independent
observations, our estimates are *too small*. Autocorrelated data carries less
independent information than the number of observations suggests. Significance
tests reject the null hypothesis far more often than their nominal rate, leading
to spurious conclusions.

Consider a simple example: regressing daily stock returns on a dummy variable
for "news day." Stock returns exhibit strong autocorrelation. If we ignore this
and apply standard OLS, we compute incorrect standard errors, likely concluding
that news has a statistically significant effect when it does not.

The solution is to use #strong[Newey-West or HAC] (heteroskedasticity- and
autocorrelation-consistent) standard errors. These estimators account for
correlation at multiple lags and asymptotically produce correct inference. When
autocorrelation is present, always use HAC standard errors, cluster standard
errors, or bootstrap methods rather than naive OLS inference.

=== Non-Stationarity and Trends

// Slide: Spurious Regression

A #strong[stationary process] has constant mean, variance, and autocovariance
over time. Many real-world economic and social time series are *not* stationary;
GDP, prices, and populations grow, creating trends that persist.

Non-stationarity leads to #strong[spurious regression], a classic pitfall. Two
entirely unrelated series that happen to trend upward will appear highly
correlated even though one does not cause the other. The example of pirates and
global temperature is famous: the number of pirates has declined over two
centuries while global temperature has increased, creating a strong negative
correlation---yet clearly one does not cause the other.

Several solutions exist. We can #strong[difference the series], working with the
first difference instead of levels. We can #strong[detrend] by removing a fitted
trend before analysis. If two series share a common stochastic trend, we can use
#strong[cointegration methods] that model their long-run relationship while
controlling for the shared trend.

The key is to test for stationarity using unit-root tests (ADF, KPSS) and
transform non-stationary series before causal inference.

=== Feedback Loops and Simultaneity

// Slide: Feedback Loops

In time series, cause and effect frequently influence each other, creating
*feedback loops*. This causes #strong[simultaneity bias] in regression: when
$Y_t$ and $T_t$ are jointly determined by unobserved shocks, OLS estimates are
inconsistent, and coefficients cannot be interpreted as causal effects.

A clear example is monetary policy and inflation. The central bank cuts rates to
stimulate the economy, which increases inflation. Policymakers then raise rates
to combat inflation, which dampens growth. The policy ($T_t$) affects inflation
($Y_t$), but inflation simultaneously influences future policy. This reciprocal
relationship invalidates naive causal inference, as illustrated in
@fig:feedback-loop.

Another example is advertising and sales. Companies increase advertising budgets
when sales rise (or to recover from low sales). Sales rise partly due to
advertising and partly due to external demand shocks. Advertising in turn
responds to sales, creating bidirectional causality that OLS cannot untangle.

// rendered_images:begin
// ```graphviz
// digraph Feedback {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.8;
//     node [shape=ellipse, style="filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
//     T_t   [label=<T<SUB>t</SUB> (Policy)>,    fillcolor="#A6E7F4"];
//     Y_t   [label=<Y<SUB>t</SUB> (Outcome)>,   fillcolor="#A6C8F4"];
//     T_tp1 [label=<T<SUB>t+1</SUB>>,         fillcolor="#A6E7F4"];
//     Y_tp1 [label=<Y<SUB>t+1</SUB>>,         fillcolor="#A6C8F4"];
//     T_t -> Y_t;
//     Y_t -> T_tp1;
//     T_tp1 -> Y_tp1;
//     T_t -> T_tp1;
//     Y_t -> Y_tp1;
// }
// ```
// label=fig:feedback-loop
// caption=Feedback loop structure where policy affects outcome contemporaneously,
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson10.2-Causal_Inference_for_Time_Series.typ.figs/Lesson10.2-Causal_Inference_for_Time_Series.2.png"),
  caption: [Feedback loop structure where policy affects outcome contemporaneously,],
) <fig:feedback-loop>
// render_images:end

=== Structural Vector Autoregressions

// Slide: VARs and Structural Shocks

#strong[Structural Vector Autoregressions] (VARs) offer one approach to feedback
loops. Rather than treating policy and outcome asymmetrically (as in
observational causal inference), VAR models treat all variables symmetrically,
estimating how they influence each other over time.

A #strong[VAR(p) model] with $k$ variables is formulated as:

```
Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + ε_t
```

Here $Y_t$ is a vector of $k$ time series, the $A_i$ are coefficient matrices
encoding dependence on lagged values, and $epsilon_t$ are structural shocks
(uncorrelated across equations). The VAR estimates how shocks propagate through
the system, revealing the causal effects of unexpected changes.

The key challenge is #strong[identification]: how do we determine causality
direction? The solution is to impose *structural assumptions*. For instance, in
monetary policy, we might assume that the Fed reacts to inflation *within* the
same quarter, but inflation responds to policy *in the next quarter*. This
timing restriction identifies the causal direction. Once identified, we can
simulate interventions (permanent shocks) and trace their effects through the
system.

=== Instrumental Variables for Time Series

// Slide: Exogenous Variation

#strong[Instrumental Variables] (IV) break feedback loops by providing exogenous
variation in treatment. The key idea is finding a variable $Z$ that affects
treatment $T$ but not outcome $Y$ directly (the #strong[exclusion restriction]).
We use $Z$ to predict $T$, obtaining , then measure how correlates with $Y$.

This approach works because $Z$ provides variation in treatment independent of
confounds, allowing us to isolate the causal effect. In education, scholars
study the effect of schooling on earnings using distance to the nearest college
as an instrument: distance affects educational access but not earnings directly
(barring unusual circumstances).

The limitation is that IV estimates effects only on #strong[compliers]---units
whose treatment changes with the instrument. Compliers may differ from the full
population, limiting generalizability. Additionally, finding a credible
instrument is difficult in practice.

=== Natural Experiments and Exogenous Shocks

// Slide: Natural Experiments

#strong[Natural experiments] exploit unexpected external events that randomly
assign treatment to some units but not others, mimicking a randomized trial in
observational data. The advantage is that randomization eliminates confounding
without requiring an instrument.

A famous example is the Vietnam War draft lottery, which randomly assigned
military service. Researchers compared drafted and non-drafted veterans on
lifetime earnings, estimating the causal effect of military service. Because the
lottery was random, we can attribute earnings differences to military service,
not selection effects.

Natural experiments are credible when the shock is truly exogenous (outside the
system's control) and affects treatment assignment but not outcomes directly.
However, they are rare, cannot be replicated, and typically estimate effects in
specific contexts that may not generalize.

=== Time-Varying Confounders

// Slide: Unobserved Confounders

#strong[Time-varying confounders] make causal identification harder. When a
confounder $U_t$ affects both treatment $T_t$ and outcome $Y_t$ and changes over
time, it contaminates causal estimates.

Consider evaluating a diet on weight loss. Season (winter vs. summer), stress,
and motivation all change over time and drive both adherence and weight change.
These are hard to measure but crucial confounders. Naive before-after
comparisons will be biased by these time-varying factors.

The solution depends on whether confounders are time-invariant or time-varying.
If a confounder is *time-invariant* (e.g., genetics), simple before-after
comparisons can difference it out. This is the core principle behind
fixed-effects models and Difference-in-Differences. If the confounder is
*time-varying* (e.g., mood, weather), differencing does not eliminate bias. More
sophisticated methods are needed: modeling the confounder explicitly, proxying
for it with measured variables, or finding exogenous variation unrelated to the
confounder.

=== Other Practical Challenges

// Slide: Additional Complications

Several additional complications arise in time series causal inference:

#strong[Anticipation effects] occur when units change behavior *before*
treatment if they expect it. When a tax cut is announced but not yet
implemented, consumers may delay purchases to time them with the cut, biasing
estimates.

#strong[Spillovers and SUTVA violations] happen when treatment on one unit
affects another. If advertising in one region spills to neighboring regions, we
cannot isolate the effect to the treated region.

#strong[Structural breaks] occur when the data-generating process changes
mid-sample. The 2008 financial crisis and COVID-19 pandemic are extreme
examples, but smaller breaks (policy regime changes, technological shifts) are
common. Models fit before the break may not hold afterward.

#strong[Missing data and irregular sampling] arise because standard time series
tools assume equispaced observations. Sensor dropouts, holidays, weekends, and
other irregularities complicate analysis. Imputation or specialized methods
(e.g., state-space models) are needed.

=== When Temporal Structure Helps

// Slide: Strengths of Temporal Data

Temporal structure is a resource for causal inference when exploited correctly.
Time orders causes before effects, eliminating many causal directions a priori.
This reduces the number of hypotheses and focuses analysis on plausible
mechanisms.

The unit serves as its own control, so *time-invariant confounders* are
automatically controlled. A person's genes, geography, and institutional
features remain constant over time, so within-person comparisons difference them
out. This is much more powerful than requiring a separate control unit.

Natural experiments are easier to spot in time series. Policy changes, shocks,
and discontinuities provide quasi-experimental variation. Interrupted Time
Series, Difference-in-Differences, and synthetic control all exploit this
variation. When studying the effect of a sugary-drink ban, for example, we
compare the same city before and after. Genes, climate, and culture are held
fixed, making the comparison much cleaner than comparing different cities.

=== When Temporal Structure Misleads

// Slide: Pitfalls of Temporal Data

Conversely, temporal structure can mislead. Trends fake causation: two series
trending upward look causally linked even if unrelated (spurious regression).
Seasonality hides effects when a seasonal bump masks or amplifies a policy
effect.

Lagged effects confuse attribution. Today's outcome may reflect yesterday's
treatment, so wrong lag specification produces wrong effect estimates. Feedback
inflates correlations: if $Y_t$ drives and $T_t$ drives $Y_t$, regressing $Y_t$
on $T_t$ picks up both the treatment effect and the reverse effect, confounding
inference.

Regime changes invalidate models. A model fit before COVID may not generalize
after. The key takeaway is that temporal data is informative, but blindly
applying cross-sectional tools to it creates more problems than it solves.

== Granger Causality

=== Intuition and Definition

// Slide: Granger Causality Concept

#strong[Granger Causality], named after Nobel laureate Clive Granger, answers a
specific question: "Does knowing the past of $X$ help me predict $Y$ beyond what
$Y$'s own past already tells me?" The intuition is that causes must come before
effects, and effects should carry information about their causes.

It is important to understand what Granger causality is NOT. It is not causation
in Pearl's do-calculus sense (i.e., causal effect in the interventional sense).
Rather, it is a predictive notion based on temporal precedence. Granger
causality answers "does $X$ help predict $Y$?" not "does changing $X$ change
$Y$?" An important consequence is that both $X \to Y$ and $Y \to X$ can hold
simultaneously in Granger causality, which is impossible in deterministic
causality but common in economic feedback systems.

A famous example illustrates the limitation: ice cream sales do not
Granger-cause drowning. Instead, *temperature* Granger-causes both ice cream
sales and drowning. They are correlated because of a common cause, not because
one causes the other. Granger causality captures this dependence but misses the
underlying mechanism.

=== Formal Definition

// Slide: Granger Causality Test

Consider two stationary time series $X_t$ and $Y_t$. We compare two predictors
of $Y_t$:

1. *Restricted model*: uses only past $Y$
```
Y_t = α + Σ_{i=1}^p φ_i Y_{t-i} + ε_t
```

2. *Unrestricted model*: adds past $X$
```
Y_t = α + Σ_{i=1}^p φ_i Y_{t-i} + Σ_{j=1}^p β_j X_{t-j} + η_t
```

#strong[Definition]: $X$ *Granger-causes* $Y$ if the unrestricted model has
*strictly smaller* forecast error variance than the restricted model.
Equivalently, an F-test on $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$
rejects, meaning at least one past value of $X$ predicts current $Y$
significantly.

The test is straightforward: fit both models by OLS, compute sum-of-squared
residuals, and test whether including $X$ reduces the residual sum of squares
significantly. If yes, we conclude $X$ Granger-causes $Y$.

=== Key Assumptions

// Slide: Stationarity and Lag Selection

Granger causality relies on several assumptions, violations of which lead to
spurious conclusions. #strong[Stationarity] is essential: both series must have
time-invariant statistical properties. Non-stationary series can produce
spurious Granger causality. The remedy is to pre-test with unit-root tests
(Augmented Dickey-Fuller, KPSS) and difference series if needed.

#strong[Correct lag order $p$] is critical. Too few lags miss the true effect,
while too many lead to overfitting and loss of power. Lag order is typically
chosen via information criteria (AIC, BIC) or cross-validation. Under- or
over-specification biases tests.

#strong[Linearity] is assumed in the basic formulation. Non-linear Granger
causality extensions exist (e.g., transfer entropy and machine-learning based
methods) but are beyond standard practice.

#strong[No omitted common causes]---this is the "big one." If an unobserved
variable $Z$ drives both $X$ and $Y$ with different lags, $X$ will appear to
Granger-cause $Y$ even though it does not. This cannot be tested; we must rely
on domain knowledge.

#strong[Correct sampling frequency] is needed. If the true effect operates
faster than the sampling interval, Granger causality may fail to detect it or
even reverse the direction.

=== Limitations

// Slide: When Granger Fails

#strong[Granger causality is not causation.] It tests *predictive*
relationships, not *interventional* ones. Granger-causal relationships can be
entirely spurious, driven by a common cause with no true causal link.

The barometer example is instructive: a barometer reading $X_t$ Granger-causes
storms because falling atmospheric pressure precedes storms. But manually
raising the barometer does *not* cause a storm. The atmosphere is the true
cause; the barometer merely signals it.

#strong[Aggregation problems] arise when sampling frequency is wrong. Monthly
data may miss daily Granger effects. If the process evolves faster than the
sampling rate, we observe spurious bidirectional causality (each series
Granger-causing the other) due to time aggregation.

#strong[Contemporaneous effects] cannot be handled in the basic form. If $X_t$
affects $Y_t$ instantaneously (same period), lag-based tests miss it. Extensions
exist (VAR formulations), but standard Granger causality tests skip
contemporaneous relationships.

#strong[Long time series] are required for reliable inference. Typically we need
$T \gg p$---for 4 lags, at least 100 observations. Short samples produce
unreliable inference and low power.

=== Practical Example: Advertising and Sales

// Slide: Granger Causality Example

Consider the question: "Does advertising spend Granger-cause sales?" Suppose we
have weekly data on ad spend $X_t$ and sales $Y_t$. We fit both the restricted
model (sales on past sales only) and unrestricted model (sales on past sales and
past ads). If the unrestricted model has significantly lower forecast error, we
conclude ad spend Granger-causes sales.

However, a critical caveat applies: this does *not* mean "if we spend more on
ads, sales will rise." Both variables could be driven by seasonal demand
patterns. A consumer goods company spends more on ads before holidays (which
also see naturally higher sales), creating an apparent Granger relationship. To
get an interventional answer---what happens if we *change* ad
spending---requires an experiment or a causal design (e.g., randomized
experiments, natural experiments).

Another widely-studied question is whether Twitter sentiment scores
Granger-cause stock returns. Research finds mixed results: sentiment does
Granger-cause short-horizon returns for some stocks, but effect sizes are small
and may reflect transaction-level feedback loops rather than true predictive
value.

== Interrupted Time Series (ITS)

=== Design and Motivation

// Slide: ITS Core Idea

#strong[Interrupted Time Series] (ITS) designs are used when a policy or
intervention is implemented at a single known date t_start across all (or most)
units. Examples include a new speed limit enforced nationwide from January 1st,
or a website redesign launched on a specific day.

The #strong[fundamental challenge] is the absence of a control group: the whole
population is treated. A naive before-after comparison might show change, but we
cannot distinguish the policy effect from concurrent trends or other events.

The #strong[solution] is to use the *pre-intervention trajectory* as a
counterfactual forecast. We fit a model to pre-intervention data, extrapolate it
forward under the assumption that the trend would have continued absent the
intervention, then compare the forecast (counterfactual) to what actually
happened. The gap is the estimated causal effect.

ITS has impressive real-world examples. Seatbelt laws reduced traffic fatalities
by approximately 15--20%. Netflix's password-sharing ban had mixed results:
growth in some regions but churn in others, illustrating that counterfactuals
are often uncertain. Apple's iOS privacy changes (requiring apps to request
tracking permission) saw opt-in rates plummet from 60--70% to 15--25%,
devastating ad targeting. Meta reported billions in revenue losses,
demonstrating the power of identifying when policy changes affect outcomes.

As illustrated in @fig:its-structure, the core ITS strategy is to extrapolate
the pre-period trend as a counterfactual, then compare it to the actual
post-period outcome. The gap represents the policy effect.

// rendered_images:begin
// ```graphviz
// digraph ITS {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.6;
//     ranksep=0.8;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     PrePeriod  [label=<Pre-period<BR/>(t &lt; t*)>,      fillcolor="#FFD1A6"];
//     Event      [label=<Intervention<BR/>at t = t*>,   fillcolor="#F4A6A6"];
//     PostPeriod [label=<Post-period<BR/>(t &gt;= t*)>,    fillcolor="#B2E2B2"];
//     Forecast   [label=<Counterfactual<BR/>forecast>,  fillcolor="#A6E7F4"];
//     Effect     [label=<Causal effect<BR/>= Actual - Forecast>, fillcolor="#A6C8F4"];
// 
//     PrePeriod -> Event -> PostPeriod;
//     PrePeriod -> Forecast [label="  fit model"];
//     Forecast -> Effect;
//     PostPeriod -> Effect;
// }
// ```
// label=fig:its-structure
// caption=Interrupted Time Series design: using pre-period trends as counterfactual
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson10.2-Causal_Inference_for_Time_Series.typ.figs/Lesson10.2-Causal_Inference_for_Time_Series.3.png"),
  caption: [Interrupted Time Series design: using pre-period trends as counterfactual],
) <fig:its-structure>
// render_images:end

=== Segmented Regression Formulation

// Slide: ITS Regression Model

The standard ITS model is #strong[segmented regression]:

`Y_t = β₀ + β₁ t + β₂ D_t + β₃ (t - t*) D_t + ε_t`

where $D_t$ is an indicator for the post-intervention period ($D_t = 1$ if
$t >= t^*$, and 0 otherwise).

The parameters have clear interpretations:
- $\beta_0$: pre-intervention intercept (baseline level)
- $\beta_1$: pre-intervention slope (trend before the intervention)
- $\beta_2$: *level change* at the intervention (immediate jump)
- $\beta_3$: *slope change* after the intervention (change in trend)

This formulation lets us test two key hypotheses:
1. Is there a jump at t_start ? Test $H_0: \beta_2 = 0$.
2. Does the trajectory change? Test $H_0: \beta_3 = 0$.

The estimated effect is $\beta_2$ (immediate level change) plus the cumulative
effect of $\beta_3$ over the post-period (slope change). A positive $\beta_2$
means the outcome jumped up at intervention; a positive $\beta_3$ means the
trend steepened.

=== Implementation Details

// Slide: ITS Best Practices

Several practical details matter for valid ITS:

*Autocorrelation* is almost always present in time series. Standard OLS standard
errors are too small, leading to overconfidence. Fit models using
autocorrelation-robust methods (Newey-West), ARIMA (AutoRegressive Integrated
Moving Average), or other time-series methods that account for serial
correlation.

*Robustness checks* strengthen ITS conclusions. Vary the pre-intervention window
(use 1, 2, 5 years of data) to see if results are stable. Run placebo tests by
pretending the intervention occurred at a fake date; if you find spurious
effects, your design is picking up something other than the true intervention.

Always check for confounding events near t_start : seasonality, holidays,
economic shocks. Visual diagnostics are crucial: plot the series with the fitted
model and the intervention date marked. Do the ITS results survive inspection?

=== Applications

// Slide: ITS Applications

ITS is widely used across domains:

- *Public health*: estimating the effect of soda taxes on sugar-sweetened
  beverage purchases.
- *Policy evaluation*: measuring the effect of congestion pricing in London
  (2003) on traffic volumes, or minimum wage increases on employment.
- *Tech and product*: quantifying the effect of a website redesign on conversion
  rate or a new recommendation algorithm on engagement.
- *Safety interventions*: did installing speed cameras reduce traffic accidents?

Each application requires careful attention to the timing of the intervention,
confounding events, and data quality.

=== Strengths and Weaknesses

// Slide: ITS Evaluation

#strong[Pros of ITS]:
- Works when *no control group exists*: policies are often applied universally
  (same rules for all units).
- Uses the unit as its own control, so *time-invariant confounders cancel*:
  genes, geography, and institutional features are held fixed.

#strong[Cons of ITS]:
- Relies on *extrapolation* of the pre-period trend, introducing model
  dependence.
- Vulnerable to *co-occurring events*: "was it the policy or the recession?"
- Autocorrelation inflates errors if ignored.
- Needs a long, stable pre-period, which is rare in practice.

#strong[Rule of thumb]: ITS is most credible when:
1. The intervention is sharply timed (clear before/after boundary).
2. No other major events occur near t_start .
3. The pre-period is long and stable.
4. The outcome has low noise relative to effect size.

== Difference-in-Differences (DiD)

=== Motivation and Setup

// Slide: DiD Core Concept
// TODO(ai_gp): Overlaps with 08.4

#strong[Motivating Example]
- Suppose a policy is rolled out in *some* units but not others
  - E.g., New Jersey raises its minimum wage in 1992; Pennsylvania does not
  - Two groups: treated (NJ) and control (PA); two periods: before and after

#strong[Definition]: Difference-in-Differences (DiD) estimates a causal effect by
comparing the change in outcomes over time between a group that received
treatment and a group that didn't
- Assume both groups would have followed the same trend in the absence of
  treatment

#strong[Procedure]
- Subtracts out time-invariant differences between groups
- Subtracts out common time trends across groups
- What's left is attributed to the policy

- #strong[Example]: Card & Krueger (1994) studied the NJ minimum wage
  - Standard theory predicted employment would fall
  - DiD estimate: essentially zero (a surprising result)

=== The Core Idea Graphically

// Slide: DiD Visual

The DiD estimator with two groups and two periods is shown in
@fig:did-structure shows the DiD estimator with two groups and two periods:

```
τ̂_{DiD} = (Ȳ^T_{post} - Ȳ^T_{pre}) - (Ȳ^C_{post} - Ȳ^C_{pre})
```

Here the treated group mean is compared against the control mean, with
subscripts denoting time periods (pre vs. post). Visually, we compute the change
in the treated group (pre to post) and subtract the change in the control group.
The residual is the estimated causal effect, with the visual arrangement
highlighting how both between-group differences and common time trends are
removed.

// rendered_images:begin
// ```graphviz
// digraph DiD {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     TreatPre   [label=<Treated<BR/>Pre>,   fillcolor="#FFD1A6"];
//     TreatPost  [label=<Treated<BR/>Post>,  fillcolor="#F4A6A6"];
//     CtrlPre    [label=<Control<BR/>Pre>,   fillcolor="#B2E2B2"];
//     CtrlPost   [label=<Control<BR/>Post>,  fillcolor="#A0D6D1"];
//     Effect     [label=<DiD =<BR/>(T_post - T_pre)<BR/>- (C_post - C_pre)>, fillcolor="#A6C8F4"];
// 
//     TreatPre -> TreatPost [label="  change in treated"];
//     CtrlPre -> CtrlPost  [label="  change in control"];
//     TreatPost -> Effect;
//     CtrlPost -> Effect;
// }
// ```
// label=fig:did-structure
// caption=Difference-in-Differences design: the causal effect is the difference
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson10.2-Causal_Inference_for_Time_Series.typ.figs/Lesson10.2-Causal_Inference_for_Time_Series.4.png"),
  caption: [Difference-in-Differences design: the causal effect is the difference],
) <fig:did-structure>
// render_images:end

=== Regression Formulation

// Slide: DiD Regression

The equivalent regression formulation with unit $i$ and time $t$ is:

```
Y_{it} = α + β·Treat_i + γ·Post_t + τ·(Treat_i × Post_t) + ε_{it}
```

The parameters are:
- $alpha$: baseline outcome in control units in pre-period
- $beta$: time-invariant difference between groups
- $gamma$: common time trend affecting both groups
- $tau$: #strong[the DiD causal effect] (coefficient on the interaction term)

A more flexible #strong[fixed-effects] generalization is:

```
Y_{it} = α_i + λ_t + τ D_{it} + ε_{it}
```

Here $alpha_i$ are unit fixed effects (absorb all time-invariant unit traits:
location, management, size), $lambda_t$ are time fixed effects (absorb all
common shocks: recessions, industry trends), and $D_{i,t}$ indicates whether
unit $i$ is treated at time $t$. The fixed-effects specification is more
flexible because it allows heterogeneous levels and trends across units, not
just between treated and control groups.

=== The Parallel Trends Assumption

// Slide: Identifying Assumption

The identifying assumption for DiD is #strong[parallel trends]: in the absence
of treatment, treated and control units would have followed the *same trend*
over time:

```
E[Y^{(0)}_{it} - Y^{(0)}_{i,t-1} | Treat_i = 1] = E[Y^{(0)}_{it} - Y^{(0)}_{i,t-1} | Treat_i = 0]
```

where $Y^{(0)}$ denotes the untreated potential outcome. The intuition is: levels can
differ (NJ may always have higher wages than PA), but *changes* should match.
Both regions should move together absent the policy.

The assumption is #strong[untestable]---it is a counterfactual statement about
what would have happened without treatment. We cannot directly observe it.
However, we can gain credibility by checking *pre-treatment* trends. If treated
and control units move in parallel before treatment, we gain confidence (though
no proof) that parallel trends hold post-treatment.

=== Checking Parallel Trends

// Slide: Parallel Trends Testing

Several methods help validate parallel trends:

#strong[Visual check]: plot the two group means over time. Do they move in
lockstep before treatment? Divergence suggests violation.

#strong[Placebo / event-study regression] is more formal:

We estimate effects at all lags and leads relative to treatment. For
pre-treatment periods, coefficients should be near zero (no pre-treatment
effect). For post-treatment periods, we capture dynamic treatment effects.

#strong[Red flags] include:
- Significant pre-trends with opposite sign from the estimated effect
- Divergence starting *before* the policy (suggesting selection or anticipation,
  not treatment effect)

If pre-trends diverge, #strong[do not just add linear trend terms and hope]. Use
a different method (synthetic control, matching on trends) instead.

=== Estimation Details

// Slide: DiD Standard Errors and Inference

Several practical issues arise in DiD estimation:

#strong[Standard errors must be clustered] at the *unit level*. Bertrand, Duflo,
and Mullainathan (2004) showed that ignoring clustering overrejects the null by
40+ percent when autocorrelation is present. Most software has clustering
options; always use them.

#strong[OLS with fixed effects] gives the same point estimate as simple group
means but allows covariates. Covariates help when they affect outcomes but are
balanced across groups. They reduce variance without introducing bias.

#strong[Balanced vs. unbalanced panels]: missing data or attrition biases
estimates if non-random. Always report attrition rates and check whether balance
is maintained across groups and time.

#strong[Common mistakes]:
- Forgetting that treatment and time indicators enter as main effects, not just
  the interaction.
- Ignoring clustering when standard errors are autocorrelated.
- Using too short a pre-period, making it impossible to check parallel trends.

=== Robustness Checks

// Slide: DiD Robustness

Strengthen DiD conclusions through robustness checks:

#strong[Placebo treatments in time]: pretend treatment happened earlier. Should
get no effect if the design is sound.

#strong[Placebo outcomes]: apply DiD to unrelated outcomes that *should not*
change. If minimum wage affects employment but not commodity prices, commodity
prices should be unaffected in a DiD estimate.

#strong[Alternative control groups]: use different comparison units. Results
should be stable across reasonable choices. If the effect flips when you switch
controls, the design is fragile.

#strong[Sensitivity to functional form]: check whether results change with
logarithmic vs. level outcomes, or linear vs. spline trend adjustments.

#strong[Falsification tests]: subset the data in ways that *should not* yield
effects. If they do, your design picks up something other than treatment.

The Card & Krueger minimum wage study exemplifies good robustness practice: they
tried different fast-food chains and different controls, and the null effect
held up, lending credibility.

=== Staggered Adoption and Modern Estimators

// Slide: Multiple Periods and Heterogeneous Effects

In reality, units are often treated at #strong[different times] (staggered
rollout). States adopt laws across different years; companies roll out products
sequentially to different regions. Traditional #strong[two-way fixed effects]
(TWFE) models all these with:

However, recent econometric research (Goodman-Bacon 2021, de Chaisemartin &
D'Haultfœuille 2020) revealed a critical flaw: with heterogeneous effects across
units and time, TWFE becomes a *weighted average where some weights are
negative*. Already-treated units serve as controls for later-treated units,
which can produce wrong-sign bias.

Modern staggered DiD estimators fix this:

- #strong[Callaway & Sant'Anna (2021)]: estimate group-time average treatment
  effects, ATT(g, t), for each cohort g at time t, then aggregate with clean
  weights.
- #strong[Sun & Abraham (2021)]: interaction-weighted event-study estimator
  avoiding negative weights.
- #strong[de Chaisemartin & D'Haultfœuille (2020)]: robust estimators under
  effect heterogeneity.
- #strong[Borusyak, Jaravel, & Spiess (2024)]: imputation estimator using
  never-treated or not-yet-treated units as controls.

The key takeaway: for staggered adoption, *do not* default to TWFE. Use a modern
estimator and report the full event-study plot showing effects by lag relative
to treatment.

=== Summary: Strengths and Limits

// Slide: DiD Evaluation

#strong[Pros of DiD]:
- Removes time-invariant confounders with unit fixed effects.
- Removes common shocks with time fixed effects.
- Uses both treated and control units, more credible than ITS alone.
- Flexible: extends to multiple periods, staggered adoption, covariates.

#strong[Cons of DiD]:
- Parallel trends assumption is untestable---can only check pre-trends.
- Time-varying confounders still bias estimates.
- Staggered adoption with heterogeneous effects breaks naive TWFE.
- Standard errors require clustering---easy to get wrong.

#strong[Rule of thumb]: DiD is credible when:
1. Pre-trends are visually parallel.
2. No co-occurring shocks hit treated and controls differently.
3. Staggered timing is handled with modern estimators.

== Synthetic Control Methods

=== Motivation and Construction

// Slide: Synthetic Control Overview

#strong[Synthetic Control] methods handle a scenario where DiD falls short: when
*no single control unit* matches the treated unit. Studying California's 1988
tobacco control program (Prop. 99) illustrates the issue. California is large,
diverse, and economically unique. Comparing it to any single state (Texas,
Florida, or any other) is unconvincing because the state does not match
California on key pre-treatment characteristics.

The #strong[synthetic control idea] (Abadie & Gardeazabal 2003; Abadie, Diamond,
& Hainmueller 2010) constructs a *weighted average* of control units that best
matches the treated unit's pre-treatment characteristics. As shown in
@fig:synthetic-control-construction, this weighted combination becomes the
#strong[synthetic control]---a data-driven, transparent counterfactual. We
compare the treated unit to the synthetic control in the post-period. The gap is
the estimated effect. The method's strength is transparency: reviewers can see
*which* donors contribute and how much, making the counterfactual explicit.

// rendered_images:begin
// ```graphviz
// digraph SC {
//     rankdir=LR;
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Treated [label=<Treated unit<BR/>(e.g., California)>,  fillcolor="#F4A6A6"];
//     Donors  [label=<Donor pool<BR/>(other states)>,         fillcolor="#FFD1A6"];
//     Weights [label=<Optimal weights<BR/>w<SUB>1</SUB>, w<SUB>2</SUB>, ..., w_J>,fillcolor="#A6E7F4"];
//     Synth   [label=<Synthetic<BR/>counterfactual>,          fillcolor="#B2E2B2"];
//     Effect  [label=<Causal effect =<BR/>Treated - Synthetic>,fillcolor="#A6C8F4"];
// 
//     Donors -> Weights;
//     Treated -> Weights [label="  pre-treatment fit"];
//     Weights -> Synth;
//     Synth -> Effect;
//     Treated -> Effect;
// }
// ```
// label=fig:synthetic-control-construction
// caption=Synthetic control construction: the donor pool is weighted optimally
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson10.2-Causal_Inference_for_Time_Series.typ.figs/Lesson10.2-Causal_Inference_for_Time_Series.5.png"),
  caption: [Synthetic control construction: the donor pool is weighted optimally],
) <fig:synthetic-control-construction>
// render_images:end

=== Formal Setup

// Slide: Synthetic Control Formal Definition

Suppose we have J+1 units: unit 1 is treated at time t_start , units 2, ..., J+1
are untreated (donors). Let X_1 be a vector of pre-treatment predictors for the
treated unit, and X_0 is a matrix of the same predictors for donors (size k by J
matrix).

#strong[Weights] must satisfy: each weight is nonnegative and sums to one.

```
w_j ≥ 0, Σ_{j=2}^{J+1} w_j = 1
```

Non-negativity and unit-sum ensure weights are a *convex combination* of donors,
placing the synthetic control in the *convex hull* of the donor pool. This
prevents extrapolation---the synthetic control is never outside the range
spanned by donors.

We choose optimal weights to minimize pre-treatment mismatch:

```
w* = argmin_w (X_1 - X_0 w)' V (X_1 - X_0 w)
```

subject to $w_j >= 0$, $sum w_j = 1$.

The predictor matrix $V$ is diagonal, containing importance weights for each
predictor, chosen to minimize mean-squared prediction error on pre-treatment
outcomes. This is a nested optimization: the inner loop finds weights for a
given predictor matrix; the outer loop finds $V$.

The #strong[estimated causal effect] at time $t >= t^*$ is:

```
τ̂_t = Y_{1t} - Σ_{j=2}^{J+1} w_j* Y_{jt}
```

the gap between the treated unit's outcome and the weighted average of donor outcomes.

=== Inference via Placebos

// Slide: Inference Strategy

A critical challenge is that there is only *one* treated unit---standard
inference does not apply. The solution is #strong[placebo tests] by permutation:

1. Pretend each donor was the treated unit.
2. Compute the synthetic control and causal-effect trajectory for each.
3. Compare the *actual* treated unit's effect against this null distribution.

If the treated unit's effect is *larger in magnitude* than most placebo effects,
we have evidence of a real effect. Compute an empirical p-value from the
placebos.

A useful variant filters placebos to keep only those with good pre-treatment
fit. A donor with poor pre-fit produces noise, not a legitimate null. In the
California tobacco study, the post-period gap for California was larger than 38
of 39 placebo runs, yielding p approximately 0.026---strong evidence the policy
worked.

=== When Synthetic Control Works

// Slide: Synthetic Control Prerequisites

Synthetic control works well under specific conditions:

#strong[Long pre-treatment period]: many time points allow matching on
pre-treatment characteristics. Rule of thumb: at least 10+ observations.

#strong[Rich donor pool]: many untreated units with diverse characteristics
enable good pre-treatment fit via convex combinations.

#strong[Small number of treated units], usually one. Synthetic control is
designed for aggregated units: countries, states, cities, or firms.

#strong[Outcome with moderate noise], low enough to detect signal above
background variation.

Canonical successes include:
- Abadie & Gardeazabal (2003): effect of ETA terrorism on Basque GDP
- Abadie et al. (2010): California Prop. 99 tobacco control
- Abadie et al. (2015): German reunification's effect on West German GDP

=== When Synthetic Control Fails

// Slide: Synthetic Control Limitations

#strong[Poor pre-treatment fit] is a fatal flaw. If no convex combination of
donors matches the treated unit, the treated unit is an *outlier* in donor
space. Reported effects are unreliable---you are extrapolating in
high-dimensional space.

#strong[Treatment contaminates donors] (spillovers). If neighboring states are
also affected by a policy, they are not valid controls. This violates the
exclusion assumption.

#strong[Short pre-period or noisy outcome] makes fit unreliable or prone to
overfitting noise.

#strong[Confounding unobserved events] near t_start masquerade as treatment
effects. If California's economy experienced a shock around the tobacco law,
synthetic control attributes it to the law.

#strong[Interpolation bias] occurs when covariates are imbalanced: synthetic
control averages over donors who differ in important but unmeasured ways.

#strong[Diagnostic]: always plot the pre-treatment fit. Poor fit means do not
trust the post-period gap.

=== Comparison with Other Methods

// Slide: Method Comparison

Each causal inference method has distinct strengths. @tbl:causal-methods
summarizes the key properties of each approach.

#show table.cell.where(y: 0): set text(weight: "bold")

#figure(
  table(
    columns: (1.1fr, 1.2fr, 2fr, 2fr),
    inset: 6pt,
    align: left + horizon,

    stroke: none,

    table.hline(stroke: 1.2pt),
    [Method], [Control Group], [Strength], [Key Assumption],
    table.hline(stroke: 0.8pt),

    [Granger], [Own history], [Predictive screen],
    [Stationarity, no common causes],

    [ITS], [Pre-period of same unit],
    [Works without controls],
    [Stable pre-trend, no co-events],

    [DiD], [Untreated comparison],
    [Removes time-invariant confounders],
    [Parallel trends],

    [Synthetic Control],
    [Weighted donor pool],
    [Data-driven counterfactual],
    [Good pre-fit, no spillovers],

    table.hline(stroke: 1.2pt),
  ),
  caption: [Comparison of causal inference methods for time series data.],
) <tbl:causal-methods>

*Granger Causality*: uses the own history as control, offering fast predictive
screening but only testing predictions, not interventions.

*ITS*: works without controls when policies are universal, but relies on stable
pre-trends and extrapolation.

*DiD*: removes time-invariant confounders using unit fixed effects and common
shocks using time fixed effects. Requires parallel pre-trends.

*Synthetic Control*: provides data-driven counterfactuals using weighted donors,
transparent and avoids extrapolation (convex hull property), but inference is
permutation-based and requires long, clean pre-periods.

#strong[Pros of synthetic control]:
- Transparent weights---reviewers see which donors contribute.
- Avoids extrapolation (convex combination stays inside donor hull).
- Works when no single comparison unit is credible.

#strong[Cons of synthetic control]:
- Inference requires placebos---less standard than standard statistical tests.
- Sensitive to donor pool choice---discipline needed.
- Requires long, clean pre-period.

=== Putting It All Together

// Slide: Method Selection Guide

Choose the appropriate method based on your data and design:

#strong[Use Granger causality] when you want a fast predictive screen on two
long time series and no intervention/experiment is available.

#strong[Use ITS] when a single intervention has a sharp date and no comparison
units exist, with a long and stable pre-period.

#strong[Use DiD] when some units are treated and others are not, parallel
pre-trends are plausible, and you have panel data with units and time.

#strong[Use Synthetic Control] when one (or few) treated units have a rich donor
pool, and no single donor is credible but a weighted combination is.

The key takeaway is that *temporal structure is a resource if exploited with the
right tool, and a trap if ignored*. Each method has specific assumptions and
conditions under which it excels. Matching method to design and checking
assumptions carefully produces credible causal inference.
