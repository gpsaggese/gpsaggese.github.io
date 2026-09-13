// Import AIMA style formatting and macros
#import "../../helpers_root/dev_scripts_helpers/typst/aima_style.typ": aima-style, chapter, algorithm, glossary, wrap-content

// Document metadata
#set document(
  title: "Using Bayesian Networks",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules)
#show: aima-style

#chapter(6, "Using Bayesian Networks")

== Semantics of Bayesian Networks

// Slide: Bayesian Networks Semantics

A #strong[Bayesian Network] encodes probabilistic relationships among variables
and can be understood through two equivalent semantic interpretations.
Understanding these interpretations is fundamental to working with Bayesian
networks effectively.

The #strong[joint distribution view] treats the network as an encoding of the
complete joint probability distribution over all variables. This distribution
can be computed as a product of local conditional probabilities:

$P(X_1, ..., X_n) = product_(i=1)^(n) P(X_i | "Parents"(X_i))$

This perspective is particularly useful for constructing models and
understanding the overall behavior of a system. It captures how variables
collectively define a complete probabilistic model.

The #strong[conditional independence view] interprets the network structure as
encoding conditional independencies between variables. In this view, a variable
is conditionally independent of its non-descendants given its parents. This
perspective is invaluable for inference and reasoning, as it guides which
information is relevant for predicting a variable's state.

=== Chain Rule for Joint Distributions

// Slide: Chain Rule for a Joint Distribution

The #strong[chain rule] provides a fundamental theorem: any joint distribution
can be expressed as a product of conditional probabilities for any ordering of
random variables. This flexibility makes the chain rule a powerful tool for
deriving Bayesian networks.

#strong[Proof outline:] For any variables $x_(1), ..., x_n$, we can express one
variable conditionally:

$Pr(x_1, ..., x_{n-1}, x_n) = Pr(x_n | x_{n-1}, ..., x_1) Pr(x_{n-1}, ..., x_1)$

Applying this formula recursively until reaching unconditional probabilities
yields:

$Pr(x_1, x_2, ..., x_(n-2), x_(n-1), x_n) = product_(i=1)^n Pr(x_i | x_(i-1), ..., x_1)$

=== Evaluating a Bayesian Network

// Slide: Evaluate a Bayesian Network

To compute probabilities using a Bayesian network:

1. Sort the nodes in topological order (multiple valid orderings may exist given
  the directed acyclic graph structure).
2. Apply the chain rule using the topological ordering:
  $Pr(X_1, ..., X_n) = product_(i=1)^n Pr(X_i | X_(i-1), ..., X_1)$
3. Leverage conditional independence: since each node is conditionally
  independent of all predecessors given its parents,
  $Pr(X_i | X_{i-1}, ..., X_1) = Pr(X_i | text("Parents")(X_i))$
4. Express the joint probability using the Conditional Probability Tables
  (CPTs): $Pr(X_1, ..., X_n) = product_(i=1)^n Pr(X_i | "Parents"(X_i))$

This systematic approach transforms the full joint distribution into a factored
form tractable by the network structure.

=== Example: Pearl's Burglary-Earthquake Network

// Slide: Evaluate a Bayesian Network: Example

Consider Pearl's classic alarm network example. Given the network structure
showing burglary and earthquake events triggering an alarm, which causes John
and Mary to call:

To compute
$Pr("JohnCalls", "MaryCalls", "Alarm", not "Burglary", not "Earthquake")$:

The computation follows from the network structure:

```
Pr(JohnCalls, MaryCalls, Alarm, ¬Burglary, ¬Earthquake)
  = Pr(JohnCalls|Alarm) ·
    Pr(MaryCalls|Alarm) ·
    Pr(Alarm|¬Burglary ∧ ¬Earthquake) ·
    Pr(¬Burglary) · Pr(¬Earthquake)
```

Each factor comes directly from the network's conditional probability tables,
allowing efficient computation without enumerating the full joint distribution.

== Constructing a Bayesian Network

// Slide: Constructing a Bayesian Network

Building an effective Bayesian network requires systematic domain engineering
and probabilistic specification. The construction process involves several key
steps:

1. #strong[Gather domain knowledge]: List all relevant random variables
  necessary to describe the system. Identify key variables and their potential
  interactions based on expert understanding or data exploration.

2. #strong[Order nodes by causality]: Arrange nodes according to cause-effect
  dependencies. This ordering ensures the resulting network is minimal in
  connectivity.

3. #strong[Specify parent relationships]: For each node, select the minimum set
  of parents $"Parents"(X_i)$ that determine its probabilistic behavior. Add
  edges to represent dependencies while avoiding redundant connections.

4. #strong[Estimate conditional probabilities]: Provide
  $Pr(X_i | "Parents"(X_i))$ for each node using data, expert opinion, or
  statistical estimation techniques.

5. #strong[Validate the model]: Have domain experts review the structure. Verify
  the network is a Directed Acyclic Graph (DAG). Test predictions against known
  outcomes and compare with actual data to ensure fidelity.

=== Properties of Bayesian Networks

// Slide: Bayesian Networks: Properties

Bayesian networks possess several desirable properties:

#strong[Completeness]: The network encodes all information in the joint
probability distribution. No relevant probabilistic relationships are implicit
or missing.

#strong[Consistency]: The representation contains no redundant probability
values. Domain experts cannot create a Bayesian network violating probability
axioms, as the formal structure enforces consistency automatically.

#strong[Compactness]: Bayesian networks are locally structured and sparse. Each
component typically interacts directly with a limited number of other
components, yielding linear rather than exponential growth in complexity.
However, in fully connected systems where every variable influences all others,
the network retains the complexity of the full joint distribution.

These properties make Bayesian networks practical for reasoning under
uncertainty in complex domains.

=== Node Ordering and Network Complexity

// Slide: Ordering of Nodes

The complexity of a Bayesian network depends critically on the node ordering
chosen during construction. Different orderings can produce networks with vastly
different edge counts and conditional probability table sizes.

#wrap-content(
  [
    // rendered_images:begin
    // ```graphviz
    // digraph BayesianNetwork {
    //     splines=true;
    //     nodesep=0.8;
    //     ranksep=0.8;
    //
    //     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
    //
    //     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
    //     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
    //     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
    //     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
    //     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
    //
    //     Burglary -> Alarm;
    //     Earthquake -> Alarm;
    //     Alarm -> JohnCalls;
    //     Alarm -> MaryCalls;
    // }
    // ```
    // rendered_images:end
    // render_images:begin
    #figure(
      image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.1.png", width: 100%),
    )
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  *Causal ordering* (Burglary, Earthquake, Alarm, JohnCalls, MaryCalls)—minimal
  edges following causal direction:
]

*Poor ordering 1*—requires backward edges, increasing complexity:

// rendered_images:begin
// ```graphviz
// digraph BayesianNetwork {
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
//     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
//     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
//     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
//     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
// 
//     MaryCalls -> Alarm;
//     JohnCalls -> Alarm;
//     Alarm -> Burglary;
//     Alarm -> Earthquake;
//     Burglary -> Alarm;
//     Earthquake -> Alarm;
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.2.png"),
)
// render_images:end

*Poor ordering 2*—dense interconnections requiring large CPTs:

// rendered_images:begin
// ```graphviz
// digraph BayesianNetwork {
//     splines=true;
//     nodesep=0.8;
//     ranksep=0.8;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     Burglary   [label="Burglary", fillcolor="#A6C8F4"];
//     Alarm      [label="Alarm", fillcolor="#FFD1A6"];
//     JohnCalls   [label="JohnCalls", fillcolor="#B2E2B2"];
//     MaryCalls   [label="MaryCalls", fillcolor="#B2E2B2"];
//     Earthquake [label="Earthquake", fillcolor="#A6C8F4"];
// 
//     MaryCalls -> Earthquake;
//     MaryCalls -> Burglary;
//     MaryCalls -> JohnCalls;
//     JohnCalls -> Earthquake;
//     Earthquake -> Burglary;
//     Earthquake -> Alarm;
//     Burglary -> Alarm;
//     Alarm -> MaryCalls;
//     Alarm -> JohnCalls;
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.3.png"),
)
// render_images:end

The graph is "minimal" in terms of connectivity when all edges represent causal
relationships. Using domain knowledge to order nodes causally leads to simpler,
more interpretable networks.

=== Causal vs. Diagnostic Models

// Slide: Causal vs Diagnostic Models

#strong[Causal models] express relationships flowing from causes to symptoms
(e.g., $"Burglary" -> "Alarm"$). These models are simpler, involve fewer and
more robust dependencies, and are easier to estimate from data or expert
judgment. The conditional probabilities capture mechanisms: "If burglary occurs,
what is the alarm probability?"

#wrap-content(
  [
    // rendered_images:begin
    // ```graphviz
    // digraph CausalModel {
    //     splines=true;
    //     nodesep=2.0;
    //     ranksep=1.5;
    //     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
    //
    //     Causes [label="Causes", fillcolor="#B2E2B2"];
    //     Symptoms [label="Symptoms", fillcolor="#F4A6A6"];
    //
    //     Causes -> Symptoms;
    //     Symptoms -> Causes;
    // }
    // ```
    // rendered_images:end
    // render_images:begin
    #figure(
      image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.4.png", width: 100%),
    )
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  #strong[Diagnostic models] work in the reverse direction, from symptoms to
  causes (e.g., $"MaryCalls" -> "Alarm"$ or $"Alarm" -> "Burglary"$). These models
  are tenuous and unstable, difficult to estimate reliably. However, they align
  with practical reasoning: given that Mary called, what caused it? To use
  diagnostic models, we apply Bayes' rule to invert the probabilities:
  $Pr("Cause"|"Symptom") = (Pr("Symptom"|"Cause")Pr("Cause"))/(Pr("Symptom"))$
]

Effective Bayesian networks typically use causal structure for specification,
then invert via Bayes' rule for diagnostic reasoning.

== Markov Blanket of a Node

// Slide: Markov Blanket of a Node

The #strong[Markov blanket] of a node $X$ is the set of variables that
completely determine its conditional distribution, rendering all other variables
irrelevant. It consists of three components:

1. #strong[Parents]: Nodes that directly influence $X$.
2. #strong[Children]: Nodes directly influenced by $X$.
3. #strong[Spouses]: Nodes that are parents of $X$'s children (co-parents).

Once the Markov blanket is known, $X$ is conditionally independent of all other
nodes in the network. This means inference about $X$ requires only information
about its parents, children, and spouses—enabling efficient and localized
computation.

The Markov blanket is fundamental to many inference algorithms (particularly
Gibbs sampling) because it identifies the minimal sufficient information for
reasoning about a variable.

=== Conditional Independence and Markov Blankets

// Slide: Conditional Independence on Markov Blanket

In a Bayesian network, each variable exhibits strong independence properties:

- A variable is conditionally independent of its predecessors given its parents.
- A variable is conditionally independent of *all other nodes* given its Markov
  blanket (parents, children, and spouses).

This means the Markov blanket contains all nodes necessary to predict the state
of $X_(i)$, making the rest of the network irrelevant for that variable. This
principle enables efficient inference: instead of considering all variables,
focus computation on the local Markov blanket.

=== Influence Through Explaining Away

// Slide: How Can a Node Be Influenced by Its Children?

Descendants can influence ancestors indirectly through a process called
#strong[explaining away]. When evidence about a descendant becomes available, it
can change beliefs about an ancestor through competing explanations.

#wrap-content(
  [
    // rendered_images:begin
    // ```graphviz
    // digraph BayesianFlow {
    //     splines=true;
    //     nodesep=1.0;
    //     ranksep=0.75;
    //     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
    //
    //     Rain [fillcolor="#A6C8F4", label="Rain"];
    //     WetGrass [fillcolor="#B2E2B2", label="WetGrass"];
    //     Sprinkler [fillcolor="#A6E7F4", label="Sprinkler"];
    //
    //     Rain -> WetGrass;
    //     Sprinkler -> WetGrass;
    // }
    // ```
    // rendered_images:end
    // render_images:begin
    #figure(
      image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.5.png", width: 100%),
    )
    // render_images:end
  ],
  align: right,
  column-gutter: 1em,
  columns: (1fr, 20%),
)[
  #strong[Example—Garden World:] Suppose you observe wet grass ($"WetGrass"$).
  Without additional information, this evidence increases the probability of
  either the sprinkler being on or rain falling. However, if you later observe
  that the sprinkler was on, this explains the wet grass, and the probability of
  rain decreases—the observed cause "explains away" the alternative. Evidence from
  a descendant thus flows backward, updating ancestors through dependent paths in
  the network.
]

This bidirectional information flow, despite the network's directed structure,
is a subtle but powerful aspect of Bayesian reasoning.

=== Medical Example: Heart Disease

// Slide: Markov Blanket: Medical Example

Consider risk factors and outcomes for heart disease:

// rendered_images:begin
// ```graphviz
// digraph HeartDiseaseGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     H [label="HeartDisease", fillcolor="#F4A6A6"];
//     A [label="Age", fillcolor="#A6C8F4"];
//     G [label="Genetics", fillcolor="#A6C8F4"];
//     D [label="Diet", fillcolor="#A6C8F4"];
//     E [label="Exercise", fillcolor="#A6C8F4"];
//     BP [label="BloodPressure", fillcolor="#B2E2B2"];
//     C [label="Cholesterol", fillcolor="#B2E2B2"];
// 
//     A -> H;
//     G -> H;
//     D -> H;
//     E -> H;
// 
//     H -> BP;
//     H -> C;
// 
//     A -> BP;
//     A -> C;
//     G -> BP;
//     G -> C;
//     D -> BP;
//     D -> C;
//     E -> BP;
//     E -> C;
// 
//     {rank=same; A; G; D; E}
//     {rank=same; BP; C}
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.6.png"),
)
// render_images:end

- #strong[Target node]: HeartDisease
- #strong[Parent nodes] (direct causes): Age, Genetics, Diet, Exercise
- #strong[Children nodes] (outcomes): BloodPressure, Cholesterol
- #strong[Spouse nodes]: The risk factors that also influence the outcomes
  directly

The Markov blanket of $"HeartDisease"$ includes all parents, children, and
spouses. Knowing only these nodes allows prediction of heart disease without
requiring information about variables outside this blanket. This structure
reflects the medical reality that certain risk factors and measurable outcomes
contain all relevant information for assessing disease presence.

=== Economic Example: House Prices

// Slide: Markov Blanket: Economic Example

For house prices in a region:

// rendered_images:begin
// ```graphviz
// digraph HousePriceGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     HP [label="HousePrices", fillcolor="#F4A6A6"];
//     E [label="EconomicGrowth", fillcolor="#A6C8F4"];
//     IR [label="InterestRate", fillcolor="#A6C8F4"];
//     UE [label="UnemploymentRate", fillcolor="#A6C8F4"];
//     DI [label="DisposableIncome", fillcolor="#B2E2B2"];
//     D [label="HousingDemand", fillcolor="#B2E2B2"];
// 
//     E -> HP;
//     IR -> HP;
//     UE -> HP;
// 
//     HP -> DI;
//     HP -> D;
// 
//     {rank=same; E; IR; UE}
//     {rank=same; DI; D}
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.7.png"),
)
// render_images:end

- #strong[Target node]: HousePrices
- #strong[Parent nodes] (direct influences): EconomicGrowth, InterestRate,
  UnemploymentRate
- #strong[Children nodes]: DisposableIncome (house price affects purchasing
  power), HousingDemand (price inversely affects demand)

The Markov blanket contains all nodes needed to estimate house prices, excluding
grandparents and other distant relatives in the network. This localized
structure shows that effective real-estate pricing depends on immediate economic
factors and observable demand signals, not on the full chain of global economic
determinants.

=== Financial Example: Stock Prices

// Slide: Markov Blanket: Finance Example

For an individual company's stock price:

// rendered_images:begin
// ```graphviz
// digraph StockPriceGraph {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.7];
// 
//     SP [label="Stock Price", fillcolor="#F4A6A6"];
//     EPS [label="Earnings Per Share", fillcolor="#A6C8F4"];
//     IP [label="Industry Performance", fillcolor="#A6C8F4"];
//     MS [label="Market Sentiment", fillcolor="#A6C8F4"];
//     TV [label="Trading Volume", fillcolor="#B2E2B2"];
//     RC [label="Regulatory Changes", fillcolor="#C6A6F4"];
//     GE [label="Global Economic Conditions", fillcolor="#C6A6F4"];
// 
//     EPS -> SP;
//     IP -> SP;
//     MS -> SP;
// 
//     SP -> TV;
// 
//     RC -> EPS;
//     RC -> IP;
//     GE -> EPS;
//     GE -> MS;
// 
//     {rank=same; EPS; IP; MS}
//     {rank=same; RC; GE}
//     {rank=same; TV}
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.8.png"),
)
// render_images:end

- #strong[Target node]: StockPrice
- #strong[Parent nodes]: IndustryPerformance, EarningsPerShare, MarketSentiment
- #strong[Children node]: TradingVolume (price changes trigger trading activity)
- #strong[Grandparents] (outside Markov blanket): RegulatoryChanges,
  GlobalEconomicConditions (influence parents but not trading volume directly)

Notably, grandparents do not belong to the Markov blanket. This means estimating
stock price requires only immediate factors (industry performance, earnings,
sentiment) and their observable effects (volume), not the global factors that
influence these immediate factors. This principle guides practical stock
analysis: focus on a company's fundamentals and market perception, not on
distant macroeconomic causes.

== Specifying Conditional Probabilities

// Slide: Specifying a Conditional Probability Table

A critical practical challenge is specifying #strong[Conditional Probability
  Tables (CPTs)]. A naive CPT for a node with $k$ parents requires $O(2^k)$
values, growing exponentially with parent count. Fortunately, many real-world
relationships exhibit special structure that compresses representation.

When nodes have many parents or relationships are complex, exploiting structure
becomes essential. Several approaches reduce the required number of probability
values:

1. #strong[Deterministic relationships]: Some nodes are functions of parents
  without uncertainty.
2. #strong[Noisy logical relationships]: Probabilistic versions of logical rules
  (e.g., noisy-OR).
3. #strong[Context-specific independence]: A variable's independence from some
  parents depends on values of others.

These techniques allow practical Bayesian networks with many variables and
dependencies.

=== Deterministic Nodes

// Slide: Deterministic Nodes

Some nodes in a Bayesian network are #strong[deterministic], meaning their
values are completely determined by parents without any uncertainty. Examples
include:

- *Logical relationships*:
  $"IsNorthAmerican" = "IsCanadian" or "IsUS" or "IsMexican"$. The child is true
  if any parent is true.
- *Numerical relationships*: $"BestPrice" = min("Price"_1, "Price"_2, ...)$. The
  child is the minimum of parent prices.

Deterministic nodes reduce CPT size to zero (or one entry) and simplify
inference, making networks more efficient when such relationships exist.

=== Noisy Logical Relationships

// Slide: Noisy Logic Relationships

#strong[Noisy logical relationships] are probabilistic versions of logical
rules, useful when causes combine with uncertainty. Consider the classic
example: Fever might result from Cold, Flu, or Malaria, but not
deterministically.

#strong[Noisy-OR model:] Under assumptions that all causes are listed (with
optional "leak" node for miscellaneous causes), probabilities of parents are
independent, we compute:

$Pr(text("Fever") | text("parents")) = 1 - Pr(text("No Fever") | text("Cold")) times Pr(text("No Fever") | text("Flu")) times Pr(text("No Fever") | text("Malaria"))$

This formulation captures the intuition: fever is absent only if *all* causes
fail to produce it. With $k$ parents, this reduces specification from $O(2^k)$
probabilities to $k+1$ parameters (one per cause, plus a base rate).

=== Context-Specific Independence

// Slide: Context-specific Independence

#strong[Context-specific independence] occurs when a variable is conditionally
independent of some parents depending on values of other parents. Rather than
specifying uniform CPTs, we condition probabilities on contexts.

#strong[Example—Vehicle Damage:] The probability of damage depends on ruggedness
and accident occurrence. When an accident occurs, damage is essentially certain
regardless of ruggedness ($d_(1)$ is high). When no accident occurs, damage
depends on ruggedness via distribution $d_(2)(text("Ruggedness"))$. This
context-specific structure dramatically reduces the CPT size compared to a full
specification, requiring only two conditional distributions instead of
specifying damage for every combination of ruggedness and accident values.

== Bayesian Networks with Continuous Variables

// Slide: Bayesian Networks with Continuous Variables

Many real-world problems involve continuous quantities (height, temperature,
income), for which Conditional Probability Tables are inappropriate. The
Bayesian network framework extends naturally to continuous variables through
alternative representations:

#strong[Discretization]: Divide continuous ranges into intervals and treat as
discrete. Simple but suffers from loss of accuracy and requires large CPTs as
discretization granularity increases.

#strong[Probability density functions]: Represent continuous parents and
children using families of distributions (e.g., Gaussian). Compact and
interpretable but requires choosing appropriate distributional families.

#strong[Non-parametric PDFs]: Use flexible density representations when standard
families are inadequate.

#strong[Hybrid networks] combine discrete and continuous variables, essential
for realistic modeling. Example: number of apples purchased (discrete) depends
on price (continuous).

=== Car Insurance: Network Structure

// Slide: Bayesian Network: Car Insurance Company (1/2)

A practical application: insurance companies must assess risk and set premiums
based on applicant and vehicle information. A Bayesian network captures the
causal structure of insurance risk:

#strong[Applicant information]: Age, Years With License, Driving Record, Good
Student status

#strong[Vehicle information]: Make/Model, Year, Airbag, Safety Features

#strong[Driving situation]: Mileage, Garage availability

#strong[Hidden (unobservable) variables]: Risk Aversion (affects how drivers use
features), Driving Behavior (results from skill and personality)

#strong[Claim types]:
- Medical Cost: injuries to applicant
- Liability Cost: lawsuits from other parties
- Property Cost: vehicle damage and theft

The network structure reflects causal influences: applicant attributes and
vehicle characteristics drive accident and theft risk, which translate to costs.

=== Car Insurance: Network Visualization

// Slide: Bayesian Network: Car Insurance Company (2/2)

// rendered_images:begin
// ```graphviz
// digraph InsuranceRiskModel {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2];
// 
//     Age [fillcolor="#A6C8F4"];
//     GoodStudent [fillcolor="#A6C8F4"];
//     YearsLicensed [fillcolor="#A6C8F4"];
//     DrivingRecord [fillcolor="#A6C8F4"];
//     Mileage [fillcolor="#A6C8F4"];
//     SafetyFeatures [fillcolor="#A6C8F4"];
//     MakeModel [fillcolor="#A6C8F4"];
//     VehicleYear [fillcolor="#A6C8F4"];
//     CarValue [fillcolor="#A6C8F4"];
//     Airbag [fillcolor="#A6C8F4"];
//     AntiTheft [fillcolor="#A6C8F4"];
//     Garaged [fillcolor="#A6C8F4"];
//     ExtraCar [fillcolor="#A6C8F4"];
// 
//     RiskAversion [fillcolor="#FFD1A6"];
//     DrivingSkill [fillcolor="#FFD1A6"];
//     DrivingBehavior [fillcolor="#FFD1A6"];
//     Ruggedness [fillcolor="#FFD1A6"];
//     Theft [fillcolor="#FFD1A6"];
//     Cushioning [fillcolor="#FFD1A6"];
//     OwnCarDamage [fillcolor="#FFD1A6"];
//     OtherCost [fillcolor="#FFD1A6"];
//     Accident [fillcolor="#FFD1A6"];
//     SocioEcon [fillcolor="#FFD1A6"];
// 
//     MedicalCost [fillcolor="#C6A6F4"];
//     LiabilityCost [fillcolor="#C6A6F4"];
//     PropertyCost [fillcolor="#C6A6F4"];
//     OwnCarCost [fillcolor="#C6A6F4"];
// 
//     Age -> YearsLicensed;
//     Age -> DrivingSkill;
//     Age -> GoodStudent;
//     Age -> RiskAversion;
//     YearsLicensed -> DrivingSkill;
//     DrivingSkill -> DrivingRecord;
//     DrivingSkill -> DrivingBehavior;
//     DrivingRecord -> DrivingBehavior;
//     DrivingBehavior -> Accident;
//     RiskAversion -> Garaged;
//     RiskAversion -> AntiTheft;
//     Garaged -> Theft;
//     AntiTheft -> Theft;
//     Mileage -> Ruggedness;
//     SafetyFeatures -> Ruggedness;
//     SocioEcon -> RiskAversion;
//     SocioEcon -> MakeModel;
//     SocioEcon -> ExtraCar;
//     MakeModel -> VehicleYear;
//     MakeModel -> SafetyFeatures;
//     MakeModel -> Ruggedness;
//     VehicleYear -> CarValue;
//     CarValue -> Ruggedness;
//     CarValue -> Airbag;
//     Ruggedness -> OwnCarDamage;
//     Airbag -> Cushioning;
//     Cushioning -> Accident;
//     Accident -> MedicalCost;
//     Accident -> LiabilityCost;
//     Accident -> PropertyCost;
//     Accident -> OtherCost;
//     OwnCarDamage -> OwnCarCost;
//     OwnCarCost -> PropertyCost;
//     Theft -> OwnCarDamage;
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson06.2-Using_Bayesian_Networks.typ.figs/Lesson06.2-Using_Bayesian_Networks.9.png"),
)
// render_images:end

The network uses color coding: blue nodes are observable applicant/vehicle
inputs, brown nodes are hidden variables, purple nodes are claim costs. This
structure enables the insurance company to reason from observable facts to
unobservable risk factors to financial outcomes.

#pagebreak()

== Exact Inference in Bayesian Networks

// Slide: Exact Inference in Bayesian Networks

#strong[Exact inference] computes posterior probabilities $P(X|E=e)$ for query
variable $X$ given evidence $E=e$, where hidden variables $Y$ may also be
present.

The fundamental approach uses the full joint distribution:

$P(X|e) = alpha sum_Y P(X,e,Y)$

This sums the joint probability over all configurations of hidden variables,
normalizing by the probability of evidence.

#strong[Variable elimination] improves efficiency by caching intermediate
results and systematically eliminating variables to avoid redundant computation.
Irrelevant variables—those not ancestors of query or evidence variables—can be
ignored entirely.

#strong[Limitations]:
- Exact inference is efficient (linear time) for tree-structured networks but
  intractable (exponential) for general networks.
- Continuous variables require mathematical integration, not enumeration.
- Motivates approximate inference methods for large or complex systems.

=== Exact Inference: Burglary Example

// Slide: Exact Inference in Bayesian Networks: Example

Consider the burglary network. Given that both John and Mary called, what is the
probability of a burglary?

$P(text("Burglary") | text("JohnCalls")=text("True"), text("MaryCalls")=text("True"))$

Using the definition of conditional probability:

$Pr(b | j, m) = alpha Pr(B, j, m) = alpha sum_e sum_a Pr(B, j, m, e, a)$

where $e$ ranges over Earthquake states and $a$ over Alarm states.

Factoring using the network's conditional probabilities:

$Pr(b | j, m) = alpha sum_e sum_a Pr(b) Pr(e) Pr(a | b, e) Pr(j | a) Pr(m | a)$

This expression shows how the joint probability decomposes. Computing this sum
yields the posterior probability that a burglary occurred, accounting for all
possible explanations consistent with the observed evidence.

#pagebreak()

== Approximate Inference in Bayesian Networks

// Slide: Monte Carlo Algorithms

When exact inference becomes intractable, #strong[Monte Carlo algorithms]
provide approximate solutions by generating random samples from the posterior
distribution. These randomized sampling methods estimate probabilities difficult
to calculate exactly.

#strong[Advantages]:
- Accuracy improves with sample count; arbitrarily close approximation possible
  with sufficient samples.
- Convergence is well-understood; used successfully across science and
  engineering.
- Flexible, working with general network structures.

#strong[Disadvantages]:
- Understanding variable interactions from samples alone is difficult.
- Computationally intensive, requiring many samples for high accuracy.
- Variance in estimates decreases slowly (proportional to $1/"sqrt"{N}$).

=== Sampling from Arbitrary Distributions

// Slide: Sampling from Arbitrary Distributions

To implement Monte Carlo methods, we must sample from arbitrary probability
distributions.

#strong[General approach]:
1. Start with uniform random number $r in [0,1]$.
2. Construct cumulative distribution function (CDF): $F(x) = Pr(X <= x)$.

For #strong[discrete distributions]:
- Create a table of outcomes with cumulative probabilities.
- Find the smallest outcome where $F(x) > r$.

For #strong[continuous distributions]:
- Use inverse transform when CDF is invertible: $x = F^{-1}(r)$.
- Example: Exponential distribution with CDF $F(x) = 1 - e^{-lambda x}$ inverts
  to $x = -1/lambda times ln(1-r)$.
- Use numerical methods if closed form is unavailable.

This inverse-transform method is efficient and fundamental to sampling-based
inference.

=== Prior Sampling

// Slide: Sampling Bayesian Network Without Evidence

#strong[Prior sampling] generates independent samples from a Bayesian network's
prior distribution (without conditioning on evidence).

#strong[Algorithm]:
1. Sample variables in topological order.
2. Source nodes (roots) sample from unconditional distributions.
3. Conditional nodes sample using their CPTs, conditioned on sampled parent
  values.

#strong[Example—Garden World]:
- Sample $"Rain"$ from $Pr("Rain") = 0.5$.
- If Rain=True, sample Sprinkler from $Pr("Sprinkler"|"Rain")$; otherwise from
  $Pr("Sprinkler"|"neg""Rain")$.
- Sample WetGrass using the parent values.

The samples from prior sampling implement the network's joint probability:

$f_("PS")(x_1, ..., x_n) = product_(i=1)^n Pr(x_i | "parents"(X_i))$

where "PS" denotes "Prior Sampling."

=== Consistency of Prior Sampling

// Slide: Consistency of Sampling

A key theoretical result: the empirical distribution from prior sampling
converges to the true distribution as sample count $N -> infinity$.

If $N_("PS")(x_1,...,x_n)$ is the number of samples matching configuration
$(x_1,...,x_n)$:

`lim N -> infinity` (N_PS(x_1,...,x_n) / N) = Pr(x_1,...,x_n)

This enables probability estimation:

Pr(x_1,...,x_n) ≈ (N_PS(x_1,...,x_n) / N)

Convergence is guaranteed at rate $O(1\/sqrt(N))$: doubling accuracy requires
quadrupling samples. This convergence guarantee justifies the use of sampling
for inference.

=== Rejection Sampling

// Slide: Rejection Sampling

#strong[Rejection sampling] handles inference with evidence by sampling from the
prior, then filtering to retained samples matching evidence.

#strong[Algorithm]:
1. Generate samples from the prior distribution.
2. Reject samples not matching evidence $E=e$.
3. Count occurrences of query variable $X=x$ among retained samples.
4. Estimate
  $Pr(X=x|E=e) = (\#("samples with " X=x, E=e))/(\#("samples with " E=e))$.

#strong[Garden World example]: To estimate $Pr("Rain"|"Sprinkler"="True")$ with
100 samples:
- 73 samples have $"Sprinkler"="False"$ → rejected.
- 27 samples have $"Sprinkler"="True"$ → retained.
- Among retained: 8 have $"Rain"$, 19 have $"neg""Rain"$.
- Estimate: $Pr("Rain"|"Sprinkler") = 8/27 "approx" 0.296$.

#strong[Advantages]: Simple to implement; consistent estimates.

#strong[Disadvantages]: Many samples rejected if evidence is rare; rejection
probability grows exponentially with evidence variables (curse of
dimensionality). Impractical for high-dimensional evidence or continuous
variables.

=== Importance Sampling

// Slide: Importance Sampling

#strong[Importance sampling] improves efficiency by deliberately sampling from a
distribution favoring evidence regions, then reweighting samples by importance
weights.

Instead of sampling from the prior $Pr(X)$, sample from a proposal distribution
$Q(X)$ and weight each sample:

$w_(i) = Pr(X_i) \/ Q(X_i)$

Estimate expectations as:

$E[f(X)] approx (1\/N) sum_{i=1}^N w_i f(X_i)$

#strong[Intuition]: A biased survey of a population can be corrected by giving
underrepresented groups higher weights. Similarly, if $Q$ oversamples
evidence-consistent regions, we reweight to recover the true posterior.

#strong[Advantage]: Improves efficiency dramatically when evidence is rare, as
samples are focused on relevant regions.

#pagebreak()

== Markov Chain Monte Carlo

// Slide: Markov Chain Monte Carlo

MCMC ranks among the top 10 most influential algorithms in history, developed
during the Manhattan Project (1940s) by Ulam, von Neumann, Metropolis, and
others for solving high-dimensional integration problems in physics.

#strong[Key insight]: Connect two distinct mathematical objects—Markov chains
and Bayesian networks—to enable powerful approximate inference.

Unlike rejection and importance sampling, which generate independent samples,
MCMC builds a sequence of samples where each depends on the previous. The chain
is constructed to have a #strong[stationary distribution] equal to the target
posterior.

Under conditions of ergodicity (chain can reach any state) and aperiodicity (no
cycles), the chain's distribution converges to the posterior distribution
$Pr(X|"mathbf"{e})$.

=== Markov Chain Construction

// Slide: Markov Chain Construction

A #strong[Markov chain] is a random walk through state space where the future
depends only on the present:

- Sequence of states: $x^{(0)}, x^{(1)}, x^{(2)}, ...$
- Initial state $x^{(0)}$: starting configuration
- Transition probabilities: $Pr(x -> x')$
- Distribution at time $t$: $pi_t(x)$
- Convergence: $pi_t(x) -> pi^*(x)$ (stationary distribution)

Two main transition operators:

#strong[Gibbs sampling]: Resample one variable given its Markov blanket (all
parents, children, and spouses). Simple and local.

#strong[Metropolis–Hastings]: Propose a new state from a proposal distribution,
then accept/reject based on probability ratio. Flexible; works with any
proposal.

#strong[Magic]: Under appropriate conditions, these chains have stationary
distribution equal to the posterior—samples after burn-in approximate the true
posterior.

=== MCMC Mixing

// Slide: Markov Chain Monte Carlo: Mixing

#strong[Mixing] describes how quickly a chain forgets its initial state and
explores the entire distribution.

A #strong[well-mixed chain]:
- Moves between high-probability regions frequently.
- Has low correlation between successive samples.
- Explores the support of the distribution uniformly.

#strong[Poor mixing]:
- Chain gets stuck in one mode for prolonged periods.
- High correlation between successive samples.
- Leads to biased estimates and high variance.

In practice:
- Discard initial samples as a #strong[burn-in period] before the chain
  converges.
- After convergence, collected samples approximate the true posterior.
- Monitor mixing diagnostics (autocorrelation, potential scale reduction) to
  ensure adequate sampling.

#strong[Example]: For a bimodal distribution, poor mixing means the chain stays
in one peak, missing the second mode. Good mixing jumps between peaks,
reflecting the true posterior.

=== Gibbs Sampling

// Slide: Gibbs Sampling in Bayesian Networks

#strong[Gibbs sampling] is a special case of MCMC that samples one variable at a
time, given all others.

#strong[Algorithm]:
1. Initialize all non-evidence variables to random states.
2. Keep evidence variables fixed.
3. Repeat: For each non-evidence variable $X_(i)$:
  - Sample $X_(i)$ from $Pr(X_i | text("MB")(X_i))$
  - where $text("MB")(X_i)$ is the Markov blanket (parents, children, spouses).

#strong[Example—Weather network]:
- Fix evidence: $"WetGrass"="true"$, $"Sprinkler"="true"$.
- Repeatedly sample $"Cloudy"$ from
  $Pr("Cloudy"|"Sprinkler", "Rain", "WetGrass")$.
- Then sample $"Rain"$ from $Pr("Rain"|"Cloudy", "Sprinkler", "WetGrass")$.
- Continue iterating.

#strong[Advantages]:
- Simple to implement for any Bayesian network.
- Requires only local Markov blanket information.
- Handles large, complex graphs with local updates.

#strong[Disadvantages]:
- Can mix slowly if variables are highly correlated.
- May require many samples for accurate estimates of highly dependent systems.

=== Metropolis–Hastings Sampling

// Slide: Metropolis–Hastings Sampling

#strong[Metropolis–Hastings] generalizes MCMC beyond Gibbs sampling, allowing
flexible proposal distributions.

#strong[Algorithm]:
1. Start at current state $x$.
2. Propose new state $x'$ from proposal distribution $q(x'|x)$. Examples:
  - 95% probability: Gibbs sampling
  - Otherwise: importance sampling or other methods
3. Compute acceptance ratio:
  $A(x, x') = min(1, (pi(x')q(x|x')) \/ (pi(x)q(x'|x)))$
4. Move to $x'$ with probability $A(x,x')$; otherwise stay at $x$.

#strong[Intuition]:
- Propose moves favoring higher-probability regions.
- Always accept moves to higher probability.
- Sometimes accept lower-probability moves to escape local modes.
- Balance exploration and exploitation via acceptance probabilities.

#strong[Advantages]:
- Very flexible: works with any proposal distribution.
- Scales to high-dimensional state spaces.
- Acceptance ratio ensures correct stationary distribution.

#strong[Disadvantages]:
- Mixing depends critically on proposal distribution choice.
- Poor proposals lead to slow mixing and inefficiency.
- Requires careful tuning for best performance.

The acceptance ratio elegantly ensures that the chain converges to the correct
posterior, regardless of proposal choice—a key theoretical guarantee.
