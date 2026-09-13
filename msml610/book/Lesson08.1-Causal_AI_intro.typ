// Import AIMA style formatting and macros
#import "../../helpers_root/dev_scripts_helpers/typst/aima_style.typ": aima-style, algorithm, chapter, glossary, wrap-content

// Document metadata
#set document(
  title: "Introduction to Causal AI",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules)
#show: aima-style

#chapter(8, "Introduction to Causal AI")

== Introduction and Motivation

=== Background

// Slide: Big Data and Traditional AI

Over the past decade, the dominant paradigm in data-driven organizations has
been to collect and organize massive volumes of data, then run analytics on that
data in the form of dashboards, statistical models, and reports. This approach
has produced tremendous value, but it also has fundamental limitations that
become apparent when organizations attempt to move from understanding the past
to shaping the future.

#strong[Data analytics] encompasses several categories of increasing
sophistication. At the most basic level, organizations maintain
#strong[collections of data]---organized datasets such as customer purchase
histories stored in CRM systems. #strong[Descriptive statistics] summarize these
datasets using key metrics like mean, median, and standard deviation, enabling
teams to track trends such as average quarterly sales. #strong[Historical
  reports] review past performance, for instance monthly sales reports from the
previous year. #strong[Dashboards] provide visual summaries of key metrics for
quick insights, such as quarterly revenue and expense overviews. Finally,
#strong[models] use statistical tools to forecast or explain phenomena, such as
predicting customer churn from behavioral data.

// Slide: Data Analytics Sophistication

As organizations mature in their analytical capabilities, they progress through
levels of sophistication defined by the business questions they can answer.
#strong[Descriptive analytics] answers "What happened?" through summary
statistics and historical reports. #strong[Predictive analytics] answers "What
will happen?" using forecasting models. #strong[Prescriptive analytics] answers
"What should we do?" through optimization programs. At the highest level,
#strong[simulation and optimization] answers "What is the best we can do?" by
modeling complex scenarios and finding optimal strategies.

#figure(
  image(
    "../lectures_source/figures/L08.1.Analytical_sophistication.png",
    width: 90%,
  ),
  caption: [
    Analytical sophistication spectrum: from descriptive statistics through
    predictive models to prescriptive and optimization-based approaches.
  ],
)

Each step up this ladder requires more sophisticated methodology and delivers
greater business value, but also demands stronger assumptions about the
underlying data-generating process.

=== What ML Systems Can and Cannot Tell You

// Slide: The Illusion of Understanding

Traditional machine learning systems create what we might call an
#strong[illusion of understanding]. These systems excel at finding patterns in
historical data and making accurate predictions, but they fundamentally cannot
explain the mechanisms behind those patterns or answer "what if" scenarios.

The critical distinction is between #strong[prediction] and
#strong[explanation]. A model may achieve 95% accuracy in predicting customer
churn, yet it cannot tell you what *causes* churn. Similarly, accurate sales
forecasts cannot predict the effects of pricing changes, because the model has
only learned correlations in historical data---it has no representation of
causal mechanisms.

This creates a real problem for decision-makers: without understanding
causation, organizations can only react to past events. They cannot proactively
change outcomes because they lack the knowledge of which interventions will
produce desired effects.

// Slide: Association, Correlation, and Causation

Understanding the distinctions between #strong[association],
#strong[correlation], and #strong[causation] is fundamental to appreciating why
traditional ML falls short.

#strong[Association] means that variables co-occur or are statistically
dependent. For example, "people who buy diapers also buy beer." Association
provides no information about direction or mechanism---it simply indicates that
knowing one variable tells you something about the other.

// rendered_images:begin
// ```tikz
// \tikzset{
//     box/.style={
//         draw=black,
//         ultra thick,
//         rounded corners=12pt,
//         minimum height=#1,
//         minimum width=#1*1.8,
//         align=center,
//         font=\Large\bfseries
//     }
// }
// 
// % Outermost box: Association.
// \node[box=4.5cm, fill=blue!15] (association) {};
// \node[font=\Large\bfseries, anchor=north] at ([yshift=-8pt]association.north) {Association};
// 
// % Middle box: Correlation.
// \node[box=3.0cm, fill=orange!20] (correlation) at ([yshift=-8pt]association.center) {};
// \node[font=\Large\bfseries, anchor=north] at ([yshift=-6pt]correlation.north) {Correlation};
// 
// % Innermost box: Causation.
// \node[box=1.5cm, fill=red!25] (causation) at ([yshift=-6pt]correlation.center) {};
// \node[font=\Large\bfseries] at (causation.center) {Causation};
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson08.1-Causal_AI_intro.typ.figs/Lesson08.1-Causal_AI_intro.1.png"),
)
// render_images:end

#strong[Correlation] is a specific form of association describing a linear
relationship between variables. The classic example is that ice cream sales and
drowning deaths correlate---both increase in summer due to a shared confounding
variable (warm weather), not because one causes the other.

#strong[Causation] means that changing one variable *directly* changes another.
For instance, asbestos causes cancer. Establishing causation requires either a
causal model or a controlled experiment. The fundamental illusion is that high
correlation *feels like* understanding, but real understanding means knowing
what happens when you *intervene*.

// Slide: What ML Systems Can Tell You

Machine learning systems provide genuine value in several domains.
#strong[Pattern recognition] allows models to find correlations in data,
identify similar items through clustering and classification, and forecast based
on past trends. ML delivers #strong[predictive accuracy on held-out test data],
meaning it can make accurate predictions as long as conditions remain
stable---that is, when training data and future data share the same
distribution. ML also identifies #strong[feature importance], revealing which
variables have the strongest statistical associations with outcomes. However,
feature importance fundamentally cannot distinguish correlation from causation.

// Slide: What ML Systems Cannot Tell You

The limitations of ML systems become clear when we consider what they *cannot*
tell us:

#strong[Causation from observational data alone]: Strong correlations may arise
from confounding, reverse causation, or mere coincidence.

#strong[Effects of interventions]: Questions like "If we change $X$, what
happens to $Y$?" cannot be answered from observational data without causal
assumptions. For example, "If we lower prices by 10%, will revenue increase?"
requires understanding the causal mechanism linking price to demand.

#strong[Counterfactuals]: Questions about what would have happened under
different decisions---"Would the customer have churned if we offered them a
discount?"---require reasoning beyond observed data.

#strong[Fairness]: A model can be *statistically* unbiased
($EE["Prediction"] = EE["True Value"]$) yet *causally* biased if it uses
variables that are proxies for protected attributes, even when those attributes
are not explicitly included.

#strong[Optimal decisions]: ML optimizes for accuracy, not business outcomes. A
90% accurate model might lead to worse decisions than an 85% accurate one,
depending on the consequences of different types of errors.

#pagebreak()

== Why Causal AI Matters

=== Problems with Traditional AI

// Slide: Problem 1: Correlation is Not Causation!

The most fundamental problem with traditional AI is that #strong[correlation is
  not causation]. Correlation describes statistical relationships between
variables: it excels at finding patterns in past data to predict the future, but
it does not explain cause. Variables may move together by coincidence or due to
hidden confounding factors.

#strong[Causation], by contrast, explains how changing one variable influences
another. It cannot be concluded from correlation alone. The key insight is that
data itself does not understand causes and effects---only humans can identify
the relevant variables and relationships from context and domain knowledge.
Without causal understanding, intelligent decision-making is impossible.

// Slide: Correlation is Not Causation: Examples

Intuitively, most people understand that association is not causation, yet
humans routinely confuse the two. Common examples include inferring that food
$X$ is bad for you because eating it preceded a stomachache, or believing you
can time markets because you happened to buy stocks before price spikes.

Sometimes both causal directions are plausible: does top-notch consulting
improve businesses, or do successful businesses hire top consultants? Without
causal analysis, the data alone cannot distinguish these explanations.

In the #strong[hotel industry], prices tend to be low when hotels are empty and
high when demand fills rooms. A naive correlation-based analysis might suggest
that increasing prices leads to selling more rooms---the exact opposite of the
true causal relationship. The confounding variable is demand: high demand causes
both high prices *and* high occupancy.

For an #strong[online marketplace], the causal question "What is the impact of
lowering prices on units sold?" requires understanding that gains from selling
more units must compensate for the loss from selling cheaper. Furthermore, price
cuts have heterogeneous effects depending on the type of business (clothing
versus pharmaceuticals) and timing (before Christmas versus mid-summer).

// Slide: Problem 2: Decision Making

Businesses need #strong[decisions, not just predictions]. Consider the question:
"What happens if a product price is reduced by 10%?" This seemingly simple
question spawns many causal sub-questions: Will more customers buy? Will revenue
decrease, and if so, what action should follow? Why are customers leaving---is
it a quality issue or an emerging competitor?

Examples of policy questions that require causal reasoning include: "Low-sugar
versus low-fat diet: which works for weight loss?", "If we increase credit
lines, how does bank margin change?", and "Should we build a library or give
tablets to boost test scores?" None of these can be answered by prediction
alone.

// Slide: Problem 3: ML Explainability

Regulators increasingly require organizations to defend their ML-based
decisions, particularly in sensitive domains like hiring and insurance policy
setup. Many ML systems---especially neural networks---function as #strong[black
  boxes] where humans cannot understand the input-to-output process. This
creates serious problems: organizations cannot explain decisions to
shareholders, and features like age, race, and sex may introduce bias that is
invisible within the model.

The consequences of unexplainable AI include regulatory fines, customer
backlash, and activist pressure. The solution lies in #strong[explainable AI]
that lets users comprehend, explain, and trust machine-generated results. Causal
models are inherently more interpretable because they represent the mechanisms
by which variables influence each other.

// Slide: Problem 4: How ML Systems Make Decisions Today

The typical ML workflow follows a simple pipeline: collect data, train a model,
make predictions, and deploy with rules like "If model predicts $X$, take action
$Y$." The critical limitation is that these systems optimize for prediction
accuracy rather than decision quality or business outcomes. A fraud detector
with 95% accuracy that blocks 50% of legitimate transactions is technically
accurate but practically catastrophic for the business.

// Slide: Data Science for Decisions: Examples

Consider three common applications where optimizing for prediction accuracy
misses the mark:

#strong[Recommendation systems] (Netflix, Amazon, TikTok) predict which items
users will click based on historical behavior, then recommend top-$N$ items. The
problem is that recommending novel items has no historical data to learn from,
and optimizing purely for clicks may harm long-term engagement through boredom,
echo chambers, and filter bubbles.

#strong[Hiring pipelines] score candidates and rank applicants, but training
data encodes historical hiring bias. The model perpetuates past discrimination
rather than identifying the best candidates.

#strong[Fraud detection] systems flag suspicious transactions and block them
automatically, but high false positive rates persist without understanding the
causal mechanisms behind fraudulent versus legitimate behavior.

// Slide: Problem 5: Feedback Loops

Prediction systems can change the world through #strong[feedback loops],
creating self-reinforcing cycles that amplify initial biases.

In #strong[recommendation systems], when a model recommends sensational content
and users click more, future training data contains more clicks on sensational
content, causing the model to recommend even more extreme material. This leads
to radicalization, echo chambers, and mental health issues.

In #strong[criminal justice relapse prediction], minorities are policed more
heavily, leading to more arrests, which biases the training data. Models trained
on arrest data (biased against minorities) are then used by judges for bail and
sentencing decisions, creating an increasingly biased system over time.

In #strong[college admissions], models trained on past acceptance decisions
encode historical racism and sexism. When used to screen applications, these
biases are perpetuated and potentially amplified through each admission cycle.

// Slide: Problem 6: Distribution Shift

The world is always changing, and ML models face several forms of
#strong[distribution shift]:

#strong[Concept drift] occurs when $Pr(Y | X)$---the relationship between inputs
and predictions---changes over time. For example, customer preferences evolve,
so a model trained on 2020 data becomes less accurate by
2023.

#strong[Covariate shift] occurs when $Pr(X)$---the input distribution---changes
while the underlying relationship remains stable. For example, if more young
customers join a platform, a model trained on an older customer base performs
worse on the new population.

#strong[Label shift] occurs when $Pr(Y)$---the outcome distribution---changes.
For example, fraud rates may increase due to new fraud tactics, rendering a
model trained on historical fraud rates less effective.

Causal models help address distribution shift because they encode *mechanisms*
rather than mere patterns. Mechanisms tend to be more stable across distribution
shifts than statistical correlations. Understanding "why" enables prediction in
new contexts where correlations may break down.

=== Optimization vs. Inference vs. Decision Theory

// Slide: Optimization vs. Inference vs. Decision Theory

Machine learning often conflates three distinct paradigms that serve different
purposes:

#strong[Optimization] finds the best action given a *known* objective. It
assumes the world's behavior is understood and seeks to minimize cost, maximize
throughput, or tune parameters. Tools include gradient descent, linear
programming, and evolutionary algorithms.

#strong[Statistical inference] estimates parameters from data. It learns from
observations given a model structure---for example, estimating treatment effects
or computing confidence intervals. Tools include regression, Bayesian
estimation, and hypothesis testing.

#strong[Decision theory] chooses actions under *uncertainty*. Questions like
"Should we launch a product with uncertain demand?" require reasoning about
interventions and their uncertain outcomes. Tools include utility functions,
expected value of information, and causal models.

// Slide: Where ML Falls Short

ML focuses primarily on inference and optimization, but real-world problems
demand decision theory---the ability to act under uncertainty with causal
knowledge.

#block(
  inset: 8pt,
)[
  #set text(size: 8.5pt)
  #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt,
    [*Paradigm*], [*Question*], [*Assumes*], [*Output*],
    [Optimization],
    [What maximizes $f(x)$?],
    [Known objective],
    [Optimal $x^*$],

    [Inference], [What is $theta$?], [Model structure], [Parameter estimates],
    [Decision theory],
    [What should we *do*?],
    [Uncertainty],
    [Recommended action],
  )
]

#strong[Causal AI bridges inference and decision theory]. Inference tells us
what the world looks like; causal models tell us what happens when we act;
decision theory tells us which action to choose. This bridge is precisely what
traditional ML lacks.

=== The Cost of Ignoring Causality

// Slide: The Cost of Ignoring Causality (1/2)

When organizations ignore causality, they make wrong decisions and waste
resources. Several well-known statistical phenomena illustrate this cost:

#strong[Simpson's Paradox] occurs when aggregate data shows the opposite trend
from subgroups. The classic example is Berkeley admissions: overall data
suggested gender bias, but within individual departments no bias existed. The
causal fix is to condition on the right variables using a directed acyclic graph
(DAG).

#strong[Confounding bias] arises when a hidden variable drives both the
treatment and the outcome. Ice cream sales and drowning deaths both rise in
summer---not because ice cream causes drowning, but because warm weather (the
confounder) causes both. The causal fix is to identify and control for
confounders.

#strong[Collider bias] occurs when conditioning on a common effect creates false
associations. In hospitalized patients, unrelated diseases may appear correlated
because both cause hospitalization (the collider). The causal fix is to avoid
conditioning on colliders.

// Slide: The Cost of Ignoring Causality (2/2)

#strong[Feedback loop amplification] occurs when biased predictions reinforce
bias in future training data. More police sent to minority neighborhoods leads
to more arrests, which makes the model learn those neighborhoods are more
dangerous, which sends more police. The causal fix is to model feedback
explicitly within the system.

#strong[Intervention backfire] happens when acting on correlation leads to wrong
actions. Hospitals with more staff have higher mortality rates---but this is
confounded by patient severity, since sicker patients require more staff. A
naive intervention ("reduce staff to lower mortality") would be catastrophic.
The causal fix is to distinguish correlation from causal effect before taking
action.

// Slide: Causal AI

#strong[Causal AI] addresses these limitations by providing a framework that:

- *Understands the why*: determines cause-and-effect relationships between
  variables, answering questions like "Did the marketing campaign increase
  sales?"
- *Identifies interventions*: finds variables and actions that change outcomes,
  such as "Which lifestyle changes reduce blood pressure?"
- *Predicts counterfactuals*: hypothesizes outcomes under different
  circumstances, such as predicting student grades if they attended a different
  school
- *Avoids bias*: ensures fairness beyond training data biases by accounting for
  confounding variables
- *Improves decisions*: understands relationships for better choices, such as
  improving supply chains by understanding the impact of decisions on logistics

// Slide: Causal AI vs Traditional AI

#strong[Current AI] uses correlation to analyze data, identify patterns, and
make predictions. Model quality depends entirely on data quality---biased or
unclean data produces poor models.

#strong[Causal AI] is the science of inferring causation from association,
understanding when and why causation and association differ, and understanding
cause-and-effect relationships to intervene on causes for desired effects.

As Judea Pearl stated in 2021: *"The next revolution of data science is the
science of interpreting reality, not of summarizing data."*

#pagebreak()

== Causal AI Fundamentals

=== The Ladder of Causation

// Slide: The Ladder of Causation

Judea Pearl provided a three-level framework for understanding causality---the
#strong[Ladder of Causation]. Each rung represents a qualitatively different
type of reasoning, and higher rungs cannot be answered using tools from lower
rungs alone.

#block(
  inset: 8pt,
)[
  #set text(size: 8.5pt)
  #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt,
    [*Level*], [*Symbol*], [*Activity*], [*Questions*],
    [1. Association], [$Pr(Y | X)$], [Observing], [What is?],
    [2. Intervention], [$Pr(Y | "do"(X), Z)$], [Intervening], [What if?],
    [3. Counterfactuals], [$Pr(Y_X | x', y')$], [Imagining], [Why?],
  )
]

This hierarchy is not merely a matter of increasing complexity---it represents
fundamentally different kinds of knowledge. No amount of observational data
(Rung 1) can answer interventional questions (Rung 2) without additional causal
assumptions.

// Slide: Rung 1: Association

#strong[Rung 1: Association] addresses the question: "How would seeing $X$
change your belief in $Y$?"

Mathematically represented as $Pr(Y|X)$, this rung involves passive observation
to determine whether $X$ and $Y$ are related. Traditional AI and ML operate
entirely at this level. Examples include observing that "trees have green leaves
in spring," asking "what does a symptom tell you about a disease?" or "what does
a survey tell you about election outcomes?"

Association is powerful for prediction---if you observe $X$, you can update your
belief about $Y$---but it provides no information about what happens if you
*change* $X$.

// Slide: Rung 2: Intervention

#strong[Rung 2: Intervention] addresses the question: "What happens to $Y$ if
you *do* $X$?"

Mathematically represented as $Pr(Y | "do"(X), Z)$, this rung requires
understanding the impact of action $X$ on $Y$ under conditions $Z$. Crucially,
this requires a causal model, not just observations. The `do()` operator
represents an active intervention that breaks the natural data-generating
process.

Examples include: "Spring makes leaves turn green" (a causal mechanism), "Did
the headache disappear from the pain reliever or from food?", "If I take
aspirin, will my headache be cured?", and "If we ban sugary sodas, what happens
to obesity rates?"

// Slide: Rung 3: Counterfactuals

#strong[Rung 3: Counterfactuals] addresses the question: "Was it $X$ that caused
$Y$?"

Represented symbolically as $Pr(Y_X | x', y')$, this is the highest form of
causal reasoning. It requires imagining what *would* have happened if the facts
were different---a capacity that demands full understanding of cause-and-effect
mechanisms.

Examples include: "What would have happened if the child received an adult drug
dose?", "What would the jury have concluded with different evidence?", and "Why
did the marketing campaign fail?" Counterfactual reasoning is essential for
attribution, explanation, and learning from past decisions.

=== Correlation vs. Causation Models

// Slide: Correlation vs Causation Model

Both correlation-based and causation-based models accept inputs, transform data,
and compute predictions. However, they differ fundamentally in their approach
and capabilities.

#strong[Correlation-based AI] works best with abundant historical data. It
identifies statistical patterns and uses them for prediction. This approach is
powerful when the goal is forecasting under stable conditions.

#strong[Causal-based AI] requires building a business model first, then
integrating data. It identifies mechanisms and enables reasoning about
interventions. This approach is necessary when the goal is decision-making or
understanding effects of actions.

// Slide: Correlation-Based Model Process

The #strong[correlation-based model process] follows a "data first" approach
where more data is assumed to be better. The typical process involves: acquiring
data, integrating and cleaning it, performing exploratory data analysis,
engineering features, building and testing models, and deploying models in
production.

Common failures of this approach include organizational issues (teams working in
silos), opaque models that lack explainability, spurious correlations that
mislead decision-makers, and a failure to ever articulate "what is the business
goal?" The data-first approach often produces technically impressive models that
solve the wrong problem.

// Slide: Causation-based Model Process

The #strong[causation-based model process] follows a "model first" approach that
requires understanding the business question before collecting data. The process
begins with fundamental questions: What is the intended outcome? What is the
proposed intervention? What are the confounding and effecting factors? Only
after answering these questions does the team create a model diagram (typically
a DAG), followed by targeted data acquisition aligned with the causal structure.

=== Data Science vs. Decision Science

// Slide: From Data Science to Decision Science

Data science and decision science represent #strong[two fundamentally different
  paradigms]:

#block(
  inset: 8pt,
)[
  #set text(size: 8.5pt)
  #table(
    columns: (auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt,
    [*Dimension*], [*Data Science*], [*Decision Science*],
    [Goal], [Predict outcomes], [Choose best actions],
    [Core question], ["What will happen?"], ["What should we do?"],
    [Method], [Statistical models, ML], [Causal models, decision theory],
    [Output],
    [Score, forecast, cluster],
    [Recommended action + expected outcome],

    [Success metric], [Accuracy, AUC, RMSE], [Business outcome, ROI, impact],
    [Data requirement],
    [Historical observations],
    [Observations + causal structure],
  )
]

The key insight is that most organizations have data science teams focused on
prediction, while business stakeholders actually need decision science. The gap
between "what will happen" and "what should we do" is precisely #strong[causal
  reasoning]. Machine learning excels at prediction but fails at "what-if"
questions and decisions involving interventions.

// Slide: Causal Questions vs. Predictive Questions

The distinction between predictive and causal questions clarifies when standard
ML suffices and when causal methods are needed:

#block(
  inset: 8pt,
)[
  #set text(size: 8.5pt)
  #table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: 0.5pt,
    [*Predictive Question*], [*Causal Question*],
    [Which patients will be readmitted?],
    [Which intervention reduces readmission?],

    [Which customers will churn?], [What action prevents churn?],
    [What will sales be next quarter?],
    [How would a 10% price cut affect sales?],

    [Which students will fail the exam?], [Does tutoring improve exam scores?],
    [Which transactions are fraudulent?], [What policy changes reduce fraud?],
  )
]

#strong[Predictive questions] require only standard ML and observational data.
They are useful for monitoring and alerting. #strong[Causal questions] require
causal models or experiments. They answer "what if" and "why" and are essential
for decisions and policy design.

// Slide: A Roadmap From Prediction to Decision Intelligence

Organizations progress through four levels of analytical maturity:

#strong[Level 0: Descriptive]---"What happened?" Dashboards, reports, and
statistics provide backward-looking summaries.

#strong[Level 1: Predictive]---"What will happen?" ML models and forecasting
assume a stable world.

#strong[Level 2: Causal]---"Why?" and "What if we intervene?" DAGs and
do-calculus enable understanding of mechanisms.

#strong[Level 3: Decision]---"What should we do?" Causal models combined with
decision theory and business goals enable optimal action selection.

Most organizations remain stuck at Level 1. Causal AI provides the methodology
to advance to Levels 2 and 3, where the greatest business value lies.

#pagebreak()

== Causal AI in Business

=== Business Context and Motivation

// Slide: Digital Transformation in Business

#strong[Digital transformation] uses digital technologies to reduce costs and
improve revenues through several mechanisms: integration of digital technology
(embedding AI, cloud, IoT, and automation into business areas), cultural and
organizational change (encouraging innovation and data-driven mindsets),
customer-centric approaches (using personalized experiences and AI-driven
insights), process automation and optimization (streamlining operations through
AI and analytics), and data-driven decision making (leveraging big data and
real-time analytics for smarter, faster decisions).

// Slide: Why AI Projects Fail?

#strong[AI projects fail] primarily because they approach problems exclusively
from an ML perspective. Data scientists focus on model accuracy and technical
metrics while working in isolation from domain experts, business users, and data
engineers. This leads to a lack of understanding of problem context, producing
irrelevant solutions.

Key failure modes include: misalignment with business goals (models solve the
wrong problem), black-box models that fail to gain stakeholder trust, data
issues that undermine performance (poor quality, missing, or biased data),
deployment and integration challenges, and lack of feedback loops for monitoring
post-deployment performance.

// Slide: How to Make AI Project Succeed

Successful AI projects follow several principles:

1. #strong[Create a hybrid team] that represents all business aspects, uses a
  collaborative framework, and communicates with a single language (such as
  DAGs). Team size varies by company and project complexity.
2. #strong[Meet regularly] to ensure continuity and alignment.
3. #strong[Find an executive sponsor] who understands the project's goals and
  potential.
4. #strong[Start with an initial pilot]---a small team tackling a targeted
  problem to demonstrate the merit of the causal AI approach.

=== The Causal AI Workflow

// Slide: AI in Business

The Causal AI workflow in business follows a cyclic process connecting eight key
components:

// rendered_images:begin
// ```graphviz
// digraph CausalFlowGrid {
//   layout=neato;
//   node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, width=1.4, height=0.6, fixedsize=true];
// 
//   BusinessGoals   [label="Business Goals",        pos="0,2!",  fillcolor="#E53935", fontcolor="white"];
//   UseCases        [label="Use Cases",             pos="2,2!",  fillcolor="#FB8C00", fontcolor="black"];
//   CausalModels    [label="Causal Models",         pos="4,2!",  fillcolor="#FDD835", fontcolor="black"];
//   DataSources     [label="Data Sources",          pos="4,1!",  fillcolor="#43A047", fontcolor="white"];
//   KnowledgeModels [label="Knowledge\nModels",     pos="4,0!",  fillcolor="#1E88E5", fontcolor="white"];
//   Algorithms      [label="Algorithms/\nLibraries", pos="2,0!", fillcolor="#5E35B1", fontcolor="white"];
//   Interventions   [label="Interventions/\nCounterfactuals", pos="0,0!", fillcolor="#8E24AA", fontcolor="white"];
//   Results         [label="Key Results/\nTakeaways", pos="0,1!", fillcolor="#D81B60", fontcolor="white"];
// 
//   Placeholder     [label="", pos="2,1!", style=invis];
// 
//   edge [color="#555555", penwidth=1.4, splines=true];
// 
//   BusinessGoals -> UseCases;
//   UseCases -> CausalModels;
//   CausalModels -> DataSources;
//   DataSources -> KnowledgeModels;
//   KnowledgeModels -> Algorithms;
//   Algorithms -> Interventions;
//   Interventions -> Results;
//   Results -> BusinessGoals;
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson08.1-Causal_AI_intro.typ.figs/Lesson08.1-Causal_AI_intro.2.png"),
)
// render_images:end

This cycle emphasizes that causal AI is not a one-shot analysis but an iterative
process that continuously refines understanding and improves decision-making.

// Slide: Step 1: What Are the Intended Outcomes?

#strong[Step 1] focuses on defining intended outcomes. Given that most AI
projects (even 90%) fail due to misalignment between business goals and the
project, this step is critical. Teams must answer: What process are you
analyzing? What happens if a course of action is taken or not? What outcomes are
positive, negative, unacceptable, or optimal? What related data can be combined?
What are possible and feasible interventions? What confounding factors are
correlated with outcomes and treatments? What factors exist but cannot be
accurately measured?

// Slide: Step 2: What Are the Proposed Interventions?

#strong[Step 2] identifies the #strong[levers] available to achieve a business
outcome. For example, in customer marketing for a product, possible
interventions include: introducing a new product, bundling multiple products
(and determining whether bundling improves or inhibits long-term sales),
focusing advertising on product quality versus alternatives, adding variations,
discontinuing the product, or divesting the product line entirely. Each
intervention represents a different causal pathway to the desired outcome.

// Slide: Step 3: What Factors Create Changes?

#strong[Step 3] identifies the factors that create changes, with particular
attention to #strong[confounders]---variables that obscure the relationship
between treatment and outcome by either muting or inflating the apparent effect.

Three types of effects must be distinguished: #strong[direct effects] (effects
through the intervention itself), #strong[indirect effects] (effects through the
environment or intervention byproducts), and the #strong[total causal effect]
(the combined effect of all factors modifying the outcome).

// Slide: Step 4: Build Causal DAG

#strong[Step 4] constructs the #strong[causal DAG] (Directed Acyclic Graph).
Causal models simplify complex systems without losing key relationships,
focusing on essential variables and interactions. Visual models abstract
complexity into interpretable formats and highlight the direction and strength
of influence.

Simplicity serves multiple purposes: it aids communication between stakeholders,
reduces cognitive overload by excluding irrelevant details, guides data
collection by identifying impactful variables, and supports hypothesis testing
through counterfactual and intervention scenarios. The key is balance---too much
simplicity loses insight, while too much complexity loses clarity.

// Slide: Marketing Example: Price Intervention

Consider a marketing example where #strong[price is the intervention] and the
goal is to #strong[maximize revenue]. The team must decide: Increase price? By
how much? Decrease price? Change price one-time or over time? Develop individual
pricing models for each customer?

The causal DAG reveals the complexity:

// rendered_images:begin
// ```graphviz
// digraph CausalDiagram {
//     splines=true;
//     nodesep=1.0;
//     ranksep=0.75;
// 
//     node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.4];
// 
//     Price [label="Price", fillcolor="#F4A6A6"];
//     Behavior [label="Behavior", fillcolor="#FFD1A6"];
//     ProductAmount [label="Product Amount", fillcolor="#B2E2B2"];
//     Revenue [label="Revenue", fillcolor="#A6C8F4"];
//     CompetitiveOffers [label="Competitive Offers", fillcolor="#C6A6F4"];
//     Supply [label="Product Supply", fillcolor="#A6E7F4"];
//     StoreDistance [label="Distance to store", fillcolor="#D2B48C"];
// 
//     Price -> Behavior;
//     Behavior -> Revenue;
//     Price -> ProductAmount;
//     ProductAmount -> Revenue;
//     Price -> CompetitiveOffers;
//     CompetitiveOffers -> ProductAmount;
//     Supply -> ProductAmount;
//     StoreDistance -> ProductAmount;
// }
// ```
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson08.1-Causal_AI_intro.typ.figs/Lesson08.1-Causal_AI_intro.3.png"),
)
// render_images:end

The DAG reveals multiple confounders: product supply and distance to store
affect product amount independently of price, and competitive offers mediate
part of the price effect. Without this causal structure, a naive analysis would
conflate these distinct pathways.

// Slide: Step 5: Data Acquisition and Integration

#strong[Step 5] involves data acquisition and integration. Organizations can
reuse data collected for correlation-based ML but must also collect data
specifically for causal AI. Data must be treated, conditioned, and transformed
to match the refined causal model.

#strong[Transformation] involves mapping observed variables to DAG nodes,
encoding categorical variables, handling missing data (through imputation or
exclusion), and normalizing or scaling values. The team must also control for
bias and confounding. The goal is to ensure that the data structure supports
causal estimation---not just predictive accuracy.

// Slide: Step 6: Model Modification

#strong[Step 6] refines the initial DAG to reduce bias and improve reliability.
After the initial design, variables must be clarified as confounders, mediators,
or outcomes.

Key pitfalls to avoid: do not control for mediators or colliders (which can
distort results), and do control for direct and indirect confounders (to prevent
biased estimates). Models are tested against both technical and business
objectives. This step determines the necessary controls to isolate causal
effects, ensures causal assumptions align with data and domain logic, and
improves model interpretability.

// Slide: Step 7: Preparing for Deployment in Business

#strong[Step 7] transitions from experimentation to integration with
decision-making processes. This involves validating the model against real-world
business outcomes and ensuring stakeholders understand and trust the causal
logic and assumptions.

Teams must develop user-friendly interfaces for business users, automate data
pipelines for timely updates, establish metrics for performance tracking and
drift detection, create feedback loops for post-deployment refinement, provide
clear documentation for auditors and users, and train stakeholders on
interpreting results. The ultimate goal is to deliver measurable business value.

// Slide: Roles in Hybrid Teams

Effective causal AI projects require #strong[hybrid teams] with diverse roles:

- #strong[Business Strategists]: align modeling with goals, sponsor projects,
  communicate insights
- #strong[Subject-Matter Experts]: provide domain expertise, identify variables,
  validate DAGs
- #strong[Data Experts]: source and clean data, map data to model variables
- #strong[Data Scientists]: construct and validate DAGs, apply causal inference
  methods, simulate decisions
- #strong[Software Developers]: build tools, interfaces, and data pipelines
- #strong[IT Professionals]: provide infrastructure, governance, and enterprise
  integration
- #strong[Project Managers]: coordinate collaboration, manage documentation,
  ensure strategic alignment

// Slide: Executing for a Hybrid Team Project

Executing a hybrid team project requires: establishing a collaborative approach
that aligns strategic goals with technical efforts and emphasizes early
stakeholder engagement; defining team goals using SMART objectives aligned with
business strategy; targeting a bounded, feasible, high-value use case that
prioritizes early wins for trust and momentum; defining the hypothesis by
building a preliminary DAG with expert input; and pursuing incremental model
development built in small stages with regular feedback.

=== Explainability and Interpretability

// Slide: The Importance of Explainability

Understanding how ML models make decisions, whether they can be trusted, and
whether they are biased are critical questions for any organization deploying
AI. Systems must foster trust, transparency, and user confidence.

Management faces serious consequences from unexplainable AI: loss of trust,
regulation violations, fines, and lawsuits. A false negative in medical
screening for cancer, for instance, represents a catastrophic failure that
demands explanation. #strong[Humans in the loop] ensure that interpretation and
context are considered alongside model outputs.

// Slide: Techniques for Interpretability

Several techniques exist for model interpretability:

#strong[LIME] (Local Interpretable Model-agnostic Explanations) focuses on
single predictions by approximating the model locally with an interpretable
surrogate, perturbing input data and observing prediction changes.

#strong[Partial Dependence Plots] (PDP) show the marginal effect of features on
predicted outcome by varying one feature while keeping others constant.

#strong[Individual Conditional Expectation] (ICE) plots show the
feature-prediction relationship for individual instances rather than averages.

#strong[SHAP] (SHapley Additive exPlanations) quantifies each feature's
contribution to a specific prediction using game-theoretic principles.

// Slide: Causal AI in Interpretable AI

#strong[Causal AI] provides a natural framework for interpretability:

- DAGs present variables, relationships, and their strengths in a visual,
  intuitive format
- Counterfactuals predict action outcomes directly
- Removing confounders prevents skewed estimates
- Both experts and non-experts can understand the output
- Hybrid teams enhance context and reduce blind spots

// Slide: The Future of Causal AI

The future of AI involves a #strong[convergence] of multiple approaches: Causal
AI moves from academia to commercial use, departing from purely data-driven
systems to reflect human thinking and decision-making.

This convergence brings together causal AI, traditional AI, deep learning, and
generative techniques into unified systems that can both predict and explain,
both correlate and reason about causation.
