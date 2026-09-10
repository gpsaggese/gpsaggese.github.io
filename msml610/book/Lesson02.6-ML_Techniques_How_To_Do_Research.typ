// git_hash=97a53972d-xas timestamp=20260910_100559
// Import AIMA style formatting and macros.
#import "/helpers_root/dev_scripts_helpers/typst/aima_style.typ": (
  aima-style, algorithm, chapter, glossary, styled-table,
)
// Import the custom citation/bibliography system.
#import "/helpers_root/dev_scripts_helpers/typst/umd_references.typ": (
  cite, references,
)

// Document metadata
#set document(
  title: "L02.6: ML Techniques - How to Do Research",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.6: ML Techniques - How to Do Research")

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:7 '# How to Do Research'
// Slide: How to Do Research
= How to Do Research

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:9 '## Simple Is Better'
// Slide: Simple Is Better
== Simple Is Better

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:11 '* Occam's Razor'
// Slide: Occam's Razor
#strong[Simple is better] when it comes to modeling. As William of Occam argued, "the
simplest model that fits the data is also the most plausible." A remark often
attributed to Einstein puts the same point with a useful caveat: "an explanation of
the data should be as simple as possible, but not simpler." The practical takeaway is
that you should trim a model to the bare minimum necessary to explain the data,
resisting the temptation to add parameters or structure beyond what the evidence
supports.

What do #emph[simple] and #emph[better] actually mean here? A model counts as simple
when it belongs to a small class of possible objects; the fewer distinct models a
hypothesis class contains, the simpler each member is. A model is better in the sense
that matters most: it achieves better #emph[out-of-sample] performance, generalizing
to data it was not trained on rather than merely memorizing the training set.

Why should simplicity lead to better generalization? A simple model is less likely to
fit a given dataset by coincidence. When a model from a small class happens to
explain the observations well, that agreement is more meaningful precisely because
there were fewer ways it could have gotten lucky. This intuition can be formalized in
information-theoretic terms: an event is more statistically significant when it is
unlikely, and the surprise it carries is quantified by entropy. A complex model that
fits the data perfectly may simply be absorbing noise, whereas a simple model that
fits nearly as well is far more likely to have captured genuine structure.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:27 '* What Is a "simple model"?'
// Slide: What Is a "simple model"?
That leaves open a natural question: what exactly counts as a #emph[simple model]? An
object is #strong[simple] when it is one of relatively few possible objects meeting
the same structural constraints. This is a counting argument: simplicity is not about
visual appearance or intuitive "cleanness," but about how large the set of
alternatives is.

Consider polynomials: a polynomial of order 2 is simpler than one of order 17,
because far fewer polynomials of order 2 exist (once you fix the number of
coefficients). Both sets are infinite, yet the order-17 family is vastly larger. A
less obvious example comes from support vector machines: the separating hyperplane an
SVM finds may look wiggly and visually "complex," but it is defined by only a handful
of support vectors, making it simple in the counting sense that matters for
generalization.

There are many #strong[measures of complexity], and they split naturally into two
levels. At the level of a single hypothesis $h$, one can measure complexity by
polynomial order, by the minimum description length (MDL) needed to encode $h$ in
bits #cite("rissanen1978mdl"), or by its Kolmogorov complexity. At the level of an
entire hypothesis set $cal(H)$, a central measure is the #emph[VC dimension] of the
model #cite("abumostafa2012learning") #cite("vapnikchervonenkis1971uniform"), which
captures the richest pattern of data points the set can shatter.

These two levels are not independent. If you need $l$ bits to specify a particular
hypothesis $h$, then $h$ is one element of a set containing at most $2^l$ hypotheses.
The bit-length of the individual hypothesis therefore bounds the size of the
hypothesis set it belongs to, linking single-hypothesis complexity directly to
set-level complexity through a counting argument. @tab:whatisasimplemodel summarizes
the main complexity measures, the level at which each operates, and a concrete
example of each.

#figure(
  styled-table(
    headers: ("Complexity Of", "Measure", "Example"),
    rows: (
      (
        "Single hypothesis $h$",
        "Description length",
        "Polynomial order, MDL, Kolmogorov complexity",
      ),
      ("Hypothesis set $\calH$", "Capacity", "VC dimension"),
    ),
  ),
  caption: [Table of Complexity Of, Measure, Example],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:whatisasimplemodel>

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:59 '* Model Soundness'
// Slide: Model Soundness
Controlling complexity is not enough on its own, however: even a simple model still
has to earn your trust by telling a coherent story. You cannot simply accept its
output at face value without interrogating it. As statistician George Box put it,
"All models are wrong, but some are useful" #cite("box1976science"). A useful
discipline is to ask yourself the same question a skeptical reviewer would: "What
criticisms would I give if this model were presented for the first time?"

One way to keep that skepticism grounded is to compare the model against a simple
benchmark rather than judging its performance in isolation. A benchmark might always
output a constant value, such as a long-only model that always predicts a stock will
rise, or it might produce purely random results, mimicking a bootstrap test of the
null hypothesis that the model has no genuine predictive power. A model that cannot
clear either baseline is not adding value, however sophisticated it looks.

#strong[A perfect fit can mean nothing], and the reason is a matter of degrees of
freedom rather than modeling skill. Take two data points on a plane: a straight line
always fits them perfectly, simply because two points determine a unique line. That
perfect fit reveals nothing about the underlying process, since the model (the line)
is exactly as complex as the dataset it was fit to (two points), leaving no room for
the data to falsify the hypothesis. A model this flexible relative to its data can
never be wrong, and a hypothesis that can never be wrong has not really been tested.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:78 '* Sampling Bias (1/2)'
// Slide: Sampling Bias (1/2)
Even a properly falsifiable model, however, is only as trustworthy as the data used
to test it. #strong[Sampling bias] occurs when the data used to train or evaluate a
model is not representative of the intended population. Because a model views the
world entirely through its training data, biased sampling leads directly to biased
outcomes. This connection is formalized by Hoeffding's inequality in ML theory, which
assumes that training and testing distributions are drawn from the same underlying
population #cite(
  "abumostafa2012learning",
). When that assumption breaks, the guarantees on generalization break with it.

Several mechanisms introduce sampling bias into a dataset, as @fig:samplingbias12
illustrates:

- #emph[Non-random sampling]: the collection process favors certain outcomes or
  groups over others, so some subpopulations are systematically overrepresented.
- #emph[Undercoverage]: some members of the target population have little or no
  chance of appearing in the sample, leaving the model blind to their
  characteristics.
- #emph[Survivorship bias]: only successful or surviving subjects make it into the
  dataset, hiding the patterns associated with failure or attrition.
- #emph[Self-selection bias]: participants opt in voluntarily, and those who choose
  to participate often differ systematically from those who do not.

#figure(
  image(
    "Lesson02.6-ML_Techniques_How_To_Do_Research.typ.figs/Lesson02.6-ML_Techniques_How_To_Do_Research.1.png",
    width: 70%,
  ),
  caption: [Diagram relating Sampling Bias],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:samplingbias12>

The consequences of these biases are far-reaching. Systematic errors distort the
relationships a model learns, so predictions that look accurate on the training set
fall apart when applied to real-world data drawn from the full population. Worse,
decisions based on those predictions can be not only incorrect but actively unfair,
reinforcing the very disparities that the biased sample failed to capture.
Recognizing which of the mechanisms above is at work is the first step toward
correcting it, whether through stratified sampling, reweighting, or collecting
additional data from underrepresented groups.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:111 '* Sampling Bias (2/2)'
// Slide: Sampling Bias (2/2)
Real-world datasets routinely reflect who was easiest to measure rather than who
matters most. Training a facial recognition system predominantly on lighter-skinned
faces, for instance, degrades its accuracy on darker-skinned faces, not because the
algorithm is inherently biased but because the training distribution does not match
the deployment population. Similarly, building an income-prediction model from
records of employed individuals silently drops everyone who is unemployed, producing
estimates that cannot generalize to the full population.

Several practical strategies can reduce the damage from selection bias:

- #emph[Compare sample statistics with population statistics] to detect
  distributional mismatches before modeling begins.
- #emph[Stratified sampling or resampling] ensures that underrepresented groups
  appear in the training set at rates closer to their true prevalence.
- #emph[Re-weighting or bias correction] adjusts the loss function or the sample
  weights so that observations from undersampled groups exert proportionally greater
  influence during training.

All of these remedies share a hard prerequisite: every relevant subgroup must have at
least some representation in the data. When certain data points have zero probability
of being observed ($Pr = 0$), no post-hoc correction can recover the missing
information; the model is structurally unable to learn about outcomes it has never
seen.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:125 '* Data Snooping (1/2)'
// Slide: Data Snooping (1/2)
Sampling bias is not the only way a model's evaluation can mislead you: a second,
more insidious problem comes from how the test set itself gets used. #strong[Data
  snooping] occurs when information from the test set improperly influences the
model-building process #cite("hastie2009elements") #cite("kaufman2012leakage"). Even
a small leak can silently inflate performance estimates, making a model look far
better than it actually is on genuinely unseen data.

Two mechanisms account for most cases. The first is #emph[train-test contamination]:
test data leaks into the training pipeline, whether through feature engineering that
peeks at test labels, model selection guided by test-set accuracy, or hyperparameter
tuning evaluated directly on the held-out split. The second is #emph[multiple
  hypothesis testing]: when a practitioner tries many models, feature sets, or
preprocessing choices and reports only the best result, the reported performance
reflects selection bias rather than true generalization. This is the same logic
behind p-hacking in statistics, but it arises in subtler forms too; even an honest
grid search over dozens of configurations can overfit to the particular test fold if
the same fold is reused for every comparison.

The consequences are predictable: evaluations become overly optimistic, and models
that appeared strong during validation fail once deployed on fresh data. What makes
data snooping especially dangerous is that it produces #emph[false confidence]. The
practitioner sees clean metrics, passes review checkpoints, and ships a model that
quietly underperforms in production. Because the numbers looked right at every stage,
the failure is harder to diagnose than a model that was obviously undertrained.
Guarding against data snooping requires strict separation of data splits from the
very first preprocessing step, combined with disciplined use of held-out or nested
cross-validation schemes whenever multiple modeling choices are compared.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:145 '* Data Snooping (2/2)'
// Slide: Data Snooping (2/2)
Choosing features based on the full dataset before splitting into training and test
sets is a common form of data snooping. For instance, if you demean the entire
dataset and then split it into training and test partitions, the test set's mean has
already influenced the preprocessing, giving the model foresight it would never have
in deployment.

The core preventive strategy is straightforward: keep training, validation, and test
sets strictly separate. Fit all preprocessing steps, including feature selection,
normalization, and imputation, only on the training data, then apply those fitted
transformations to the validation and test data without refitting. Cross-validation
must follow the same discipline; every preprocessing step belongs inside the
cross-validation loop so that each fold's held-out portion remains untouched during
fitting.

@tab:datasnooping22 summarizes specific snooping sources alongside the corresponding
preventive strategy for each.

#figure(
  styled-table(
    headers: ("Source", "Preventive Strategy"),
    rows: (
      (
        "Train-test contamination",
        "Keep training, validation, and test sets separate",
      ),
      (
        "Multiple hypothesis testing",
        "Use cross-validation properly; fit preprocessing on training data only",
      ),
    ),
  ),
  caption: [Table of Source, Preventive Strategy],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:datasnooping22>

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:168 '* "Burning the Test Set"'
// Slide: "Burning the Test Set"
Keeping the splits separate guards against contamination, but even a properly
separated test set can be compromised through overuse. When a test set is consulted
repeatedly during model development, information about its specific examples
gradually leaks into the training process #cite(
  "hastie2009elements",
). Each time a practitioner checks test-set accuracy, tweaks a hyperparameter, and
checks again, the model drifts toward memorizing quirks of that particular sample
rather than learning the underlying pattern. As Ronald Coase quipped, "if you torture
the data long enough, it will confess whatever you want." The observable symptom is a
test accuracy that keeps climbing while the model's true, out-of-sample performance
stagnates or even degrades. Worse, because the only scorecard you have is the
now-contaminated test set, you lose the ability to trust your own evaluation process.

Several safeguards exist to guard against this failure mode:

- #emph[One-time use principle]: reserve the test set for a single, final evaluation
  after all tuning is complete. Every intermediate decision (architecture search,
  hyperparameter selection, feature engineering) should rely on a separate validation
  split or cross-validation within the training data.
- #emph[Statistical adjustments]: when multiple hypotheses or model variants are
  compared on the same held-out data, raw p-values overstate significance.
  Corrections such as the Bonferroni method or False Discovery Rate control account
  for the number of comparisons and reduce the chance of a spurious "winner."
- #emph[Theoretical capacity bounds]: the VC dimension quantifies a model class's
  learning capacity in a way that accounts for repeated trials on the same data,
  providing distribution-free generalization guarantees. Similarly, the Minimum
  Description Length (MDL) principle penalizes both model complexity and repeated
  fitting by framing learning as compression: a model that needs many bits to
  describe itself and its fit to a particular test set is, by construction, less
  trustworthy than a simpler one.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:189 '## Research Methodology'
// Slide: Research Methodology
== Research Methodology

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:191 '* How to Achieve Out-Of-Sample Fit'
// Slide: How to Achieve Out-Of-Sample Fit
The goal of learning is to choose a hypothesis $g in cal(H)$ that approximates the
unknown target function $f$ #cite("abumostafa2012learning"). Saying $g approx f$
means, concretely, that the out-of-sample error is near zero: $E_"out" (g) approx 0$.

Achieving that requires a model with two properties working together. First, the
model needs good in-sample performance, $E_"in" (g) approx 0$, meaning it fits the
training data well. Second, it needs good generalization,
$E_"out" (g) approx E_"in" (g)$, meaning its performance on unseen data closely
tracks what it achieved on the training set. When both conditions hold
simultaneously, they combine to deliver the real objective: good out-of-sample
performance, $E_"out" (g) approx 0$. Neither condition alone is sufficient; a model
can memorize every training example yet fail on new data (low $E_"in"$ but poor
generalization), or it can generalize perfectly from a model so simple that it never
fit the data in the first place (good generalization but high $E_"in"$).

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:203 '* What If Out-Of-Sample Fit Is Poor?'
// Slide: What If Out-Of-Sample Fit Is Poor?
A common frustration in practice is that the model performs well on the training set
($E_"in" approx 0$) but poorly on unseen data ($E_"out" >> E_"in"$) #cite(
  "abumostafa2012learning",
). In-sample performance is optimistic: the model has been tuned to the particular
quirks of the training examples, so it is overfitted and fails to generalize.

The remedy has two stages. First, run diagnostics to gain insight into what is
working and what is not before committing to a long-term improvement effort.
Bias-variance decomposition curves and learning curves are the most informative tools
here: they reveal whether the dominant source of error is bias (underfitting) or
variance (overfitting), and they show how performance changes as training-set size
grows.

Second, once the diagnosis is clear, address the problem by pulling one of several
levers, each targeting a specific failure mode:

- #emph[Training data]: collecting more training data is the most direct fix for high
  variance, because a larger sample leaves less room for the model to memorize noise.
- #emph[Feature selection]: removing features reduces high variance by shrinking the
  hypothesis space, while adding features or constructing derived features (for
  instance, polynomial terms) reduces high bias by giving the model enough expressive
  power to capture the true relationship.
- #emph[Regularization]: decreasing the regularization parameter $lambda$ relaxes the
  complexity penalty and fixes high bias; increasing $lambda$ tightens it and fixes
  high variance.

@tab:whatifoutofsamplefitispoor summarizes these levers, the specific action to take,
and whether each action targets high bias or high variance.

#figure(
  styled-table(
    headers: ("Lever", "Action", "Effect"),
    rows: (
      ("Training data", "Get more data", "Fixes high variance"),
      ("Features", "Remove features", "Fixes high variance"),
      ("Features", "Add (derived) features", "Fixes high bias"),
      ("Regularization", "Decrease $\lambda$", "Fixes high bias"),
      ("Regularization", "Increase $\lambda$", "Fixes high variance"),
    ),
  ),
  caption: [Table of Lever, Action, Effect],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:whatifoutofsamplefitispoor>

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:240 '* Why Using a Lot of Data?'
// Slide: Why Using a Lot of Data?
Of these levers, increasing the amount of training data deserves special attention.
Several studies #cite("abumostafa2012learning") #cite("banko2001data") have
demonstrated a striking empirical pattern: when different learning algorithms and
model architectures are compared on the same task, their performance levels tend to
be remarkably similar. What consistently makes the bigger difference is not the
choice of algorithm but the size of the training set; increasing the amount of
training data reliably improves performance across the board #cite(
  "halevy2009unreasonable",
).

The practical consequence is straightforward: pair a high-capacity model with a
massive training set, and you can expect strong performance. This is, in essence, the
recipe behind much of modern machine learning's success.

The intuition follows from the bias-variance perspective. A high-capacity model with
many parameters (a deep neural network, for instance) can drive the in-sample error
to near zero:

$ E_("in") approx 0 $

This happens because such models have very low bias; they are flexible enough to fit
virtually any training set. The tradeoff is high variance: without enough data, the
model memorizes noise and generalizes poorly.

A large dataset counteracts that variance by constraining the model to patterns that
genuinely recur, effectively preventing overfitting. When the training set is large
enough, out-of-sample error tracks in-sample error closely:

$ E_("out") approx E_("in") $

Combining these two conditions yields the desired result:

$ E_("out") approx E_("in") approx 0 => E_("out") approx 0 $

In other words, a model that is expressive enough to achieve near-zero training
error, trained on enough data to keep generalization error in check, will achieve
near-zero test error as well. This chain of reasoning explains why the modern "scale
up both the model and the data" strategy works so reliably in practice.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:264 '* What to Do When You Have Lots of Data?'
// Slide: What to Do When You Have Lots of Data?
Having 100 million examples in a dataset sounds like an embarrassment of riches, but
it comes with real costs: training on that volume is slow, demands substantial
compute, and requires serious infrastructure work to manage data pipelines,
distributed training, and storage.

The practical question is whether all that data is actually necessary. The answer
depends on where the model currently sits on the bias-variance spectrum, and learning
curves are the diagnostic tool for finding out.

1. Plot in-sample and out-of-sample performance (training error and validation error)
  for increasing subsets of the data: $m$ = 1k, 10k, 100k, 1M.
2. Diagnose the current situation from the shape of those curves:
  - If the model has #emph[large bias], training and validation performance will be
    close together even at $m$ = 1M, both plateauing at an unsatisfactory level.
    Adding more data will not help; the model is too simple to capture the underlying
    pattern. The next step is to use a more complex model rather than throwing all
    100M instances at the existing one.
  - If the model has #emph[large variance], there will be a persistent gap between
    training and validation performance at $m$ = 1M, with validation error still
    visibly decreasing as $m$ grows. In this regime more data genuinely helps, so the
    next step is to train on all 100M instances to close that gap.

@fig:learningcurves illustrates both scenarios side by side: in the high-bias case
the two curves converge early and flatten, while in the high-variance case the
validation curve keeps improving as data increases, signaling that the full dataset
is worth the infrastructure investment.

#figure(
  image("../lectures_source/figures/L02.6.Learning_curves.png", width: 80%),
  caption: [Learning curves],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningcurves>

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:292 '* Right and Wrong Approach to Research'
// Slide: Right and Wrong Approach to Research
Diagnosing bias and variance is one skill; deciding what to actually work on next is
another, broader challenge. When facing a machine learning project, it is rarely
clear how to prioritize among the many possible tasks: feature engineering, data
collection, algorithm selection, hyperparameter tuning, and so on. A tempting but
unreliable strategy is to pick a task based on gut feeling, complete it, then
re-evaluate performance. This cycle wastes effort because it gives no systematic
signal about where the real bottleneck lies.

A more disciplined approach proceeds in stages:

1. #emph[Build a simple algorithm quickly], ideally within a single day. The goal is
  not perfection; it is to have a concrete baseline that exposes the shape of the
  problem.
2. #emph[Set up a rigorous performance evaluation process.] Use cross-validation and
  commit to a single numeric metric with confidence bounds so that every subsequent
  change can be measured against a clear target.
3. #emph[Set up diagnostic tools.] Compute learning curves and bias-variance curves
  before attempting any fix. Understanding whether the model suffers from high bias
  or high variance tells you which interventions can actually help and prevents
  premature optimization #cite("knuth1974premature").
4. #emph[Understand the problem by debugging.] For instance, in a spam classifier,
  manually review the misclassified emails and ask: what types are they? Are they
  short messages, image-heavy emails, or messages that use unusual formatting? This
  qualitative inspection often reveals patterns that no aggregate metric can surface.
5. #emph[Accept that some approaches simply need to be tried.] Not every decision can
  be made from first principles; occasionally, running an experiment is the fastest
  way to learn whether an idea has merit.

The key difference between these two strategies is that the disciplined approach
front-loads measurement and diagnosis. By establishing evaluation infrastructure
early, every subsequent decision is informed by evidence rather than intuition, and
effort flows toward the changes most likely to improve performance.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:313 '* Example: Spam Filter Classification'
// Slide: Example: Spam Filter Classification
A concrete example makes this disciplined approach easier to see in action. Suppose
you build a spam classifier using logistic regression with $N = 5$ fixed indicator
features, one per word: `buy`, `now`, `deal`, `discount`, and your own name. Each
feature is 1 if the word appears in the email and 0 otherwise. This is a reasonable
starting point, but five hand-picked words leave a lot of signal on the table. How
can you do better?

There are several complementary strategies for improving classifier performance.
First, you can #strong[collect more training data]. One practical technique is a
#emph[honeypot]: you set up a fake email account, publish its address in places
spammers scrape, and harvest the incoming messages as labeled spam. This gives you a
large, naturally distributed negative class without manually labeling thousands of
emails.

Second, you can #strong[use better features] drawn from the email metadata rather
than the message body. Spammers often route messages through unusual relay servers or
forge headers to make the sender look legitimate. Features extracted from routing
information, such as whether the originating IP belongs to a known open relay, or
whether the "From" domain matches the actual envelope sender, capture structural
patterns that body-word features miss entirely.

Third, you can #strong[expand the feature set from the message body itself] by moving
well beyond five words. Instead of hand-picking a small vocabulary, you might use the
full bag-of-words representation, or at least the top few hundred words ranked by
mutual information with the spam label. A richer vocabulary lets the model pick up on
subtler cues: words like "unsubscribe," "congratulations," or "wire transfer" each
carry some discriminative weight that a five-word model simply cannot access.

Fourth, you can #strong[detect intentional misspellings]. Spammers routinely
substitute characters to evade keyword filters: `w4tch` for "watch," `v1agra` for a
well-known pharmaceutical name, or `fr33` for "free." A naive exact-match feature
will miss all of these variants. Applying #emph[stemming software], which reduces
words to their root form, helps collapse inflected variants (e.g., "buying," "buys,"
"bought" all map to "buy"). For the more creative letter-substitution tricks, you can
add a normalization step that maps common character swaps (digits for visually
similar letters, repeated characters, inserted punctuation) back to a canonical
spelling before feature extraction. Together, stemming and normalization let the
classifier see through obfuscation that would otherwise split one concept across
dozens of surface forms.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:331 '* Why Do We Do Things?'
// Slide: Why Do We Do Things?
Chasing every one of these improvements, however, is only worthwhile if you first
know why you are chasing it. Before diving into any data science task, you must
understand its purpose. Ask yourself: "Why are we doing this?" and "What do we hope
to determine by performing this task?" Clarifying goals and expected outcomes up
front prevents wasted effort and ensures that every analytical step connects to a
meaningful business or research objective. Think about your actions with the bigger
picture in mind, avoid going through the motions mechanically, and prioritize tasks
by their importance and impact.

Consider a few concrete scenarios. Suppose your team is collecting customer feedback
through surveys. The natural question is: why surveys, and what is the desired
outcome? The answer might be to improve product features and customer service, with
the specific goal of identifying areas for improvement and innovation. Or suppose a
marketing campaign is being launched. Why is this campaign running? Perhaps to
increase brand awareness or drive sales. What are the specific goals? Setting a
target number of new leads or achieving a particular click-through rate. In each
case, starting with purpose transforms a generic task into a focused, measurable
effort whose success you can later evaluate.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:352 '* Summary of the Results, Next Steps'
// Slide: Summary of the Results, Next Steps
Evaluating that success is only useful if it gets communicated clearly once the work
is done. Every analysis report should open with a summary of results: a high-level
list of the major discoveries and findings stated in plain language. For instance, a
summary might note that "smoothing model coefficients helps the model generalize to
held-out data." Beyond simply listing what happened, interpret those results for the
reader. If sales rose during a particular quarter, say so and offer a plausible
explanation: the increase in sales is likely due to the new marketing strategy
launched that same period. Close the summary with conclusions that state whether the
data confirmed or contradicted the hypothesis under investigation. A conclusion such
as "the hypothesis that user engagement increases retention is supported by the
observed effect size" gives stakeholders a clear verdict without requiring them to
parse every table.

Pair that summary with a reference to the detailed results that follow. Busy
executives and senior decision-makers need a quick, digestible overview (a TLDR)
before they decide whether to read further. Structuring the report this way, summary
first, details second, respects their time while still preserving the full evidence
trail for anyone who wants it.

Finally, always think about the next steps. A good analyst anticipates what should
happen after the current findings land, much like thinking $n$ moves ahead in chess.
If the analysis revealed that a particular demographic segment drove most of the
sales growth, the natural follow-up is to conduct a detailed analysis on the
demographics contributing most to that growth. Outline potential experiments or
analyses that would validate the findings further, sharpen the estimates, or test the
causal story you proposed in the interpretation. Ending with concrete next steps
turns a static report into a living part of the decision cycle.

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:374 '* Incremental vs Iterative'
// Slide: Incremental vs Iterative
Those next steps are usually delivered through one of two broad development styles.
#strong[Incremental development] builds a system piece by piece: each increment adds
a distinct functional component to what already exists. This approach requires
careful upfront planning to divide features into meaningful slices, and integrating
successive increments can grow complex as the system expands.
@fig:monalisaincremental illustrates the idea: imagine painting the Mona Lisa by
completing one section of the canvas at a time, left to right, until the full picture
emerges.

#figure(
  image(
    "../lectures_source/figures/L02.6.Monalisa_incremental.png",
    width: 80%,
  ),
  caption: [Monalisa incremental],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:monalisaincremental>

#strong[Iterative development] takes a different path. Every cycle delivers a usable,
if rough, version of the entire system, which is then refined and improved through
repeated passes. The key advantage is that each iteration invites feedback, helping
the team uncover and adjust for requirements that were unknown at the outset.
@fig:monalisaiterative captures this nicely: the Mona Lisa starts as a coarse sketch
of the whole subject, then gains detail and polish with each successive iteration.

#figure(
  image("../lectures_source/figures/L02.6.Monalisa_iterative.png", width: 80%),
  caption: [Monalisa iterative],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:monalisaiterative>

In practice, incremental development is almost always stronger than purely iterative
development ($"incremental" >> "iterative"$), but the most effective strategies
combine both. @fig:skateboard shows the classic contrast: rather than delivering an
unusable quarter of a car in each increment, deliver a skateboard first, then a
scooter, then a bicycle, and finally a car. Each delivery is #emph[usable] (iterative
thinking), and each one #emph[adds capability] (incremental thinking). The skateboard
may not be a car, but it gets the user moving and generates the feedback needed to
build the right thing next.

#figure(
  image("../lectures_source/figures/L02.6.Skateboard.png", width: 80%),
  caption: [Skateboard],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:skateboard>

// From: msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd:416 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
