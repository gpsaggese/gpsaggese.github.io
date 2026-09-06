// git_hash=6d65371d2-h2c timestamp=20260904_160218
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
  title: "L02.5: ML Techniques - Model Evaluation",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.5: ML Techniques - Model Evaluation")

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:11 '# Model Evaluation'
// Slide: Model Evaluation
#strong[Model Evaluation]

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:13 '## Why Evaluate'
// Slide: Why Evaluate
== Why Evaluate

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:15 '* How to Make Progress in ML Research'
// Slide: How to Make Progress in ML Research
#strong[How to Make Progress in ML Research]

Machine learning offers a vast landscape of choices at every stage of a project:
how to preprocess the data, which features to extract or engineer, which model
family to use, which training algorithm to apply, and how to evaluate the
result. Each of these axes can be varied independently, producing a
combinatorial explosion of possible pipelines. Faced with so many degrees of
freedom, how should a practitioner decide which combination actually works best?

The answer is to evaluate models systematically using a single number. Rather
than relying on intuition or anecdotal comparisons, each candidate pipeline is
scored with a well-defined #strong[metric] such as accuracy or F1 score, giving
a clear, comparable summary of performance. A single scalar makes it
straightforward to rank alternatives and to communicate results to stakeholders
who need a bottom line.

A metric alone, however, is only as trustworthy as the evaluation protocol
behind it. #emph[Cross-validation] provides a principled way to estimate how
well a model will generalize: by repeatedly splitting the data into training and
validation folds, it guards against the optimistic bias that comes from
evaluating on the same data used for fitting. Beyond cross-validation,
#emph[statistical tests] are essential for confirming that an observed
improvement is genuine rather than an artifact of random variation. Hypothesis
testing quantifies the probability that the difference between two models'
scores could have arisen by chance, while #emph[A/B testing] extends this
reasoning into production settings, where a new model is compared against a
baseline on live traffic under controlled conditions. Together, these tools turn
model selection from guesswork into an evidence-based process.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:33 '* In-Sample vs Out-Of-Sample Error Expressions'
// Slide: In-Sample vs Out-Of-Sample Error Expressions
#strong[In-Sample vs Out-Of-Sample Error Expressions]

The goal of learning is to find a function $h$ that approximates an unknown
target function $f$ over the space of inputs $x in cal(X)$. Because $f$ is
unknown, we cannot compare $h$ to it everywhere; instead, we measure how well
$h$ matches $f$ at individual points through a pointwise error function:

$ e(h(bold(x)_i), f(bold(x)_i)) $

Common choices for this error function include the squared error
$e(bold(x)) = (h(bold(x)) - f(bold(x)))^2$, which penalizes large deviations
quadratically; the 0-1 binary error
$e(bold(x)) = I[h(bold(x)) eq.not f(bold(x))]$, which simply counts
misclassifications; and the negative log probability
$e(bold(x)) = -log(Pr(h(bold(x)) = f(bold(x))))$, which measures how surprised
the model is by the correct answer. Each choice shapes what "good approximation"
means in practice: squared error rewards getting close, 0-1 error cares only
about being exactly right or wrong, and log probability rewards well-calibrated
confidence.

Given a pointwise error, we can aggregate it in two fundamentally different
ways. #strong[In-sample error] is computed using all $N$ points in the training
set:

$ E_("in")(h) = 1 / N sum_(i=1)^N e(h(bold(x)_i), f(bold(x)_i)) $

This quantity tells us how well $h$ fits the data we have already seen. It is
always computable, since we know both $h(bold(x)_i)$ and the target value at
every training point. However, a low in-sample error does not guarantee that $h$
will perform well on new data; a sufficiently complex hypothesis can memorize
the training set while failing everywhere else.

#strong[Out-of-sample error] is computed over the entire input space $cal(X)$:

$ E_("out")(h) = EE_(bold(x) in cal(X))[e(h(bold(x)), f(bold(x)))] $

This is the quantity we truly care about: it measures how well $h$ generalizes
to inputs it has never encountered. The expectation is taken with respect to the
underlying distribution over $cal(X)$, so $E_("out")$ weights each region of the
input space by how likely it is to arise in practice. The central challenge of
learning theory is understanding when and why minimizing $E_("in")$ leads to a
small $E_("out")$, since we can only ever compute the former directly from our
finite training data.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:58 '* Training vs Test Set'
// Slide: Training vs Test Set
#strong[Training vs Test Set]

Performance on the training set $E_(i n)$ is an #strong[optimistic estimate] of
the true out-of-sample error $E_(o u t)$. A model can achieve a 0% error rate on
its training data simply by memorizing every response, yet still answer
essentially at random on unseen examples, yielding a 50% error rate on a test
set. The gap between these two numbers is the core danger of overfitting:
in-sample success tells you almost nothing about real-world reliability.

To #strong[properly evaluate model performance], the data used for evaluation
must be kept entirely separate from the data used for training. Both the
training and test sets should be representative samples drawn from the same
underlying population. Consider a credit-risk problem: if a model is built using
transaction data from a New York bank branch, evaluating it on data from a
Florida branch may be misleading, because the two populations can have very
different income distributions, spending patterns, and default rates. A test set
that does not reflect the deployment population will produce error estimates
that are just as unreliable as testing on the training data itself.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:74 '## Data Splitting'
// Slide: Data Splitting
== Data Splitting

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:76 '* Lots of Data Scenario vs Scarce Data Scenario'
// Slide: Lots of Data Scenario vs Scarce Data Scenario
#strong[Lots of Data Scenario vs Scarce Data Scenario]

When working with machine learning models, the amount of available data
fundamentally shapes every decision about how to train, evaluate, and deploy
them. In an ideal world, you would have an essentially unlimited supply of
labeled examples. With enough data, you can fit all the degrees of freedom of a
complex model without worrying about overfitting, and you can set aside a large
holdout set to get a precise, low-variance estimate of out-of-sample
performance. In practice, this scenario is rare outside a few domains (web-scale
text, ad clicks, sensor telemetry) where data is generated continuously and
cheaply.

Far more often, data is #emph[scarce]. Consider facial recognition: collecting
and annotating images across diverse demographics, lighting conditions, and
poses is expensive and raises privacy concerns, so the resulting dataset may
contain only a few thousand labeled examples. In such settings, you cannot
afford to use every example for training, because you would have nothing left to
honestly assess how well the model generalizes. The standard remedy is to
#strong[hold out] a portion of the data exclusively for evaluation.

The simplest version of this idea is a single #emph[train/test split]: partition
the dataset into a training set used to fit the model and a test set used only
once, at the end, to estimate performance metrics and confidence bounds.
Choosing the split ratio involves a tension: a larger training set lets the
model learn more, but a larger test set gives a more reliable performance
estimate. Common heuristics (80/20 or 70/30) balance these concerns, but the
right ratio depends on the total dataset size and the model's complexity.

When data is especially limited, even a single split wastes information.
#strong[Cross-validation] addresses this by rotating the role of the holdout
fold across the entire dataset, so every example eventually serves as both a
training point and a validation point. $k$-fold cross-validation, for instance,
partitions the data into $k$ equally sized folds, trains $k$ separate models
each holding out one fold, and averages their performance estimates.

Beyond smarter splitting, two other strategies help stretch a small dataset
further:

- #emph[Data augmentation]: create synthetic training examples by applying
  label-preserving transformations to existing ones. In image processing, this
  might mean random crops, flips, rotations, or color jitter; each
  transformation yields a "new" example that teaches the model invariance
  without requiring additional annotation effort.
- #emph[Transfer learning]: start from a model that was pre-trained on a large,
  related task (for example, ImageNet classification) and fine-tune it on the
  smaller target dataset. The pre-trained weights already encode useful
  low-level features (edges, textures, shapes), so the model needs far fewer
  target-domain examples to reach good performance.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:96 '* Splitting Data Into Training, Validation, Test Sets'
// Slide: Splitting Data Into Training, Validation, Test Sets
#strong[Splitting Data Into Training, Validation, Test Sets]

Training, validation, and test sets must satisfy three core requirements. First,
they must be #strong[distinct]: no data point can appear in more than one set,
or the model's evaluation will be contaminated by information it has already
seen. Second, each set must be #emph[representative] of the problem, meaning the
distribution of classes, feature ranges, and edge cases in every set should
mirror the overall dataset. If a class makes up 15% of the original data, it
should make up roughly 15% of each split, not be concentrated in one set and
absent from another. Third, the relative sizes of the three sets should be
chosen based on the volume of available data and the complexity of the problem:
a small dataset may need a larger validation share to produce stable estimates,
while a very large dataset can afford a thin validation slice.

Several techniques help ensure that every split shares the #emph[same underlying
  distribution]:

- #emph[Stratified sampling]: split the data so that each class label (or, more
  generally, each important subgroup) appears in every set in proportion to its
  frequency in the full dataset. This is especially important when some classes
  are rare; a naive random split could leave a minority class entirely out of
  the test set.
- #emph[Shuffle and sample]: randomly permute the entire dataset before drawing
  the train, validation, and test portions. Shuffling breaks any ordering
  artifacts (for instance, data sorted by date or by class) and, combined with
  stratification, keeps the resulting splits well balanced.
- #emph[Sample and check statistics]: after splitting, compare summary
  statistics of key variables across all three sets. Examining the mean,
  standard deviation, and probability density function (PDF) of each feature
  confirms that no split has drifted from the others. A large discrepancy in any
  of these signals that the split should be redone or that stratification
  constraints need tightening.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:113 '* Rule of Thumbs for Data Set Splits'
// Slide: Rule of Thumbs for Data Set Splits
#strong[Rule of Thumbs for Data Set Splits]

When the dataset is large, a common strategy is the 60-20-20 split: 60% of the
data goes to training, 20% to validation for tuning hyperparameters, and the
remaining 20% to a held-out test set for final evaluation. This three-way
partition gives enough examples in each subset that the estimates of both
hyperparameter quality and generalization error are reliable.

When the dataset is of moderate size, the validation set becomes a luxury the
practitioner can no longer afford. A 60-40 split between training and test is
typical here. Because there is no separate validation set, hyperparameter tuning
must rely on other means (or be foregone entirely), and the model is evaluated
directly on the 40% test portion.

When the dataset is small, even a single fixed split wastes too many examples.
#emph[K-fold cross-validation] is the standard remedy: the data is partitioned
into $K$ equally sized folds, and the model is trained $K$ times, each time
holding out a different fold for evaluation and averaging the results. Even so,
small datasets carry a heightened risk that a seemingly high accuracy is an
artifact of chance rather than genuine learning. At this scale it is worth
asking whether machine learning is even the right tool for the problem, or
whether the sample size is simply too limited for any data-driven model to
generalize reliably. Reporting the small data size alongside any results is good
practice, so that readers can calibrate their confidence accordingly.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:130 '* Training Data'
// Slide: Training Data
#strong[Training Data]

How much data is enough? More data is generally better, though returns diminish
after a certain volume. Initially, increasing the size of a training set
significantly improves model performance: patterns become clearer, variance
drops, and the model generalizes more reliably to unseen examples. Eventually,
however, adding more data yields smaller and smaller accuracy gains, and the
computational cost of processing that additional data may no longer be justified
by the marginal improvement.

You should use #emph[learning curves] to track this effect. A learning curve
plots model performance (on the vertical axis) against training set size (on the
horizontal axis), making it easy to see where the curve begins to flatten.
@fig:learningcurvesexample illustrates a typical learning curve: early on the
slope is steep, reflecting rapid improvement, but as the dataset grows the curve
levels off, signaling diminishing returns. By inspecting this plot, you can make
a practical decision about whether collecting or labeling more data is worth the
effort, or whether your time is better spent improving the model architecture or
feature engineering instead.

#figure(
  image(
    "../lectures_source/figures/L02.5.Learning_Curves_Example.png",
    width: 80%,
  ),
  caption: [Learning Curves Example],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:learningcurvesexample>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:144 '* Using Test Data'
// Slide: Using Test Data
#strong[Using Test Data]

Can you use the test set for training? The short answer is: not until you have
finished evaluating your model. The test set exists to give you a single,
unbiased estimate of generalization performance. If you train on it before that
estimate is locked in, you contaminate the very measurement you need.

Once that final performance number $E_"test"$ has been recorded, however, you
can and often should fold the test data back into the training pool before
deploying the model. The reasoning is straightforward: you have already
committed to the modeling procedure (the algorithm, the hyperparameters, the
preprocessing pipeline), so there is no further selection bias to worry about.
Retraining on the combined dataset simply gives the deployed model access to
every available example, which generally improves its accuracy.

There is a subtle but important shift in interpretation that comes with this
practice. The reported $E_"test"$ no longer describes the exact model sitting in
production, because that model was retrained on a larger dataset. Instead,
$E_"test"$ estimates the quality of the #emph[procedure]: the entire pipeline
that, given data of this size and shape, produces a fitted model. In practice,
the deployed model will usually perform at least as well as the estimate
suggests, since it saw strictly more training data, but the guarantee is about
the procedure rather than the artifact.

@fig:usingtestdata illustrates this workflow: evaluate first on held-out test
data, lock in the performance estimate, then retrain on everything for
deployment.

// rendered_images:begin
// ```tikz
// % Timeline showing test set usage - professional version
// \draw[thick, ->] (0,0) -- (11,0);
// \node[below] at (5.5,-0.5) {Development Pipeline Timeline};
// 
// % Axes and grid
// \draw[thin, gray] (0,0) grid[step=0.5] (11,0);
// 
// % Phase 1: Training
// \draw[fill=blue!40, draw=blue, thick] (0.5,0.3) rectangle (3,0.9);
// \node[font=\bfseries] at (1.75,0.6) {Training};
// \node[below, font=\small] at (1.75,-0.8) {60\% data};
// 
// % Phase 2: Validation
// \draw[fill=green!40, draw=green, thick] (3.5,0.3) rectangle (5.5,0.9);
// \node[font=\bfseries] at (4.5,0.6) {Validation};
// \node[below, font=\small] at (4.5,-0.8) {20\% data};
// 
// % Phase 3: Testing
// \draw[fill=red!40, draw=red, thick] (6,0.3) rectangle (8,0.9);
// \node[font=\bfseries] at (7,0.6) {Testing};
// \node[below, font=\small] at (7,-0.8) {20\% data};
// 
// % Phase 4: Deployment
// \draw[fill=purple!40, draw=purple, thick] (8.5,0.3) rectangle (10.5,0.9);
// \node[font=\bfseries] at (9.5,0.6) {Deploy (Retrain)};
// \node[below, font=\small] at (9.5,-0.8) {100\% data};
// 
// % Annotations above phases
// \node[above, font=\small] at (1.75,1.3) {Model selection};
// \node[above, font=\small] at (7,1.3) {Final evaluation};
// \node[above, font=\small] at (9.5,1.3) {Production};
// 
// % Arrow showing data reuse
// \draw[dashed, thick, ->] (8,0.15) to (8.5,0.15);
// \node[font=\small, gray] at (8.25,-1.1) {Reuse test data};
// 
// % Legend
// \node[anchor=west, font=\small] at (0.5,-1.8) {Note: Only retrain model with full dataset after test set evaluation is complete};
// ```
// label=fig:usingtestdata
// caption=Diagram illustrating Using Test Data
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.5-ML_Techniques_Model_Evaluation.typ.figs/Lesson02.5-ML_Techniques_Model_Evaluation.1.png"),
  caption: [Diagram illustrating Using Test Data],
) <fig:usingtestdata>
// render_images:end

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:196 '## Choosing an Error Measure'
// Slide: Choosing an Error Measure
== Choosing an Error Measure

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:198 '* How to Choose an Error Measure?'
// Slide: How to Choose an Error Measure?
#strong[How to Choose an Error Measure?]

The choice of error measure is not purely a technical decision; it depends on
the #strong[application] domain and should ultimately be defined by the customer
or end user in terms of an acceptable error level. A medical diagnostic system,
for instance, demands extremely low error tolerance because a missed diagnosis
can be life-threatening, whereas a movie recommendation engine can tolerate a
much higher rate of mistakes without serious consequences.

One powerful way to customize evaluation is through a #strong[utility function]
that assigns different costs to different kinds of errors and successes. In a
binary classification setting, the four possible outcomes (true positives, true
negatives, false positives, and false negatives) rarely matter equally. For
medical diagnosis, minimizing false negatives (cases where a disease is present
but the model fails to detect it) is far more important than minimizing false
positives, because sending a healthy patient for an extra test is a minor
inconvenience compared to missing a serious illness. A well-chosen utility
function encodes these priorities directly into the training or evaluation
objective.

Beyond application-specific costs, a good error measure should satisfy two
general properties. First, it should be #emph[plausible]: the measure should
match the statistical assumptions underlying the data. Squared error, for
example, is a natural choice when the noise in the observations follows a
Gaussian distribution, because minimizing squared error is equivalent to maximum
likelihood estimation under that model. Second, it should be #emph[friendly]:
the measure should be mathematically convenient to work with. Measures that
admit closed-form solutions simplify calculations considerably and reduce
computational cost. Measures that are convex are especially valuable because
convex optimization guarantees that any local minimum is also the global
minimum, so standard gradient-based algorithms can find the best solution
reliably without getting trapped in suboptimal basins.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:218 '* Error Measures: Fingerprint Verification Example'
// Slide: Error Measures: Fingerprint Verification Example
#strong[Error Measures: Fingerprint Verification Example]

In fingerprint verification, correctly recognizing a valid fingerprint produces
no error. When the system gets it wrong, however, the mistake falls into one of
two categories: a #strong[false positive] (accepting an invalid fingerprint) or
a #strong[false negative] (rejecting a valid one). The weight assigned to each
type of error depends entirely on the application.

Consider a supermarket loyalty program that uses fingerprint scanning to verify
discount eligibility. A false positive here (✓) is a minor issue:
one customer receives an undeserved discount. A false negative (⚠)
is far more costly: a legitimate customer is denied their discount, leading to
frustration and checkout delays. Now consider access control at a CIA facility.
The calculus reverses completely. A false positive (💀) is critical: an
unauthorized person gains entry, creating a security breach. A false negative
(✓) is acceptable: a legitimate employee is simply asked to verify
their identity through an additional check.

#figure(
  styled-table(
    headers: ("Application", "False Positive", "False Negative"),
    rows: (
      (
        "Supermarket",
        "Minor issue: one extra discount",
        "Costly: annoyed customer, delays",
      ),
      (
        "CIA Building",
        "Critical: security breach",
        "Acceptable: triggers further checks",
      ),
    ),
  ),
  caption: [Error severity for fingerprint verification in two different
    settings.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:errorweights>

As @tab:errorweights summarizes, the very same classification problem in two
different settings demands the #emph[opposite] error measure. What counts as the
tolerable mistake in a supermarket becomes the catastrophic one in a secure
facility, and vice versa. This means that any practical classifier must be tuned
not just for overall accuracy, but for the specific cost structure of the domain
in which it operates.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:235 '* How to Measure Classifier's Performance?'
// Slide: How to Measure Classifier's Performance?
#strong[How to Measure Classifier's Performance?]

#strong[Success rate] (also called hit rate or win rate) measures the proportion
of correct predictions out of all predictions made. It is computed as

$ "accuracy" = frac("TP" + "TN", "TP" + "TN" + "FP" + "FN") $

where TP and TN are true positives and true negatives, while FP and FN are false
positives and false negatives. This single number gives an at-a-glance sense of
how often the model gets the right answer. For instance, if a classifier labels
80 out of 100 examples correctly, it achieves an 80% success rate. The
complement of this quantity, the error or miss rate, simply counts the fraction
of incorrect predictions.

#strong[Log probability loss], more commonly known as #strong[cross-entropy
  loss], evaluates a classifier that outputs probabilities between 0 and 1
rather than hard labels. It is defined as

$
  "cross-entropy" = -frac(1, N) sum_(i=1)^N [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
$

where $y_i$ is the true binary label and $p_i$ is the predicted probability for
observation $i$. Cross-entropy penalizes confident wrong predictions far more
heavily than uncertain ones: predicting 0.99 for a true negative incurs a much
larger loss than predicting 0.6. Lower cross-entropy therefore signals
better-calibrated probabilistic predictions, making it the standard training
objective for logistic regression and neural-network classifiers alike.

#strong[Precision, recall, and F-score] become essential when the class
distribution is imbalanced, because raw accuracy can be misleading in such
settings (a model that always predicts the majority class may look accurate yet
be useless). #emph[Precision] is the fraction of predicted positives that are
actually positive, answering "when the model says yes, how often is it right?"
#emph[Recall] is the fraction of actual positives that the model correctly
identifies, answering "of all the real positives, how many did the model catch?"
These two metrics trade off against each other: raising the decision threshold
typically increases precision at the expense of recall, and vice versa. The
#emph[F-score] reconciles the two by computing their weighted harmonic mean,
providing a single number that rewards balanced performance on both axes. Formal
definitions of each metric appear in later sections.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:259 '### Regression Metrics'
// Slide: Regression Metrics
=== Regression Metrics

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:261 '* Mean Squared Error (MSE)'
// Slide: Mean Squared Error (MSE)
#strong[Mean Squared Error (MSE)]

#strong[Mean squared error] (MSE) is the average squared difference between
predicted and actual values:

$ "MSE" equiv 1 / N sum_(i=1)^N (h(bold(x)_i) - f(bold(x)_i))^2 $

For each data point, the model's prediction $h(bold(x)_i)$ is compared against
the true value $f(bold(x)_i)$, the difference is squared, and the results are
averaged over all $N$ observations. Consider house price prediction: if the
model estimates a home at \$320,000 but it sells for \$300,000, the squared
error for that single observation is $(20{,}000)^2 = 4 times 10^8$. MSE
aggregates contributions like this across every house in the dataset, producing
a single number that summarizes overall prediction quality.

MSE is popular in part because of its connection to the Gaussian error model.
When residuals are normally distributed, minimizing MSE is equivalent to
maximizing the likelihood of the data, giving the metric a principled
statistical justification. It is also optimization-friendly: the squared term is
everywhere differentiable, producing smooth gradients that work well with
gradient descent and closed-form solutions such as the normal equations in
linear regression.

These advantages come with real costs. Because the errors are squared before
averaging, MSE is reported in squared units of the target variable (e.g.,
dollars-squared for house prices), which makes the raw number hard to interpret
on the original scale. This is one reason practitioners often report root mean
squared error (RMSE) instead, restoring the original units. MSE also assumes a
roughly symmetric error distribution; for skewed targets, such as income or
insurance claims, it can paint a misleading picture of model performance.
Perhaps its most consequential drawback is sensitivity to outliers. A single
prediction that misses by a large margin contributes a disproportionately large
squared term, inflating the overall metric even when the model performs well on
most observations. When outlier robustness matters, alternatives like
#emph[median absolute deviation] (MAD) or the median of squared errors provide
summaries that are far less influenced by extreme values.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:283 '* Root Mean Squared Error (RMSE)'
// Slide: Root Mean Squared Error (RMSE)
#strong[Root Mean Squared Error (RMSE)]

#strong[Root Mean Squared Error] (RMSE) is defined as the square root of the
Mean Squared Error:

$
  "RMSE" equiv sqrt("MSE") = sqrt(1/N sum_(i=1)^N (h(bold(x)_i) - f(bold(x)_i))^2)
$

Because the square root undoes the squaring of units introduced by MSE, RMSE is
expressed in the same units as the target variable. This makes it far easier to
interpret: if you are predicting house prices in dollars, an RMSE of 15,000
immediately tells you that predictions are off by roughly \$15,000 on average, a
statement you can compare against the mean sale price to judge whether the error
is large or small. That unit-level interpretability also makes RMSE a natural
choice when comparing models trained on different data sets or predicting
outputs at different scales, since the metric is already normalized to the
output's own scale.

The tradeoff is that RMSE inherits every drawback of MSE. It still squares
individual errors before averaging, so a single large residual can dominate the
sum under the radical and inflate the reported error well beyond what the
typical prediction experiences. For data with a skewed error distribution, where
most predictions are close but a few are far off, RMSE overstates how poorly the
model performs on the bulk of examples while simultaneously underrepresenting
just how extreme the worst cases are.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:299 '* Median-Based Metrics'
// Slide: Median-Based Metrics
#strong[Median-Based Metrics]

One natural alternative to mean-based loss functions is to replace the mean with
the median, shifting from an $L_2$ perspective to an $L_1$ one. Because the
median is insensitive to extreme values, metrics built on it resist the pull of
outliers that can dominate squared-error averages.

#strong[Median absolute deviation] (MAD) is defined as

$ "MAD" eq.def "median"_i (|h(bold(x)_i) - f(bold(x)_i)|) $

A closely related quantity is the #strong[median squared error]:

$ "median"_i (|h(bold(x)_i) - f(bold(x)_i)|^2) $

Both measures report the "typical" residual rather than the average one, so a
handful of poorly predicted points cannot inflate the overall score. This
robustness to outliers is their chief advantage: in data sets where a few
observations are corrupted, mislabeled, or genuinely extreme, median-based
metrics give a far more stable picture of model quality than their mean-based
counterparts. The tradeoff is computational: medians do not decompose as neatly
as sums, making optimization and analytical manipulation more difficult to carry
out. Gradient-based training, for instance, relies on smooth, differentiable
objectives, and the median introduces a sorting step that complicates both the
loss surface and its gradients.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:321 '### Classification Metrics'
// Slide: Classification Metrics
=== Classification Metrics

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:323 '* Error Metrics for Skewed Classes'
// Slide: Error Metrics for Skewed Classes
#strong[Error Metrics for Skewed Classes]

Accuracy can be misleading as a standalone performance metric, particularly when
the classes in a dataset are not evenly distributed. To get a more honest
picture of how well a classifier performs, practitioners turn to tools like the
#strong[confusion matrix], #strong[precision], and #strong[recall].

Consider training a classifier to distinguish tumors as malignant ($y = 1$) or
benign ($y = 0$). Suppose the classifier achieves an error rate of just 1%,
meaning it guesses correctly 99% of the time. That sounds impressive at first
glance. But now consider the underlying data: only 0.5% of patients actually
have cancer. A trivial classifier that always outputs $y = 0$ (predicting every
patient is healthy, regardless of any input) would achieve an error rate of only
0.5%, outperforming the trained model without learning anything at all.
Suddenly, that 1% error rate no longer looks so good; the model is actually
doing worse than a completely naive baseline that ignores the input entirely.

This example illustrates a fundamental pitfall of #emph[accuracy] (or
equivalently, error rate) on #emph[imbalanced datasets]. When one class
dominates the data, a classifier can achieve high accuracy simply by predicting
the majority class every time. The metric rewards the model for doing nothing
useful. To detect whether a classifier has genuinely learned to identify the
rare but important class, we need metrics that separately measure how well it
handles positives and negatives, which is exactly what precision, recall, and
the confusion matrix provide.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:339 '* Confusion Matrix'
// Slide: Confusion Matrix
#strong[Confusion Matrix]

Binary classification problems rest on two assumptions: the actual and predicted
class labels both belong to $\{0, 1\}$, and by convention $y = 1$ encodes the
rare, positive class the model is trying to detect.

Given those two possible labels for both the true outcome and the model's
prediction, every single example falls into exactly one of four cases:

- #emph[True positive (TP)]: actual $= 1$ and predicted $= 1$. The model
  correctly identifies a positive instance.
- #emph[True negative (TN)]: actual $= 0$ and predicted $= 0$. The model
  correctly identifies a negative instance.
- #emph[False negative (FN)]: actual $= 1$ but predicted $= 0$. The model says
  negative, but it is wrong; a positive case was missed.
- #emph[False positive (FP)]: actual $= 0$ but predicted $= 1$. The model says
  positive, but it is wrong; a negative case was incorrectly flagged.

These four counts are arranged into a two-by-two grid called the
#strong[confusion matrix], illustrated in @fig:confusionmatrix. Reading the
matrix row by row (grouped by actual class) or column by column (grouped by
predicted class) immediately reveals how the model's errors are distributed
between the two kinds of mistakes.

// rendered_images:begin
// ```tikz
// % Draw matrix
// \draw[thick] (0,0) rectangle (4,4);
// \draw[thick] (0,2) -- (4,2); % horizontal middle
// \draw[thick] (2,0) -- (2,4); % vertical middle
// 
// % Labels for actual class
// \node[rotate=90] at (-0.8,3) {act = 1};
// \node[rotate=90] at (-0.8,1) {act = 0};
// 
// % Labels for predicted class
// \node at (1,4.3) {pred = 1};
// \node at (3,4.3) {pred = 0};
// 
// % Cell labels
// \node at (1,3) {\textbf{TP}};
// \node at (3,3) {\textbf{FN}};
// \node at (1,1) {\textbf{FP}};
// \node at (3,1) {\textbf{TN}};
// ```
// label=fig:confusionmatrix
// caption=Diagram illustrating Confusion Matrix
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.5-ML_Techniques_Model_Evaluation.typ.figs/Lesson02.5-ML_Techniques_Model_Evaluation.2.png"),
  caption: [Diagram illustrating Confusion Matrix],
) <fig:confusionmatrix>
// render_images:end

The confusion matrix is more than a bookkeeping device: its four cells can be
aggregated into the two most widely used classification metrics,
#emph[precision] and #emph[recall], each of which highlights a different axis of
model quality. Precision asks "of everything the model called positive, how much
really was?" while recall asks "of everything that truly was positive, how much
did the model find?" Both are direct functions of the TP, FP, and FN counts, so
the confusion matrix is the single structure from which nearly all
binary-classification evaluation flows.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:385 '* Precision vs Recall: Definition'
// Slide: Precision vs Recall: Definition
#strong[Precision vs Recall: Definition]

Throughout this discussion we assume that $y = 1$ encodes the rare event, the
class we most care about detecting correctly.

#strong[Precision] measures how often a positive prediction is actually correct.
Formally, it is the conditional probability of a true positive given that the
model predicted positive:

$
  "precision" := Pr("TP" | "pred = 1") = frac(|"pred = 1" and "act = 1"|, |"pred = 1"|) = frac("TP", "TP" + "FP")
$

A model with high precision rarely cries wolf: when it flags an instance as
positive, that flag is usually right. The denominator counts everything the
model labeled positive, so every false positive drags precision down.

#strong[Recall] measures how often the model catches actual positives. It is the
conditional probability of a true positive given that the actual label is
positive:

$
  "recall" := Pr("TP" | "act = 1") = frac("TP", |"act = 1"|) = frac("TP", "TP" + "FN")
$

A model with high recall misses very few real positives: it finds most of the
needles in the haystack. Here the denominator counts every truly positive
instance, so every false negative drags recall down.

Both metrics are conditional probabilities that measure the fraction of true
positives, but they condition on different events. Precision conditions on the
model's prediction ($"pred" = 1$), asking "of everything I flagged, how much was
real?" Recall conditions on the ground truth ($"act" = 1$), asking "of
everything that was real, how much did I find?" A useful mnemonic is that
#emph[precision] starts with "pre" for #emph[prediction], while #emph[recall]
contains "a" for #emph[actual]. Keeping this distinction clear is essential,
because improving one often comes at the expense of the other: a model that
predicts positive very sparingly can achieve near-perfect precision while
missing many true cases (low recall), and a model that predicts positive
liberally can achieve near-perfect recall while generating many false alarms
(low precision).

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:414 '* Precision / Recall as Quality / Quantity'
// Slide: Precision / Recall as Quality / Quantity
#strong[Precision / Recall as Quality / Quantity]

Precision and recall find heavy use in information retrieval. Consider a search
engine that returns 30 pages in response to a query, but only 20 of those pages
are actually relevant. The precision is $20 / 30 = 2 / 3$: two-thirds of what
the engine returned was useful. Now suppose there were 60 relevant pages in
total, meaning 40 relevant pages were missed. The recall is $20 / 60 = 1 / 3$:
the engine found only one-third of everything it should have found.

#strong[Increasing precision] means that when the model predicts a positive
outcome, it is more likely to be correct. Higher precision corresponds to fewer
false positives. In a spam email detection system, for instance, "precision is
90%" means that 90% of the emails the filter flagged as spam truly are spam.
Precision therefore measures the #emph[quality] of positive predictions.

#strong[Increasing recall] means the model captures more of the actual positive
instances. Higher recall corresponds to fewer false negatives. In the same spam
detection system, "recall is 80%" indicates that 80% of all actual spam emails
were correctly identified as spam, while the remaining 20% slipped through to
the inbox. Recall therefore measures the #emph[quantity], or coverage, of
positive predictions.

These two goals are in tension: aggressively labeling more emails as spam raises
recall but risks lowering precision as legitimate messages get caught, while a
conservative filter keeps precision high but lets more spam through, lowering
recall. @fig:precisionrecallroc illustrates this tradeoff visually, showing how
precision and recall shift as the decision threshold changes.

#figure(
  image(
    "../lectures_source/figures/L02.5.Precision_Recall_ROC.png",
    width: 80%,
  ),
  caption: [Precision Recall ROC],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:precisionrecallroc>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:439 '* Precision / Recall for Trivial Classifiers'
// Slide: Precision / Recall for Trivial Classifiers
#strong[Precision / Recall for Trivial Classifiers]

Consider two degenerate classifiers to see why precision and recall, taken
together, guard against trivial strategies.

A classifier that #strong[always predicts the majority class] never produces a
true positive, since it never predicts the rare class at all. Both precision and
recall collapse to zero:

$
  "precision" & equiv Pr("TP" | "pred" = 1) = 0 quad & "(since TP" = 0")" \
     "recall" & equiv Pr("TP" | "act" = 1) = 0 quad  & "(since TP" = 0")"
$

A classifier that #strong[always predicts the rare class] takes the opposite
shortcut. Because it never withholds a positive prediction, every actually
positive instance is caught, giving perfect recall. Precision, however,
plummets: every sample is called positive, so the denominator equals the full
dataset size $n$, and the numerator is just the count of truly positive cases:

$
  "recall" &= 1 quad &"(since FN" = 0")" \
  "precision" &equiv Pr("TP" | "pred" = 1) = frac("TP", "TP + FP") = frac(hash "(y = 1)", n) = Pr("pos") approx 0
$

The numerator $hash (y = 1)$ is tiny relative to $n$ precisely because the
positive class is rare, so precision is near zero.

The key takeaway is that any trivial classifier will drive at least one of
precision or recall close to zero. A model that genuinely identifies the rare
class must keep both metrics respectably high, which is exactly why the two are
reported as a pair (or combined into an F-score) rather than examined in
isolation.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:473 '* Trading Off Precision and Recall'
// Slide: Trading Off Precision and Recall
#strong[Trading Off Precision and Recall]

In theory, the goal is to increase #strong[precision] and #strong[recall]
simultaneously. In practice, a probabilistic classifier outputs a score, and you
choose a threshold θ above which you predict the positive class. Moving that
threshold lets you trade one metric for the other. Recall that precision is
$"TP" / |"pred" = 1|$, the fraction of positive predictions that are correct,
while recall is $"TP" / |"act" = 1|$, the fraction of actual positives that the
classifier finds.

Raising θ means the classifier only predicts 1 when it is more confident, so the
positive predictions it does make are more likely correct: precision goes up.
Lowering θ means the classifier predicts 1 more liberally, reducing the chance
of missing a rare positive event: recall goes up. The two objectives pull in
opposite directions, and the right balance depends on the application. A spam
filter can tolerate a few false positives (lower precision) if it catches nearly
all spam (high recall), whereas a medical screening test that triggers an
expensive follow-up procedure may need high precision to avoid unnecessary
costs.

These threshold-based metrics give genuine insight into a classifier's behavior.
A confusion matrix, for instance, reveals exactly where errors concentrate, and
precision and recall together prevent you from mistaking a trivial classifier
(one that always predicts the majority class) for a useful one, since that
trivial classifier will have zero recall on the minority class. The tradeoff is
that you now have two numbers instead of one, which makes comparing classifiers
less straightforward. If classifier A has higher precision but lower recall than
classifier B, which is better? Composite measures such as the #emph[F-score]
(typically the harmonic mean of precision and recall) or the #emph[area under
  the ROC curve] (AUC) compress the tradeoff back into a single number, making
comparison possible at the cost of choosing how to weight the two concerns.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:495 '* Precision-Recall Curves'
// Slide: Precision-Recall Curves
#strong[Precision-Recall Curves]

A #strong[precision-recall curve] shows the trade-off between precision and
recall for a classifier as its decision threshold θ varies. For a model like
logistic regression, each threshold value produces a different pair of precision
and recall scores; plotting all such pairs on a plane with recall on the
horizontal axis and precision on the vertical axis traces out the curve.

Several features of the precision-recall plane help compare classifiers. The
ideal classifier, achieving both perfect precision and perfect recall, sits in
the top-right corner of the plot. When one curve lies entirely above another,
the higher curve represents a strictly better classifier: at every level of
recall it delivers higher precision. The baseline for comparison is a horizontal
line drawn at the fraction of positive examples in the dataset, which
corresponds to the performance of a random classifier. Any curve that stays
above this baseline is doing better than chance, while a curve that dips below
it is performing worse than random guessing on at least part of the operating
range.

@fig:precisionrecallcurves illustrates these properties, showing how different
classifiers trace out curves at different heights in the precision-recall plane
and how the random baseline provides a floor for useful performance.

// rendered_images:begin
// ```tikz[width=55%]
// % Precision-Recall curve
// \draw[thick, ->] (0,0) -- (8,0);
// \draw[thick, ->] (0,0) -- (0,6);
// 
// \node[below] at (4,-0.3) {Recall};
// \node[left] at (-0.3,3) {Precision};
// 
// % Axis labels
// \node[below] at (0,-0.5) {0};
// \node[below] at (8,-0.5) {1};
// \node[left] at (-0.4,0) {0};
// \node[left] at (-0.4,6) {1};
// 
// % Baseline (random classifier)
// \draw[dashed, gray] (0,1.5) -- (8,1.5);
// \node[gray, right] at (8,1.5) {Baseline (random)};
// 
// % Good classifier curve
// \draw[thick, blue, smooth] (0.5,5.8) to (1.5,5.5) to (3,4.8) to (5,3.5) to (7,2.2) to (8,1.6);
// \node[blue] at (5.5,5.5) {\small Good classifier};
// 
// % Excellent classifier curve
// \draw[thick, green, smooth] (0.2,5.9) to (1,5.95) to (2,5.9) to (4,5.5) to (6,4.2) to (7.8,2.8);
// \node[green] at (2,6.5) {\small Excellent classifier};
// 
// % Best point
// \draw[fill=red] (0.3,5.95) circle (0.15);
// \node[red, above] at (0.3,6.3) {Best};
// ```
// label=fig:precisionrecallcurves
// caption=Diagram illustrating Precision-Recall
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.5-ML_Techniques_Model_Evaluation.typ.figs/Lesson02.5-ML_Techniques_Model_Evaluation.3.png"),
  caption: [Diagram illustrating Precision-Recall],
) <fig:precisionrecallcurves>
// render_images:end
Curves

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:541 '* ROC Curves'
// Slide: ROC Curves
#strong[ROC Curves]

A #strong[ROC curve] (short for "Receiver Operating Characteristic") plots the
true positive rate against the false positive rate as the classification
threshold θ varies. The true positive rate (also called recall) is defined as
$"TPR" = "TP" / ("TP" + "FN")$, measuring the fraction of actual positives the
classifier correctly identifies. The false positive rate is
$"FPR" = "FP" / ("FP" + "TN")$, measuring the fraction of actual negatives the
classifier incorrectly flags as positive. By sweeping the decision threshold
from its most permissive to its most restrictive value, each setting produces
one (FPR, TPR) pair, and connecting these pairs traces the ROC curve.

The curve lives on a unit square where the x-axis is FPR and the y-axis is TPR.
The diagonal line $y = x$ represents the performance of a classifier that
guesses randomly: at every threshold it accepts positives and negatives in equal
proportion, so its true positive rate always matches its false positive rate.
Any curve that bows above this diagonal does better than chance, and the ideal
classifier sits at the top-left corner of the plot, where TPR = 1 and FPR = 0,
meaning it catches every positive while producing no false alarms. The area
under the ROC curve (AUC) summarizes this in a single number: an AUC of 0.5
corresponds to random guessing, and an AUC of 1.0 corresponds to perfect
separation.

ROC curves have a useful invariance property: because TPR is normalized by the
total number of actual positives and FPR is normalized by the total number of
actual negatives, the curve's shape does not shift when the class balance
changes. A dataset with 1% positives and one with 50% positives can produce the
same ROC curve if the classifier's discriminative power is the same. This makes
ROC analysis a reliable tool for comparing models across datasets with different
prevalence rates. On the other hand, when the positive class is rare,
precision-recall curves are often more informative. Precision is sensitive to
the absolute number of false positives relative to true positives, so a small
number of false alarms that barely moves the FPR can still devastate precision.
In such imbalanced settings, a model can look excellent on a ROC curve while
performing poorly in practice, and the precision-recall curve exposes that gap.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:560 '* Area Under the Curve'
// Slide: Area Under the Curve
#strong[Area Under the Curve]

#strong[AUC] is the _area under the precision-recall curve_, a single scalar
that summarizes classifier performance by integrating precision and recall over
every possible decision threshold. A higher AUC indicates that the model
achieves better separation between the positive and negative classes across the
full range of operating points. When the AUC sits near the positive-class
_prevalence_, the classifier is doing no better than random guessing, because a
naive model that predicts the positive class at the base rate would reach that
same score. As AUC approaches 1.0, the model demonstrates strong discriminative
ability at nearly every threshold.

This metric offers several practical advantages. It condenses the entire
precision-recall tradeoff into a single number, making it straightforward to
rank competing models against one another. Because the integration sweeps over
all thresholds, there is no need to commit to a specific cutoff before
evaluating performance. That property is especially valuable when working with
imbalanced datasets, where a fixed threshold chosen on one class distribution
can be misleading on another.

Consider a clinical screening task in which a model must distinguish patients
who have a disease from those who do not. The AUC tells clinicians how well the
model separates the two groups _in aggregate_: a high AUC means that, for most
threshold choices, the model will assign higher risk scores to truly diseased
patients than to healthy ones. @fig:aucroccurve illustrates a typical ROC curve
whose shaded area corresponds to the AUC; a curve that hugs the upper-left
corner encloses more area and reflects stronger overall discrimination.

#figure(
  image("../lectures_source/figures/L02.5.AUC_ROC_Curve.png", width: 80%),
  caption: [AUC ROC Curve],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:aucroccurve>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:580 '* F-Score'
// Slide: F-Score
#strong[F-Score]

The #strong[F-score] is defined as the harmonic mean of precision and recall:

$ "F-score" eq frac(2, frac(1, P) + frac(1, R)) = 2 frac(P dot.op R, P + R) $

This single number captures the balance between precision and recall. Trivial
classifiers that achieve either $P = 0$ or $R = 0$ receive an F-score of exactly
0, while a perfect classifier with $P = R = 1$ achieves an F-score of 1. For the
F-score to be large, both precision and recall must be high simultaneously; a
strong value in one cannot compensate for a weak value in the other.

Why not simply average precision and recall with an arithmetic mean? Consider a
classifier that always predicts the positive class. Its recall is perfect
($R = 1$) because it never misses a positive example, but its precision is near
zero ($P approx 0$) when the positive class is rare. The arithmetic mean
$frac(P + R, 2) approx frac(1, 2)$ makes this trivial classifier look passable.
That is misleading: a classifier with no discriminative ability should score
close to 0. The harmonic mean solves this problem. Because it is dominated by
the smaller of its two inputs, the F-score for this degenerate classifier stays
close to 0, correctly flagging it as useless. This property makes the F-score a
far more reliable summary than the ordinary average whenever precision and
recall diverge sharply.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:602 '## Model Selection'
// Slide: Model Selection
== Model Selection

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:604 '* The Problem of Model Selection'
// Slide: The Problem of Model Selection
#strong[The Problem of Model Selection]

#strong[Model selection] is the process of choosing the best model from a set of
candidates based on their performance, particularly when multiple hypotheses can
explain the same data equally well. In practice, a machine learning pipeline
involves many decisions that go beyond simply fitting parameters to a training
set; the practitioner must decide which structural choices yield the most
reliable predictions on unseen data.

Several components typically require selection across candidate models:

- #emph[Set of features]: from a pool of, say, 100 available variables, only a
  subset may be informative, and choosing which to include is itself a modeling
  decision.
- #emph[Learning algorithms]: even for a fixed architecture such as a neural
  network, the training procedure (optimizer, learning rate schedule, batch
  size) can vary.
- #emph[Model types]: fundamentally different model families may be compared,
  for instance linear regression against a support vector machine.
- #emph[Model complexity]: within a single family, complexity can be tuned, such
  as restricting polynomials to degree $d < 10$.
- #emph[Regularization strength]: values of the regularization parameter (e.g.,
  0.01, 0.1, 1.0) control the tradeoff between fitting the training data closely
  and keeping the model simple enough to generalize.

The standard technique for evaluating these choices is cross-validation, which
estimates how well each candidate generalizes by repeatedly partitioning the
data into training and validation folds. The metrics used for comparison, such
as accuracy, precision, and recall, are computed on the held-out folds rather
than on the data the model was trained on, giving a less biased estimate of
real-world performance. By systematically varying the components listed above
and comparing cross-validated scores, model selection turns what would otherwise
be an ad hoc series of judgment calls into a principled, reproducible search
over the space of candidate pipelines.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:625 '* Model Selection Process'
// Slide: Model Selection Process
#strong[Model Selection Process]

The standard model selection procedure begins by partitioning the available data
into three disjoint subsets, typically in a 60-20-20 ratio: a training set
$D_"train"$, a validation set $D_"val"$, and a test set $D_"test"$. With this
partition in hand, the process unfolds in a clear sequence of steps:

1. Learn $N$ candidate hypotheses $g_1, dots, g_N$ on $D_"train"$ alone.
2. Evaluate each hypothesis on $D_"val"$, producing validation errors
  $E_"val"^((1)), dots, E_"val"^((N))$.
3. Select the model $g_m$ that achieves the minimum validation error
  $E_"val"^((m))$.
4. Estimate the chosen model's true out-of-sample performance using the held-out
  test set: $E_"test" approx E_"out"$.
5. Retrain the selected model architecture on all available data,
  $D = D_"train" union D_"val" union D_"test"$, to produce the final hypothesis
  $g_m^*$.

The reason for retraining at the end is straightforward: once model selection is
complete and a fair performance estimate has been recorded, there is no further
need to hold data back. The final model benefits from every example the
practitioner has, which generally reduces its out-of-sample error below what the
validation-stage model could achieve. The test-set estimate recorded in step 4
remains a trustworthy, unbiased measure of generalization because it was
computed before the test data touched the learning algorithm.
@fig:modelselectionprocess illustrates this full pipeline, from the initial
three-way split through candidate training, validation-based selection, test-set
evaluation, and final retraining on the combined dataset.

// rendered_images:begin
// ```tikz[width=65%]
// % Model Selection Process Diagram
// % Horizontal layout showing data splitting and evaluation flow
// 
// % Data sources
// \node[rectangle, draw, thick, fill=blue!30] at (1,3) {Dataset};
// 
// % Splitting
// \draw[thick, ->] (1,2.7) to (0.5,2.2);
// \draw[thick, ->] (1,2.7) to (1.5,2.2);
// 
// % Training data
// \node[rectangle, draw, thick, fill=blue!20] at (0.5,1.5) {$D_{train}$ (60\%)};
// 
// % Validation data
// \node[rectangle, draw, thick, fill=green!20] at (1.5,1.5) {$D_{val}$ (20\%)};
// 
// % Learning process
// \draw[thick, ->] (0.5,1.2) to (0.5,0.8);
// \node[above, font=\small] at (0.7,1.0) {Learn N models};
// 
// % Learned models
// \node[rectangle, draw] at (0.5,0.2) {$g_1, ..., g_N$};
// 
// % Evaluation on validation
// \draw[thick, ->] (1.2,1.2) to (1.2,0.8);
// \node[above, font=\small] at (1.4,1.0) {Evaluate};
// 
// % Selection
// \node[rectangle, draw, fill=yellow!20] at (1.2,0.2) {Best: $g_m$};
// 
// % Test set
// \node[rectangle, draw, thick, fill=red!20] at (3,1.5) {$D_{test}$ (20\%)};
// 
// % Final evaluation
// \draw[thick, ->] (1.7,0.2) to (2.5,0.2);
// \draw[thick, ->] (2.9,1.2) to (2.9,0.5);
// 
// % Final performance
// \node[rectangle, draw, fill=red!30, thick] at (3,0.2) {$E_{test}(g_m)$};
// 
// % Retraining
// \draw[thick, ->] (1,2.5) to (4.5,2.5);
// \draw[thick, ->] (4.5,2.5) to (4.5,1.2);
// 
// % Full data retrain
// \node[rectangle, draw] at (4.5,1.5) {$D_{train+val+test}$};
// \draw[thick, ->] (4.5,1.2) to (4.5,0.5);
// \node[rectangle, draw, fill=purple!30, thick] at (4.5,0.2) {Deploy: $g_m^*$};
// ```
// label=fig:modelselectionprocess
// caption=Diagram illustrating Model Selection
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.5-ML_Techniques_Model_Evaluation.typ.figs/Lesson02.5-ML_Techniques_Model_Evaluation.4.png"),
  caption: [Diagram illustrating Model Selection],
) <fig:modelselectionprocess>
// render_images:end
Process

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:687 '* Model Selection as Learning'
// Slide: Model Selection as Learning
#strong[Model Selection as Learning]

Selecting the model with the smallest $E_("val")$ is itself a #strong[form of
  learning]. The hypothesis set is the finite collection $\{g_1, dots, g_N\}$ of
candidate models, the training data is the validation set $D_("val")$, and the
learning algorithm simply picks the model $g_m$ that performs best on that data.
Viewing model selection through this lens, as shown in
@fig:modelselectionaslearning, makes it clear why the validation error alone
cannot be trusted as a final performance estimate.

// rendered_images:begin
// ```tikz[width=55%]
// % Bias in model selection: E_val vs E_out relationship
// \draw[thick, ->] (0,0) -- (10,0);
// \draw[thick, ->] (0,0) -- (0,7);
// 
// \node[below] at (5,-0.3) {Model Complexity};
// \node[left] at (-0.3,3.5) {Error};
// 
// % Axis labels
// \node[below] at (0,-0.5) {0};
// \node[below] at (10,-0.5) {$\lambda$ (regularization)};
// \node[left] at (-0.4,0) {0};
// 
// % E_out curve (true error)
// \draw[thick, red, smooth] (1,3) to (3,2.2) to (5,2) to (7,2.5) to (9,4);
// \node[red, above] at (5,2) {$E_{out}(g_m)$};
// 
// % E_val curve (validation error - lower due to bias)
// \draw[thick, blue, smooth] (1,2) to (3,1.2) to (5,1) to (7,1.5) to (9,3);
// \node[blue, above] at (5,1) {$E_{val}(g_m)$};
// 
// % Gap between curves
// \draw[dashed, gray] (5,1) -- (5,2);
// \node[gray, right] at (5.3,1.5) {Bias from};
// \node[gray, right] at (5.3,1.3) {model selection};
// 
// % Optimal point
// \draw[fill=red] (5,2) circle (0.2);
// \node[red, above] at (5,2.5) {Optimal $\lambda$};
// ```
// label=fig:modelselectionaslearning
// caption=Diagram illustrating Model Selection
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.5-ML_Techniques_Model_Evaluation.typ.figs/Lesson02.5-ML_Techniques_Model_Evaluation.5.png"),
  caption: [Diagram illustrating Model Selection],
) <fig:modelselectionaslearning>
// render_images:end
as Learning

Once a model has been selected, its true performance must still be assessed on a
separate test set $D_("test")$. The validation error $E_("val")(g_m)$ is an
optimistically biased estimate of the out-of-sample error because the selection
process specifically chose the model that looked best on $D_("val")$; formally,
$E_("val")(g_m) < E_("out")(g_m)$. From a theoretical standpoint, the penalty
for searching over a finite hypothesis set of size $N$ using $K$ validation
examples is bounded by

$ E_("out")(g_m) lt.eq E_("val")(g_m) + O(sqrt(log(N) / K)) $

This bound grows only logarithmically in $N$, so even a moderately large model
zoo incurs a mild penalty as long as the validation set is sizeable. When the
number of hypotheses is effectively infinite, for instance when choosing a
continuous regularization parameter λ, the VC dimension replaces $log(N)$ in the
complexity term and provides an analogous guarantee.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:735 '## Ensemble Learning'
// Slide: Ensemble Learning
== Ensemble Learning

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:737 '* Ensemble Learning: Intuition'
// Slide: Ensemble Learning: Intuition
#strong[Ensemble Learning: Intuition]

The core intuition behind ensemble methods is straightforward: a group of weak
learners, each only slightly better than random guessing, can be combined into a
strong learner with substantially higher accuracy. No single model needs to be
perfect; what matters is that the models disagree in useful ways.

#strong[Ensemble learning] combines the outputs of multiple models $X_i$ to
construct a composite model $X^*$ that outperforms any individual $X_i$. The key
mechanism is diversity in predictions: because each constituent model captures a
slightly different aspect of the data or makes errors in different regions of
the input space, aggregating their outputs tends to cancel out individual
mistakes. The result is reduced variance and less overfitting compared to
relying on any single model alone. A helpful analogy is a panel of voting
experts: each expert has blind spots, but a majority vote across the panel is
more reliable than any one opinion.

Consider face detection in computer vision, a task that was notoriously
difficult before roughly 2010. Rather than building a single monolithic
classifier, an ensemble approach breaks the problem into simpler sub-questions.
One detector might ask whether the image region contains eyes; another checks
for a nose; a third verifies that the eyes and nose appear in the correct
spatial arrangement relative to each other. Individually, each of these feature
detectors is a weak learner: checking for eyes alone produces many false
positives (other round, dark regions in an image can look like eyes), and
checking for a nose alone is similarly unreliable. Yet when their outputs are
combined, the ensemble becomes a dependable face detector, because a region that
simultaneously satisfies all three checks is far more likely to be an actual
face than one that satisfies only one.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:755 '* Ensemble Learning: Different Techniques'
// Slide: Ensemble Learning: Different Techniques
#strong[Ensemble Learning: Different Techniques]

#strong[Bagging] (bootstrap aggregating) reduces variance by averaging
predictions from multiple models trained on different random samples of the
original data. The canonical example is the random forest: the algorithm creates
many decision trees, each fitted to a bootstrapped subset of the training set,
and then averages their predictions (for regression) or takes a majority vote
(for classification). Because each tree sees a slightly different sample, the
individual trees' errors are partially uncorrelated, and the average is more
stable than any single tree would be on its own.

#strong[Boosting] attacks the opposite problem: it reduces bias by sequentially
adding models, where each new model focuses on the mistakes its predecessors
made. AdaBoost, for instance, increases the weights of incorrectly classified
data points after each round, so the next learner in the sequence concentrates
on the cases the ensemble still gets wrong. Over many rounds, this drives down
the training error and often produces a highly accurate combined classifier,
though it can overfit if run too long on noisy data.

#strong[Stacking] (stacked generalization) takes a different approach: instead
of averaging or sequencing, it trains a meta-model that learns how best to
combine the outputs of several base models. For example, a stacking ensemble
might use logistic regression as its meta-learner, taking as input the
predictions of a decision tree, a support vector machine, and a neural network.
The meta-model discovers which base learner to trust more in which region of the
input space, often outperforming any single base model or a simple average.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:775 '* When Ensemble Learning Works'
// Slide: When Ensemble Learning Works
#strong[When Ensemble Learning Works]

Ensemble learning succeeds when its constituent models satisfy three key
conditions. First, the individual models must be #emph[substantially different
  from each other]: if every model in the ensemble makes the same mistakes,
combining them offers no advantage over using just one. This diversity can come
from different algorithm families (a decision tree, a neural network, and a
support vector machine), different training subsets, or different feature
representations.

Second, each model must #emph[treat a reasonable percentage of the data
  correctly]. There is a floor below which combination cannot help: if every
classifier in the ensemble hovers around 50% accuracy on a binary task,
aggregating their predictions amounts to aggregating coin flips. The ensemble's
power comes from the fact that each member gets most cases right and only
stumbles on a minority; majority voting or averaging then cancels out those
minority errors.

Third, and most critically, the models must #emph[complement each other]. Each
model should be a specialist in some region of the input space where the others
perform poorly. One classifier might handle noisy inputs well but struggle with
rare classes, while another might excel at rare classes but falter on noise.
When their strengths cover each other's weaknesses, the ensemble as a whole
achieves accuracy that no single member could reach alone. This complementarity
is what distinguishes a genuinely useful ensemble from a collection of redundant
models that all fail on the same hard examples.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:785 '* How to Combine Outputs in Ensemble Learning'
// Slide: How to Combine Outputs in Ensemble Learning
#strong[How to Combine Outputs in Ensemble Learning]

Ensemble methods combine the outputs of multiple base models, and the specific
aggregation strategy depends on the type of prediction task at hand.

For #emph[regression] problems, the ensemble produces a weighted average of the
individual predictions, where the weights might reflect each model's estimated
accuracy or encode a Bayesian prior over model quality. For
#emph[classification], the ensemble takes a weighted vote over the predicted
class labels. Because a tie among voters would leave the ensemble undecided,
classification ensembles typically use an odd number of base models. When the
base classifiers output calibrated class probabilities rather than hard labels,
the ensemble can instead compute a #emph[weighted average of class
  probabilities], which preserves richer information than a simple vote. A more
flexible alternative is to #strong[learn a meta-learner] on top of the base
models, an approach known as #emph[stacking]: the meta-learner takes the base
models' outputs as its own input features and learns how best to combine them,
potentially capturing interactions that a fixed weighting scheme would miss.

To see why even a naive majority vote can help, consider three independent
classifiers that are each correct with probability 0.7. The ensemble predicts
whichever class at least two of the three agree on, so

$
  Pr("majority correct") & = Pr("at least 2 classifiers correct") \
                         & = binom(3, 2) 0.7^2 dot.op 0.3 + 0.7^3 \
                         & = 3 times 0.7^2 times 0.3 + 0.7^3 \
                         & approx 0.78 > 0.7
$

The ensemble's accuracy (≈ 0.78) exceeds every individual model's accuracy
(0.70), even though each voter uses the same information. The key assumption is
#emph[independence of errors]: if two classifiers always fail on the same
examples, their votes add no new information, and the majority vote gains
nothing. In practice, true independence is rare, but ensembles still help
whenever the base models' error patterns are at least partially uncorrelated,
which is why diversity among base learners is a central design goal in ensemble
construction.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:810 '* Ensemble Learning: Pros and Cons'
// Slide: Ensemble Learning: Pros and Cons
#strong[Ensemble Learning: Pros and Cons]

Ensemble methods offer a compelling advantage: by combining hypotheses from
different base models, they effectively expand the hypothesis set $cal(H)$
beyond what any single learner can represent, giving the combined system access
to richer decision boundaries and better approximations of the true target
function.

This expanded capacity comes with real costs, however. Training and evaluating
multiple models is more computationally intensive than fitting a single one,
since each base learner must be built and its predictions aggregated at
inference time. The combined model also sacrifices interpretability: while a
single decision tree or linear model can be read and explained directly, an
ensemble of hundreds of trees or a mixture of diverse classifiers becomes a
black box whose reasoning is difficult to communicate. There is also the risk of
overfitting, since the overall model complexity grows with the number and
diversity of base learners; without careful regularization or validation, the
ensemble can memorize training noise rather than capture genuine patterns.
Finally, ensemble learning sits in fundamental tension with #emph[Occam's
  razor], the principle that simpler explanations should be preferred. Combining
many models into one prediction system is, by definition, not simple, and a
practitioner must weigh whether the accuracy gains justify the added complexity.
In practice, techniques like pruning the ensemble or using structured
combination rules (such as the weighted vote in boosting) help keep this
tradeoff manageable, but the tension never fully disappears.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:821 '### Bagging'
// Slide: Bagging
=== Bagging

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:823 '* Bagging'
// Slide: Bagging
#strong[Bagging]

#strong[Bagging] stands for "Bootstrap AGGregation," a technique that builds a
stronger model by training several learners on resampled versions of the same
dataset and combining their predictions.

The learning procedure works as follows:

1. Draw several training datasets at random from the original data, sampling
  _with replacement_ (the bootstrap step).
2. Train one model on each of these bootstrapped datasets.
3. Combine the individual outputs: for classification, take a majority vote; for
  regression, average the predictions.

The result is a composite model that typically outperforms any single
constituent learner.

Why does this help? The bias-variance decomposition offers a clean explanation.
Each bootstrapped model sees a slightly different sample, so its errors are
partly independent of the others. Averaging over many such models reduces the
_variance_ component of the prediction error without compromising _bias_,
because the individual models are themselves trained on datasets drawn from the
same underlying distribution and remain approximately unbiased. In effect,
bagging mimics the ideal scenario of having access to multiple independent
training sets drawn from the true data-generating process. The bootstrap samples
are not truly independent, since they all come from the same finite dataset, but
the diversity they introduce is enough to yield a meaningful variance reduction
in practice.

@fig:baggingclassifier illustrates the overall pipeline: the original dataset is
resampled into several bootstrap copies, a separate classifier is trained on
each copy, and the individual predictions are aggregated into a single final
output.

#figure(
  image("../lectures_source/figures/L02.5.Bagging_Classifier.png", width: 80%),
  caption: [Bagging Classifier],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:baggingclassifier>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:843 '* Bagging and Instability in Learning Algorithms'
// Slide: Bagging and Instability in Learning Algorithms
#strong[Bagging and Instability in Learning Algorithms]

Bagging delivers the largest gains when the base learners are highly diverse,
which in practice means using models that are sensitive to small changes in the
training data. Non-linear models are ideal candidates precisely because their
decision boundaries can shift substantially from one bootstrap sample to the
next. Beyond relying on the natural variance of the learner, you can also inject
randomization into the learning algorithm on purpose to push the ensemble
members further apart.

#strong[Decision trees] are the canonical example. To maximize diversity, you
disable pruning so that each tree grows deep enough to overfit its particular
bootstrap sample. You also break ties randomly when selecting the best attribute
to split on: two trees trained on the same data but resolving ties differently
can end up with substantially different structures. Combining bagged, unpruned,
randomized trees yields what is known as a #emph[random forest], one of the most
widely used ensemble methods in practice.

#strong[Multilayer perceptrons] offer a different source of diversity. Because
backpropagation converges to a local minimum of the loss surface, starting each
network from a different random weight initialization sends each copy down a
different optimization path. The resulting networks land in different local
minima and therefore learn different internal representations of the same data,
which is exactly the diversity bagging needs.

#strong[Nearest-neighbor classifiers] present an interesting contrast.
Resampling the training set has only a limited effect on a nearest-neighbor
model because changing which copies of a point appear in the bootstrap sample is
roughly equivalent to changing example weights; the decision boundary barely
moves. A more effective strategy is to give each base classifier a random subset
of the input features, so that different classifiers measure distance in
different subspaces and thereby produce genuinely different predictions.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:862 '### Boosting'
// Slide: Boosting
=== Boosting

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:864 '* Boosting'
// Slide: Boosting
#strong[Boosting]

#strong[Boosting] builds models that complement each other, typically using
homogeneous learners drawn from the same hypothesis class $cal(H)$. The core
insight is that strong classifiers can be assembled from weak ones. A decision
tree, for instance, can be constructed out of #emph[decision stumps]: decision
trees with only a single split level. Each stump alone is a poor classifier, but
their sequential combination can achieve high accuracy.

Why does this work? Boosting implements #emph[forward stagewise additive
  modeling]. At each step, a new base learner is trained not on the original
targets but on the #emph[residuals]: the difference between the current
ensemble's predictions and the true values. Each successive model therefore
focuses on exactly the mistakes its predecessors made, and the ensemble steadily
closes the gap.

Boosting does have a notable limitation: it offers no benefit for #emph[linear
  regression]. A linear combination of linear models is still a linear model,
and ordinary least squares already finds the optimal weights in a single step,
so there are no residual patterns for a second linear model to exploit.
@fig:boostingconcept illustrates the general boosting concept, showing how
sequentially added weak learners collectively form a strong predictor.

#figure(
  image("../lectures_source/figures/L02.5.Boosting_Concept.png", width: 80%),
  caption: [Boosting Concept],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:boostingconcept>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:883 '* Adaboost.M1'
// Slide: Adaboost.M1
#strong[Adaboost.M1]

#strong[AdaBoost] (Adaptive Boosting) is a powerful ensemble method widely used
for classification. The core idea is to train a sequence of weak learners, each
one focusing harder on the examples its predecessors got wrong, then combine
their predictions into a single strong classifier.

The method assumes that the learning algorithm can accept weighted examples in
its cost function, so that some training points count more heavily than others
during fitting. If the base learner does not natively support instance weights,
the same effect can be achieved through resampling: drawing a new training set
where misclassified points appear more frequently in proportion to their
weights.

@fig:adaboostdiagram illustrates the overall flow of the AdaBoost procedure,
showing how example weights and model weights interact across successive rounds
of boosting.

#figure(
  image("../lectures_source/figures/L02.5.AdaBoost_Diagram.png", width: 80%),
  caption: [AdaBoost Diagram],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:adaboostdiagram>

The learning procedure follows a clear iterative pattern:

1. Initialize all training examples with equal weight, so the first weak learner
  treats every point as equally important.
2. At each round, train a new classifier using the current weight distribution
  over the examples.
3. Evaluate that classifier's performance and assign it a model weight
  reflecting its overall accuracy: a more accurate learner earns a higher vote
  in the final ensemble.
4. Update the example weights, increasing the weight of points the new
  classifier got wrong and decreasing the weight of those it classified
  correctly. This forces the next round's learner to concentrate on the hard
  cases.
5. Repeat until a predetermined number of rounds is reached or the ensemble's
  training error drops to zero.

The final prediction is a weighted majority vote (for classification) across all
the learners produced during these rounds. Because each successive model is
steered toward the mistakes of the ensemble so far, AdaBoost can achieve high
accuracy even when every individual learner is only slightly better than random
guessing. The adaptive reweighting is what gives the algorithm its name and its
strength: rather than treating all data uniformly, it directs learning capacity
precisely where it is most needed.

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:902 '### Stacking'
// Slide: Stacking
=== Stacking

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:904 '* Stacking'
// Slide: Stacking
#strong[Stacking]

#strong[Stacking] is the idea of learning how to combine models, which need not
even be of the same type. With simple voting or averaging, every base model gets
equal say (or a fixed weight), and there is no principled way to know which
model to trust on which kinds of inputs. Stacking solves this by introducing a
#emph[meta-learner] (called the level-1 model) whose job is to discover how best
to pick or mix the predictions of several base learners (the level-0 models).

The learning procedure has two stages. First, train the level-0 models in the
usual way. Second, build a new training set for the level-1 model using hold-out
predictions from the level-0 stage, much like the validation splits used in
model selection. Concretely, each level-0 model produces predicted values (or,
better yet, predicted probabilities) on data it was not trained on, and those
predictions become the features of the level-1 training set. The level-1 model
then learns from these features how to weight or route the base models' outputs.
Because the meta-learner sits on top of already-expressive base models, it
should itself be simple: a linear model or a shallow decision tree is typical,
keeping the risk of overfitting low. Using probability outputs rather than hard
class labels at level 0 is preferred, because it lets the meta-learner assess
the confidence of each base model, not just its point prediction.
@fig:stackingensemble illustrates this two-tier architecture, showing how
base-model outputs feed into the meta-learner that produces the final
prediction.

#figure(
  image("../lectures_source/figures/L02.5.Stacking_Ensemble.png", width: 80%),
  caption: [Stacking Ensemble],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:stackingensemble>

// From: msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd:927 '* Boosting vs Bagging vs Stacking'
// Slide: Boosting vs Bagging vs Stacking
#strong[Boosting vs Bagging vs Stacking]

#figure(
  styled-table(
    headers: ("Aspect", "Bagging", "Boosting", "Stacking"),
    rows: (
      (
        "Combines",
        "Models of same type",
        "Models of same type",
        "Different types",
      ),
      ("Learning", "Independent", "Iterative", "Independent"),
      (
        "Predicting",
        "Uniform or data-driven weights",
        "Learned weights",
        "Learned weights/confidence",
      ),
      (
        "Main Objective",
        "Reduce variance",
        "Reduce bias",
        "Improve generalization via diversity",
      ),
      (
        "Base Learners",
        "Often strong",
        "Often weak",
        "Any type (heterogeneous)",
      ),
      ("Sensitivity to Noise", "Low", "High", "Medium"),
      ("Parallelizable", "Yes", "No (sequential)", "Partially (base models)"),
      ("Meta-model", "Not used", "Not used", "Required"),
      (
        "Examples",
        "Random Forest",
        "AdaBoost, Gradient Boosting",
        "Stacked Generalization, Blending",
      ),
    ),
  ),
  caption: [Comparison of ensemble learning techniques.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
)
