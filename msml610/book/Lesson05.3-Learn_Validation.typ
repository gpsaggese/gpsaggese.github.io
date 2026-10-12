// git_hash=6da0f339-ixn timestamp=20260924_112335
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
  title: "L05.3: Learn-Validation",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L05.3: Learn-Validation")

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:9 '* Roadmap'
// Slide: Roadmap
= Roadmap

This lesson covers the #strong[learn-validation approach], which estimates
out-of-sample error directly by holding out a portion of the data, along with
the fundamental trade-off this creates between training set size and validation
set size. It then introduces #strong[cross-validation] and its variants
(repeated cross-validation, leave-one-out), which reuse data more efficiently by
cycling through $K$ train/validation splits. The chapter closes with the
#strong[bootstrap], a resampling technique for estimating the sampling
distribution of any statistic, not just out-of-sample error.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:20 '# Learn-Validation Approach'
// Slide: Learn-Validation Approach
#strong[Learn-Validation Approach]

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:22 '## Train / Test'
// Slide: Train / Test
= Train / Test

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:24 '* Estimating Out-Of-Sample Error with One Point'
// Slide: Estimating Out-Of-Sample Error with One Point
#strong[Estimating Out-Of-Sample Error with One Point]

Consider picking a single out-of-sample point $(bold(x)', y)$ and measuring how
well a learned hypothesis $h$ performs on it. The validation error for that one
point is

$ E_("val")(h) eq.delta e(h(bold(x)'), y) $

where $e$ is whichever pointwise error function suits the task: the squared
error $(h(bold(x)) - y)^2$ for regression, or the binary indicator
$I[h(bold(x)) eq.not y]$ for classification.

A single held-out point still carries real information: the error on an
out-of-sample point is an #strong[unbiased estimate] of $E_("out")$ #cite(
  "abumostafa2012learning",
). Formally,

$ EE[E_("val")(h)] = EE[e(h(bold(x)), y)] = E_("out") $

The expectation is taken over the random draw of $(bold(x)', y)$ from the same
distribution the training data came from. Since $h$ was fixed before this point
was chosen, the point carries no optimistic bias: it is genuinely new evidence
about how $h$ generalizes.

Unbiasedness, however, does not guarantee precision. The quality of this
single-point estimate depends on the standard error,
$sqrt(VV[e(h(bold(x)), y)])$, which is itself unknown. A single observation can
land far from the true $E_("out")$ even though, on average across many such
draws, it would be exactly right. Reducing that variance is the motivation for
using more than one validation point, as we will see next.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:43 '* Estimating Out-Of-Sample Error with $K$ Points'
// Slide: Estimating Out-Of-Sample Error with $K$ Points
#strong[Estimating Out-Of-Sample Error with $K$ Points]

To improve the estimate of out-of-sample performance, we set aside a
#strong[validation set] of $K$ points $(bold(x)_1, y_1), dots, (bold(x)_K, y_K)$
drawn independently and identically distributed from the same data-generating
process. Because these points were not used during training, evaluating the
learned hypothesis $h$ on them gives an honest picture of how well $h$
generalizes.

The #strong[validation error] is the average pointwise error over this held-out
set:

$ E_("val")(h) eq.delta 1/K sum_(i=1)^K e(h(bold(x)_i), y_i) $

This quantity is an #strong[unbiased estimate] of the true out-of-sample error,
meaning that its expectation equals the out-of-sample error exactly:

$ EE[E_("val")(h)] = E_("out")(h) $

Unbiasedness alone does not guarantee a tight estimate; we also need the
variance to be small. Because the validation points are independent, the
covariance terms between distinct pointwise errors vanish, and the variance of
the validation error shrinks by a factor of $K$:

$
  "Var"[E_("val")(h)] = 1/K^2 sum_i "Var"[e(h(bold(x)_i), y_i)] = (K sigma^2)/K^2 = sigma^2/K
$

The standard error of the validation estimate is therefore
$sigma slash sqrt(K)$. Larger validation sets produce tighter estimates, but at
the cost of leaving fewer points for training. This tradeoff between a reliable
error estimate and a well-trained model is a recurring tension in model
selection, and it motivates techniques such as cross-validation that reuse data
across multiple splits.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:71 '* Trade-Off Between Training and Validation Set'
// Slide: Trade-Off Between Training and Validation Set
#strong[Trade-Off Between Training and Validation Set]

With a finite dataset of $N$ points, every point assigned to the validation set
$D_("val")$ is one fewer point available for training. Formally, if the
validation set contains $K$ points, the split is:

$ D_("val") = {K "points"}, quad D_("train") = {N - K "points"} $

This creates a fundamental tension. Increasing $K$ tightens the gap
$|E_("val") - E_("out")|$, giving a more reliable estimate of out-of-sample
performance. But a larger $K$ simultaneously shrinks the training set to $N - K$
points, which drives both $E_("in")$ and $E_("out")$ upward because the model
has less data to learn from. Taken to the extreme, a very large validation set
produces a reliable estimate of a bad number: you know with high confidence that
your model performs poorly, which is not especially useful.

#figure(
  image(
    "../lectures_source/figures/L05.3.Train_Validation_Set_Trade_Off.png",
    width: 80%,
  ),
  caption: [Train Validation Set Trade Off],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:trainvalidationsettradeoff>

As @fig:trainvalidationsettradeoff illustrates, the sweet spot lies between
these two extremes. The standard rule of thumb is a 70/30 or 80/20 split between
training and validation data. This heuristic balances having enough training
examples for the model to learn a reasonable hypothesis against reserving enough
held-out examples for the validation error to be a trustworthy proxy for true
out-of-sample performance.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:106 '* Error From VC Analysis vs Learn-Validation Approach'
// Slide: Error From VC Analysis vs Learn-Validation Approach
#strong[Error From VC Analysis vs Learn-Validation Approach]

In general, the relationship between in-sample and out-of-sample error can be
expressed as

$ E_("out")(h) = E_("in")(h) + "generalization error" $

The central question in learning theory is how to estimate that generalization
error term. Two complementary approaches address this from different angles.

#emph[VC analysis] takes a theoretical route: it bounds the generalization error
as an "overfit penalty" derived from the complexity of the hypothesis set. The
bound grows with the VC dimension and shrinks with the number of training
examples, giving a worst-case guarantee that holds uniformly over every
hypothesis in the set. This is valuable for understanding how capacity and
sample size interact, but the bound is often loose in practice because it must
cover every possible dataset, not just the one at hand.

#emph[Learn-validation] takes an empirical route instead: it estimates
$E_("out")$ directly by holding out a portion of the data that the learning
algorithm never sees during training. The validation error $E_("val")$ then
serves as an unbiased estimate of $E_("out")$:

$ E_("val") approx E_("out") $

This held-out strategy appears at two distinct points in the modeling workflow.
First, a #emph[validation set] is used during model development to select
hyperparameters: the algorithm trains on the remaining data under each candidate
setting, and whichever setting yields the lowest validation error is chosen.
Second, a #emph[test set], kept entirely separate from both training and
validation, is reserved to estimate the final performance of the fully selected
model. Conflating these two roles (reusing validation data as the test set)
introduces optimistic bias, because the validation set already influenced model
selection.

#figure(
  styled-table(
    headers: ("Approach", "Estimates via", "Basis"),
    rows: (
      (
        "VC analysis",
        "Overfit penalty from hypothesis set complexity",
        "Theoretical bound",
      ),
      (
        "Learn-validation",
        [Held-out data ($E_("val") approx E_("out")$)],
        "Empirical measurement",
      ),
    ),
    bold-first-col: true,
  ),
  caption: [VC analysis and learn-validation as two ways to estimate
    generalization error.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:vcvsvalidation>

@tab:vcvsvalidation summarizes the distinction: VC analysis provides a
theoretical bound rooted in hypothesis-set complexity, while learn-validation
provides an empirical measurement grounded in held-out data. In practice,
learn-validation is the workhorse of applied machine learning, since it gives
tighter, data-specific estimates, but VC analysis remains essential for
understanding why generalization is possible at all and for guiding the design
of learning algorithms.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:130 '* Reusing Validation / Test Set for Training'
// Slide: Reusing Validation / Test Set for Training
#strong[Reusing Validation / Test Set for Training]

Any data used during learning, whether for model training or model selection,
carries an inherent optimistic bias: it has already influenced the choices that
produced the model, so it cannot serve as a fair judge of that model's quality.
This is why $D_("val")$ and $D_("test")$ must never be used for training, and
$D_("train")$ must never be used for evaluation. Mixing these roles destroys the
statistical independence that makes an estimate trustworthy.

Once all research decisions are finalized, however, that separation has served
its purpose. At deployment time, every available data point can and should be
folded back into a single training run to squeeze maximum performance out of the
chosen model form.

The full validation workflow proceeds in four steps:

1. #emph[Train on $N - K$ points.] Hold out $K$ observations and learn a
  preliminary hypothesis $g^-$ from the remaining $N - K$ data points.
2. #emph[Estimate out-of-sample error.] Evaluate $g^-$ on the $K$ held-out
  points to obtain $E_("val") (g^-)$, which serves as an unbiased estimate of
  $E_("out") (g^-)$.
3. #emph[Retrain on all $N$ points.] Once the model form is finalized, combine
  the training and validation (and test) data and learn a final hypothesis $g$
  from the full $N$ observations. Because $g$ sees strictly more data than
  $g^-$, it fits at least as well: $E_("val") (g) < E_("val") (g^-)$.
4. #emph[Deliver two artifacts.] Hand the customer the final hypothesis $g$
  together with the validation-set estimate $E_("val") (g^-)$ as a conservative
  upper bound on $g$'s true out-of-sample performance. The bound is conservative
  precisely because it was measured on the weaker model $g^-$, not on the
  improved $g$.

@fig:usevalidationsettotrain illustrates this workflow, showing how the held-out
validation partition first provides an honest error estimate and then rejoins
the training data for the final model fit.

#figure(
  image(
    "../lectures_source/figures/L05.3.Use_Validation_Set_To_Train.png",
    width: 80%,
  ),
  caption: [Use Validation Set To Train],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:usevalidationsettotrain>

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:164 '* Learn-Validation Approach: Pros and Cons'
// Slide: Learn-Validation Approach: Pros and Cons
#strong[Learn-Validation Approach: Pros and Cons]

Validation offers a practical way to estimate $E_("out")$ by computing
$E_("val")$ on held-out data, and unlike VC-dimension analysis, it requires no
complex theoretical machinery: just train on one portion of the data and measure
error on the rest. The tradeoff is that validation forces a compromise in how
the data is allocated. Every example set aside for validation is one fewer
example available for learning, so the model trained on the reduced dataset may
be weaker than one trained on all the data. Compounding this, both the learned
model and the validation error $E_("val")$ depend on which specific examples end
up in each partition. A different random split can yield a noticeably different
error estimate, making the result somewhat unstable, particularly when the total
dataset is small.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:174 '## Cross-Validation'
// Slide: Cross-Validation
= Cross-Validation

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:176 '* Cross-validation'
// Slide: Cross-validation
#strong[Cross-validation]

The $K$-fold cross-validation procedure #cite("kohavi1995crossvalidation")
begins by dividing the full dataset of $N$ samples into $K$ equally sized folds,
each containing roughly $N / K$ samples. These folds should reflect the overall
dataset statistics; for a classification task, this typically means using
#emph[stratified sampling] so that each fold preserves the class distribution of
the whole dataset.

The algorithm then runs $K$ iterations. In iteration $i$, folds
$1, dots, i - 1, i + 1, dots, K$ serve as the training set, giving
$(K - 1) / K dot N$ training points that produce a hypothesis
$g^((-i))(bold(x))$. The held-out fold $i$ serves as the validation set, and the
validation error for that iteration is

$ E_("val")^((i)) = E_("val") [g^((-i))(bold(x))] $

After all $K$ iterations, the final cross-validation estimate is the average of
the $K$ per-fold errors:

$ E_("val") = 1 / K sum_i E_("val")^((i)) $

This average provides a more stable estimate of generalization performance than
any single train/test split, because every sample appears in the validation set
exactly once across the $K$ rounds. @fig:crossvalidation illustrates the
simplest baseline: a single train/test split, where one contiguous block is held
out for testing and the rest is used for training. @fig:crossvalidation-2 then
shows how 5-fold cross-validation rotates the held-out block through all five
positions, so that the entire dataset contributes to both training and
validation. The tighter the folds track the true data distribution, the more
reliable the resulting error bounds become.

// rendered_images:begin
// ```tikz
// \bfseries
//     \def\blockwidth{1.2}
//     \def\blockheight{0.7}
// 
//     % Label
//     \node[anchor=east] at (-0.2, -0.5*\blockheight) {\textbf{Train/Test Split}};
// 
//     % Create 6 blocks
//     \foreach \j in {0,...,5} {
//         \pgfmathsetmacro{\x}{\j * \blockwidth}
//         \pgfmathsetmacro{\y}{0}
// 
//         % Mark first 4 as train, last 2 as test
//         \ifnum\j<4
//             \fill[blue!70] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//             \node at (\x + 0.5*\blockwidth, \y - 0.5*\blockheight) {\textbf{Train}};
//         \else
//             \fill[red!80] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//             \node at (\x + 0.5*\blockwidth, \y - 0.5*\blockheight) {\textbf{Test}};
//         \fi
// 
//         \draw[black] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//     }
// ```
// label=fig:crossvalidation
// caption=Diagram illustrating cross-validation
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.3-Learn_Validation.typ.figs/Lesson05.3-Learn_Validation.1.png",
    width: 70%,
  ),
  caption: [Diagram illustrating cross-validation],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:crossvalidation>
// render_images:end

// rendered_images:begin
// ```tikz
// \bfseries
// \def\blockwidth{1.2}
// \def\blockheight{0.7}
// \def\vspacing{1.2} % vertical spacing between iterations
// \def\nfolds{6}
// 
// \foreach \i in {1,...,5} {
//     % Compute vertical center for this row
//     \pgfmathsetmacro{\ycenter}{-\i * \vspacing - 0.5 * \blockheight}
// 
//     % Label each iteration (vertically centered)
//     \node[anchor=east] at (-0.2, \ycenter) {\textbf{Iteration \i}};
// 
//     \foreach \j in {0,...,4} {
//         \pgfmathtruncatemacro{\jj}{\j + 1}
// 
//         % Compute block position
//         \pgfmathsetmacro{\x}{\j * \blockwidth}
//         \pgfmathsetmacro{\y}{-\i * \vspacing}
// 
//         % Fill and label blocks
//         \ifnum\i=\jj
//             \fill[red!80] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//             \node at (\x + 0.5*\blockwidth, \y - 0.5*\blockheight) {\textbf{Test}};
//         \else
//             \fill[blue!70] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//             \node at (\x + 0.5*\blockwidth, \y - 0.5*\blockheight) {\textbf{Train}};
//         \fi
// 
//         % Draw borders
//         \draw[black] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//     }
// }
// ```
// label=fig:crossvalidation-2
// caption=Diagram illustrating 5-fold cross-validation
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.3-Learn_Validation.typ.figs/Lesson05.3-Learn_Validation.2.png",
    width: 70%,
  ),
  caption: [Diagram illustrating 5-fold cross-validation],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:crossvalidation-2>
// render_images:end

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:278 '* Cross-Validation: Pros and Cons'
// Slide: Cross-Validation: Pros and Cons
#strong[Cross-Validation: Pros and Cons]

Cross-validation offers several genuine advantages. It makes efficient use of
available data, since every example serves in both training and validation roles
across the full set of folds. This produces a better estimate of $E_("val")$
than a single train/validation split, which can be heavily influenced by which
examples happen to land on each side. Additionally, folds can be stratified to
ensure each one mirrors the overall class distribution, reducing the variance
that comes from unlucky partitions.

The tradeoff is computational cost: $K$-fold cross-validation requires $K$
separate learning phases, each training a model from scratch on a different
subset. The final estimate also depends on how the folds were chosen; a
different random partition can yield a noticeably different score. A subtler
issue is that the per-fold validation errors $E_("val")$ are not statistically
independent, because the training sets of any two folds overlap heavily. Two
folds that share most of their training data will produce similar models and,
therefore, correlated errors. Empirically, however, these errors are not
completely correlated either, so averaging across folds still reduces variance
compared to a single split.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:292 '* Repeated Cross-Validation'
// Slide: Repeated Cross-Validation
#strong[Repeated Cross-Validation]

Cross-validation results can vary depending on how the data happens to be
partitioned into folds. A particular random split might place an unusual cluster
of hard examples into one fold, skewing the estimate. To remove this dependency
on any single partition, you can repeat the entire cross-validation procedure
multiple times (for example, ten times) and average the resulting performance
estimates. Each repetition uses a fresh random partition of the data into folds,
so the final averaged score smooths out the variance introduced by any one
unlucky split.

#emph[10 times 10-fold cross-validation] is not the same thing as #emph[1 time
100-fold cross-validation]. In the former, you run ten independent 10-fold
procedures, each with its own random partition, and average
across all one hundred individual fold estimates. In the latter, you split the
data into 100 folds once: each training set uses 99% of the data and each test
set only 1%, which changes the bias-variance profile of the estimate entirely.
The repeated scheme preserves the training-set size characteristic of 10-fold CV
(90% of the data per fold) while reducing estimator variance through repetition,
making it a more robust evaluation strategy in practice.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:300 '* Leave-One-Out Cross-Validation'
// Slide: Leave-One-Out Cross-Validation
#strong[Leave-One-Out Cross-Validation]

#strong[Leave-one-out] (LOO) cross-validation pushes the $K$-fold idea to its
logical extreme: set $K = N$, where $N$ is the total number of examples in the
dataset. Each of the $N$ training sessions holds out exactly one data point for
validation and trains on the remaining $N - 1$ points. Because nearly the entire
dataset is used for training in every fold, each learned hypothesis $g^(-i)$ is
very close to the hypothesis $g$ that would be learned from the full dataset.
That closeness is the key advantage: the training set bias is as small as it can
get.

The validation estimate from any single fold, however, is unreliable. A single
held-out point gives

$ E_("val")[g^(-i)] = e(g^(-i)(bold(x)_i), y_i) approx.not E_("out")[g^(-i)] $

because one observation tells you almost nothing about out-of-sample
performance. The remedy is to average across all $N$ folds:

$ E_("val") = 1 / N sum_(i=1)^K E_("val")[g^(-i)] $

This average smooths out the noise from individual points and yields a much more
stable estimate of $E_("out")$. As @fig:leaveoneoutcrossvalidation illustrates,
each iteration designates exactly one point as the validation sample while the
rest form the training set, cycling through every point in turn.

// rendered_images:begin
// ```tikz
// \bfseries
// \tikzset{every node/.style={font=\sffamily}}
// \def\blockwidth{0.7}
// \def\blockheight{0.6}
// \def\vspacing{1.0}
// \def\nblocks{6}
// 
// \foreach \i in {1,2} {
//     \pgfmathsetmacro{\ycenter}{-\i * \vspacing - 0.5 * \blockheight}
//     \node[anchor=east] at (-0.3, \ycenter) {\footnotesize Iteration \i};
// 
//     \foreach \j in {1,...,\nblocks} {
//         \pgfmathsetmacro{\x}{(\j - 1) * \blockwidth}
//         \pgfmathsetmacro{\y}{-\i * \vspacing}
//         \ifnum\i=\j
//             \fill[red!80] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//         \else
//             \fill[blue!70] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//         \fi
//         \draw[black] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//     }
// }
// 
// \node[anchor=east] at (-0.3, -3 * \vspacing - 0.5*\blockheight) {\footnotesize ...};
// 
// \pgfmathsetmacro{\ycenterN}{-4 * \vspacing - 0.5 * \blockheight}
// \node[anchor=east] at (-0.3, \ycenterN) {\footnotesize Iteration $N$};
// \foreach \j in {1,...,\nblocks} {
//     \pgfmathsetmacro{\x}{(\j - 1) * \blockwidth}
//     \pgfmathsetmacro{\y}{-4 * \vspacing}
//     \ifnum\j=\nblocks
//         \fill[red!80] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//     \else
//         \fill[blue!70] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
//     \fi
//     \draw[black] (\x, \y) rectangle ++(\blockwidth, -\blockheight);
// }
// ```
// label=fig:leaveoneoutcrossvalidation
// caption=Diagram illustrating leave-one-out cross-validation
// rendered_images:end
// render_images:begin
#figure(
  image(
    "Lesson05.3-Learn_Validation.typ.figs/Lesson05.3-Learn_Validation.3.png",
    width: 70%,
  ),
  caption: [Diagram illustrating leave-one-out cross-validation],
  kind: "figure",
  supplement: [Fig.],
  placement: auto,
) <fig:leaveoneoutcrossvalidation>
// render_images:end

LOO cross-validation is particularly useful when the dataset is small and you
cannot afford to set aside a large fraction for validation. Its cost is
computational: you must train the model $N$ separate times, which can be
prohibitive for large $N$ or expensive-to-fit models. For certain model families
(linear regression, for instance), closed-form shortcuts exist that compute the
LOO error without literally retraining $N$ times, making it practical even at
scale.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:375 '* Leave-One-Out Cross-Validation: Pros and Cons'
// Slide: Leave-One-Out Cross-Validation: Pros and Cons
#strong[Leave-One-Out Cross-Validation: Pros and Cons]

Leave-one-out cross-validation (LOOCV) uses every available data point for
training except one, which offers the maximum possible training set size in each
fold and makes the procedure entirely deterministic: there is no random fold
assignment to worry about, so repeating the experiment always yields the same
result.

The cost, though, is computational: because the number of folds equals the
number of data points, the model must be retrained once per observation, which
can be prohibitive for large datasets or expensive learners. LOOCV also cannot
be stratified, since each validation fold contains exactly one example and
therefore cannot preserve class proportions. A subtler issue is that the
training sets across folds overlap almost completely (each pair shares all but
two points), which induces high correlation among the individual error estimates
and can inflate the variance of the overall cross-validation estimate.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:386 '## Bootstrap'
// Slide: Bootstrap
= Bootstrap

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:388 '* Bootstrap'
// Slide: Bootstrap
#strong[Bootstrap]

The #strong[bootstrap validation] method #cite("kohavi1995crossvalidation")
works by drawing $N$ samples with replacement from a dataset that itself
contains $N$ instances; this drawn set becomes the training set. Because
sampling is done with replacement, some original instances will be selected
multiple times while others will never appear. Those never-chosen instances,
called #emph[out-of-bag] samples, form the test set. A well-known probability
argument shows that roughly 63.2% of the original instances end up in the
training set (counting unique instances), leaving approximately 36.8% for
testing.

#figure(
  styled-table(
    headers: (
      "Original indices",
      "Bootstrap sample (with replacement)",
      "Out-of-bag",
    ),
    rows: (
      ("1, 2, 3, 4, 5, 6, 7, 8", "3, 1, 3, 7, 5, 2, 8, 3", "4, 6"),
    ),
    bold-first-col: true,
  ),
  caption: [One bootstrap draw of eight indices, with indices 4 and 6 never
    selected and left out-of-bag.],
  kind: "table",
  supplement: [Table.],
  placement: auto,
) <tab:bootstrapexample>

@tab:bootstrapexample shows one such draw: indices 4 and 6 are never selected
and become the out-of-bag test set. This approach offers a genuine advantage
for small datasets, since sampling with replacement effectively "expands" the
available training data by allowing duplicates. On the other hand, bootstrap validation is less flexible than
$k$-fold cross-validation, and the 63.2% unique-instance ratio means the
training set actually uses a smaller fraction of distinct samples than a
standard 10-fold split (which trains on 90% of the data in each fold). For
datasets large enough to support $k$-fold partitioning, the bootstrap's
duplication benefit rarely outweighs that cost.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:407 '* Bootstrap: Problem'
// Slide: Bootstrap: Problem
#strong[Bootstrap: Problem]

The starting assumption is straightforward: suppose you have a random variable
$X$ drawn from some distribution $F$, and you collect $n$ independent,
identically distributed samples $X_1, X_2, dots, X_n$ from $F$.

The goal is to reason about a #strong[sampling statistic]
$T = g(X_1, dots, X_n)$, which is any function of those $n$ samples. This could
be the sample mean, the median, ordinary least squares regression coefficients,
or any other quantity computed from the data. Specifically, you want to do one
or more of the following:

- #emph[Estimate the distribution of $T$]: find its cumulative distribution
  function $F_T (x)$ or its density $f_T (x)$.
- #emph[Estimate a statistic of $T$]: for instance, its expected value or
  standard error.
- #emph[Construct confidence intervals]: express uncertainty as an interval
  $mu plus.minus epsilon$ that contains the true parameter with a specified
  probability.
- #emph[Calculate standard errors]: quantify the variability $sigma(hat(T)_n)$
  of the estimator across repeated samples.
- #emph[Perform hypothesis testing]: determine whether the observed value of $T$
  is consistent with a null hypothesis about $F$.

All of these tasks require knowing, or at least approximating, how $T$ varies
from one sample of size $n$ to another. When $F$ is known and simple (say, a
normal distribution), you can often derive these quantities analytically. The
challenge arises when $F$ is unknown or when $g$ is complicated enough that no
closed-form sampling distribution exists. That is precisely the setting where
resampling methods such as the bootstrap become essential: they let you
approximate the behavior of $T$ by drawing repeated samples from the data
itself, rather than from a theoretical model of $F$.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:422 '* Bootstrap Procedure: Algorithm'
// Slide: Bootstrap Procedure: Algorithm
#strong[Bootstrap Procedure: Algorithm]

The bootstrap procedure begins with observed data $X_1, dots, X_n$, which serve
as the basis for constructing an estimated population distribution $hat(F)_n$
#cite("hastie2009elements"). Since the true population distribution $F$ is
unknown, $hat(F)_n$ acts as a stand-in: every value in the original sample
receives equal probability $1\/n$, making the empirical distribution function
the best available approximation of the process that generated the data.

From this estimated distribution, the algorithm proceeds by repeating the
following steps $B$ times:

- Draw $n$ samples with replacement from $hat(F)_n$. Because sampling with
  replacement from the observed values $X_i$ is equivalent to sampling from
  $hat(F)_n$, no parametric assumptions about the population shape are needed.
- Compute the sample statistic of interest from each bootstrap sample:
  $T^((i)) = g(X_1^((i)), dots, X_n^((i)))$, where $g$ is whatever function
  defines the statistic (a mean, a median, a regression coefficient, or any
  other quantity).

After all $B$ iterations, the collection of bootstrap statistics
$T^((1)), dots, T^((B))$ forms an empirical distribution $hat(T)$ that
approximates the true sampling distribution of $T$. The final step is to read
off whatever summary is needed from $hat(T)$: a confidence interval (typically
by taking the appropriate percentiles of the $B$ values), a standard error (the
standard deviation of the $B$ values), or a bias estimate (the difference
between the mean of the bootstrap distribution and the original point estimate).
The power of this approach is that no closed-form formula for the sampling
distribution of $T$ is required; the resampling procedure lets the data
themselves reveal how much variability the statistic carries.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:438 '* Bootstrap: Pros'
// Slide: Bootstrap: Pros
#strong[Bootstrap: Pros]

The bootstrap earns its keep by demanding far fewer assumptions than classical
approaches. There is no need for simplifying
assumptions that yield closed-form formulas, and the data need not follow a
Gaussian distribution. It also offers broad generality: the same resampling
logic applies to any sample statistic, even non-linear ones such as the median,
for which analytic standard errors are notoriously difficult to derive.

In essence, the bootstrap replaces math with simulation. It is a
#strong[non-parametric method], meaning it makes no assumption about the
underlying population distribution. It does not rely on large-sample results
like the Central Limit Theorem (CLT) or the Law of Large Numbers (LLN) to
justify its confidence intervals. By shifting the burden from closed-form
derivations to computational resampling, the bootstrap frees data scientists
from complex math, approximations, and asymptotic arguments, letting the data
speak for itself.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:453 '* Bootstrap: Example of Die Rolls'
// Slide: Bootstrap: Example of Die Rolls
#strong[Bootstrap: Example of Die Rolls]

How would you compute the distribution of a sum like $Y = sum_(i=1)^(50) X_i$
when rolling a die 50 times? Sample statistics such as sample means, variances,
or other summaries all share this structure: they are functions
$g(X_1, dots, X_50)$ of the observed data, and understanding their distribution
is the core challenge.

There are three broad strategies for tackling this:

1. #emph[By mathematics]: if the PMF of the die is known exactly, you can derive
  the distribution analytically. The theorem of the lazy statistician (LOTUS)
  gives the mean and variance of $Y$ without first finding its full
  distribution. For the complete PDF, you convolve the individual PDFs of the
  $X_i$, which is exact but becomes unwieldy as the number of terms grows.

2. #emph[By sampling] (either physical or simulated): roll the die 50 times,
  compute the sample statistic, then repeat the entire procedure many times.
  Plotting the resulting values gives an approximate distribution of $Y$. This
  is straightforward when you can generate as many fresh samples as you like.

3. #emph[By bootstrapping]: when only a single set of 50 observed values is
  available and no further sampling is possible, the bootstrap provides a way to
  approximate the sampling distribution by resampling from the data you already
  have.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:471 '* Bootstrap of the Median: Pseudo-Code'
// Slide: Bootstrap of the Median: Pseudo-Code
#strong[Bootstrap of the Median: Pseudo-Code]

#algorithm("Bootstrap Median", [
  *function* `bootstrap_median`($x$, $n_"boot"$): \
  #h(1em) *for* $i arrow.l 1$ *to* $n_"boot"$: \
  #h(2em) $x^* arrow.l$ sample with replacement from $x$ \
  #h(2em) $"median_boot"[i] arrow.l "median"(x^*)$ \
  #h(1em) $m_"median" arrow.l "mean"("median_boot")$ \
  #h(1em) $"se"_"median" arrow.l "std"("median_boot")$ \
  #h(1em) *return* $m_"median"$, $"se"_"median"$
])

The bootstrap procedure for estimating the median works by repeatedly resampling
the original dataset $x$ with replacement. For each of $n_"boot"$ iterations,
the algorithm draws a bootstrap sample $x^*$ of the same size as $x$, computes
the median of that resample, and stores it. After all iterations complete, the
mean of the stored medians serves as the point estimate $m_"median"$, and their
standard deviation provides the bootstrap standard error $"se"_"median"$. This
standard error quantifies how much the sample median would fluctuate across
hypothetical repeated samples from the same population, without requiring any
closed-form formula for the median's sampling distribution.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:492 '* Bootstrap for Variance of Sample Statistics'
// Slide: Bootstrap for Variance of Sample Statistics
#strong[Bootstrap for Variance of Sample Statistics]

The bootstrap rests on a chain of two approximations, each replacing something
unknown or intractable with something computable. Understanding where each
approximation enters, and how much error it introduces, is essential for using
the bootstrap responsibly.

Start from the setup. A random variable $X$ follows some distribution $F$. We
draw $n$ independent, identically distributed samples $X_1, dots, X_n$ from $F$
and compute a statistic $T = g(X_1, dots, X_n)$. The quantity we want is
$V_F [T]$, the variance of that statistic under the true (and unknown)
distribution $F$.

The #strong[first approximation] addresses the fact that $F$ is unknown. All we
have are the $n$ observed samples, so we replace $F$ with the empirical
cumulative distribution function $hat(F)$, which places equal mass $1\/n$ on
each observed value. By the Glivenko-Cantelli theorem, $hat(F)$ converges
uniformly to $F$ as $n$ grows, so the substitution

$ V_F [T] approx V_(hat(F)) [T] $

is justified asymptotically. This approximation is not negligible in finite
samples, however: its magnitude depends on both the sample size $n$ and the
shape of the true distribution $F$. Heavy-tailed or multimodal distributions,
for instance, need larger $n$ before $hat(F)$ captures their structure well
enough for the variance estimate to be reliable.

The #strong[second approximation] deals with the computational difficulty that
$V_(hat(F)) [T]$ may have no closed-form expression. Even though $hat(F)$ is
fully known (it is just the data), the function $g$ defining the statistic can
be arbitrarily complex: a median, a ratio of means, a regression coefficient. To
handle this, we appeal to the law of large numbers via Monte Carlo simulation.
We draw $B$ bootstrap samples from $hat(F)$ (that is, we resample with
replacement from the observed data), compute $T_i$ on each, and estimate

$ V_(hat(F)) [T] approx v_"boot" = 1 / B sum_i (T_i - overline(T))^2 $

where $overline(T)$ is the mean of the $B$ bootstrap statistics. Unlike the
first approximation, the error introduced here is entirely under the analyst's
control: as $B arrow.r oo$, $v_"boot" arrow.r V_(hat(F)) [T]$ by the law of
large numbers. In practice, $B$ in the range of a few hundred to a few thousand
suffices to make this second-stage error negligible relative to the first-stage
sampling error. The computational cost of each additional bootstrap replicate is
the only limiting factor, and for most statistics it is modest.

Putting the two stages together, the bootstrap estimate $v_"boot"$ approximates
the true variance $V_F [T]$ through a composition:
$V_F [T] approx V_(hat(F)) [T] approx v_"boot"$. The first link depends on
having enough data; the second depends on running enough resamples. Recognizing
this separation helps diagnose failures: if a bootstrap interval is unreliable,
the remedy is more original data (first approximation) or more replicates
(second approximation), and knowing which one dominates guides the fix.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:518 '* Summary'
// Slide: Summary
= Summary

The central idea behind the learn-validation approach is that it estimates
$E_("out")$ directly by holding out a portion of the data, deliberately trading
some training data for a more reliable estimate of how the model will perform on
unseen examples.

To recap the three main strategies covered in this chapter:

- #emph[Train/validation split]: the validation error $E_("val")$ serves as an
  unbiased estimate of $E_("out")$, with its standard error shrinking as
  $1 / sqrt(K)$. The tradeoff is that reserving too large a validation set $K$
  yields a reliable estimate, but of a worse model trained on less data.

- #emph[Cross-validation]: all $N$ data points are reused for both training and
  validation across $K$ folds, at the cost of running $K$ separate training
  procedures. Variants such as repeated cross-validation and leave-one-out
  cross-validation navigate the tension between reducing fold-selection
  dependency and managing computational expense.

- #emph[Bootstrap]: resampling with replacement estimates the sampling
  distribution of any statistic, not just $E_("out")$, replacing the need for
  closed-form mathematical derivations with simulation. This generality makes it
  applicable well beyond model selection, wherever uncertainty quantification of
  an estimator is needed.

// From: msml610/lectures_source/Lesson05.3-Learn_Validation.smd:534 '* References'
// Slide: References
= References

#set text(size: 0.75em)
#references("/msml610/lectures_source/refs.bib")
