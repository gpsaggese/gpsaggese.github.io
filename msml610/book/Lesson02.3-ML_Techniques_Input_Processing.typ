// git_hash=1cb8c0e4d-wzv timestamp=20260903_160831
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
  title: "L02.3: ML Techniques - Input Processing",
  author: "MSML610: Advanced Machine Learning",
)

// Apply the AIMA document template (page/text/heading set + show rules).
#show: aima-style

#chapter("L02.3: ML Techniques - Input Processing")

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:11 '# Input Processing'
// Slide: Input Processing

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:13 '## Overview'
// Slide: Overview
= Overview

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:15 '* From Raw Data to Model Input'
// Slide: From Raw Data to Model Input

Raw data rarely arrives in the shape a model needs. Sensors drop readings,
categorical fields use inconsistent labels, and numeric features span wildly
different ranges. A fixed sequence of transformations is needed to turn messy
input into clean, well-scaled, informative features, yielding better model
performance, stronger generalization to unseen data, and more stable training
dynamics.

This lesson walks through that sequence in the order you would typically apply
it. First comes #emph[data quality]: removing duplicates and errors, denoising,
and handling outliers and missing values so the dataset is trustworthy before
any modeling begins. Next is #emph[scaling and encoding], which puts numeric
features on comparable scales, converts categories into numbers a model can
consume, and optionally bins continuous values into discrete intervals. The
third stage is #emph[feature-space engineering]: constructing new features from
existing ones and, when the dimensionality grows too large, reducing it back
down to a manageable size. Finally, we discuss #emph[safe application], the
discipline of fitting every transformation on the training set only and then
applying it identically to validation and test data, along with techniques for
augmenting the training set when labeled examples are scarce.

@fig:fromrawdatatomodelinput shows the pipeline: raw data flows through
cleaning, outlier and missing-value handling, and scaling and encoding before
reaching the model.

// rendered_images:begin
// ```graphviz[width=95%]
// digraph InputProcessingPipeline {
//   bgcolor="transparent";
//   pad="0.15";
//   splines=spline;
//   nodesep=0.28;
//   ranksep=0.4;
//   rankdir=LR;
// 
//   node [shape=box, style="rounded,filled", penwidth=1.6,
//         fontname="Helvetica", fontsize=10, margin="0.14,0.10", height=0.42];
//   edge [color="#A3B1C0", penwidth=1.2, arrowhead=vee, arrowsize=0.65,
//         fontname="Helvetica", fontsize=9, fontcolor="#7B8794"];
// 
//   raw   [label="Raw Data", fillcolor="#FFD1A6", color="#D9902B", fontcolor="#6B4517"];
//   clean [label="Clean &\nDenoise", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   fix   [label="Handle Outliers\n& Missing Values", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   scale [label="Scale &\nEncode", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   eng   [label="Construct &\nReduce Features", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   aug   [label="Augment", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//   ready [label="Model-Ready\nData", fillcolor="#B2E2B2", color="#4F9A5C", fontcolor="#1F4E2E"];
// 
//   raw -> clean -> fix -> scale -> eng -> aug -> ready;
// }
// ```
// label=fig:fromrawdatatomodelinput
// caption=Diagram relating Raw Data, Clean & Denoise, Handle Outliers & Missing Values and Scale & Encode
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.3-ML_Techniques_Input_Processing.typ.figs/Lesson02.3-ML_Techniques_Input_Processing.1.png"),
  caption: [Diagram relating Raw Data, Clean & Denoise, Handle Outliers & Missing Values and Scale & Encode],
) <fig:fromrawdatatomodelinput>
// render_images:end

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:55 '## Data Quality'
// Slide: Data Quality
== Data Quality

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:57 '* Data Cleaning'
// Slide: Data Cleaning

The goal of #strong[data cleaning] is to ensure data quality so that models train on
accurate, consistent inputs. Errors or inconsistencies in the raw dataset, if
left uncorrected, propagate directly into learned parameters and degrade every
downstream prediction.

Several techniques address the most common problems:

- #emph[Remove duplicates]: identical records that appear more than once inflate
  the effective weight of those examples during training, skewing the learned
  distribution.
- #emph[Correct data entry errors]: misspellings, misformatted entries, and
  transposed fields introduce noise that the model either memorizes or averages
  over, neither of which is desirable.
- #emph[Standardize data]: when the same semantic value is recorded in multiple
  surface forms, the model treats them as distinct. Date formats are a
  particularly insidious example: a column that mixes `MM/DD/YYYY` with
  `DD-MM-YYYY` can silently swap month and day. The entry `01/02/2024` is valid
  under both conventions, so no single-row check catches the inconsistency; only
  a column-wide policy resolves it. Beyond dates, standardization includes
  normalizing strings (for instance, lowercasing all text so that "New York" and
  "new york" merge into one category), converting types (casting digit strings
  to integers so arithmetic works correctly), and handling unexpected characters
  or encoding artifacts that creep in from heterogeneous data sources.

Removing mislabeled and noisy records has a dual benefit. It lowers the
#emph[variance] of the fitted model, because the learner no longer has to
contort its decision surface to accommodate contradictory examples. It also
reduces the #emph[irreducible error floor]: mislabeled points set a hard limit
on how low any model's loss can go on the training set, and purging them lets
that floor drop closer to the true Bayes error.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:77 '* Noise Removal and Smoothing'
// Slide: Noise Removal and Smoothing

The goal of #strong[denoising] is to remove irrelevant or corrupt variation from a signal
so that the underlying pattern becomes clearer and models can learn from it.
A familiar example is cleaning noisy speech: a recording picked up in a crowded
room contains high-frequency hiss and crackle that obscures the speaker's words,
and stripping that noise out makes the signal usable for transcription or
further analysis.

Two broad families of techniques handle most practical denoising tasks.
#emph[Smoothing] replaces each sample with the average of its neighbors inside a
sliding window, dampening rapid fluctuations that are unlikely to be part of the
true signal. #emph[Filtering] applies a frequency-domain or rank-order rule: a
low-pass filter for audio attenuates everything above a chosen cutoff frequency,
while a median filter replaces each sample with the median of its local
neighborhood, which is particularly effective at eliminating isolated spikes
without blurring edges the way a mean would.

Both approaches come with real costs. A moving average is indiscriminate: it
suppresses noise and genuine signal features alike, so a sharp but meaningful
peak in the data gets flattened along with the random jitter around it. A
low-pass filter introduces lag, because the smoothed output responds more slowly
than the raw signal to sudden legitimate changes; in a control loop or a
real-time monitoring system, that delay can matter. A centered smoothing window
is even more problematic in time-series forecasting: it computes each output
value using samples from both the past and the future, which means the model has
access to information it would never have at prediction time. This is a form of
data leakage that inflates validation metrics and then fails silently in
production.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:92 '* Outliers'
// Slide: Outliers

An #strong[outlier] is a data point that falls outside the interval
$[Q_1 - 1.5 dot.op "IQR", Q_3 + 1.5 dot.op "IQR"]$, where IQR is the
interquartile range #cite("tukey1977eda"), or one whose Z-score satisfies
$|z| > 3$. Outliers arise either from measurement errors (a misplaced decimal, a
sensor glitch) or from genuine variability in the underlying process.

Detecting outliers typically relies on one of three approaches: visual
inspection via box plots, the IQR rule just described, or the Z-score threshold.
The Z-score test deserves a caveat: it assumes the data are approximately
normal, and the very outliers it is trying to flag inflate the sample mean and
standard deviation, pulling those statistics toward the extreme values and
potentially hiding (masking) the points that should be caught. The IQR rule,
built on quartiles rather than moments, is more robust to this distortion. Once
outliers are identified, common treatments include outright removal, capping
values at the fence boundaries (sometimes called Winsorizing), or applying a
variance-stabilizing transform such as a log scale that compresses the tails and
reduces the leverage of extreme points.

Before applying any of these treatments, ask whether a given outlier is an
error or a rare but genuine event. In fraud detection, anomaly detection, and
risk modeling, the outlier #emph[is] the signal: a suspiciously large
transaction or an unusual sensor reading may be exactly the case the model needs
to catch. Removing it would destroy the information the analysis was built to
find. The decision to keep or discard an outlier should be guided by domain
knowledge about its likely cause, not applied as a blanket preprocessing step.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:111 '* Missing Data and Imputation'
// Slide: Missing Data and Imputation

// TODO(ai_gp): Invert MAR -> #emph[Missing at random] (MAR)
Pieces of data are often #strong[missing] in data sets.
Three mechanisms explain why a value may be absent #cite("rubin1976missingdata",). #emph[MCAR] (missing
completely at random) means the probability of missingness has no relationship
to any variable, observed or unobserved, so simple imputation yields unbiased
estimates. #emph[MAR] (missing at random) means missingness depends on other
observed variables but not on the missing value itself; imputation conditioned
on those observed variables remains unbiased. #emph[MNAR] (missing not at
random) means the missingness depends on the unobserved value, so any
imputation is biased unless the analyst explicitly models the missingness
mechanism.

Several practical techniques address missing data:

- #emph[Deletion]: remove rows or columns whose missing-value fraction exceeds a
  chosen threshold. This is simple but discards potentially useful information
  and can introduce bias when the data are not MCAR.
- #emph[Imputation]: fill gaps with mean, median, or mode values, with K-nearest
  neighbors (KNN) estimates drawn from similar records, or with predictions from
  a fitted model such as iterative regression (MICE).
- #emph[Missingness indicator]: add a binary `was_missing` column that flags
  which entries were originally absent, preserving the information that a value
  was missing even after imputation fills the cell.

Among these, mean or median substitution is the quickest to implement but
carries real costs: it shrinks the variance of the imputed feature, attenuates
correlations with other variables, and biases any downstream estimate that
depends on the true spread of the data. For that reason, model-based imputation
or KNN imputation is generally preferred whenever the dataset is large enough to
support it.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:126 '## Feature Scaling and Encoding'
// Slide: Feature Scaling and Encoding
= Feature Scaling and Encoding

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:128 '* Feature Scaling'
// Slide: Feature Scaling

Feature scaling puts all input dimensions on comparable footing so that no
single feature dominates distance calculations or gradient updates simply
because its raw numbers are larger. Two classical approaches handle this.
#[Min-max normalization] rescales every value to the interval $[0, 1]$
using

$ x' = (x - x_(min)) / (x_(max) - x_(min)) $

while #strong[z-score standardization] centers each feature at zero mean with
unit variance:

$ x' = (x - mu) / sigma $

A third option, #strong[RobustScaler], centers and scales by the median and
interquartile range instead of the mean and full range, making it far less
sensitive to outliers.

Each technique carries a different tradeoff. Min-max normalization is intuitive
and preserves the original distribution shape, but a single extreme value can
compress every other point into a narrow band, and unseen test values are not
guaranteed to stay inside $[0, 1]$. Z-score standardization handles this
somewhat better because the mean and standard deviation are less sensitive than
the absolute minimum and maximum, yet heavy-tailed outliers still pull both
statistics and distort the result. RobustScaler sidesteps this problem almost
entirely: because the median and IQR are order statistics, a handful of extreme
points barely move them. In practice, choosing among the three comes down to how
clean the data is and whether downstream models (such as $k$-nearest neighbors
or gradient-based optimizers) assume a specific scale.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:147 '* Feature Scaling'
// Slide: Feature Scaling

When choosing a preprocessing strategy, it helps to know which models actually
care about the scale of their inputs. #emph[Scale-sensitive] models include KNN,
SVM, k-means, PCA, and any model trained with gradient-based optimization. These
methods rely on distance computations or regularization terms that treat raw
numeric magnitude as meaningful. A feature whose values range from 0 to 1 will
contribute far less to a Euclidean distance, or be penalized far more heavily by
an L1 or L2 regularization term, than a feature ranging from 0 to 10,000. The
penalty is not scale-invariant: shrinking a feature's range effectively
increases the relative weight of regularization on that feature, distorting the
model's learned coefficients.

#emph[Scale-invariant] models, by contrast, include tree-based methods such as
decision trees, random forests, and gradient boosting. These algorithms split
nodes by comparing a single feature's values against a threshold, so only the
rank ordering of values within each feature matters, not their absolute
magnitude. Multiplying a feature by a constant changes every candidate threshold
by the same factor but leaves the optimal split point, and therefore the tree's
structure, unchanged. For this reason, tree-based pipelines can often skip the
scaling step entirely, simplifying preprocessing without any loss in
performance.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:158 '* Categorical Encoding'
// Slide: Categorical Encoding

Most machine learning algorithms operate on numeric inputs, so categorical
features (color names, country codes, size labels) must be converted into
numbers before a model can use them. Several #strong[categorical encoding]
strategies exist, each with different assumptions and tradeoffs.

#emph[Label encoding] assigns a distinct integer to each category: for instance,
`red`, `green`, `blue` become `1`, `2`, `3`. This is simple and
memory-efficient, but it imposes both an ordering and equal spacing on
categories that may have neither. The encoding implicitly asserts that
$"green" - "red" = "blue" - "green"$, a relationship that is meaningless for
nominal categories like color. Tree-based models can tolerate this artifact
because they split on thresholds rather than computing differences, but linear
and distance-based models will treat the fabricated ordering as real signal.

#emph[One-hot encoding] sidesteps the ordinal assumption entirely by creating a
binary indicator vector for each category: `red` becomes `[1,0,0]`, `green`
becomes `[0,1,0]`, and `blue` becomes `[0,0,1]`. No category is "closer" to any
other in this representation, which makes it safe for any model family.

#emph[Ordinal encoding] is label encoding applied in situations where the order
genuinely exists: `small < medium < large` is a real ranking, so mapping these
to `1, 2, 3` preserves information rather than fabricating it. The spacing
assumption (is the gap from small to medium the same as from medium to large?)
may still be approximate, but the ordering itself is meaningful.

Beyond these three basic strategies, #emph[target (mean) encoding] replaces each
category with a statistic of the target variable (typically the conditional
mean), and #emph[learned embeddings] map each category to a dense vector trained
jointly with the model. Both handle high-cardinality features gracefully and can
capture richer relationships than a single integer or a sparse binary vector.

One-hot encoding comes with a practical cost: a column with thousands of
distinct values (zip codes, product IDs) produces thousands of new binary
columns, inflating dimensionality and memory usage. Target encoding and
embeddings are common remedies for this. A second concern spans all encoding
methods: a category that appears for the first time at inference, absent from
the training vocabulary, has no learned representation. The standard defense is
to reserve an explicit `unknown` bucket during training, mapping any novel
category to it at serving time, so the model always receives a valid input.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:180 '* Discretization'
// Slide: Discretization

#strong[Discretization] converts a continuous quantity into a categorical one by
partitioning its range into a finite set of bins. Several techniques exist for
choosing where to place the cut points. #emph[Equal-width binning] splits the
range into bins of identical width, which is simple but can leave some bins
empty if the data are unevenly distributed. #emph[Quantile binning] instead
places the cuts so that each bin holds roughly the same number of observations,
guaranteeing that no bin is starved of data. #emph[Supervised binning] fits a
shallow decision tree on the target variable and uses the tree's split points as
bin edges, aligning the discretization with the prediction task. Finally,
#emph[k-means binning] clusters the values along a single dimension, letting the
data's own density structure determine the boundaries.

Consider discretizing age into four categories: `Child` for the interval
$[0, 13)$, `Teen` for $[13, 20)$, `Adult` for $[20, 65)$, and `Senior` for
$[65, oo)$. Under this scheme an age of 32 maps to `Adult`, as shown in
@fig:discretization.

// TODO(ai_gp): Use wrap it
// rendered_images:begin
// ```graphviz[width=90%]
// digraph AgeBinning {
//   bgcolor="transparent";
//   pad="0.1";
//   splines=spline;
//   nodesep=0.22;
//   ranksep=0.3;
//   rankdir=LR;
// 
//   node [shape=box, style="rounded,filled", penwidth=1.4,
//         fontname="Helvetica", fontsize=10, margin="0.14,0.09", height=0.4];
//   edge [color="#A3B1C0", penwidth=1.1, arrowhead=vee, arrowsize=0.6,
//         fontname="Helvetica", fontsize=9, fontcolor="#7B8794"];
// 
//   child  [label="Child\n[0, 13)",    fillcolor="#E8F1FB", color="#7CA6CE", fontcolor="#1F4E79"];
//   teen   [label="Teen\n[13, 20)",    fillcolor="#E8F1FB", color="#7CA6CE", fontcolor="#1F4E79"];
//   adult  [label="Adult\n[20, 65)",   fillcolor="#B2E2B2", color="#4F9A5C", fontcolor="#1F4E2E"];
//   senior [label="Senior\n[65, inf)", fillcolor="#E8F1FB", color="#7CA6CE", fontcolor="#1F4E79"];
// 
//   child -> teen -> adult -> senior [style=invis];
//   { rank=same; child; teen; adult; senior; }
// 
//   age32 [label="Age = 32", shape=ellipse, fillcolor="#FFD1A6", color="#D9902B", fontcolor="#6B4517"];
//   age32 -> adult [label="mapped to", color="#D9902B", fontcolor="#6B4517"];
// }
// ```
// label=fig:discretization
// caption=Diagram relating Child [0, 13), Teen [13, 20), Adult [20, 65) and Senior [65, inf)
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.3-ML_Techniques_Input_Processing.typ.figs/Lesson02.3-ML_Techniques_Input_Processing.2.png"),
  caption: [Diagram relating Child \[0, 13), Teen \[13, 20), Adult \[20, 65) and Senior \[65, inf)],
) <fig:discretization>
// render_images:end

The convenience of discretization comes with real costs. All within-bin
variation is discarded: ages 21 and 64 both become `Adult`, even though they
represent very different life stages. The cut points themselves are often
arbitrary, chosen by convention rather than by any property of the data or the
target. Equal-width bins are particularly fragile, since a bin covering a
sparsely populated part of the range may end up with zero observations. Perhaps
most problematically, a genuine threshold effect in the relationship between the
feature and the target can fall in the interior of a bin rather than at its
edge, hiding the very signal the analyst hoped to capture.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:236 '## Feature Space Engineering'
// Slide: Feature Space Engineering
== Feature Space Engineering

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:238 '* Feature Construction'
// Slide: Feature Construction

#strong[Feature engineering] is the practice of deriving more informative
variables from raw inputs, encoding domain knowledge that the original columns
do not expose directly. Several techniques are common: combining existing
variables (for instance, computing $"area" = "height" times "width"$ from two
separate columns), extracting components (pulling the year out of a full date),
and deriving logical features that capture higher-level structure. A date such
as `2023-04-15`, for example, can be transformed into the pair (`Saturday`,
`is_weekend = True`), giving a model access to weekly seasonality it could not
easily learn from the raw timestamp alone.

Hand-built features are precisely the component that #emph[representation
  learning] in deep learning replaces: a convolutional network learns its own
edge detectors rather than requiring a human to specify them. Where deep
architectures are not feasible or not warranted, however, manual feature
engineering remains the primary lever for injecting expert knowledge into a
pipeline.

One serious pitfall is #emph[data leakage]. A feature built from the target
variable, or from rows that occur after the prediction point in time, injects
information the model would never have at inference. The result is a validation
score that looks excellent but collapses on deployment, because the signal the
model relied on simply does not exist when predictions must be made in real
time.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:256 '* Dimensionality Reduction'
// Slide: Dimensionality Reduction

The goal of #strong[dimensionality reduction] is to shrink the number of
features in a dataset while preserving the information that matters most.
Consider a 1024×640 image: rather than working with every pixel, one can
compress the representation down to just 10 principal components that capture
the dominant patterns. Similarly, collapsing three color channels (red, green,
blue) into a single luminance value produces a grayscale image that retains the
structural content at a fraction of the dimensionality.

Two broad strategies exist for achieving this reduction. #emph[Feature
  selection] keeps a subset of the original columns intact, so the surviving
features remain directly interpretable. #emph[Feature extraction], by contrast,
constructs entirely new features as combinations of the originals; the result is
typically more compact but no longer maps one-to-one onto any single measured
quantity.

Several concrete techniques fall under feature extraction. #emph[Principal
  Component Analysis] (PCA) #cite("pearson1901pca") is an unsupervised method
that finds linear combinations of the original features ordered by the amount of
variance they explain: the first component captures the direction of greatest
spread, the second captures the most remaining spread orthogonal to the first,
and so on. #emph[Linear Discriminant Analysis] (LDA) takes a supervised
approach, projecting the data onto axes that maximize the separation between
known classes rather than overall variance. For visualization, non-linear
techniques such as #emph[t-SNE] #cite("vandermaaten2008tsne") and #emph[UMAP]
#cite("mcinnes2018umap") embed high-dimensional data into two or three
dimensions while attempting to preserve local neighborhood structure. Several
of these methods are scale-dependent: features measured in different units or
with very different ranges should be standardized before the transformation is
applied, or the result will be dominated by whichever feature happens to have
the largest numeric spread.

Dimensionality reduction delivers practical benefits across the modeling
pipeline. Fewer features reduce the risk of overfitting by eliminating noise
dimensions that a model might latch onto. Redundant or highly correlated columns
are collapsed, which tightens the effective size of the dataset. Training and
inference both speed up because the model operates on a lower-dimensional input.
Finally, projecting data into two or three dimensions makes it possible to
visualize clusters, outliers, and class boundaries that would be invisible in
the original high-dimensional space.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:280 '## Applying Transformations Safely'
// Slide: Applying Transformations Safely
== Applying Transformations Safely

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:282 '* Fit on Train, Apply to Validation and Test'
// Slide: Fit on Train, Apply to Validation and Test

Every fitted preprocessing step, whether a scaler, an imputer, an encoder, or a
PCA projection, learns summary statistics from the data it sees: the column
mean, the min/max range, the set of known categories, or the principal axes. If
any of these transformers is fitted on the full dataset before the train/test
split, those statistics carry information from the validation and test rows into
the training pipeline. This is a form of #strong[data leakage] #cite(
  "kaufman2012leakage",
), and it inflates evaluation scores because the model's preprocessing has
already "seen" the held-out data indirectly.

// rendered_images:begin
// ```graphviz
// digraph FitTransformLeakage {
//   bgcolor="transparent";
//   pad="0.15";
//   splines=spline;
//   nodesep=0.3;
//   ranksep=0.4;
//   rankdir=LR;
// 
//   node [shape=box, style="rounded,filled", penwidth=1.5,
//         fontname="Helvetica", fontsize=10, margin="0.14,0.10", height=0.42];
//   edge [color="#A3B1C0", penwidth=1.2, arrowhead=vee, arrowsize=0.6,
//         fontname="Helvetica", fontsize=9, fontcolor="#7B8794"];
// 
//   subgraph cluster_wrong {
//     label     = "Wrong: fit before split";
//     labelloc  = "t";
//     fontname  = "Helvetica-Bold";
//     fontsize  = 11;
//     fontcolor = "#6B1F1F";
//     style     = "rounded,filled";
//     fillcolor = "#FBE5E5";
//     color     = "#D98C8C";
//     margin    = 14;
// 
//     all_data [label="All Data", fillcolor="#FFD1A6", color="#D9902B", fontcolor="#6B4517"];
//     fit_all  [label="Fit scaler/imputer\non ALL rows", fillcolor="#FBC6C6", color="#D64545", fontcolor="#6B1F1F"];
//     split_w  [label="Split", fillcolor="#FBC6C6", color="#D64545", fontcolor="#6B1F1F"];
//     train_w  [label="Train", fillcolor="#F6C6C6", color="#D98C8C", fontcolor="#6B2A2A"];
//     test_w   [label="Test\n(already leaked into)", fillcolor="#F6C6C6", color="#D98C8C", fontcolor="#6B2A2A"];
// 
//     all_data -> fit_all -> split_w;
//     split_w -> train_w;
//     split_w -> test_w;
//   }
// }
// ```
// label=fig:fitontrainapplytovalidationandtest
// caption=Diagram relating Wrong: fit before split, All Data, Fit scaler/imputer on ALL rows and Split
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.3-ML_Techniques_Input_Processing.typ.figs/Lesson02.3-ML_Techniques_Input_Processing.3.png"),
  caption: [Diagram relating Wrong: fit before split, All Data, Fit scaler/imputer on ALL rows and Split],
) <fig:fitontrainapplytovalidationandtest>
// render_images:end

@fig:fitontrainapplytovalidationandtest shows the incorrect workflow: fitting
a scaler or imputer on all rows before splitting means the validation fold's
own distribution has already shaped the transform.

The fix is straightforward: split first, then fit every transformer on the
training fold only, and apply the already-fitted transformer to the validation
and test sets without refitting. Inside cross-validation this discipline must
hold for every fold individually; the scaler fitted on fold 1's training rows is
not reused for fold 2. Scikit-learn's `Pipeline` object enforces this
automatically by chaining the fitted steps so that a single `.fit()` call on the
training fold cascades through every transformer in order, and `.transform()` on
the held-out fold applies each one without refitting.

// rendered_images:begin
// ```graphviz
// digraph FitTransformLeakage {
//   bgcolor="transparent";
//   pad="0.15";
//   splines=spline;
//   nodesep=0.3;
//   ranksep=0.4;
//   rankdir=LR;
// 
//   node [shape=box, style="rounded,filled", penwidth=1.5,
//         fontname="Helvetica", fontsize=10, margin="0.14,0.10", height=0.42];
//   edge [color="#A3B1C0", penwidth=1.2, arrowhead=vee, arrowsize=0.6,
//         fontname="Helvetica", fontsize=9, fontcolor="#7B8794"];
// 
//   subgraph cluster_correct {
//     label     = "Correct: split before fit";
//     labelloc  = "t";
//     fontname  = "Helvetica-Bold";
//     fontsize  = 11;
//     fontcolor = "#1F4E2E";
//     style     = "rounded,filled";
//     fillcolor = "#E5F4EE";
//     color     = "#8FB79A";
//     margin    = 14;
// 
//     all_data2 [label="All Data", fillcolor="#FFD1A6", color="#D9902B", fontcolor="#6B4517"];
//     split_c   [label="Split", fillcolor="#B7DDD0", color="#6FA890", fontcolor="#1F4E39"];
//     train_c   [label="Train", fillcolor="#B2E2B2", color="#4F9A5C", fontcolor="#1F4E2E"];
//     fit_c     [label="Fit scaler/imputer\non TRAIN only", fillcolor="#B2E2B2", color="#4F9A5C", fontcolor="#1F4E2E"];
//     test_c    [label="Test", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
//     apply_c   [label="Apply fitted\ntransform", fillcolor="#D3E3F3", color="#7CA6CE", fontcolor="#1F4E79"];
// 
//     all_data2 -> split_c;
//     split_c -> train_c -> fit_c;
//     split_c -> test_c -> apply_c;
//     fit_c -> apply_c [style=dashed, label="reuse", color="#6FA890"];
//   }
// }
// ```
// label=fig:fitontrainapplytovalidationandtest-2
// caption=Diagram relating Correct: split before fit, All Data, Split and Train
// rendered_images:end
// render_images:begin
#figure(
  image("Lesson02.3-ML_Techniques_Input_Processing.typ.figs/Lesson02.3-ML_Techniques_Input_Processing.4.png"),
  caption: [Diagram relating Correct: split before fit, All Data, Split and Train],
) <fig:fitontrainapplytovalidationandtest-2>
// render_images:end

@fig:fitontrainapplytovalidationandtest-2 shows the corrected flow: the data is
split before any fitting occurs, so the training fold's statistics never leak
into evaluation.

// From: msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd:390 '* Data Augmentation'
// Slide: Data Augmentation

The goal of #strong[data augmentation] is to increase the effective size and
diversity of a training dataset by applying label-preserving transformations to
existing examples. Rather than collecting and labeling new data from scratch,
augmentation generates synthetic variants of what you already have, giving the
model a richer set of training signals at essentially zero annotation cost.

The specific transformations depend on the data modality:

- #emph[Vision]: random cropping, horizontal/vertical flips, rotation, color
  jitter, and cutout (masking a random patch of the image).
- #emph[Text]: synonym replacement, back-translation (translating a sentence to
  another language and back), and random word dropout.
- #emph[Audio]: pitch shifting, time stretching, and adding background noise.
- #emph[Tabular]: SMOTE-style synthetic minority oversampling #cite(
    "chawla2002smote",
  ), which interpolates between existing minority-class examples to balance
  skewed class distributions.

In each case the transformation is chosen so that a human would still assign the
same label to the modified example as to the original. A cropped photo of a cat
is still a cat; a sentence with one synonym swapped still carries the same
sentiment.

Augmentation also acts as a form of regularization. By exposing the model to
more variation than the raw dataset contains, it discourages the learner from
memorizing surface-level patterns (a particular background color, a specific
phrasing) that happen to correlate with the label in the original sample but
would not generalize. This effect comes without collecting a single new label.

The technique does carry a real risk, however: if a transformation inadvertently
changes the true label, the model trains on incorrect data. A classic example in
digit recognition is flipping an image of a "6," which produces something that
looks like a "9." Training on that flipped image with the original "6" label
injects noise that directly hurts accuracy. More subtly, aggressive color jitter
might make a "red light" look green in a traffic-scene classifier. The guiding
principle is that synthetic examples must preserve the same distribution as the
true data; any augmentation pipeline should be audited to confirm that every
transformation it applies is genuinely label-preserving for the task at hand.
