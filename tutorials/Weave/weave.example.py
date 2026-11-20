# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # W&B Weave: Evaluating FOMC Tone and Treasury Yield Changes
#
# - This notebook uses W&B Weave to evaluate monetary policy
#   classification models and investigate whether changes in FOMC
#   statement tone are associated with Treasury yield changes.
# - The pedagogical arc:
#   - Load labeled FOMC sentences.
#   - Train and version two classification models.
#   - Evaluate classification performance using Weave.
#   - Collect FOMC statements and Treasury yield observations.
#   - Trace statement-level tone predictions.
#   - Compare downstream financial correlations.
#   - Interpret model performance and statistical uncertainty.
# - Estimated completion time: 25 minutes.
# - Reference: [Weave documentation](https://weave-docs.wandb.ai/).

# %%
# !pip install -q weave==0.54.0 python-dotenv==1.2.4 \
#     pandas==3.1.1 numpy==2.5.5 pydantic==2.14.3 \
#     scikit-learn==1.9.1 scipy==1.18.1 \
#     requests==2.34.2 beautifulsoup4==4.15.0 \
#     matplotlib==3.11.2 seaborn==0.13.2

import weave

print("Weave version:", weave.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import pandas as pd

# %%
import helpers.hnotebook as hnotebook

import weave_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - **Application problem**:
#   - Federal Reserve communications provide information about
#     monetary policy and economic conditions.
#   - Classification models can identify hawkish, dovish, and
#     neutral language.
#   - Classification performance does not necessarily determine
#     downstream financial usefulness.
# - **How Weave helps**:
#   - Record predictions and model execution times.
#   - Version model configurations and evaluation datasets.
#   - Compare models using consistent evaluation metrics.
#   - Trace the operations used to calculate statement tone.
#
# | Component | Purpose |
# |-----------|---------|
# | `weave.Dataset` | Store labeled FOMC evaluation sentences |
# | `weave.Model` | Represent a classification model |
# | `weave.op` | Trace predictions and tone calculations |
# | `weave.Scorer` | Calculate classification metrics |
# | `weave.Evaluation` | Compare model performance |
#
# - **Application workflow**:
#
# ```
# Labeled FOMC sentences
#          |
#          v
# Train classification models
#          |
#          v
# Evaluate models with Weave
#          |
#          v
# Classify complete FOMC statements
#          |
#          v
# Calculate statement tone changes
#          |
#          v
# Compare with Treasury yield changes
# ```

# %% [markdown]
# # Part 1: Initialize Weave and Load Data

# %% [markdown]
# ## Cell 1.1: Connect to a Weave project
#
# - Initialize a dedicated Weave project for the application.
# - The project stores model traces, evaluation results, and
#   versioned objects.
# - Set `WANDB_API_KEY` before running this notebook.

# %%
utils.configure_weave()

weave_client = weave.init("weave-60-mins-fomc")

print("Weave application project initialized.")

# %% [markdown]
# ## Cell 1.2: Load labeled FOMC sentences
#
# - Use the publicly available FOMC communication dataset.
# - Train the models using the training split.
# - Reserve the test split for classification evaluation.
# - Cache downloaded data locally for subsequent runs.
#
# Source: [FOMC communication dataset](https://huggingface.co/datasets/gtfintechlab/fomc_communication).

# %%
train_df = utils.load_fomc_split("train")
test_df = utils.load_fomc_split("test")

print("Training dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

display(train_df.head())

# %% [markdown]
# ## Cell 1.3: Examine the class distribution
#
# - Dovish language generally favors monetary policy easing.
# - Hawkish language generally favors monetary policy tightening.
# - Neutral language does not express a clear directional stance.
# - Inspect class frequencies to identify potential imbalance.

# %%
class_counts = pd.DataFrame({
    "train": train_df["label"].value_counts(),
    "test": test_df["label"].value_counts(),
})

class_counts = class_counts.sort_index()
class_counts.index = class_counts.index.map(utils.LABEL_NAMES)
class_counts.index.name = "label"

display(class_counts)

# %% [markdown]
# ## Cell 1.4: Publish the evaluation dataset
#
# - Convert the labeled test sentences into a Weave dataset.
# - Publish the dataset so both models can be evaluated using
#   the same version of the evaluation examples.

# %%
test_dataset = utils.make_fomc_test_dataset(test_df)

dataset_ref = weave.publish(test_dataset)

print("Evaluation examples:", len(test_dataset.rows))
print("Dataset reference:", dataset_ref.uri())

# %% [markdown]
# # Part 2: Train and Version Classification Models

# %% [markdown]
# ## Cell 2.1: Identify the training data
#
# - Compute a SHA-256 fingerprint of the training sentences
#   and labels.
# - Include the fingerprint in each model configuration.
# - This identifies the training data used by the model.

# %%
training_digest = utils.training_data_digest(train_df)

print("Training data SHA-256:", training_digest)

# %% [markdown]
# ## Cell 2.2: Train the baseline model
#
# - Convert sentences into TF-IDF features.
# - Train a logistic regression classifier.
# - Use the default class weighting as the baseline.
# - Keep the model lightweight enough to train on a CPU.

# %%
baseline_model = utils.TfidfLogRegModel(
    name="fomc-tfidf",
    training_digest=training_digest,
)

baseline_model.fit(train_df)

print("Baseline model trained.")

# %% [markdown]
# ## Cell 2.3: Train the class-balanced model
#
# - Train a second model using balanced class weights.
# - Increase the relative contribution of less frequent classes
#   during training.
# - Keep the other model settings unchanged.

# %%
balanced_model = utils.TfidfLogRegModel(
    name="fomc-tfidf",
    training_digest=training_digest,
    class_weight="balanced",
)

balanced_model.fit(train_df)

print("Balanced model trained.")

# %% [markdown]
# ## Cell 2.4: Publish the model configurations
#
# - Publish both models using the same object name.
# - Weave records different versions when their configurations
#   differ.
# - The fitted scikit-learn pipelines are private attributes.
# - Publishing these configurations does not serialize the fitted
#   pipelines. Call `fit()` before making predictions.

# %%
models = {
    "baseline": baseline_model,
    "balanced": balanced_model,
}

model_refs = {}

for name, model in models.items():
    model_refs[name] = weave.publish(model)

    print(f"{name}: {model_refs[name].uri()}")

# %% [markdown]
# ## Cell 2.5: Compare individual predictions
#
# - Classify the same test sentence using both models.
# - Inspect the predicted label and confidence.
# - Open the Weave project to examine the prediction traces.

# %%
sample_sentence = str(test_df["sentence"].iloc[0])
true_label = int(test_df["label"].iloc[0])

print("Sentence:", sample_sentence)
print("True label:", utils.LABEL_NAMES[true_label])

for name, model in models.items():
    prediction = model.predict(sample_sentence)

    print(f"{name}: {prediction}")

# %% [markdown]
# # Part 3: Evaluate Classification Performance

# %% [markdown]
# ## Cell 3.1: Define the evaluation
#
# - Use the same versioned test dataset for both models.
# - Accuracy measures the fraction of correct predictions.
# - Macro-F1 gives equal weight to the three classes.
# - Per-class F1 scores reveal differences in performance across
#   dovish, hawkish, and neutral sentences.
# - Weave records prediction latency automatically.

# %%
evaluation = weave.Evaluation(
    name="fomc-tone-evaluation",
    dataset=test_dataset,
    scorers=[
        utils.accuracy_scorer,
        utils.MacroF1Scorer(),
    ],
)

print("Evaluation configured.")
print("Evaluation examples:", len(test_dataset.rows))

# %% [markdown]
# ## Cell 3.2: Evaluate the baseline model
#
# - Run the baseline model on all 496 test sentences.
# - Record predictions, scorer outputs, and model latency.
# - Use a descriptive display name to identify the evaluation
#   in the Weave interface.

# %%
baseline_summary = utils.run_evaluation(
    evaluation,
    baseline_model,
    display_name="baseline-tfidf",
)

print("Baseline evaluation complete.")
display(baseline_summary)

# %% [markdown]
# ## Cell 3.3: Evaluate the class-balanced model
#
# - Evaluate the balanced model using the same dataset and scorers.
# - Keep the evaluation conditions consistent across models.

# %%
balanced_summary = utils.run_evaluation(
    evaluation,
    balanced_model,
    display_name="balanced-tfidf",
)

print("Balanced evaluation complete.")
display(balanced_summary)

# %% [markdown]
# ## Cell 3.4: Compare evaluation results
#
# - Combine both evaluation summaries into one table.
# - Verify that every test example was scored.
# - Compare macro-F1, accuracy, per-class F1, and latency.
# - Use macro-F1 as the primary classification comparison metric
#   because the dataset has an imbalanced class distribution.

# %%
evaluation_summaries = {
    "baseline": baseline_summary,
    "balanced": balanced_summary,
}

classification_results = utils.summarize_fomc_evaluations(
    evaluation_summaries,
    expected_rows=len(test_df),
)

display(classification_results.round(4))

# %% [markdown]
# ## Cell 3.5: Identify the higher-scoring model
#
# - Identify the model with the highest test macro-F1.
# - Continue using both models in the financial analysis.
# - This comparison is descriptive: selecting a model using test
#   performance does not provide an independent final estimate
#   of the selected model's generalization performance.

# %%
best_model_name = classification_results["macro_f1"].idxmax()

best_macro_f1 = classification_results.loc[
    best_model_name,
    "macro_f1",
]

print("Highest-scoring model:", best_model_name)
print("Test macro-F1:", round(best_macro_f1, 4))

# %% [markdown]
# ## Cell 3.6: Inspect evaluations in Weave
#
# - Open the Weave project in your browser.
# - Navigate to the evaluation results.
# - Compare the baseline and balanced evaluation runs.
# - Inspect individual predictions and scorer outputs.
# - Find examples where the models disagree.
# - Examine the latency recorded for each prediction.
#
# Weave connects aggregate evaluation metrics to the individual
# predictions that produced them.

# %% [markdown]
# # Part 4: Collect and Align Financial Data

# %% [markdown]
# ## Cell 4.1: Load FOMC statements
#
# - Collect monetary policy statements from 2015 through 2024.
# - Preserve the announcement date and original statement text.
# - Cache the downloaded statements for reproducibility.
#
# Source: [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).

# %%
statements_df = utils.load_fomc_statements()

print("FOMC statements:", len(statements_df))
print("First announcement:", statements_df["date"].min())
print("Last announcement:", statements_df["date"].max())

display(
    statements_df[["date", "source_url"]].head()
)

# %% [markdown]
# ## Cell 4.2: Load Treasury yields
#
# - Collect daily two-year Treasury par yields from 2014
#   through 2024.
# - Include 2014 to provide preceding observations for the
#   beginning of the announcement period.
# - Yield observations are expressed in percentage points.
#
# Source: [U.S. Treasury interest rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates).

# %%
treasury_df = utils.load_treasury_yields()

print("Treasury observations:", len(treasury_df))

display(treasury_df.head())

# %% [markdown]
# ## Cell 4.3: Align statements with yield observations
#
# - Match each announcement to its same-day Treasury observation.
# - Use the next available observation when the announcement
#   occurs on a non-trading day.
# - Calculate the yield change relative to the preceding
#   available Treasury observation.
# - Multiply percentage-point changes by 100 to obtain basis
#   points.
# - These daily par yields are not intraday market prices.

# %%
aligned_df = utils.align_fomc_yields(
    statements_df,
    treasury_df,
)

print("Aligned announcements:", len(aligned_df))

print(
    "Same-day matches:",
    int(aligned_df["same_day_yield"].sum()),
)

print(
    "Next-observation matches:",
    int((~aligned_df["same_day_yield"]).sum()),
)

display(
    aligned_df[
        [
            "date",
            "yield_observation_date",
            "yield_change_bps",
            "same_day_yield",
        ]
    ].head()
)

# %% [markdown]
# ## Cell 4.4: Examine yield changes
#
# - Inspect the distribution of announcement-associated
#   Treasury yield changes.
# - Identify the announcements that required a next-observation
#   match.

# %%
display(aligned_df["yield_change_bps"].describe())

display(
    aligned_df.loc[
        ~aligned_df["same_day_yield"],
        [
            "date",
            "yield_observation_date",
            "yield_change_bps",
        ],
    ]
)

# %% [markdown]
# # Part 5: Calculate Statement-Level Tone

# %% [markdown]
# ## Cell 5.1: Prepare statement sentences
#
# - Remove publication dates and sharing metadata.
# - Preserve the substantive monetary policy language.
# - Split each statement into individual sentences.
# - Protect common abbreviations during sentence splitting.
# - Use a lightweight rule-based splitter rather than a
#   dedicated linguistic tokenizer.

# %%
sample_statement = statements_df["statement_text"].iloc[0]

sample_sentences = utils.split_sentences(sample_statement)

print("Sentences in first statement:", len(sample_sentences))

for sentence in sample_sentences[:3]:
    print(sentence)

# %% [markdown]
# ## Cell 5.2: Trace statement-level tone scoring
#
# - Classify each sentence as dovish, hawkish, or neutral.
# - Calculate tone as the hawkish share minus the dovish share.
#
# $$
# \mathrm{Tone} =
# \frac{N_{\mathrm{hawkish}}-N_{\mathrm{dovish}}}
# {N_{\mathrm{total}}}
# $$
#
# - Tone ranges from -1 (entirely dovish) to +1
#   (entirely hawkish).
# - Neutral sentences contribute to the denominator.
# - Weave traces the statement scoring operation and its
#   nested batch prediction.

# %%
sample_tone = utils.score_statement_tone(
    baseline_model,
    sample_statement,
)

display(sample_tone)

# %% [markdown]
# ## Cell 5.3: Score all statements
#
# - Apply both fitted models to every FOMC statement.
# - Use batch prediction to reduce tracing overhead.
# - Calculate the change in tone relative to the preceding
#   statement.
# - The first statement has no preceding tone observation.

# %%
tones_df = utils.calculate_statement_tones(
    aligned_df,
    models,
)

print("Tone dataset shape:", tones_df.shape)

display(tones_df.head())

# %% [markdown]
# ## Cell 5.4: Examine statement tone distributions
#
# - Compare the range and variability of tone scores.
# - Verify that the first tone change is missing for each model.

# %%
display(
    tones_df[
        ["baseline_tone", "balanced_tone"]
    ].describe()
)

display(
    tones_df[
        ["baseline_tone_change", "balanced_tone_change"]
    ].isna().sum()
)

# %% [markdown]
# # Part 6: Analyze Financial Correlations

# %% [markdown]
# ## Cell 6.1: Calculate Spearman correlations
#
# - Measure the monotonic association between changes in
#   statement tone and changes in Treasury yields.
# - Use the same valid announcement observations for both models.
# - A positive coefficient indicates that more hawkish tone
#   changes tend to accompany higher yield changes.
# - Correlation does not establish causation.

# %%
model_names = list(models)

correlation_results = utils.calculate_tone_correlations(
    tones_df,
    model_names,
)

display(correlation_results.round(4))

# %% [markdown]
# ## Cell 6.2: Estimate uncertainty with paired bootstrap
#
# - Resample announcement observations with replacement.
# - Use identical sampled indices for both models.
# - Calculate the difference between their Spearman correlations.
# - Repeat the process 2,000 times using a fixed random seed.
# - Report a percentile-based 95% confidence interval.
# - The bootstrap assumes independent announcement observations
#   and does not account for time-series dependence.

# %%
bootstrap_results = utils.bootstrap_correlation_difference(
    tones_df,
    model_a="baseline",
    model_b="balanced",
    n_bootstrap=2000,
    random_state=42,
)

display(pd.Series(bootstrap_results))

# %% [markdown]
# ## Cell 6.3: Compare classification and financial performance
#
# - Join the classification and financial evaluation results.
# - Examine whether higher classification macro-F1 corresponds
#   to a stronger financial correlation.
# - Do not select models based on their observed yield
#   correlations.

# %%
combined_results = classification_results.join(
    correlation_results
)

display(
    combined_results[
        [
            "macro_f1",
            "accuracy",
            "spearman_rho",
            "p_value",
            "n_observations",
        ]
    ].round(4)
)

# %% [markdown]
# # Part 7: Visualize and Interpret Results

# %% [markdown]
# ## Cell 7.1: Visualize tone and yield changes
#
# - Plot announcement-level tone changes against yield changes.
# - Use identical horizontal and vertical scales for both models.
# - Each point represents one announcement.
# - The fitted lines are descriptive linear fits.
# - The reported coefficients are Spearman rank correlations.

# %%
fig = utils.plot_tone_correlations(
    tones_df,
    model_names,
)

from IPython.display import display
from IPython.core.pylabtools import print_figure

display(
    {
        "image/png": print_figure(fig, fmt="png"),
    },
    raw=True,
    metadata={
        "image/png": {
            "alt": (
                "Side-by-side scatter plots comparing FOMC tone changes "
                "and Treasury yield changes for the baseline and "
                "balanced models, with descriptive linear fits."
            )
        }
    },
)
plt.close(fig)

# %% [markdown]
# ## Cell 7.2: Summarize the comparison
#
# - Compare classification macro-F1 and financial correlations.
# - Examine whether the confidence interval excludes zero.
# - Avoid interpreting small observed differences as conclusive
#   evidence of superior financial performance.

# %%
macro_f1_difference = (
    classification_results.loc["balanced", "macro_f1"]
    - classification_results.loc["baseline", "macro_f1"]
)

rho_difference = bootstrap_results["observed_difference"]
ci_lower = bootstrap_results["ci_lower"]
ci_upper = bootstrap_results["ci_upper"]

print(
    "Balanced minus baseline macro-F1:",
    round(macro_f1_difference, 4),
)

print(
    "Balanced minus baseline Spearman rho:",
    round(rho_difference, 4),
)

print(
    f"95% bootstrap confidence interval: "
    f"[{ci_lower:.4f}, {ci_upper:.4f}]"
)

if ci_lower <= 0 <= ci_upper:
    print(
        "The confidence interval includes zero. "
        "The correlation difference is inconclusive."
    )
else:
    print(
        "The confidence interval excludes zero "
        "under the bootstrap assumptions."
    )

# %% [markdown]
# ## Cell 7.3: Inspect application traces in Weave
#
# - Open the application project in the Weave interface.
# - Locate a `score_statement_tone` operation.
# - Expand the nested `predict_batch` operation.
# - Inspect the statement text, predicted labels, and tone score.
# - Compare traces from the two model configurations.
# - Review execution times and published model versions.
#
# These traces connect the downstream financial analysis to
# the individual classification operations.

# %% [markdown]
# ## Limitations
#
# - The classification dataset contains an imbalanced
#   distribution of monetary policy labels.
# - Comparing models using test performance does not provide
#   an independent final estimate for a selected model.
# - The rule-based sentence splitter may produce imperfect
#   sentence boundaries.
# - Hard-label tone scores do not incorporate prediction
#   uncertainty.
# - Daily Treasury yield changes may reflect other economic
#   information besides FOMC announcements.
# - Next-observation matches may include additional market
#   developments.
# - The bootstrap does not account for temporal dependence.
# - The analysis is exploratory and does not establish
#   a causal or predictive financial relationship.
# - The data sources may change over time. Local caching and
#   model fingerprints help identify the data used in a run.

# %% [markdown]
# ## Summary: The Mental Model
#
# - Weave traces predictions and connects them to versioned
#   model configurations and evaluation datasets.
# - Custom scorers support evaluation metrics appropriate
#   for the application, including macro-F1.
# - Downstream analysis provides a separate perspective on
#   model usefulness beyond classification accuracy.
# - Statistical uncertainty and application limitations
#   remain important when interpreting model comparisons.
