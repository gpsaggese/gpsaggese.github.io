# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %%

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # W&B Weave API Overview
#
# - This notebook introduces W&B Weave through a simple monetary policy
#   sentiment classification example.
# - The pedagogical arc:
#   - Connect to a Weave project.
#   - Trace individual and nested function calls.
#   - Create and version evaluation datasets.
#   - Define and version models.
#   - Evaluate models using custom scorers.
#   - Compare results in the Weave interface.
# - Estimated completion time: 20 minutes.
# - Reference: [Weave documentation](https://weave-docs.wandb.ai/).

# %%
# !pip install -q weave==0.54.0 python-dotenv==1.2.4

import weave

print("Weave version:", weave.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

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
# - **What problem it solves**:
#   - Machine learning applications can be difficult to debug and evaluate.
#   - Weave records model behavior and supports reproducible comparisons.
# - **Key abstraction**:
#   - A trace records the execution of an operation.
#   - Versioned objects connect predictions to their model and dataset.
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `weave.init()` | Connect to a project | Initializes a Weave client |
# | `weave.op` | Trace a function | Records inputs, outputs, and timing |
# | `weave.Dataset` | Store evaluation examples | Versioned collection of rows |
# | `weave.Model` | Represent a model | Configuration and prediction method |
# | `weave.Scorer` | Define evaluation metrics | Supports custom aggregation |
# | `weave.Evaluation` | Evaluate a model | Combines datasets and scorers |
# | `weave.publish()` | Publish an object | Returns a versioned reference |
#
# - **How the components fit together**:
#
# ```
# weave.init()
#      |
#      v
# Weave project
#      |
#      +--> Traced operations
#      |
#      +--> Versioned models --------+
#      |                             |
#      +--> Versioned datasets ------+--> Evaluation
#                                    |         |
#           Scorers -----------------+         v
#                                         Comparison
# ```
#
# - The running example classifies central bank sentences:
#   - Hawkish: favors tighter monetary policy.
#   - Dovish: favors easier monetary policy.
#   - Neutral: does not clearly favor either direction.

# %% [markdown]
# # Part 1: Connecting to Weave

# %% [markdown]
# ## Cell 1.1: Authentication
#
# - Create a W&B account at https://wandb.ai/.
# - Obtain an API key at https://wandb.ai/authorize.
# - Set the `WANDB_API_KEY` environment variable or create a local `.env`
#   file containing `WANDB_API_KEY=<your-key>`.
# - Never commit the API key to Git.
# - Weave sends traced inputs and outputs to the configured project.
#   Use only data that you are authorized to share.

# %%
# Load credentials and configure tracing.
utils.configure_weave()

print("Weave configuration complete.")
# Outcome: Credentials are available without printing the API key.

# %% [markdown]
# ## Cell 1.2: Initialize a project
#
# - `weave.init()` connects subsequent operations to a named project.
# - The project groups traces, datasets, models, and evaluations.

# %%
# Connect to the tutorial project.
weave_client = weave.init("weave-60-mins-api")

print("type(weave_client)=", type(weave_client))
# Outcome: A Weave client and a link to the project.

# %%
# Inspect the initialization function.
utils.print_api_info(weave.init)

# %% [markdown]
# # Part 2: Tracing operations

# %% [markdown]
# ## Cell 2.1: Trace a simple function
#
# - The `@weave.op` decorator records a function call.
# - The function remains callable like an ordinary Python function.
# - Start with a function that counts words in a sentence.

# %%
@weave.op
def count_words(sentence: str) -> int:
    """
    Count whitespace-separated words.

    :param sentence: text to inspect
    :return: number of words
    """
    return len(sentence.split())


# Call the traced function.
word_count = count_words("The Committee raised interest rates.")

print("word_count=", word_count)
# Outcome: Five words and a recorded trace.

# %%
# Inspect the tracing decorator.
utils.print_api_info(weave.op)

# %% [markdown]
# ## Cell 2.2: Trace a keyword matching operation
#
# - A simple classifier can count words associated with monetary policy.
# - The utility function matches keywords against the beginning of tokens.
# - For example, `raise` matches `raised`.

# %%
# Define the initial keyword lists.
hawkish_words = ["raise", "inflation"]
dovish_words = ["lower", "cut"]

# Count the hawkish keywords in one sentence.
sentence = "The Committee decided to raise rates to curb inflation."
hits = utils.count_keyword_hits(sentence, hawkish_words)

print("hits=", hits)
# Outcome: Two matching hawkish keywords.

# %% [markdown]
# ## Cell 2.3: Compose nested operations
#
# - When one traced operation calls another, Weave records their relationship.
# - The classifier calls `count_keyword_hits()` twice.
# - The resulting trace contains a parent operation and two child operations.

# %%
# Classify the sentence using both keyword lists.
prediction = utils.keyword_tone(
    sentence,
    hawkish_words,
    dovish_words,
)

print("prediction=", prediction)
# Outcome: A hawkish prediction with two hawkish keyword hits.

# %% [markdown]
# - Open the project's Traces page.
# - Select the `keyword_tone` call.
# - Expand its two child calls to inspect their inputs and outputs.
# - Try replacing the sentence with "The Committee decided to lower rates."
#   - The predicted label should change to dovish.

# %% [markdown]
# # Part 3: Managing datasets

# %% [markdown]
# ## Cell 3.1: Construct a dataset
#
# - A `weave.Dataset` contains named rows of evaluation examples.
# - Each row is a dictionary.
# - Dictionary keys become dataset columns.
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `weave.Dataset` | Dataset class | Stores evaluation examples |
# | `.rows` | Dataset contents | Collection of dictionaries |
# | `weave.publish()` | Publish an object | Returns an object reference |
# | `.get()` | Retrieve a reference | Loads the referenced object |

# %%
# Construct a minimal dataset with one example per class.
toy_rows = [
    {
        "sentence": "The Committee decided to raise rates.",
        "label": "hawkish",
    },
    {
        "sentence": "The Committee decided to lower rates.",
        "label": "dovish",
    },
    {
        "sentence": "The Committee will monitor incoming data.",
        "label": "neutral",
    },
]

toy_dataset = weave.Dataset(
    name="toy-tone-sentences",
    rows=toy_rows,
)

display(pd.DataFrame(list(toy_dataset.rows)))
# Outcome: A dataset containing three labeled sentences.

# %%
# Inspect the dataset class and its public interface.
utils.print_api_info(weave.Dataset)

# %% [markdown]
# ## Cell 3.2: Publish a dataset
#
# - Publishing stores an object in the Weave project.
# - The returned reference identifies a specific version.

# %%
# Publish the dataset and inspect its reference.
toy_dataset_ref = weave.publish(toy_dataset)

print("toy_dataset_ref=", toy_dataset_ref.uri())
# Outcome: A version-specific Weave reference.

# %%
# Inspect the publishing function.
utils.print_api_info(weave.publish)

# %% [markdown]
# ## Cell 3.3: Retrieve a published dataset
#
# - A reference allows another session to retrieve the same dataset version.
# - This supports reproducible evaluations.

# %%
# Retrieve the published dataset.
loaded_dataset = toy_dataset_ref.get()

display(pd.DataFrame(list(loaded_dataset.rows)))
# Outcome: The same three rows from the published version.

# %% [markdown]
# ## Cell 3.4: Create a second dataset version
#
# - Add examples that are harder for the keyword classifier.
# - Publish the updated dataset under the same name.
# - The original dataset reference remains available.

# %%
# Extend the original dataset with additional examples.
evaluation_rows = toy_rows + [
    {
        "sentence": "Inflation remains elevated.",
        "label": "hawkish",
    },
    {
        "sentence": "Tighter policy is needed.",
        "label": "hawkish",
    },
    {
        "sentence": "Further rate increases may be appropriate.",
        "label": "hawkish",
    },
    {
        "sentence": "Rate cuts may be needed.",
        "label": "dovish",
    },
    {
        "sentence": "Economic activity weakened and inflation eased.",
        "label": "dovish",
    },
    {
        "sentence": "Accommodative policy will support recovery.",
        "label": "dovish",
    },
    {
        "sentence": "The meeting was held on Tuesday.",
        "label": "neutral",
    },
    {
        "sentence": "Household spending grew at a moderate pace.",
        "label": "neutral",
    },
    {
        "sentence": "The target range remains unchanged.",
        "label": "neutral",
    },
]

evaluation_dataset = weave.Dataset(
    name="toy-tone-sentences",
    rows=evaluation_rows,
)

display(pd.DataFrame(list(evaluation_dataset.rows)))
# Outcome: Twelve examples, with four examples per class.

# %%
# Publish the expanded dataset.
evaluation_dataset_ref = weave.publish(evaluation_dataset)

print("Original version:", toy_dataset_ref.uri())
print("Expanded version:", evaluation_dataset_ref.uri())
# Outcome: Two distinct versions of the same named dataset.

# %% [markdown]
# # Part 4: Versioning models

# %% [markdown]
# ## Cell 4.1: Construct a model
#
# - A `weave.Model` combines configuration with a prediction method.
# - Configuration fields describe the model's behavior.
# - The prediction method is decorated with `@weave.op`.
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `weave.Model` | Base model class | Supports versioned configuration |
# | `KeywordToneModel` | Tutorial model | Stores keyword lists |
# | `.predict()` | Prediction method | Returns a label and keyword counts |
#
# - The implementation is in `weave_utils.py`.

# %%
# Create the initial model using the original keyword lists.
model_v1 = utils.KeywordToneModel(
    name="keyword-tone",
    hawkish_words=hawkish_words,
    dovish_words=dovish_words,
)

print("model_v1=", model_v1)
# Outcome: A model with the initial keyword configuration.

# %%
# Inspect the base model class.
utils.print_api_info(weave.Model)

# %% [markdown]
# ## Cell 4.2: Generate a prediction
#
# - Call `predict()` with a sentence.
# - Consider a dovish sentence that contains the word "inflation".

# %%
# Inspect the initial model's prediction.
example_sentence = (
    "Economic activity weakened and inflation eased."
)

prediction_v1 = model_v1.predict(example_sentence)

print("prediction_v1=", prediction_v1)
# Outcome: The model incorrectly predicts hawkish.

# %% [markdown]
# ## Cell 4.3: Create an improved model configuration
#
# - The word "inflation" can occur in either hawkish or dovish statements.
# - Remove that ambiguous keyword.
# - Add keywords associated with tightening and easing policy.

# %%
# Construct a second configuration.
model_v2 = utils.KeywordToneModel(
    name="keyword-tone",
    hawkish_words=["raise", "increase", "tight", "elevated"],
    dovish_words=[
        "lower",
        "cut",
        "support",
        "weak",
        "ease",
        "accommodative",
    ],
)

print("model_v2=", model_v2)
# Outcome: A model with an expanded keyword configuration.

# %% [markdown]
# ## Cell 4.4: Publish both model versions
#
# - Publishing models under the same name records their configurations.
# - Different configurations produce different version references.

# %%
# Publish the two model configurations.
model_v1_ref = weave.publish(model_v1)
model_v2_ref = weave.publish(model_v2)

print("Version 1:", model_v1_ref.uri())
print("Version 2:", model_v2_ref.uri())
# Outcome: Two distinct model versions.

# %% [markdown]
# ## Cell 4.5: Compare predictions
#
# - Apply the improved model to the same sentence.
# - Inspect how changing the configuration affects the result.

# %%
# Generate a prediction with the improved model.
prediction_v2 = model_v2.predict(example_sentence)

display(
    pd.DataFrame(
        [prediction_v1, prediction_v2],
        index=["v1", "v2"],
    )
)
# Outcome: Version 2 correctly identifies the sentence as dovish.

# %% [markdown]
# # Part 5: Evaluating models

# %% [markdown]
# ## Cell 5.1: Define an accuracy scorer
#
# - A scorer compares a model output with a reference answer.
# - Weave supports functions decorated with `@weave.op`.
# - The `exact_match` scorer returns whether the predicted label is correct.

# %%
# Score the improved model's prediction.
score = utils.exact_match(
    label="dovish",
    output=prediction_v2,
)

print("score=", score)
# Outcome: The scorer returns {"correct": True}.

# %% [markdown]
# ## Cell 5.2: Define a custom aggregation scorer
#
# - A subclass of `weave.Scorer` supports custom aggregation.
# - `score()` computes the result for one example.
# - `summarize()` aggregates results across examples.
# - Recall measures the fraction of examples of a given class that are
#   classified correctly.

# %%
# Construct the custom recall scorer.
recall_scorer = utils.PerClassRecall()

print("type(recall_scorer)=", type(recall_scorer))
# Outcome: A scorer that aggregates results by reference class.

# %%
# Inspect the scorer base class.
utils.print_api_info(weave.Scorer)

# %% [markdown]
# ## Cell 5.3: Construct an evaluation
#
# - An evaluation combines a dataset with one or more scorers.
# - Weave matches dataset column names to function arguments.
#   - `sentence` is passed to the model's prediction method.
#   - `label` is passed to the scorers.
#   - `output` contains the model prediction.
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `weave.Evaluation` | Evaluation configuration | Dataset and scorers |
# | `.evaluate()` | Execute an evaluation | Asynchronous method |
# | `weave.Scorer` | Custom scorer | Per-example and aggregate metrics |

# %%
# Configure the evaluation.
evaluation = weave.Evaluation(
    name="toy-tone-evaluation",
    dataset=evaluation_dataset,
    scorers=[
        utils.exact_match,
        recall_scorer,
    ],
)

print("type(evaluation)=", type(evaluation))
# Outcome: An evaluation configured with two scorers.

# %%
# Inspect the evaluation class.
utils.print_api_info(weave.Evaluation)

# %% [markdown]
# ## Cell 5.4: Evaluate the initial model
#
# - `evaluation.evaluate(model)` runs predictions and scoring.
# - The method is asynchronous.
# - The utility runner calls this method in a way that supports both
#   Jupyter notebooks and standalone Python scripts.

# %%
# Evaluate the first model configuration.
summary_v1 = utils.run_evaluation(
    evaluation,
    model_v1,
    "keyword-tone-v1",
)

print("summary_v1=", summary_v1)
# Outcome: Aggregate accuracy, recall, and latency metrics.

# %% [markdown]
# ## Cell 5.5: Evaluate the improved model
#
# - Reuse the same dataset and scorers.
# - Keeping the evaluation configuration fixed makes the comparison fair.

# %%
# Evaluate the second model configuration.
summary_v2 = utils.run_evaluation(
    evaluation,
    model_v2,
    "keyword-tone-v2",
)

print("summary_v2=", summary_v2)
# Outcome: Metrics for the improved keyword configuration.

# %% [markdown]
# ## Cell 5.6: Compare evaluation results
#
# - Accuracy measures the fraction of correct predictions.
# - Per-class recall reveals which sentiment classes are difficult.
# - Mean latency measures the average prediction duration in seconds.

# %%
# Convert the evaluation summaries into a comparison table.
comparison = utils.summarize_keyword_evaluations(
    {
        "v1": summary_v1,
        "v2": summary_v2,
    }
)

display(comparison)
# Outcome: Side-by-side metrics for both model versions.

# %% [markdown]
# - The improved keyword configuration should achieve higher accuracy.
# - These examples were used to design the improved keyword lists.
#   - The results demonstrate the evaluation API.
#   - They do not establish performance on unseen data.
# - The application notebook uses a separate held-out test dataset.

# %% [markdown]
# # Part 6: Exploring the Weave interface

# %% [markdown]
# ## Cell 6.1: Inspect the recorded objects
#
# Open the project link printed by `weave.init()`.
#
# - **Traces**:
#   - Find the `keyword_tone` operation.
#   - Expand the nested keyword counting operations.
#   - Inspect the inputs, outputs, and timing.
# - **Datasets**:
#   - Find `toy-tone-sentences`.
#   - Compare the three-row and twelve-row versions.
# - **Models**:
#   - Find `keyword-tone`.
#   - Compare the two keyword configurations.
# - **Evaluations**:
#   - Select the two evaluation runs.
#   - Compare their aggregate metrics.
#   - Inspect individual predictions to identify classification errors.
#
# Experiment with an additional keyword or sentence.
# Publish the modified object and observe how its version changes.

# %% [markdown]
# ## Summary: The Mental Model
#
# - Weave traces operations to make model behavior observable.
# - Versioned datasets and models make evaluation inputs reproducible.
# - Evaluations combine model predictions with scorers to measure quality.
# - The Weave interface connects aggregate metrics to individual predictions.
#
# Continue to `weave.example.ipynb` to apply these concepts to real FOMC
# statements and Treasury yield data.
