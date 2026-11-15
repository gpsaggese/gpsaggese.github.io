# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gensim: Sentiment Classification of Financial News
#
# - This notebook classifies sentences from financial news as negative,
#   neutral, or positive, using the Financial PhraseBank dataset
#   (Malo et al., 2014)
# - The pedagogical arc:
#   - Download and load the 4,846 labeled sentences
#   - Find and drop the duplicate sentences
#   - Tokenize the sentences with `simple_preprocess()` and the stop words
#   - Check which sentiment words the stop words remove
#   - Make a stratified 80/20 train / test split
#   - Summarize the vocabulary, the tokens per sentence, and the classes

# %%
# !pip install -q gensim==4.4.0 scikit-learn==1.7.2

import gensim
import sklearn

print("gensim version: ", gensim.__version__)
print("scikit-learn version: ", sklearn.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import gensim_utils as utils

# %%
import helpers.hnotebook as hnotebook

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
# # Part 1: Load and Prepare the Data

# %% [markdown]
# ## Cell 1.1: Download and load the sentences
#
# **Goal**
# - Download `Sentences_50Agree.txt` once and cache it in `data/`
#   - The dataset is licensed CC BY-NC-SA 3.0, so `data/` is not committed
#   - "50Agree" keeps every sentence on which at least 50% of the annotators
#     agreed, the largest of the 4 agreement levels
# - Load one row per sentence, with its label

# %%
# Download the dataset into `data/` on the first run, then load it.
df = utils.cell1_1_load_sentences("data")

print("df.shape=", df.shape)
display(df.head())
# Outcome: 4,846 sentences, each with one of the 3 labels.

# %% [markdown]
# ## Cell 1.2: Find duplicate sentences
#
# **Goal**
# - A sentence that appears twice can land in both the train and the test
#   set, which inflates the test score
# - A sentence with 2 different labels is label noise

# %%
# List the repeated sentences and their labels.
dups_df = utils.cell1_2_find_duplicates(df)

print("len(dups_df)=", len(dups_df))
display(dups_df)
# Outcome: 8 sentences appear twice.
# - 6 are repeated with the same label, all neutral
# - 2 are labeled neutral once and positive once

# %% [markdown]
# ## Cell 1.3: Drop duplicate sentences
#
# **Goal**
# - Keep one copy of a sentence repeated with the same label
# - Drop every copy of a sentence repeated with different labels, since there
#   is no way to know which label is right

# %%
# Drop the duplicates before the split.
df = utils.cell1_3_drop_duplicates(df)

print("df.shape=", df.shape)
# Outcome: 4,836 unique sentences remain.
# - 10 rows are dropped: the 6 extra copies, plus the 4 copies of the 2
#   conflicting sentences
# - Only exact repeats are dropped: sentences that differ only in their
#   numbers (e.g., the daily FTSE 100 moves) stay, since they are different
#   sentences

# %% [markdown]
# ## Cell 1.4: Tokenize the sentences
#
# **Goal**
# - Turn each sentence into a list of tokens with `simple_preprocess()`
#   - Lowercases the text and splits it on non-letters
#   - Keeps the tokens with 2 to 15 characters, so numbers are dropped
# - Remove the Gensim stop words, e.g., "the", "of", "and"

# %%
# Tokenize every sentence.
df = utils.cell1_4_tokenize_sentences(df)

display(df[["sentence", "tokens"]].head())
# Outcome: each sentence becomes a short list of lowercase content words.
# - Numbers disappear, so sentences that differ only in their numbers get the
#   same tokens: 56 groups of sentences share a token list, and 8 of these
#   groups mix labels

# %% [markdown]
# ## Cell 1.5: Check what the stop words remove
#
# **Goal**
# - The Gensim stop word list is a general English list, not one made for
#   sentiment
# - It also removes words that carry the direction of a change or a negation
# - Count how many sentences lose each of these words, per label

# %%
# Count the sentences that contain each removed direction or negation word.
words = ["up", "down", "not", "no", "less", "more", "above", "below", "without"]
removed_df = utils.cell1_5_count_removed_sentiment_words(df, words)

display(removed_df)
# Outcome: some removed words are strong sentiment clues.
# - "down" is in 96 sentences, 81 of them negative
# - "up" is in 190 sentences, 109 of them positive
# - Part 1 follows the project spec and removes them; the later parts can
#   compare against a model that keeps them

# %% [markdown]
# ## Cell 1.6: Stratified train / test split
#
# **Goal**
# - Hold out 20% of the sentences as the test set, with `random_state=42`
# - Stratify by label, so the 2 sets have the same class proportions

# %%
# Split the sentences, stratified by label.
train_df, test_df = utils.cell1_6_split_train_test(df)

print("train_df.shape=", train_df.shape)
print("test_df.shape=", test_df.shape)
# Outcome: 3,868 train and 968 test sentences, with no sentence in both.

# %% [markdown]
# ## Cell 1.7: Data summary table
#
# **Goal**
# - Summarize each set in one table:
#   - `n_sentences`: number of sentences
#   - `vocab_size`: number of distinct tokens
#   - `tokens_mean`, `tokens_median`, `tokens_min`, `tokens_max`: tokens per
#     sentence
#   - `n_empty`: sentences left with no token
#   - `negative`, `neutral`, `positive`: class counts, with their share

# %%
# Build the summary table of the train, test, and full sets.
table_df = utils.cell1_7_build_data_table(train_df, test_df)

display(table_df)
# Outcome: the train and test sets have the same class shares.
# - About 11.5 tokens per sentence, from a vocabulary of 8,763 tokens
# - Neutral is 59.4% of each set, so always predicting neutral already
#   scores 59.4% accuracy: the baseline to beat
# - 4 train sentences have no token left, e.g., "It 's not .", since every
#   word in them is a stop word
