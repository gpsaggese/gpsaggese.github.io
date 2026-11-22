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
#   - Fit a majority-class and a TF-IDF plus logistic regression baseline
#   - Train `Word2Vec` on the train tokens and inspect 5 finance terms
#   - Classify the mean `Word2Vec` vector of each sentence
#   - Compare the 3 models on the test set, with their confusion matrices
#   - Project the 200 most frequent words to 2D with PCA
#   - Refit TF-IDF with the direction and negation words kept, as an extra
#     check of the stop words

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

import gensim.corpora
import gensim.models
import numpy as np
import pandas as pd
import sklearn.dummy
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

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
# - Part 1 follows the project spec and removes them; Part 7 compares the
#   TF-IDF model against a variant that keeps them

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

# %% [markdown]
# # Part 2: Baselines

# %% [markdown]
# ## Cell 2.1: Quiet the Gensim progress logs
#
# **Goal**
# - Gensim reports the progress of every model through the `logging` module,
#   at the `INFO` level
# - Raise the level of the `gensim` logger to `WARNING`, keeping its warnings

# %%
# Show only the warnings and errors of Gensim.
gensim_log = logging.getLogger("gensim")
gensim_log.setLevel(logging.WARNING)

print("gensim log level=", logging.getLevelName(gensim_log.level))
# Outcome: the level is WARNING, so the next cells show only their own
# output.

# %% [markdown]
# ## Cell 2.2: Majority-class baseline
#
# **Goal**
# - The simplest model always predicts the most frequent train label
# - Any useful model must beat it on accuracy, and above all on macro-F1,
#   since it never predicts negative or positive

# %%
# Fit the majority-class model on the train labels only.
y_train = train_df["label"]
y_test = test_df["label"]
# `DummyClassifier` ignores the features, so a single dummy column is enough.
majority = sklearn.dummy.DummyClassifier(strategy="most_frequent")
majority.fit(np.zeros((len(train_df), 1)), y_train)

print("majority.classes_=", majority.classes_)
print("majority prediction=", majority.predict(np.zeros((1, 1)))[0])
# Outcome: the model always predicts "neutral", 59.4% of the train set.

# %% [markdown]
# ## Cell 2.3: Build a Dictionary on the train tokens
#
# **Goal**
# - `Dictionary` maps each train token to an integer id
# - Fit it on the train sentences only:
#   - A test token never seen in the train set gets no id
#   - This is what happens to a new sentence in production

# %%
# Map the train tokens to ids.
dictionary = gensim.corpora.Dictionary(train_df["tokens"])

print("dictionary=", dictionary)
# Outcome: 7,778 token ids, the train vocabulary size of Cell 1.7.

# %% [markdown]
# ## Cell 2.4: Fit a TfidfModel on the train corpus
#
# **Goal**
# - `TfidfModel` learns the document frequency of each token from the
#   bag-of-words corpus
#   - A token in few sentences gets a large weight
#   - A token in many sentences gets a small weight
# - The weights come from the train sentences only

# %%
# Count the tokens of each train sentence, then fit the TF-IDF weights.
train_bow = [dictionary.doc2bow(tokens) for tokens in train_df["tokens"]]
tfidf = gensim.models.TfidfModel(train_bow)

print("tfidf=", tfidf)
# Outcome: IDF weights from the 3,868 train sentences.
# - 41,820 non-zero entries, about 11 distinct tokens per sentence

# %% [markdown]
# ## Cell 2.5: Turn the TF-IDF vectors into a feature matrix
#
# **Goal**
# - scikit-learn needs a matrix with one row per sentence and one column per
#   token
# - `cell2_5_build_tfidf_matrix()` builds it:
#   - It runs `doc2bow()` and `tfidf[bow]` on each sentence
#   - It stacks the vectors with `gensim.matutils.corpus2csc()`

# %%
# Build the sparse TF-IDF matrices of the train and test sets.
X_train_tfidf = utils.cell2_5_build_tfidf_matrix(
    dictionary, tfidf, train_df["tokens"].tolist()
)
X_test_tfidf = utils.cell2_5_build_tfidf_matrix(
    dictionary, tfidf, test_df["tokens"].tolist()
)

print("X_train_tfidf.shape=", X_train_tfidf.shape)
print("X_test_tfidf.shape=", X_test_tfidf.shape)
# Outcome: both matrices have the same 7,778 columns of the train
# dictionary.
# - The test tokens that are not in the train set are dropped

# %% [markdown]
# ## Cell 2.6: Fit a logistic regression on the TF-IDF features
#
# **Goal**
# - Fit a logistic regression with the default settings of scikit-learn
#   - Nothing is tuned, so the test set stays untouched until Part 5
# - Report only the train accuracy here, to check that the model learned

# %%
# Fit the TF-IDF baseline on the train set.
tfidf_clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
tfidf_clf.fit(X_train_tfidf, y_train)
train_accuracy = tfidf_clf.score(X_train_tfidf, y_train)

print("train_accuracy=", round(train_accuracy, 3))
# Outcome: 85.2% accuracy on the train set, so the model learned from the
# features; the test score comes in Part 5.

# %% [markdown]
# ## Cell 2.7: Words with the largest weights for each class
#
# **Goal**
# - A linear model on TF-IDF is easy to read: each token has one weight per
#   class
# - Show the tokens that push the model most toward each class with
#   `cell2_7_top_words_per_class()`

# %%
# List the 8 tokens with the largest weight for each class.
top_words_df = utils.cell2_7_top_words_per_class(tfidf_clf, dictionary, topn=8)

# Show the whole word lists, without truncating the column.
with pd.option_context("display.max_colwidth", None):
    display(top_words_df)
# Outcome: the model relies on the verbs of change.
# - Negative: "decreased", "fell", "lower", "dropped", and layoffs ("staff",
#   "lay")
# - Positive: "rose", "increase", "increased", "grew", and new deals
#   ("signed", "awarded")
# - "down" and "up" cannot appear: the stop words removed them in Part 1,
#   although "down" is in 81 negative sentences

# %% [markdown]
# # Part 3: Word Embeddings with Word2Vec

# %% [markdown]
# ## Cell 3.1: Train Word2Vec on the train tokens
#
# **Goal**
# - Train on the train sentences only, so the test sentences stay unseen
# - Use the settings of the API notebook:
#   - `vector_size=50`, `window=5`, `min_count=2`
#   - `sg=1`: skip-gram, since the default CBOW collapses on a small corpus
#   - `epochs=30`
#   - `workers=1` with `seed=42`, so the run is reproducible

# %%
# Train the word vectors on the full train set.
w2v = gensim.models.Word2Vec(
    sentences=train_df["tokens"].tolist(),
    vector_size=50,
    window=5,
    min_count=2,
    sg=1,
    epochs=30,
    workers=1,
    seed=42,
)

print("w2v=", w2v)
# Outcome: 3,772 of the 7,778 train tokens get a vector.
# - The other tokens appear only once, so `min_count=2` drops them

# %% [markdown]
# ## Cell 3.2: Nearest neighbors of the 5 finance terms
#
# **Goal**
# - `most_similar()` returns the words with the highest cosine similarity
# - Check the 5 finance terms of the project with
#   `cell6_3_most_similar_to_df()`, from the API notebook

# %%
# Find the 5 nearest neighbors of each finance term.
terms = ["profit", "loss", "debt", "sales", "shares"]

display(utils.cell6_3_most_similar_to_df(w2v.wv, terms, topn=5))
# Outcome: the neighbors share the context, not the sentiment.
# - "loss" is the 2nd neighbor of "profit" (0.81): opposite words of the
#   same reports get close vectors
# - "loss" -> "narrowed", "slipped"; "shares" -> "voting", "outstanding",
#   "repurchased"
# - "debt" -> bank names ("swedbank", "deutsche", "seb") and "loan"
# - "sales" gets noisy neighbors, e.g., "seppala", "uncertainties"

# %% [markdown]
# # Part 4: Sentence Features from Word2Vec

# %% [markdown]
# ## Cell 4.1: Mean vector of each sentence
#
# **Goal**
# - Represent each sentence as the mean of the vectors of its words, with
#   `cell9_2_build_mean_vectors()` from the API notebook
#   - Words without a vector (seen once in train, or only in test) are
#     skipped
#   - A sentence with no known word gets the zero vector

# %%
# Build the 50-dimensional features of the train and test sets.
X_train_w2v = utils.cell9_2_build_mean_vectors(
    w2v.wv, train_df["tokens"].tolist()
)
X_test_w2v = utils.cell9_2_build_mean_vectors(w2v.wv, test_df["tokens"].tolist())

print("X_train_w2v.shape=", X_train_w2v.shape)
print("X_test_w2v.shape=", X_test_w2v.shape)
# Outcome: one 50-dimensional row per sentence.
# - 6 train sentences get the zero vector: the 4 empty ones of Cell 1.7, and
#   2 with only words seen once
# - Every test sentence has at least one known word

# %% [markdown]
# ## Cell 4.2: Fit a logistic regression on the mean vectors
#
# **Goal**
# - The mean of unit word vectors has small, uneven coordinates, so
#   standardize each feature before the logistic regression
#   - `StandardScaler` learns the mean and the scale on the train set only,
#     inside the pipeline
# - Keep the default settings of the logistic regression, as for TF-IDF

# %%
# Fit the scaler and the classifier on the train set.
w2v_clf = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.linear_model.LogisticRegression(max_iter=1000),
)
w2v_clf.fit(X_train_w2v, y_train)
train_accuracy = w2v_clf.score(X_train_w2v, y_train)

print("train_accuracy=", round(train_accuracy, 3))
# Outcome: 67.0% accuracy on the train set, against 85.2% for TF-IDF.
# - 50 averaged dimensions cannot separate the train sentences as well as
#   7,778 token weights, so this model underfits

# %% [markdown]
# # Part 5: Evaluate the Models on the Test Set

# %% [markdown]
# ## Cell 5.1: Predict the test labels
#
# **Goal**
# - Run the 3 fitted models once on the test set
# - Nothing was chosen by looking at the test set before this cell

# %%
# Predict the test labels with each model.
preds = {
    "majority_class": majority.predict(np.zeros((len(test_df), 1))),
    "tfidf_logreg": tfidf_clf.predict(X_test_tfidf),
    "word2vec_logreg": w2v_clf.predict(X_test_w2v),
}
# Count the predicted labels of each model, next to the true labels.
pred_counts_df = pd.DataFrame(
    {name: pd.Series(y_pred).value_counts() for name, y_pred in preds.items()}
)
pred_counts_df["true"] = y_test.value_counts()

display(pred_counts_df.fillna(0).astype(int).loc[utils.LABELS])
# Outcome: both trained models predict neutral too often.
# - TF-IDF predicts 733 neutral and Word2Vec 730, against 575 true ones
# - They predict only 68 and 59 negatives, against 121 true ones

# %% [markdown]
# ## Cell 5.2: Accuracy and macro-F1 of every model
#
# **Goal**
# - Accuracy rewards the neutral class, which is 59.4% of the test set
# - Macro-F1 averages the F1 of the 3 classes, so the small negative class
#   counts as much as neutral

# %%
# Score the 3 models on the test set.
results_df = utils.cell5_2_build_results_table(y_test, preds)

display(results_df)
# Outcome: TF-IDF wins, and Word2Vec does not beat it.
# - majority_class: 0.594 accuracy, 0.248 macro-F1
# - tfidf_logreg: 0.737 accuracy, 0.638 macro-F1
# - word2vec_logreg: 0.665 accuracy, 0.530 macro-F1
# - Word2Vec beats the majority class, but trails TF-IDF by 0.072 accuracy
#   and 0.108 macro-F1

# %% [markdown]
# ## Cell 5.3: Confusion matrix of every model
#
# **Goal**
# - See which classes each model confuses
# - Rows are the true labels, columns the predicted labels

# %%
# Plot the 3 confusion matrices side by side.
utils.cell5_3_plot_confusion_matrices(y_test, preds, figsize=(15, 4.5))
# Outcome: most errors of both trained models are sentences pushed to
# neutral.
# - TF-IDF finds 50 of the 121 negatives (41%) and 123 of the 272 positives
#   (45%)
# - Word2Vec finds 34 negatives (28%) and 96 positives (35%)
# - Word2Vec calls 37 negatives positive, against 17 for TF-IDF: likely
#   because the mean vector mixes up opposite words such as "profit" and
#   "loss"

# %% [markdown]
# ## Cell 5.4: Test sentences whose tokens are also in the train set
#
# **Goal**
# - Part 1 dropped the exact duplicates, but `tokenize()` drops the numbers
#   - 2 different sentences can then get the same tokens, e.g., 2 quarterly
#     reports
# - Count the test sentences whose token list is also a train token list

# %%
# Flag the test sentences with a token list seen in train.
is_seen = utils.cell5_4_find_seen_token_lists(train_df, test_df)

print("is_seen.sum()=", int(is_seen.sum()))
display(test_df.loc[is_seen, ["sentence", "label"]].head())
# Outcome: 23 of the 968 test sentences (2.4%) have a train token list.
# - E.g., "Operating profit rose to EUR 13.5 mn from EUR ...", which differs
#   from a train sentence only in its numbers

# %% [markdown]
# ## Cell 5.5: Scores on the test sentences with new token lists
#
# **Goal**
# - Score the models again without the sentences flagged in Cell 5.4
# - If the scores barely move, the near-duplicates do not inflate the results

# %%
# Score the 3 models on the test sentences with an unseen token list.
is_new = ~is_seen.to_numpy()
preds_new = {name: y_pred[is_new] for name, y_pred in preds.items()}
results_new_df = utils.cell5_2_build_results_table(y_test[is_new], preds_new)

display(results_new_df)
# Outcome: every score moves by 0.01 or less, and the ranking is the same.
# - tfidf_logreg: 0.733 accuracy, 0.628 macro-F1
# - word2vec_logreg: 0.661 accuracy, 0.521 macro-F1

# %% [markdown]
# # Part 6: Visualize the Embeddings

# %% [markdown]
# ## Cell 6.1: Project the 200 most frequent words to 2D
#
# **Goal**
# - PCA keeps the 2 directions along which the word vectors vary most
# - Fit it on the 200 most frequent train words with
#   `cell6_1_project_top_words()`
#   - A finance term outside the top 200 is projected with the same PCA

# %%
# Project the vectors of the most frequent words.
words_df, explained_ratio = utils.cell6_1_project_top_words(
    w2v.wv, terms, n_words=200
)

print("explained_ratio=", explained_ratio.round(3))
display(words_df[words_df["word"].isin(terms)])
# Outcome: the 2 axes explain only 12.6% of the variance (8.0% and 4.6%).
# - "sales", "profit", "shares", and "loss" are in the top 200
# - "debt" appears only 10 times in train, so it is projected without being
#   part of the fit

# %% [markdown]
# ## Cell 6.2: Word map with the finance terms labeled
#
# **Goal**
# - Plot the 2D coordinates of the frequent words, with the 5 finance terms
#   in red

# %%
# Plot the word map.
utils.cell6_2_plot_word_map(words_df, terms, explained_ratio, figsize=(10, 7))
# Outcome: "profit" and "loss" sit next to each other, far right on PC1.
# - The map groups words by the reports they appear in, not by sentiment
# - With only 12.6% of the variance shown, the distances are rough

# %% [markdown]
# # Part 7: Do the Removed Stop Words Matter?
#
# - This is an extra analysis: the main results of Part 5 stay as they are
# - Cell 1.5 showed that the stop words remove "down", "up", and "not"
# - Refit the TF-IDF model with these words kept, and change nothing else:
#   - Same train and test rows, same default logistic regression
#   - Dictionary, TF-IDF weights, and classifier fit on the train set only
# - Nothing is chosen with the test set: both models are reported as they
#   are, and Part 5 keeps the original one

# %% [markdown]
# ## Cell 7.1: Tokenize again, keeping the direction and negation words
#
# **Goal**
# - Keep the 9 words of Cell 1.5 and remove every other Gensim stop word
# - `cell7_1_tokenize_keeping_words()` adds the column `tokens_kept`, and
#   leaves the column `tokens` of the earlier parts as it is

# %%
# Tokenize the train and test sentences, keeping the words of Cell 1.5.
train_df = utils.cell7_1_tokenize_keeping_words(train_df, words)
test_df = utils.cell7_1_tokenize_keeping_words(test_df, words)
# Count the train tokens that the kept words add.
n_added = train_df["tokens_kept"].apply(len) - train_df["tokens"].apply(len)
has_down = train_df["tokens_kept"].apply(lambda tokens: "down" in tokens)

print("n_added.sum()=", int(n_added.sum()))
print("n_empty=", int((train_df["tokens_kept"].apply(len) == 0).sum()))
display(train_df.loc[has_down, ["sentence", "tokens_kept"]].head(3))
# Outcome: keeping the 9 words adds 499 tokens to the train sentences.
# - The train sentences with no token drop from 4 to 2, e.g., "It 's not ."
#   now keeps "not"
# - "down" and "up" now stay, e.g., in the "Operating profit totalled ...,
#   down from ..." reports

# %% [markdown]
# ## Cell 7.2: Build the TF-IDF features of the new tokens
#
# **Goal**
# - Fit a new `Dictionary` and `TfidfModel` on the train tokens only, as in
#   Cells 2.3 and 2.4
# - Build the train and test matrices with `cell2_5_build_tfidf_matrix()`

# %%
# Fit the dictionary and the TF-IDF weights on the train tokens.
dictionary_kept = gensim.corpora.Dictionary(train_df["tokens_kept"])
train_bow_kept = [
    dictionary_kept.doc2bow(tokens) for tokens in train_df["tokens_kept"]
]
tfidf_kept = gensim.models.TfidfModel(train_bow_kept)
# Turn the train and test sentences into TF-IDF matrices.
X_train_kept = utils.cell2_5_build_tfidf_matrix(
    dictionary_kept, tfidf_kept, train_df["tokens_kept"].tolist()
)
X_test_kept = utils.cell2_5_build_tfidf_matrix(
    dictionary_kept, tfidf_kept, test_df["tokens_kept"].tolist()
)

print("X_train_kept.shape=", X_train_kept.shape)
print("X_test_kept.shape=", X_test_kept.shape)
# Outcome: 7,787 columns, 9 more than the 7,778 of Cell 2.5: one per kept
# word.

# %% [markdown]
# ## Cell 7.3: Fit the same logistic regression on the new features
#
# **Goal**
# - Use the settings of Cell 2.6, so only the tokens change
# - Report only the train accuracy here, as in Cell 2.6

# %%
# Fit the classifier on the train set.
kept_clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
kept_clf.fit(X_train_kept, y_train)
train_accuracy = kept_clf.score(X_train_kept, y_train)

print("train_accuracy=", round(train_accuracy, 3))
# Outcome: 86.7% accuracy on the train set, against 85.2% in Cell 2.6.

# %% [markdown]
# ## Cell 7.4: Words with the largest weights for each class
#
# **Goal**
# - Check whether the kept words, e.g., "down" and "up", now get large
#   weights, with `cell2_7_top_words_per_class()` as in Cell 2.7

# %%
# List the 8 tokens with the largest weight for each class.
top_words_kept_df = utils.cell2_7_top_words_per_class(
    kept_clf, dictionary_kept, topn=8
)

with pd.option_context("display.max_colwidth", None):
    display(top_words_kept_df)
# Outcome: the kept words get some of the largest weights.
# - "down" is now the top negative word (4.4), above "decreased" (4.1)
# - "up" is the 3rd positive word (2.9)
# - "no" and "not" push toward neutral, not negative

# %% [markdown]
# ## Cell 7.5: Compare the 2 TF-IDF models on the test set
#
# **Goal**
# - Score the original TF-IDF model of Part 5 and the new one on the same
#   968 test sentences, in one table

# %%
# Score both TF-IDF models on the test set.
preds_kept = {
    "tfidf_logreg": preds["tfidf_logreg"],
    "tfidf_logreg_kept_words": kept_clf.predict(X_test_kept),
}
results_kept_df = utils.cell5_2_build_results_table(y_test, preds_kept)

display(results_kept_df)
# Outcome: keeping the words helps a little on both metrics.
# - Accuracy: 0.749 against 0.737 (+0.012, about 12 of the 968 sentences)
# - Macro-F1: 0.667 against 0.638 (+0.029)
# - This is 1 split of 968 sentences, so a gain this small may not hold on
#   another split
# - Part 5 still reports the original model, as the project spec asks

# %% [markdown]
# ## Summary: What the Results Show
#
# - TF-IDF plus logistic regression is the best model on the test set
#   - 0.737 accuracy and 0.638 macro-F1, against 0.594 and 0.248 for always
#     predicting neutral
# - The mean `Word2Vec` vector does not beat TF-IDF
#   - 0.665 accuracy and 0.530 macro-F1
#   - 3,868 train sentences are likely too few to learn good word vectors
#   - Opposite words such as "profit" and "loss" share their contexts, so
#     they get close vectors, and the mean may blur them further
# - Both trained models miss most of the small negative class
#   - TF-IDF finds 41% of the negatives, and Word2Vec 28%
#   - This is why macro-F1 is lower than accuracy
# - Keeping the direction and negation words helps TF-IDF a little (Part 7)
#   - The stop words of Part 1 remove "down" (81 negative sentences), "up",
#     and "not"
#   - Kept, "down" becomes the top negative word and "up" a top positive word
#   - Test accuracy goes from 0.737 to 0.749 and macro-F1 from 0.638 to
#     0.667, on 1 split, so the gain may not hold on another one
# - The duplicates do not inflate the scores
#   - The exact duplicates were dropped before the split
#   - The 23 test sentences whose tokens match a train sentence change every
#     score by 0.01 or less
# - Nothing was tuned on the test set: all models use default settings, and
#   `Word2Vec` reuses the settings of the API notebook
