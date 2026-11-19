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
# # Gensim API
#
# - This notebook explores the core Gensim API on a small slice of the
#   Financial PhraseBank: the same tokenized sentences go through every model
# - The pedagogical arc:
#   - Load a small slice of tokenized sentences with the utils of the example
#     notebook
#   - Map tokens to ids and count them with `Dictionary` and `doc2bow()`
#   - Reweight the counts with `TfidfModel`
#   - Search for similar sentences with `MatrixSimilarity`
#   - Find topics with `LdaModel` and score them with `CoherenceModel`
#   - Learn word vectors with `Word2Vec` and `FastText`
#   - Save, load, and query word vectors with `KeyedVectors`
#   - Turn word vectors into sentence features
#   - Compare the Gensim representations with NLTK and SBert

# %%
# !pip install -q gensim==4.4.0 scikit-learn==1.7.2 nltk==3.9.1

import gensim
import nltk
import sklearn

print("gensim version: ", gensim.__version__)
print("scikit-learn version: ", sklearn.__version__)
print("nltk version: ", nltk.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os

import gensim.corpora
import gensim.models
import gensim.similarities
import gensim.utils
import nltk.tokenize
import numpy as np
import pandas as pd
import sklearn.linear_model

# %%
import helpers.hnotebook as hnotebook

import gensim_utils as utils

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
# - **What problem it solves**: turn raw text into vectors and topics without
#   labels
#   - Each model streams the corpus, so the corpus does not need to fit in
#     memory
# - **Key abstraction**: a corpus is an iterable of sparse vectors, and a model
#   is a transformation from one vector space to another
#   - `model[vector]` transforms one document
#   - `model[corpus]` transforms a whole corpus lazily
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `Dictionary(texts)` | Maps each token to an integer id | Also counts the document frequency of each token |
# | `dictionary.doc2bow(tokens)` | Bag-of-words vector of one document | Sparse list of `(token_id, count)`, unknown tokens dropped |
# | `TfidfModel(corpus)` | Reweights counts by rarity | `tfidf[bow]` is a sparse list of `(token_id, weight)` |
# | `MatrixSimilarity(corpus)` | Index for cosine similarity search | `index[query]` returns one score per indexed document |
# | `LdaModel(corpus, id2word)` | Topic model | Each topic is a distribution over words |
# | `CoherenceModel(model, texts)` | Scores how interpretable the topics are | Higher `c_v` is better |
# | `Word2Vec(sentences)` | Learns one dense vector per word | Words in similar contexts get close vectors |
# | `FastText(sentences)` | Word2Vec on character n-grams | Builds vectors for unseen words |
# | `KeyedVectors` | The trained vectors alone, as `model.wv` | Query, save, and load without the model |
#
# - **Key classes**:
#   - `gensim.corpora.Dictionary`: the vocabulary and the bag-of-words
#     conversion
#   - `gensim.models.TfidfModel`, `gensim.models.LdaModel`: sparse models of
#     documents
#   - `gensim.models.Word2Vec`, `gensim.models.FastText`: dense models of words
#   - `gensim.models.KeyedVectors`: the word vectors and the queries on them
#   - `gensim.similarities.MatrixSimilarity`: similarity search over documents

# %% [markdown]
# # Part 1: Load a Small Slice of the Data

# %% [markdown]
# ## Cell 1.1: Quiet the Gensim progress logs
#
# **Goal**
# - Gensim reports the progress of every model through the `logging` module,
#   at the `INFO` level
#   - E.g., one line per training epoch, which floods the notebook
# - Raise the level of the `gensim` logger to `WARNING`, keeping its warnings
#   - Set it back to `INFO` to follow a long training run

# %%
# Show only the warnings and errors of Gensim.
gensim_log = logging.getLogger("gensim")
gensim_log.setLevel(logging.WARNING)

print("gensim log level=", logging.getLevelName(gensim_log.level))
# Outcome: the level is WARNING, so the next cells show only their own
# output.

# %% [markdown]
# ## Cell 1.2: Load, clean, and tokenize the sentences
#
# **Goal**
# - Reuse the functions of the example notebook from `gensim_utils.py`:
#   - `cell1_1_load_sentences()` downloads and loads the 4,846 sentences
#   - `cell1_3_drop_duplicates()` drops the repeated sentences
#   - `cell1_4_tokenize_sentences()` adds the `tokens` column, with
#     `simple_preprocess()` and the Gensim stop words removed

# %%
# Load the sentences, drop the duplicates, and tokenize them.
df = utils.cell1_1_load_sentences("data")
df = utils.cell1_3_drop_duplicates(df)
df = utils.cell1_4_tokenize_sentences(df)

print("df.shape=", df.shape)
display(df[["sentence", "label", "tokens"]].head(3))
# Outcome: 4,836 unique sentences, each with its label and its tokens.

# %% [markdown]
# ## Cell 1.3: Take a small slice of the train and test sets
#
# **Goal**
# - Split the sentences with the same stratified split as the example notebook
# - Keep the first 2,000 train and 500 test sentences, so that every model
#   trains in seconds
#   - The split shuffles the sentences, so the first rows are a random sample

# %%
# Split the sentences, then keep a slice of each set.
train_df, test_df = utils.cell1_6_split_train_test(df)
train_df = train_df.head(2000)
test_df = test_df.head(500)
train_texts = train_df["tokens"].tolist()
test_texts = test_df["tokens"].tolist()

print("len(train_texts)=", len(train_texts))
print("len(test_texts)=", len(test_texts))
display(train_df["label"].value_counts(normalize=True).round(3))
# Outcome: 2,000 train and 500 test sentences.
# - The slice keeps the class shares of the full data: 59.2% neutral, 27.8%
#   positive, and 12.9% negative

# %% [markdown]
# # Part 2: Bag of Words with Dictionary

# %% [markdown]
# ## Cell 2.1: Build a Dictionary
#
# **Goal**
# - `Dictionary` gives each distinct token an integer id
# - It also counts in how many sentences each token appears (`dfs`), which
#   the next cells use to filter tokens and to compute TF-IDF

# %%
# Build the vocabulary of the train slice.
dictionary = gensim.corpora.Dictionary(train_texts)

print("dictionary=", dictionary)
print("dictionary.token2id['profit']=", dictionary.token2id["profit"])
# Outcome: 5,469 distinct tokens, and "profit" has the id 130.

# %% [markdown]
# ## Cell 2.2: Filter rare and common tokens
#
# **Goal**
# - `filter_extremes()` drops the tokens that carry little information:
#   - `no_below=2`: tokens in fewer than 2 sentences, mostly names and typos
#   - `no_above=0.5`: tokens in more than half of the sentences
# - It changes the dictionary in place and gives new, compact ids

# %%
# Drop the rare and the very common tokens.
n_before = len(dictionary)
dictionary.filter_extremes(no_below=2, no_above=0.5)

print("n_before=", n_before)
print("len(dictionary)=", len(dictionary))
# Outcome: 2,380 tokens remain out of 5,469.
# - 3,012 of the 3,089 dropped tokens appear only once in the whole slice

# %% [markdown]
# ## Cell 2.3: Convert one sentence with doc2bow
#
# **Goal**
# - `doc2bow()` turns a list of tokens into a sparse vector of
#   `(token_id, count)` pairs
#   - Word order is lost: only the counts remain
# - `cell2_3_bow_to_df()` puts the token back next to each id
# - The next parts reuse this example sentence

# %%
# Convert an example sentence with repeated words to a bag of words.
example_pos = 361
bow = dictionary.doc2bow(train_texts[example_pos])

print("example sentence=", train_df["sentence"].iloc[example_pos])
print("bow=", bow)
display(utils.cell2_3_bow_to_df(dictionary, bow))
# Outcome: 6 `(token_id, count)` pairs for 8 tokens.
# - "eur" and "million" appear twice, so they get a count of 2
# - The numbers are already gone, since the tokenizer drops them

# %% [markdown]
# ## Cell 2.4: Unknown tokens are dropped
#
# **Goal**
# - A token that is not in the dictionary has no id, so `doc2bow()` drops it
#   silently
# - `return_missing=True` also returns the dropped tokens, which is a quick
#   check of the vocabulary coverage

# %%
# Convert a sentence with a word that never appears in the news.
bow_new, missing = dictionary.doc2bow(
    ["profit", "rose", "blockchain"], return_missing=True
)

print("bow_new=", bow_new)
print("missing=", missing)
# Outcome: "profit" and "rose" get their ids, and "blockchain" is
# dropped from the vector and reported in `missing`.

# %% [markdown]
# ## Cell 2.5: Build the bag-of-words corpus
#
# **Goal**
# - A Gensim corpus is any iterable of bag-of-words vectors
#   - A list works for a small corpus
#   - A generator that reads a file works for a corpus that does not fit in
#     memory

# %%
# Convert every train sentence.
train_corpus = [dictionary.doc2bow(tokens) for tokens in train_texts]

print("len(train_corpus)=", len(train_corpus))
print("train_corpus[1]=", train_corpus[1])
# Outcome: 2,000 sparse vectors, one per sentence.

# %% [markdown]
# # Part 3: Reweight the Counts with TfidfModel

# %% [markdown]
# ## Cell 3.1: Fit a TfidfModel
#
# **Goal**
# - `TfidfModel` learns the inverse document frequency of each token from the
#   corpus
#   - $\text{idf}(t) = \log_2 \frac{N}{\text{df}(t)}$, with $N$ sentences and
#     $\text{df}(t)$ sentences containing the token $t$
# - Each vector is then normalized to unit length

# %%
# Learn the idf of each token from the train corpus.
tfidf = gensim.models.TfidfModel(train_corpus)

print("tfidf=", tfidf)
# Outcome: the model has seen 2,000 sentences with 18,250 non-zero
# entries, about 9 distinct tokens per sentence.

# %% [markdown]
# ## Cell 3.2: Reweight one sentence with TF-IDF
#
# **Goal**
# - `tfidf[bow]` transforms one bag-of-words vector into a TF-IDF vector
# - Compare the weight of each token with its count and its document
#   frequency, with `cell3_2_tfidf_to_df()`

# %%
# Transform the example sentence.
tfidf_vec = tfidf[bow]

display(utils.cell3_2_tfidf_to_df(dictionary, bow, tfidf_vec))
# Outcome: the weight follows rarity, not only the count.
# - "million" gets the top weight: count 2, in 127 sentences
# - "eur" also has count 2, but is in 355 sentences, so it weighs less than
#   "rose", which appears once and is in 46 sentences

# %% [markdown]
# # Part 4: Similarity Search with MatrixSimilarity

# %% [markdown]
# ## Cell 4.1: Index the TF-IDF vectors
#
# **Goal**
# - `MatrixSimilarity` stores the corpus as a dense matrix of unit vectors
#   - A query is then one matrix product, giving the cosine similarity with
#     every sentence
#   - `num_features` is the size of the vocabulary
# - For a corpus that does not fit in memory, `Similarity` shards the index on
#   disk with the same API

# %%
# Index the TF-IDF vectors of the train slice.
index = gensim.similarities.MatrixSimilarity(
    tfidf[train_corpus], num_features=len(dictionary)
)

print("index=", index)
print("index.index.shape=", index.index.shape)
# Outcome: a dense 2,000 x 2,380 matrix, one unit vector per sentence.

# %% [markdown]
# ## Cell 4.2: Find the sentences most similar to a query
#
# **Goal**
# - Query the index with a test sentence, after the same 2 steps as the
#   corpus: `doc2bow()`, then `tfidf[...]`
# - `cell4_2_top_matches()` shows the train sentences with the highest scores

# %%
# Query the index with a test sentence about sales growth.
query_pos = 4
query_tokens = test_texts[query_pos]
sims = index[tfidf[dictionary.doc2bow(query_tokens)]]

print("query sentence=", test_df["sentence"].iloc[query_pos])
print("query label=", test_df["label"].iloc[query_pos])
# Show the whole sentences, without truncating the column.
with pd.option_context("display.max_colwidth", None):
    display(utils.cell4_2_top_matches(sims, train_df, topn=5))
# Outcome: the matches share the words of the query, not its sentiment.
# - All 5 match on "net sales", "eur", "mn", and "corresponding period"
# - The top match is negative: "down" is a stop word, so TF-IDF cannot see
#   the direction of the change

# %% [markdown]
# # Part 5: Topics with LdaModel and CoherenceModel

# %% [markdown]
# ## Cell 5.1: Fit an LdaModel
#
# **Goal**
# - LDA explains each sentence as a mix of topics, and each topic as a
#   distribution over words
# - It works on counts, so it takes the bag-of-words corpus, not the TF-IDF
#   one
# - `random_state` fixes the initialization, so that the topics can be
#   reproduced

# %%
# Fit 5 topics on the bag-of-words corpus.
lda = gensim.models.LdaModel(
    train_corpus,
    id2word=dictionary,
    num_topics=5,
    passes=10,
    random_state=42,
)

print("lda=", lda)
# Outcome: an LDA model with 5 topics over the 2,380 tokens.

# %% [markdown]
# ## Cell 5.2: Show the top words of each topic
#
# **Goal**
# - `show_topic()` returns the most probable words of a topic
# - `cell5_2_topics_to_df()` collects them in one table

# %%
# List the 8 most probable words of each topic.
topics_df = utils.cell5_2_topics_to_df(lda, topn=8)

# Show the whole word lists, without truncating the column.
with pd.option_context("display.max_colwidth", None):
    display(topics_df)
# Outcome: 1 clear topic and 4 mixed ones.
# - Topic 3 is the financial results: "eur", "mn", "profit", "sales"
# - The others share generic words such as "company", "finnish", and
#   "said", since all the sentences come from the same kind of news

# %% [markdown]
# ## Cell 5.3: Topic mix of one sentence
#
# **Goal**
# - `get_document_topics()` returns the topic weights of one sentence
# - Short sentences usually get most of their weight on 1 or 2 topics

# %%
# Show the topic weights of the example sentence.
doc_topics = lda.get_document_topics(bow)

print("example sentence=", train_df["sentence"].iloc[example_pos])
print("doc_topics=", [(topic, round(float(w), 3)) for topic, w in doc_topics])
# Outcome: 91% of the weight is on topic 3, the financial results topic.

# %% [markdown]
# ## Cell 5.4: Score the topics with CoherenceModel
#
# **Goal**
# - The `c_v` coherence checks how often the top words of each topic appear
#   together in the sentences
#   - It needs the tokenized `texts`, not only the corpus
#   - It ranges from 0 to 1, and higher means more interpretable topics
# - `processes=1` keeps the computation in this process

# %%
# Score the coherence of the 5 topics.
coherence_model = gensim.models.CoherenceModel(
    model=lda,
    texts=train_texts,
    dictionary=dictionary,
    coherence="c_v",
    processes=1,
)
c_v = coherence_model.get_coherence()

print("c_v=", round(c_v, 3))
# Outcome: c_v is 0.37, a low score, as expected for short sentences from
# one domain.

# %% [markdown]
# ## Cell 5.5: Choose the number of topics with the coherence
#
# **Goal**
# - Fit one model per number of topics, with the same settings, and compare
#   their `c_v`
# - `cell5_5_coherence_by_num_topics()` runs the same 2 steps as Cells 5.1
#   and 5.4 for each number

# %%
# Compare 3, 5, and 8 topics.
coherence_df = utils.cell5_5_coherence_by_num_topics(
    train_corpus, train_texts, dictionary, [3, 5, 8]
)

display(coherence_df)
# Outcome: the coherence drops as the number of topics grows.
# - 3 topics score best, with 0.386
# - 2,000 short sentences do not hold enough distinct themes for 8 topics

# %% [markdown]
# # Part 6: Word Embeddings with Word2Vec

# %% [markdown]
# ## Cell 6.1: Train a Word2Vec model
#
# **Goal**
# - `Word2Vec` learns one dense vector per word, by predicting each word from
#   the words around it
#   - `vector_size=50`: the length of each vector
#   - `window=5`: how many words on each side count as context
#   - `min_count=2`: words seen once get no vector
#   - `sg=1`: skip-gram, which predicts the context from the word
#   - `epochs=30`: passes over the sentences
#   - `workers=1` with `seed=42`: one thread, so the run is reproducible
# - It takes the token lists directly, not the bag-of-words corpus, since
#   word order matters
# - On a corpus this small, the default CBOW (`sg=0`) collapses: every pair
#   of words gets a similarity close to 1, so skip-gram with more epochs is
#   used

# %%
# Train the word vectors on the train slice.
w2v = gensim.models.Word2Vec(
    sentences=train_texts,
    vector_size=50,
    window=5,
    min_count=2,
    sg=1,
    epochs=30,
    workers=1,
    seed=42,
)

print("w2v=", w2v)
# Outcome: 2,457 words get a 50-dimensional vector.
# - `min_count` counts occurrences, so the vocabulary differs from the 2,380
#   tokens of the dictionary, which counts sentences

# %% [markdown]
# ## Cell 6.2: Look up a word vector
#
# **Goal**
# - The trained vectors live in `w2v.wv`, a `KeyedVectors` object
# - `wv[word]` returns the vector of a word as a numpy array

# %%
# Get the vector of "profit".
vec = w2v.wv["profit"]

print("vec.shape=", vec.shape)
print("vec[:5]=", vec[:5])
# Outcome: a dense vector of 50 floats.

# %% [markdown]
# ## Cell 6.3: Nearest neighbors of finance terms
#
# **Goal**
# - `most_similar()` returns the words with the highest cosine similarity
# - Check the neighbors of 5 finance terms, as the project asks, with
#   `cell6_3_most_similar_to_df()`

# %%
# Find the 5 nearest neighbors of each term.
terms = ["profit", "loss", "debt", "sales", "shares"]

display(utils.cell6_3_most_similar_to_df(w2v.wv, terms, topn=5))
# Outcome: most neighbors are finance words used in the same contexts.
# - "loss" -> "narrowed", "slipped"; "debt" -> "loan", "bond"
# - "shares" -> "voting", "outstanding"
# - "sales" gets noisy neighbors, e.g., "hobby", "breweries": 2,000
#   sentences are few for a common word

# %% [markdown]
# ## Cell 6.4: Similarity of word pairs
#
# **Goal**
# - `similarity()` returns the cosine similarity of 2 words
# - Opposite words often get close vectors, since they appear in the same
#   contexts (e.g., "profit rose" and "profit fell")

# %%
# Compare pairs of related and opposite words.
pairs = [("profit", "loss"), ("rose", "fell"), ("increased", "decreased")]
pairs_df = pd.DataFrame(pairs, columns=["word_1", "word_2"])
pairs_df["similarity"] = [
    round(float(w2v.wv.similarity(a, b)), 3) for a, b in pairs
]

display(pairs_df)
# Outcome: the opposite pairs are all close, from 0.77 to 0.90.
# - Word2Vec learns that the words are used in the same way, not what they
#   mean, so on its own it does not separate "profit" from "loss"

# %% [markdown]
# # Part 7: Subword Embeddings with FastText

# %% [markdown]
# ## Cell 7.1: Train a FastText model
#
# **Goal**
# - `FastText` has the same API as `Word2Vec`, but it also learns a vector
#   for each character n-gram of 3 to 6 characters (`min_n`, `max_n`)
# - A word vector is the sum of the vectors of the word and its n-grams
# - The n-grams are hashed into `bucket` slots
#   - The default of 2,000,000 slots takes about 400 MB for 50 dimensions
#   - 100,000 slots are enough for a vocabulary of a few thousand words

# %%
# Train FastText with the same settings as Word2Vec.
ft = gensim.models.FastText(
    sentences=train_texts,
    vector_size=50,
    window=5,
    min_count=2,
    sg=1,
    epochs=30,
    min_n=3,
    max_n=6,
    bucket=100000,
    workers=1,
    seed=42,
)

print("ft=", ft)
# Outcome: the same vocabulary of 2,457 words as Word2Vec.

# %% [markdown]
# ## Cell 7.2: Word2Vec has no vector for an unseen word
#
# **Goal**
# - A misspelled word is not in the Word2Vec vocabulary, so it has no vector

# %%
# Check a misspelling of "profit" in both vocabularies.
word = "proffit"

print("word in w2v.wv=", word in w2v.wv)
print("word in ft.wv.key_to_index=", word in ft.wv.key_to_index)
# Outcome: False for both: the misspelling is in neither vocabulary.

# %% [markdown]
# ## Cell 7.3: FastText builds a vector for an unseen word
#
# **Goal**
# - FastText builds the vector of an unseen word from its n-grams, so it can
#   still answer `most_similar()`

# %%
# Find the nearest neighbors of the misspelled word.
print("ft.wv.most_similar(word)=", ft.wv.most_similar(word, topn=5))
# Outcome: FastText still finds the right neighbors from the n-grams:
# "profits", "profit", and "profitability".

# %% [markdown]
# # Part 8: Save and Load with KeyedVectors

# %% [markdown]
# ## Cell 8.1: Save the word vectors
#
# **Goal**
# - `wv.save()` writes only the vectors and the vocabulary, not the training
#   state, so the file is smaller than the full model
# - The file goes in `data/`, which is not committed

# %%
# Save the Word2Vec vectors to a file.
kv_path = os.path.join("data", "gensim_api.word2vec.kv")
w2v.wv.save(kv_path)

print("kv_path=", kv_path)
print("file size (KB)=", round(os.path.getsize(kv_path) / 1024, 1))
# Outcome: a file of about 561 KB in `data/`.

# %% [markdown]
# ## Cell 8.2: Load the word vectors
#
# **Goal**
# - `KeyedVectors.load()` reads the file back
# - The loaded vectors are the same as the trained ones

# %%
# Load the vectors and compare them with the trained ones.
kv = gensim.models.KeyedVectors.load(kv_path)

print("kv=", kv)
print("same vectors=", np.allclose(kv.vectors, w2v.wv.vectors))
# Outcome: the 2,457 vectors come back identical.

# %% [markdown]
# ## Cell 8.3: Query the loaded vectors
#
# **Goal**
# - The loaded `KeyedVectors` answer the same queries as `w2v.wv`, without
#   the model
# - Pretrained vectors, e.g., `glove-wiki-gigaword-100` from
#   `gensim.downloader`, load as the same `KeyedVectors` type
#   - They are not downloaded here, to keep the notebook fast

# %%
# Find the nearest neighbors of "loss" with the loaded vectors.
neighbors = kv.most_similar("loss", topn=3)

print("neighbors=", [(word, round(score, 3)) for word, score in neighbors])
# Outcome: the same neighbors as in Cell 6.3: "narrowed", "slipped", and
# "versus".

# %% [markdown]
# ## Cell 8.4: Find the word that does not fit
#
# **Goal**
# - `doesnt_match()` returns the word farthest from the mean of a list of
#   words

# %%
# Ask the loaded vectors which word does not fit with the others.
odd_one = kv.doesnt_match(["profit", "sales", "net", "finland"])

print("odd_one=", odd_one)
# Outcome: "finland" is the odd one out among the finance words.

# %% [markdown]
# # Part 9: Sentence Features from Word Vectors

# %% [markdown]
# ## Cell 9.1: Mean vector of one sentence
#
# **Goal**
# - `get_mean_vector()` averages the vectors of a list of words
# - The mean is a simple sentence vector: same length as a word vector, word
#   order lost

# %%
# Average the vectors of the words of the example sentence.
tokens = [token for token in train_texts[example_pos] if token in kv]
sentence_vec = kv.get_mean_vector(tokens)

print("tokens=", tokens)
print("sentence_vec.shape=", sentence_vec.shape)
# Outcome: one 50-dimensional vector for the sentence.
# - "eur" and "million" appear twice, so they count twice in the mean

# %% [markdown]
# ## Cell 9.2: Sentence features for all the sentences
#
# **Goal**
# - Build one mean vector per sentence with `cell9_2_build_mean_vectors()`
#   - A sentence with no known word gets the zero vector

# %%
# Build the features of the train and test slices.
X_train = utils.cell9_2_build_mean_vectors(kv, train_texts)
X_test = utils.cell9_2_build_mean_vectors(kv, test_texts)

print("X_train.shape=", X_train.shape)
print("X_test.shape=", X_test.shape)
# Outcome: one 50-dimensional row per sentence.
# - 4 train sentences have no token, so they get the zero vector

# %% [markdown]
# ## Cell 9.3: Fit a logistic regression on the sentence features
#
# **Goal**
# - Check that the features carry sentiment: fit a logistic regression on the
#   train slice and score it on the test slice
# - The full comparison with the TF-IDF and majority-class baselines is in
#   the example notebook

# %%
# Fit the classifier and score its accuracy.
clf = sklearn.linear_model.LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label"])
accuracy = clf.score(X_test, test_df["label"])

print("accuracy=", round(accuracy, 3))
# Outcome: 63.2% accuracy, against 59.2% for always predicting neutral.
# - The mean vectors carry some sentiment, but the opposite words of Cell
#   6.4 get close vectors, which limits them

# %% [markdown]
# # Part 10: Compare with NLTK and SBert

# %% [markdown]
# ## Cell 10.1: Tokenize the same sentence with NLTK and Gensim
#
# **Goal**
# - NLTK is a toolkit of text processing steps, e.g., tokenizers, stemmers,
#   and taggers, but it does not learn vectors
# - Compare on one sentence of the dataset:
#   - The NLTK Treebank tokenizer
#   - `simple_preprocess()` of Gensim
#   - `tokenize()` of `gensim_utils.py`, which also removes the stop words

# %%
# Tokenize one sentence with a negation and a year in 3 ways.
sentence = (
    "The retail chain launched in 2007 does not correspond to profit and "
    "sales forecasts of the company ."
)
tokens_df = pd.DataFrame(
    {
        "tokenizer": [
            "nltk TreebankWordTokenizer",
            "gensim simple_preprocess",
            "utils.tokenize",
        ],
        "tokens": [
            nltk.tokenize.TreebankWordTokenizer().tokenize(sentence),
            gensim.utils.simple_preprocess(sentence),
            utils.tokenize(sentence),
        ],
    }
)

print("sentence=", sentence)
# Show the whole token lists, without truncating the column.
with pd.option_context("display.max_colwidth", None):
    display(tokens_df)
# Outcome: 3 different views of the same sentence.
# - NLTK keeps the case, the year "2007", the stop words, and the period
# - `simple_preprocess()` lowercases and drops "2007" and "."
# - `tokenize()` also drops "not", so the negation is lost

# %% [markdown]
# ## Cell 10.2: Gensim, NLTK, and SBert as text representations
#
# | Library | Representation | Trained on | Unseen words | Word order | Cost |
# |---------|----------------|------------|--------------|------------|------|
# | NLTK | Tokens, stems, tags | Nothing, rules and lexicons | Kept as tokens | Kept | Very low |
# | Gensim TF-IDF | Sparse vector, one weight per token | Your corpus | Dropped | Lost | Low |
# | Gensim Word2Vec / FastText | Dense vector per word, averaged per sentence | Your corpus | FastText only | Lost in the mean | Low, CPU in seconds |
# | SBert | Dense vector per sentence | Large pretrained transformer | Handled by subwords | Kept | High, a GPU helps |
#
# - NLTK prepares the text, while Gensim and SBert turn it into vectors
# - Gensim learns from your own sentences:
#   - It is fast and needs no labels
#   - Its quality is limited by the size of the corpus, about 4,800 sentences
#     here
# - SBert brings knowledge from a large pretrained model:
#   - It reads the whole sentence, so "profit did not rise" and "profit rose"
#     get different vectors
#   - It needs `sentence-transformers` and PyTorch, too large for this API
#     notebook

# %% [markdown]
# ## Summary: The Mental Model
#
# - Every Gensim model is a transformation between vector spaces, applied with
#   `model[vector]` to one document or `model[corpus]` to a whole corpus
# - The sparse pipeline is `Dictionary` -> `doc2bow()` -> `TfidfModel` or
#   `LdaModel` -> `MatrixSimilarity`, and the query goes through the same steps
#   as the corpus
# - `Word2Vec` and `FastText` learn dense word vectors from the token lists,
#   and `FastText` also covers unseen words through character n-grams
# - `KeyedVectors` holds the vectors alone: it saves, loads, and answers
#   `most_similar()` without the model, and is also how pretrained vectors load
