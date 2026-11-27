"""
Utility functions for the Gensim sentiment classification of financial news.

The notebook `gensim.example.ipynb` calls these functions, one per cell, to:
- Download and cache the Financial PhraseBank sentences
- Find and drop duplicate sentences
- Tokenize the sentences with `simple_preprocess()` and the Gensim stop words
- Make a stratified train / test split
- Summarize the data in one table
- Turn the TF-IDF vectors into a feature matrix for the baseline
- Score the models and plot their confusion matrices
- Project the Word2Vec vectors of the frequent words to 2D with PCA
- Tokenize again, keeping the direction and negation stop words

The notebook `gensim.API.ipynb` reuses the loading and tokenizing functions
above, and calls the functions in the `gensim.API.ipynb` sections below to
turn the outputs of the Gensim models into tables. The example notebook also
reuses `cell6_3_most_similar_to_df()` and `cell9_2_build_mean_vectors()`
from these sections.

Import as:

import class_project.msml610.Fall2026.projects.UmdTask782_MSML610_Fall2026_Gensim_Project1.gensim_utils as cmfpugpgu
"""

import collections
import logging
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import gensim.corpora
import gensim.matutils
import gensim.models
import gensim.parsing.preprocessing
import gensim.utils
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

_LOG = logging.getLogger(__name__)

# The 3 sentiment classes, in the order used by every table.
LABELS = ["negative", "neutral", "positive"]


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    :return: None, since the loggers are configured in place
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1.1: Download and load the sentences
# #############################################################################


def _download_sentences_file(data_dir: str) -> str:
    """
    Download `Sentences_50Agree.txt` into `data_dir`, unless it is cached there.

    The file comes from the zip archive of the Financial PhraseBank dataset
    repo on Hugging Face. Only the 50% agreement file is extracted, since it
    is the one used by the project.

    :param data_dir: dir where the zip archive and the text file are cached
    :return: path of the cached `Sentences_50Agree.txt`
    """
    _LOG.debug(hprint.to_str("data_dir"))
    url = (
        "https://huggingface.co/datasets/takala/financial_phrasebank/"
        "resolve/main/data/FinancialPhraseBank-v1.0.zip"
    )
    file_name = "Sentences_50Agree.txt"
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        _LOG.info("Using cached '%s'", file_path)
    else:
        # Create the dir incrementally, keeping any file already in it.
        hio.create_dir(data_dir, True)
        # Download the archive once and keep it, so that a later run can
        # extract the other agreement levels without downloading again.
        zip_path = os.path.join(data_dir, "FinancialPhraseBank-v1.0.zip")
        if not os.path.exists(zip_path):
            _LOG.info("Downloading '%s' to '%s'", url, zip_path)
            urllib.request.urlretrieve(url, zip_path)
        # Extract only the 50% agreement file, flattening the archive dir.
        with zipfile.ZipFile(zip_path) as zip_file:
            member = "FinancialPhraseBank-v1.0/" + file_name
            hdbg.dassert_in(
                member,
                zip_file.namelist(),
                "The archive layout changed upstream",
            )
            with open(file_path, "wb") as f:
                f.write(zip_file.read(member))
        _LOG.info("Saved '%s'", file_path)
    _LOG.debug("return=%s", file_path)
    return file_path


def cell1_1_load_sentences(data_dir: str) -> pd.DataFrame:
    """
    Load the Financial PhraseBank sentences into a dataframe.

    Each line of the file is `<sentence>@<label>`, encoded in Latin-1.

    :param data_dir: dir where the dataset is downloaded and cached
    :return: dataframe with one row per line of the file
        ```
        sentence                                            label
        According to Gran , the company has no plans ...   neutral
        ```
    """
    _LOG.debug(hprint.to_str("data_dir"))
    file_path = _download_sentences_file(data_dir)
    # The file is not valid UTF-8: it is Latin-1 (e.g., the byte 0xf1 is "n"
    # with a tilde).
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    # Split on the last `@`, so that an `@` inside a sentence stays in it.
    rows = []
    for line in lines:
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label.strip()})
    df = pd.DataFrame(rows)
    # Check the file against its published description.
    hdbg.dassert_eq(len(df), 4846, "Unexpected number of sentences")
    hdbg.dassert_set_eq(df["label"].unique(), LABELS, "Unexpected labels")
    _LOG.debug("return=%s", df.shape)
    return df


# #############################################################################
# Cell 1.2: Find duplicate sentences
# #############################################################################


def cell1_2_find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    List the sentences that appear more than once, with their labels.

    :param df: dataframe with the columns `sentence` and `label`
    :return: one row per repeated sentence, with the number of copies and the
        set of labels it received
        ```
        sentence                     n_copies  labels              conflict
        The Group 's business ...    2         neutral, positive   True
        ```
    """
    _LOG.debug(hprint.to_str("df.shape"))
    # Count the copies of each sentence and collect its labels.
    grouped = df.groupby("sentence")["label"]
    dups_df = pd.DataFrame(
        {
            "n_copies": grouped.size(),
            "labels": grouped.agg(lambda labels: ", ".join(sorted(set(labels)))),
        }
    )
    # Keep only the repeated sentences.
    dups_df = dups_df[dups_df["n_copies"] > 1]
    # A repeated sentence with more than one label has a conflicting label.
    dups_df["conflict"] = dups_df["labels"].str.contains(",")
    dups_df = dups_df.reset_index()
    _LOG.debug("return=%s", dups_df.shape)
    return dups_df


# #############################################################################
# Cell 1.3: Drop duplicate sentences
# #############################################################################


def cell1_3_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the repeated sentences, before the split.

    - A sentence repeated with the same label keeps one copy
    - A sentence repeated with different labels is dropped entirely, since
      there is no way to know which label is right

    Dropping before the split prevents the same sentence from landing in both
    the train and the test set.

    :param df: dataframe with the columns `sentence` and `label`
    :return: dataframe with unique sentences, with a fresh index
    """
    _LOG.debug(hprint.to_str("df.shape"))
    # Find the sentences with more than one label.
    n_labels = df.groupby("sentence")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    is_conflicting = df["sentence"].isin(conflicting)
    # Drop every copy of a conflicting sentence, then the exact duplicates.
    df_out = df[~is_conflicting]
    df_out = df_out.drop_duplicates(subset="sentence", keep="first")
    df_out = df_out.reset_index(drop=True)
    hdbg.dassert_no_duplicates(
        df_out["sentence"].tolist(), "Sentences must be unique after dropping"
    )
    # Report how many rows each rule removed.
    n_dropped = len(df) - len(df_out)
    n_conflicting = int(is_conflicting.sum())
    _LOG.info(
        "Dropped '%s' rows: '%s' copies of conflicting sentences, '%s' exact "
        "duplicates",
        n_dropped,
        n_conflicting,
        n_dropped - n_conflicting,
    )
    _LOG.debug("return=%s", df_out.shape)
    return df_out


# #############################################################################
# Cell 1.4: Tokenize the sentences
# #############################################################################


def tokenize(sentence: str) -> List[str]:
    """
    Tokenize a sentence and remove the Gensim stop words.

    `simple_preprocess()` lowercases the text, splits it on non-letters, and
    keeps the tokens with 2 to 15 characters, so numbers are dropped.

    :param sentence: raw sentence
    :return: list of tokens
        ```
        "Operating profit rose to EUR 13.1 mn ."
        -> ["operating", "profit", "rose", "eur", "mn"]
        ```
    """
    tokens = gensim.utils.simple_preprocess(sentence)
    tokens = [
        token
        for token in tokens
        if token not in gensim.parsing.preprocessing.STOPWORDS
    ]
    return tokens


def cell1_4_tokenize_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the column `tokens` with the tokens of each sentence.

    :param df: dataframe with the column `sentence`
    :return: copy of `df` with the extra column `tokens`
    """
    _LOG.debug(hprint.to_str("df.shape"))
    df_out = df.copy()
    df_out["tokens"] = df_out["sentence"].apply(tokenize)
    _LOG.debug("return=%s", df_out.shape)
    return df_out


# #############################################################################
# Cell 1.5: Check what the stop words remove
# #############################################################################


def cell1_5_count_removed_sentiment_words(
    df: pd.DataFrame, words: List[str]
) -> pd.DataFrame:
    """
    Count the sentences that contain each word removed by the stop words.

    The Gensim stop words include words that carry the direction of a change
    or a negation (e.g., "up", "down", "not"), which matter for sentiment.

    :param df: dataframe with the columns `sentence` and `label`
    :param words: words to check; each must be a Gensim stop word
    :return: one row per word, with the number of sentences containing it,
        overall and per label
        ```
        word  n_sentences  negative  neutral  positive
        down  96           81        14       1
        ```
    """
    _LOG.debug(hprint.to_str("df.shape words"))
    hdbg.dassert_is_subset(
        words,
        gensim.parsing.preprocessing.STOPWORDS,
        "Only stop words are removed by `tokenize()`",
    )
    # Tokenize without removing the stop words, to see where they occur.
    raw_tokens = df["sentence"].apply(
        lambda sentence: set(gensim.utils.simple_preprocess(sentence))
    )
    # Count the sentences that contain each word, overall and per label.
    rows = []
    for word in words:
        has_word = raw_tokens.apply(lambda tokens: word in tokens)
        counts = df.loc[has_word, "label"].value_counts()
        row = {"word": word, "n_sentences": int(has_word.sum())}
        for label in LABELS:
            row[label] = int(counts.get(label, 0))
        rows.append(row)
    counts_df = pd.DataFrame(rows)
    _LOG.debug("return=%s", counts_df.shape)
    return counts_df


# #############################################################################
# Cell 1.6: Stratified train / test split
# #############################################################################


def cell1_6_split_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the sentences into a train and a test set, stratified by label.

    Stratifying keeps the class proportions equal in the two sets, which
    matters because neutral sentences are about 60% of the data.

    :param df: dataframe with the column `label`
    :param test_size: fraction of the sentences in the test set
    :param random_state: seed of the split, so that it can be reproduced
    :return: train and test dataframes, each with a fresh index
    """
    _LOG.debug(hprint.to_str("df.shape test_size random_state"))
    train_df, test_df = sklearn.model_selection.train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # The two sets must not share a sentence.
    overlap = set(train_df["sentence"]) & set(test_df["sentence"])
    hdbg.dassert_eq(len(overlap), 0, "Train and test share sentences")
    _LOG.debug("return=%s", (train_df.shape, test_df.shape))
    return train_df, test_df


# #############################################################################
# Cell 1.7: Data summary table
# #############################################################################


def _summarize_split(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the summary statistics of one set of sentences.

    :param name: name of the set (e.g., "train")
    :param df: dataframe with the columns `tokens` and `label`
    :return: dict with one entry per column of the summary table
    """
    n_tokens = df["tokens"].apply(len)
    vocab = collections.Counter(
        token for tokens in df["tokens"] for token in tokens
    )
    row = {
        "split": name,
        "n_sentences": len(df),
        "vocab_size": len(vocab),
        "tokens_mean": round(n_tokens.mean(), 1),
        "tokens_median": int(n_tokens.median()),
        "tokens_min": int(n_tokens.min()),
        "tokens_max": int(n_tokens.max()),
        # A sentence with no token gets no Word2Vec vector later.
        "n_empty": int((n_tokens == 0).sum()),
    }
    # Class counts, with the share of the set in parentheses.
    counts = df["label"].value_counts()
    for label in LABELS:
        count = int(counts.get(label, 0))
        row[label] = f"{count} ({count / len(df):.1%})"
    return row


def cell1_7_build_data_table(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the table with the vocabulary size, tokens per sentence, and classes.

    :param train_df: train set, with the columns `tokens` and `label`
    :param test_df: test set, with the columns `tokens` and `label`
    :return: one row per set (train, test, all)
        ```
        split  n_sentences  vocab_size  tokens_mean  ...  positive
        train  3868         7778        11.6         ...  1089 (28.2%)
        ```
    """
    _LOG.debug(hprint.to_str("train_df.shape test_df.shape"))
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    rows = [
        _summarize_split("train", train_df),
        _summarize_split("test", test_df),
        _summarize_split("all", all_df),
    ]
    table_df = pd.DataFrame(rows).set_index("split")
    _LOG.debug("return=%s", table_df.shape)
    return table_df


# #############################################################################
# Cell 2.5: Turn the TF-IDF vectors into a feature matrix
# #############################################################################


def cell2_5_build_tfidf_matrix(
    dictionary: gensim.corpora.Dictionary,
    tfidf: gensim.models.TfidfModel,
    texts: List[List[str]],
) -> scipy.sparse.csr_matrix:
    """
    Turn tokenized sentences into a sparse TF-IDF matrix for scikit-learn.

    Each sentence goes through `doc2bow()` and then `tfidf[bow]`, so the
    tokens that are not in `dictionary` (e.g., test tokens never seen in the
    train set) are dropped.

    :param dictionary: dictionary fitted on the train sentences
    :param tfidf: TF-IDF model fitted on the train bag-of-words corpus
    :param texts: tokenized sentences
    :return: sparse matrix with one row per sentence and one column per token
        of `dictionary`
    """
    _LOG.debug(hprint.to_str("len(dictionary) len(texts)"))
    tfidf_corpus = [tfidf[dictionary.doc2bow(tokens)] for tokens in texts]
    # `corpus2csc()` puts the sentences in the columns, so transpose it.
    matrix = gensim.matutils.corpus2csc(
        tfidf_corpus, num_terms=len(dictionary), num_docs=len(texts)
    )
    matrix = matrix.T.tocsr()
    hdbg.dassert_eq(
        matrix.shape[0], len(texts), "One row per sentence is expected"
    )
    _LOG.debug("return=%s", matrix.shape)
    return matrix


# #############################################################################
# Cell 2.7: Words with the largest weights for each class
# #############################################################################


def cell2_7_top_words_per_class(
    clf: sklearn.linear_model.LogisticRegression,
    dictionary: gensim.corpora.Dictionary,
    *,
    topn: int = 8,
) -> pd.DataFrame:
    """
    Show the words that push a logistic regression most toward each class.

    :param clf: logistic regression fitted on the TF-IDF matrix, whose
        columns are the token ids of `dictionary`
    :param dictionary: dictionary that produced the TF-IDF matrix
    :param topn: number of words per class
    :return: one row per class, with its words, largest weight first, and the
        weight in parentheses
        ```
        label     top_words
        negative  decreased (4.1), fell (3.1), ...
        ```
    """
    _LOG.debug(hprint.to_str("clf.classes_ topn"))
    hdbg.dassert_eq(
        clf.coef_.shape[1],
        len(dictionary),
        "The classifier must be fitted on one column per token",
    )
    rows = []
    for label, coefs in zip(clf.classes_, clf.coef_):
        # Sort the token ids by decreasing weight for this class.
        top_ids = np.argsort(-coefs)[:topn]
        words = [f"{dictionary[i]} ({coefs[i]:.1f})" for i in top_ids]
        rows.append({"label": label, "top_words": ", ".join(words)})
    words_df = pd.DataFrame(rows).set_index("label")
    _LOG.debug("return=%s", words_df.shape)
    return words_df


# #############################################################################
# Cell 5.2: Accuracy and macro-F1 of every model
# #############################################################################


def cell5_2_build_results_table(
    y_true: pd.Series, preds: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Score the predictions of several models on the same sentences.

    The macro-F1 is the mean of the F1 of the 3 classes, so a model that
    ignores the small negative class scores low even with a high accuracy.

    :param y_true: true labels
    :param preds: predicted labels of each model, keyed by the model name,
        in the same order as `y_true`
    :return: one row per model
        ```
        model           accuracy  macro_f1
        majority_class  0.594     0.248
        ```
    """
    _LOG.debug(hprint.to_str("len(y_true) list(preds)"))
    rows = []
    for name, y_pred in preds.items():
        hdbg.dassert_eq(
            len(y_pred),
            len(y_true),
            "Model '%s' has one label per sentence",
            name,
        )
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        # Fix the labels, so a class that is never predicted still counts.
        macro_f1 = sklearn.metrics.f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )
        rows.append(
            {
                "model": name,
                "accuracy": round(accuracy, 3),
                "macro_f1": round(macro_f1, 3),
            }
        )
    results_df = pd.DataFrame(rows).set_index("model")
    _LOG.debug("return=%s", results_df.shape)
    return results_df


# #############################################################################
# Cell 5.3: Confusion matrix of every model
# #############################################################################


def cell5_3_plot_confusion_matrices(
    y_true: pd.Series,
    preds: Dict[str, np.ndarray],
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot one confusion matrix per model, side by side.

    The rows are the true labels and the columns the predicted labels, so the
    diagonal holds the correct predictions. The title of each panel shows the
    accuracy and the macro-F1 of the model.

    :param y_true: true labels
    :param preds: predicted labels of each model, keyed by the model name
    :param figsize: size of the whole figure
    """
    _LOG.debug(hprint.to_str("len(y_true) list(preds)"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    scores_df = cell5_2_build_results_table(y_true, preds)
    _, axes = plt.subplots(1, len(preds), figsize=figsize, sharey=True)
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        counts = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=LABELS)
        counts_df = pd.DataFrame(counts, index=LABELS, columns=LABELS)
        sns.heatmap(
            counts_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax
        )
        accuracy = scores_df.loc[name, "accuracy"]
        macro_f1 = scores_df.loc[name, "macro_f1"]
        ax.set_title(f"{name}\naccuracy={accuracy:.3f}, macro-F1={macro_f1:.3f}")
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 5.4: Test sentences whose tokens are also in the train set
# #############################################################################


def cell5_4_find_seen_token_lists(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.Series:
    """
    Flag the test sentences whose token list is also a train token list.

    The duplicates are dropped before the split, but `tokenize()` drops the
    numbers, so 2 different sentences (e.g., 2 daily market reports) can get
    the same tokens. Such a test sentence is not new to the models.

    :param train_df: train set, with the column `tokens`
    :param test_df: test set, with the column `tokens`
    :return: one boolean per test sentence, `True` if its token list is in
        the train set
    """
    _LOG.debug(hprint.to_str("train_df.shape test_df.shape"))
    train_token_lists = set(train_df["tokens"].apply(tuple))
    is_seen = test_df["tokens"].apply(
        lambda tokens: tuple(tokens) in train_token_lists
    )
    _LOG.debug("return=%s", int(is_seen.sum()))
    return is_seen


# #############################################################################
# Cell 6.1: Project the most frequent words to 2D with PCA
# #############################################################################


def cell6_1_project_top_words(
    kv: gensim.models.KeyedVectors,
    terms: List[str],
    *,
    n_words: int = 200,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Project the vectors of the most frequent words to 2D with PCA.

    The PCA is fitted on the `n_words` most frequent words only. A term of
    `terms` outside them is projected with the same PCA, so that every term
    can be labeled on the plot.

    :param kv: word vectors, whose `index_to_key` is sorted by frequency, as
        in `model.wv` of a `Word2Vec` model
    :param terms: words to label; each must be in the vocabulary of `kv`
    :param n_words: number of most frequent words to fit the PCA on
    :return:
        - one row per word, with its count, its 2 coordinates, and whether it
          is among the `n_words` most frequent words
          ```
          word  count  pc1    pc2    in_top
          eur   1580   0.41  -0.12   True
          ```
        - the share of the variance explained by each of the 2 components
    """
    _LOG.debug(hprint.to_str("terms n_words"))
    missing = [term for term in terms if term not in kv]
    hdbg.dassert_eq(len(missing), 0, "Terms not in the vocabulary: %s", missing)
    top_words = kv.index_to_key[:n_words]
    extra_words = [term for term in terms if term not in top_words]
    # Fit the PCA on the top words, then project the extra terms with it.
    pca = sklearn.decomposition.PCA(n_components=2)
    top_coords = pca.fit_transform(kv[top_words])
    words = top_words + extra_words
    coords = top_coords
    if extra_words:
        coords = np.vstack([top_coords, pca.transform(kv[extra_words])])
    words_df = pd.DataFrame(
        {
            "word": words,
            "count": [kv.get_vecattr(word, "count") for word in words],
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "in_top": [word in top_words for word in words],
        }
    )
    _LOG.debug("return=%s", words_df.shape)
    return words_df, pca.explained_variance_ratio_


# #############################################################################
# Cell 6.2: Plot the word map with the finance terms labeled
# #############################################################################


def cell6_2_plot_word_map(
    words_df: pd.DataFrame,
    terms: List[str],
    explained_ratio: np.ndarray,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the 2D word map, with the finance terms highlighted and labeled.

    :param words_df: word coordinates, from `cell6_1_project_top_words()`
    :param terms: words to highlight and label
    :param explained_ratio: share of the variance explained by each axis,
        from `cell6_1_project_top_words()`
    :param figsize: size of the figure
    """
    _LOG.debug(hprint.to_str("words_df.shape terms"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    _, ax = plt.subplots(figsize=figsize)
    is_term = words_df["word"].isin(terms)
    # Draw the frequent words as small gray points.
    sns.scatterplot(
        data=words_df[~is_term],
        x="pc1",
        y="pc2",
        color="lightgray",
        s=25,
        label=f"{int(words_df['in_top'].sum())} most frequent words",
        ax=ax,
    )
    # Draw the finance terms on top, each with its label.
    sns.scatterplot(
        data=words_df[is_term],
        x="pc1",
        y="pc2",
        color="tab:red",
        s=60,
        label="finance terms",
        ax=ax,
    )
    for _, row in words_df[is_term].iterrows():
        # Flag a term that was projected without being part of the PCA fit.
        text = row["word"] if row["in_top"] else f"{row['word']} (not in top)"
        ax.annotate(
            text,
            (row["pc1"], row["pc2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel(f"PC1 ({explained_ratio[0]:.1%} of the variance)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1]:.1%} of the variance)")
    ax.set_title("Word2Vec vectors of the most frequent train words, PCA")
    plt.tight_layout()
    plt.show()


# #############################################################################
# Cell 7.1: Tokenize again, keeping the direction and negation words
# #############################################################################


def cell7_1_tokenize_keeping_words(
    df: pd.DataFrame, keep_words: List[str]
) -> pd.DataFrame:
    """
    Tokenize the sentences again, keeping some of the stop words.

    The new column `tokens_kept` is built like `tokenize()`: the sentences go
    through the same `simple_preprocess()`, and the Gensim stop words are
    removed, except the words in `keep_words` (e.g., "up", "down", "not").
    The column `tokens` of `df` is left as it is.

    :param df: dataframe with the column `sentence`
    :param keep_words: stop words to keep; each must be a Gensim stop word
    :return: copy of `df` with the extra column `tokens_kept`
        ```
        "Operating profit fell , down from EUR 4.4 mn ."
        -> ["operating", "profit", "fell", "down", "eur", "mn"]
        ```
    """
    _LOG.debug(hprint.to_str("df.shape keep_words"))
    hdbg.dassert_is_subset(
        keep_words,
        gensim.parsing.preprocessing.STOPWORDS,
        "Only stop words can be kept",
    )
    # Remove every stop word, except the ones to keep.
    stop_words = gensim.parsing.preprocessing.STOPWORDS - set(keep_words)
    df_out = df.copy()
    df_out["tokens_kept"] = df_out["sentence"].apply(
        lambda sentence: [
            token
            for token in gensim.utils.simple_preprocess(sentence)
            if token not in stop_words
        ]
    )
    _LOG.debug("return=%s", df_out.shape)
    return df_out


# #############################################################################
# gensim.API.ipynb: Cell 2.3: Convert one sentence with doc2bow
# #############################################################################


def cell2_3_bow_to_df(
    dictionary: gensim.corpora.Dictionary, bow: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Show a bag-of-words vector with the token behind each id.

    :param dictionary: dictionary that produced `bow`
    :param bow: sparse vector as `(token_id, count)` pairs, from `doc2bow()`
    :return: one row per token of the sentence
        ```
        token_id  token   count
        12        profit  1
        ```
    """
    _LOG.debug(hprint.to_str("len(bow)"))
    bow_df = pd.DataFrame(bow, columns=["token_id", "count"])
    bow_df.insert(1, "token", bow_df["token_id"].map(dictionary.get))
    _LOG.debug("return=%s", bow_df.shape)
    return bow_df


# #############################################################################
# gensim.API.ipynb: Cell 3.2: Reweight one sentence with TF-IDF
# #############################################################################


def cell3_2_tfidf_to_df(
    dictionary: gensim.corpora.Dictionary,
    bow: List[Tuple[int, int]],
    tfidf_vec: List[Tuple[int, float]],
) -> pd.DataFrame:
    """
    Show the count and the TF-IDF weight of each token of a sentence.

    :param dictionary: dictionary that produced `bow`
    :param bow: sparse count vector, from `doc2bow()`
    :param tfidf_vec: sparse TF-IDF vector of the same sentence, from
        `tfidf[bow]`
    :return: one row per token, sorted by decreasing weight
        ```
        token     count  doc_freq  tfidf
        cramo     1      2         0.62
        ```
    """
    _LOG.debug(hprint.to_str("len(bow) len(tfidf_vec)"))
    weights = dict(tfidf_vec)
    rows = []
    for token_id, count in bow:
        rows.append(
            {
                "token": dictionary[token_id],
                "count": count,
                # Number of sentences of the corpus that contain the token.
                "doc_freq": dictionary.dfs[token_id],
                # A token in every sentence gets weight 0 and is dropped by
                # the model, so it is missing from `weights`.
                "tfidf": round(weights.get(token_id, 0.0), 3),
            }
        )
    tfidf_df = pd.DataFrame(rows)
    tfidf_df = tfidf_df.sort_values("tfidf", ascending=False)
    tfidf_df = tfidf_df.reset_index(drop=True)
    _LOG.debug("return=%s", tfidf_df.shape)
    return tfidf_df


# #############################################################################
# gensim.API.ipynb: Cell 4.2: Find the sentences most similar to a query
# #############################################################################


def cell4_2_top_matches(
    sims: np.ndarray, df: pd.DataFrame, *, topn: int = 5
) -> pd.DataFrame:
    """
    Show the sentences with the highest similarity to a query.

    :param sims: one cosine similarity per sentence of `df`, from
        `index[query_vec]`
    :param df: sentences that were indexed, with the columns `sentence` and
        `label`, in the same order as the index
    :param topn: number of sentences to return
    :return: the `topn` most similar sentences, best first
        ```
        score  label     sentence
        0.71   positive  Operating profit rose to EUR 13.1 mn ...
        ```
    """
    _LOG.debug(hprint.to_str("sims.shape df.shape topn"))
    hdbg.dassert_eq(len(sims), len(df), "One score per indexed sentence")
    # Sort the sentence positions by decreasing score.
    top_idx = np.argsort(-sims)[:topn]
    matches_df = df.iloc[top_idx][["label", "sentence"]].copy()
    matches_df.insert(0, "score", np.round(sims[top_idx], 3))
    matches_df = matches_df.reset_index(drop=True)
    _LOG.debug("return=%s", matches_df.shape)
    return matches_df


# #############################################################################
# gensim.API.ipynb: Cell 5.2: Show the top words of each topic
# #############################################################################


def cell5_2_topics_to_df(
    lda: gensim.models.LdaModel, *, topn: int = 8
) -> pd.DataFrame:
    """
    Show the top words of each topic of an LDA model.

    :param lda: trained LDA model
    :param topn: number of words per topic
    :return: one row per topic, with its top words, most probable first
        ```
        topic  top_words
        0      eur, mn, profit, net, sales, period, quarter, million
        ```
    """
    _LOG.debug(hprint.to_str("lda.num_topics topn"))
    rows = []
    for topic_id in range(lda.num_topics):
        word_probs = lda.show_topic(topic_id, topn=topn)
        words = [word for word, _ in word_probs]
        rows.append({"topic": topic_id, "top_words": ", ".join(words)})
    topics_df = pd.DataFrame(rows).set_index("topic")
    _LOG.debug("return=%s", topics_df.shape)
    return topics_df


# #############################################################################
# gensim.API.ipynb: Cell 5.5: Compare the coherence of several topic counts
# #############################################################################


def cell5_5_coherence_by_num_topics(
    corpus: List[List[Tuple[int, int]]],
    texts: List[List[str]],
    dictionary: gensim.corpora.Dictionary,
    num_topics_list: List[int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit one LDA model per number of topics and score its coherence.

    Each model uses the same settings as the model of the notebook, so that
    only the number of topics changes.

    :param corpus: bag-of-words corpus
    :param texts: tokenized sentences behind `corpus`, used by the `c_v`
        coherence to count the co-occurrences of the top words
    :param dictionary: dictionary of `corpus`
    :param num_topics_list: numbers of topics to try
    :param random_state: seed of every LDA model
    :return: one row per number of topics
        ```
        num_topics  c_v
        3           0.386
        ```
    """
    _LOG.debug(hprint.to_str("len(corpus) num_topics_list random_state"))
    rows = []
    for num_topics in num_topics_list:
        lda = gensim.models.LdaModel(
            corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            random_state=random_state,
        )
        # Use 1 process, so that the result does not depend on the machine.
        coherence_model = gensim.models.CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        c_v = coherence_model.get_coherence()
        _LOG.debug("num_topics=%s c_v=%s", num_topics, c_v)
        rows.append({"num_topics": num_topics, "c_v": round(c_v, 3)})
    coherence_df = pd.DataFrame(rows).set_index("num_topics")
    _LOG.debug("return=%s", coherence_df.shape)
    return coherence_df


# #############################################################################
# gensim.API.ipynb: Cell 6.3: Nearest neighbors of finance terms
# #############################################################################


def cell6_3_most_similar_to_df(
    kv: gensim.models.KeyedVectors, terms: List[str], *, topn: int = 5
) -> pd.DataFrame:
    """
    Show the nearest neighbors of several words, one row per word.

    :param kv: word vectors, e.g., `model.wv` of a `Word2Vec` model
    :param terms: words to query; each must be in the vocabulary of `kv`
    :param topn: number of neighbors per word
    :return: one row per term, with one column per neighbor, closest first,
        and the cosine similarity in parentheses
        ```
        term    1           2                 ...
        profit  net (0.91)  thousand (0.90)   ...
        ```
    """
    _LOG.debug(hprint.to_str("terms topn"))
    missing = [term for term in terms if term not in kv]
    hdbg.dassert_eq(len(missing), 0, "Terms not in the vocabulary: %s", missing)
    rows = []
    for term in terms:
        row = {"term": term}
        neighbors = kv.most_similar(term, topn=topn)
        for rank, (word, score) in enumerate(neighbors, start=1):
            row[str(rank)] = f"{word} ({score:.2f})"
        rows.append(row)
    neighbors_df = pd.DataFrame(rows).set_index("term")
    _LOG.debug("return=%s", neighbors_df.shape)
    return neighbors_df


# #############################################################################
# gensim.API.ipynb: Cell 9.2: Sentence features for all the sentences
# #############################################################################


def cell9_2_build_mean_vectors(
    kv: gensim.models.KeyedVectors, texts: List[List[str]]
) -> np.ndarray:
    """
    Represent each sentence as the mean of the vectors of its words.

    The words that are not in the vocabulary of `kv` are skipped. A sentence
    with no word in the vocabulary (e.g., no token left after removing the
    stop words) gets the zero vector, so that every sentence has a row.

    :param kv: word vectors
    :param texts: tokenized sentences
    :return: array with one row per sentence and `kv.vector_size` columns
    """
    _LOG.debug(hprint.to_str("len(texts)"))
    n_dims = kv.vector_size
    vectors: np.ndarray = np.zeros((len(texts), n_dims), dtype=np.float32)
    n_zero = 0
    for i, tokens in enumerate(texts):
        known = [token for token in tokens if token in kv]
        if known:
            vectors[i] = kv.get_mean_vector(known)
        else:
            n_zero += 1
    _LOG.info("Sentences with no known word: '%s'", n_zero)
    _LOG.debug("return=%s", vectors.shape)
    return vectors
