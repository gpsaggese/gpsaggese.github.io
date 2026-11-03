"""
Utility functions for the Gensim sentiment classification of financial news.

The notebook `gensim.example.ipynb` calls these functions, one per cell, to:
- Download and cache the Financial PhraseBank sentences
- Find and drop duplicate sentences
- Tokenize the sentences with `simple_preprocess()` and the Gensim stop words
- Make a stratified train / test split
- Summarize the data in one table

Import as:

import class_project.msml610.Fall2026.projects.UmdTask782_MSML610_Fall2026_Gensim_Project1.gensim_utils as cmfpugpgu
"""

import collections
import logging
import os
import urllib.request
import zipfile
from typing import Any, Dict, List, Tuple

import gensim.parsing.preprocessing
import gensim.utils
import pandas as pd
import sklearn.model_selection

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# The 3 sentiment classes, in the order used by every table.
LABELS = ["negative", "neutral", "positive"]


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
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
