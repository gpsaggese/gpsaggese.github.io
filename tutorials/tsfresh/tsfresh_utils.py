"""
Utility functions for tsfresh-based time series feature extraction workflows.

Import as:

import tsfresh_utils as ttsfuti
"""

import logging
import pathlib
import zipfile

import numpy as np
import pandas as pd
import requests
import tqdm

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# #############################################################################
# Data Download
# #############################################################################

_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)

_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def fetch_har_data(url=_DATA_URL, out_dir="./uci_har_data"):
    """
    Download and extract the UCI HAR dataset.

    :param url: URL of the UCI HAR dataset zip file.
    :param out_dir: Directory path to download and extract the dataset into.
    :return: pathlib.Path to the extracted "UCI HAR Dataset" root directory.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "UCI_HAR_Dataset.zip"
    if not zip_path.exists():
        _LOG.info("Downloading UCI HAR dataset from %s ...", url)
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc="UCI HAR"
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        _LOG.info("Dataset archive already present at %s", zip_path)
    extracted_root = out_dir / "UCI HAR Dataset"
    if not extracted_root.exists():
        _LOG.info("Extracting archive ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    return extracted_root


# #############################################################################
# Data Loading
# #############################################################################


def load_inertial_split(split, root):
    """
    Load raw inertial sensor data for a train or test split.

    :param split: One of "train" or "test".
    :param root: pathlib.Path to the extracted "UCI HAR Dataset" directory.
    :return: Tuple of (arrays dict, y_labels numpy array, subject_ids numpy
        array).  Arrays dict maps channel name to 2D array of shape
        (n_samples, n_timesteps).
    """
    hdbg.dassert_in(split, ["train", "test"])
    root = pathlib.Path(root)
    base = root / split / "Inertial Signals"
    arrays = {}
    for ch in _CHANNELS:
        fn = base / f"{ch}_{split}.txt"
        arrays[ch] = np.loadtxt(fn)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int)
    subjects = np.loadtxt(
        root / split / f"subject_{split}.txt"
    ).astype(int)
    return arrays, y, subjects


# #############################################################################
# Data Transformation
# #############################################################################


def to_long_format(arrays, start_id=0):
    """
    Convert sensor arrays to long-format DataFrame suitable for tsfresh.

    :param arrays: Dict mapping channel names to 2D arrays of shape
        (n_samples, n_timesteps).
    :param start_id: Starting integer ID for the first sample row.
    :return: Long-format DataFrame with columns [id, time, kind, value].
    """
    n_samples = next(iter(arrays.values())).shape[0]
    n_time = next(iter(arrays.values())).shape[1]
    frames = []
    for i in range(n_samples):
        for ch, mat in arrays.items():
            df = pd.DataFrame(
                {
                    "id": start_id + i,
                    "time": np.arange(n_time, dtype=int),
                    "kind": ch,
                    "value": mat[i, :].astype(float),
                }
            )
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# #############################################################################
# Feature Engineering
# #############################################################################


def extract_tsfresh_features(df_long, fc_parameters=None, n_jobs=0):
    """
    Extract features from a long-format time series DataFrame using tsfresh.

    :param df_long: Long-format DataFrame with columns [id, time, kind, value].
    :param fc_parameters: tsfresh feature calculator parameters object.
        Defaults to MinimalFCParameters for speed.
    :param n_jobs: Number of parallel jobs for tsfresh (0 = single process).
    :return: Wide-format DataFrame with one row per id and one column per
        extracted feature.
    """
    import tsfresh
    import tsfresh.feature_extraction as fe

    if fc_parameters is None:
        fc_parameters = fe.MinimalFCParameters()
    X_features = tsfresh.extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters,
        n_jobs=n_jobs,
        disable_progressbar=False,
    )
    # Replace NaN/inf with 0.
    X_features = X_features.fillna(0).replace(
        [float("inf"), float("-inf")], 0
    )
    return X_features


def select_tsfresh_features(X_features, y):
    """
    Select statistically relevant features using tsfresh hypothesis tests.

    :param X_features: Wide-format feature DataFrame (output of
        extract_tsfresh_features).
    :param y: Target Series with the same index as X_features.
    :return: Wide-format DataFrame containing only the selected features.
    """
    from tsfresh import select_features

    X_selected = select_features(X_features, y)
    _LOG.info(
        "Selected %d / %d features",
        X_selected.shape[1],
        X_features.shape[1],
    )
    return X_selected


# #############################################################################
# Model Training
# #############################################################################


def train_classifier(X_train, y_train, n_estimators=500, random_state=42):
    """
    Train a RandomForestClassifier on extracted features.

    :param X_train: Training feature DataFrame.
    :param y_train: Training labels Series aligned by id.
    :param n_estimators: Number of decision trees.
    :param random_state: Random seed for reproducibility.
    :return: Trained RandomForestClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate a trained classifier and return performance metrics.

    :param clf: Trained sklearn classifier.
    :param X_test: Test feature DataFrame.
    :param y_test: Test labels Series.
    :return: Dict with keys "accuracy", "macro_f1", "macro_roc_auc".
    """
    from sklearn import metrics
    from sklearn.preprocessing import label_binarize

    proba = clf.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    classes = np.arange(len(clf.classes_))
    acc = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average="macro")
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_auc = metrics.roc_auc_score(
        y_test_bin, proba, average="macro", multi_class="ovr"
    )
    results = {"accuracy": acc, "macro_f1": f1, "macro_roc_auc": roc_auc}
    _LOG.info(
        "Accuracy=%.3f  Macro-F1=%.3f  Macro-ROC-AUC=%.3f",
        acc,
        f1,
        roc_auc,
    )
    return results
