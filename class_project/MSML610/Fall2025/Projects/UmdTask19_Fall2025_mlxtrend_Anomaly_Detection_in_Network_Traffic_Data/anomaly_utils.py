# anomaly_utils.py

"""
Utility helpers for the UNSW-NB15 anomaly detection project.

This module is the single place where we:
- Load and clean the UNSW-NB15 data from a zip file
- Do quick EDA checks
- Build the preprocessing pipeline + train/test split
- Run a **fast** numeric feature selection (MI-based, no heavy EFS)
"""

from pathlib import Path
import os
import glob
import zipfile

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif


# ------------------------------------------------------------------
# Column names for UNSW-NB15 CSVs
# ------------------------------------------------------------------

UNSW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]


# ------------------------------------------------------------------
# 1) Data loading + quick EDA
# ------------------------------------------------------------------

def load_unsw_from_zip(zip_path: str, extract_dir: str = "./data",
                       columns=UNSW_COLS) -> pd.DataFrame:
    """
    Extract UNSW-NB15 CSVs from a zip file and return a clean DataFrame.

    - Skips the metadata header rows that appear in some UNSW files
    - Casts numeric columns to floats
    - Forces 'label' to int
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract all files
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find all CSV parts in the extracted folder
    parts = sorted(glob.glob(str(extract_dir / "**" / "*.csv"), recursive=True))
    if not parts:
        raise FileNotFoundError(
            f"No CSVs found after extracting archive into {extract_dir}"
        )

    dfs = []
    for p in parts:
        df_part = pd.read_csv(
            p,
            encoding="latin1",
            header=None,
            names=columns,
            low_memory=False,
            dtype=str,
        )

        # Drop UNSW metadata rows (those weird "Name / Type / Description" lines)
        drop_mask = df_part["srcip"].str.contains(
            "No|Name|Type|Description|nominal|integer",
            case=False,
            na=False,
        )
        df_part = df_part[~drop_mask]
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Convert numeric columns
    non_numeric = ["srcip", "dstip", "proto", "state", "service", "attack_cat"]
    numeric_like = [c for c in df.columns if c not in non_numeric]

    for c in numeric_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean label
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # Drop rows that are almost entirely NaN
    df = df.dropna(thresh=int(len(df.columns) * 0.3))

    return df


def basic_eda(df: pd.DataFrame) -> None:
    """Simple dataset sanity check."""
    print("✅ Dataset Loaded Successfully")
    print(f"Shape: {df.shape}")

    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())

    print("\n--- Missing Values (Top 10) ---")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss.head(10))

    print("\n--- Sample Rows ---")
    print(df.head(3))


# ------------------------------------------------------------------
# 2) Preprocessing pipeline + train/test split
# ------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocess_and_split(df, test_size=0.25, random_state=42):
    """
    Take the cleaned df from Step 1 and:

    - split into train / test
    - build a preprocessing pipeline
    - fit on train, transform both
    - return everything we need for later steps
    """

    # --- target + features ---
    y = df["label"].astype(int)
    X = df.drop(columns=["label"]).copy()

    # Categorical and numeric columns
    cat_cols = ["proto", "service", "state", "attack_cat"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 🔧 Force numeric columns to numeric; bad strings -> NaN (imputer will handle)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # --- train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # --- Pipelines ---
    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # with_mean=False is important when combined with sparse one-hot features
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    # --- fit + transform ---
    X_train_proc = preprocess.fit_transform(X_train, y_train)
    X_test_proc = preprocess.transform(X_test)

    print("Preprocessing pipeline ready.")
    print("Train:", X_train_proc.shape, " Test:", X_test_proc.shape)

    return (
        preprocess,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_proc,
        X_test_proc,
        num_cols,
        cat_cols,
    )



# ------------------------------------------------------------------
# 3) FAST numeric feature selection (MI-based only)
# ------------------------------------------------------------------

def fast_numeric_feature_selection(
    X_train,
    y_train,
    numeric_columns,
    out_dir="outputs",
    top_k_mi=10,
    min_k=3,
    max_k=5,
    sample_size=50000,
):
    """
    Fast numeric feature selection using:
    1. Mutual Information (top_k_mi features)
    2. Exhaustive Feature Selector (EFS) on a small sample

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series
    numeric_columns : list of str
    out_dir : directory to save outputs
    top_k_mi : number of MI features to keep
    min_k, max_k : subset sizes for EFS
    sample_size : number of rows to run EFS on (small for speed)
    """

    import numpy as np
    import pandas as pd
    import json
    import os
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

    # ------------------------------
    # 1) Compute Mutual Information
    # ------------------------------
    X_num = X_train[numeric_columns].copy()
    X_num_filled = X_num.fillna(0)

    mi = mutual_info_classif(X_num_filled, y_train, random_state=42)
    mi_series = pd.Series(mi, index=numeric_columns).sort_values(ascending=False)

    top_mi_features = mi_series.head(top_k_mi).index.tolist()
    print("Top MI features:", top_mi_features)

    # ------------------------------
    # 2) EFS on a small sample
    # ------------------------------
    if sample_size < len(X_train):
        sample_idx = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train.iloc[sample_idx][top_mi_features]
        y_sample = y_train.iloc[sample_idx]
    else:
        X_sample = X_train[top_mi_features]
        y_sample = y_train

    imp = SimpleImputer(strategy="median")
    scal = StandardScaler(with_mean=False)

    Xs = scal.fit_transform(imp.fit_transform(X_sample))

    clf = LogisticRegression(max_iter=400, solver="liblinear")

    efs = EFS(
        clf,
        min_features=min_k,
        max_features=max_k,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        print_progress=False,
    )

    efs.fit(Xs, y_sample)

    best_subset_features = [top_mi_features[i] for i in efs.best_idx_]
    print("Best EFS subset:", best_subset_features)

    # ------------------------------
    # 3) Save output to JSON
    # ------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "selected_numeric.json")

    with open(out_path, "w") as f:
        json.dump(best_subset_features, f, indent=2)

    print(f"Saved to: {out_path}")

    return best_subset_features
