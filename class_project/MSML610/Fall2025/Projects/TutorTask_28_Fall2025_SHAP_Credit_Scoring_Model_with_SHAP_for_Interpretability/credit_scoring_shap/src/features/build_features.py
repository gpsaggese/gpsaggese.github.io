"""
Build a preprocessing pipeline and transform train/test into model-ready arrays.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy import sparse

from src.utils.config import Paths
from src.utils.io import ensure_dirs
from src.utils.seed import set_seed

TARGET_COL = "target"

def detect_types(df: pd.DataFrame):
    # Simple heuristic: treat object and category as categoricals.
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)
    num_cols = [c for c in df.columns if c not in cat_cols + [TARGET_COL]]
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[('scaler', StandardScaler())])
    cat_pipe = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols),
        ]
    )
    return pre

def save_sparse_matrix(path: Path, X):
    path.parent.mkdir(parents=True, exist_ok=True)
    if sparse.issparse(X):
        np.savez(path, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
    else:
        np.savez(path, data=X)

def main():
    set_seed(42)
    P = Paths()
    ensure_dirs(P.processed_dir, P.models_dir)
    train_df = pd.read_parquet(P.interim_dir / 'split_train.parquet')
    test_df = pd.read_parquet(P.interim_dir / 'split_test.parquet')

    num_cols, cat_cols = detect_types(train_df.drop(columns=[TARGET_COL]))
    pre = build_pipeline(num_cols, cat_cols)

    X_train = pre.fit_transform(train_df.drop(columns=[TARGET_COL]))
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = pre.transform(test_df.drop(columns=[TARGET_COL]))
    y_test = test_df[TARGET_COL].to_numpy()

    dump(pre, P.pipeline_path)
    save_sparse_matrix(P.processed_dir / 'X_train.npz', X_train)
    np.save(P.processed_dir / 'y_train.npy', y_train)
    save_sparse_matrix(P.processed_dir / 'X_test.npz', X_test)
    np.save(P.processed_dir / 'y_test.npy', y_test)

    print("Saved pipeline and processed arrays to:", P.processed_dir)

if __name__ == "__main__":
    main()
