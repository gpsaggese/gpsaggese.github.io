"""
Train an XGBoost classifier. Minimal baseline with a few sensible defaults.
"""
import json
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.utils.config import Paths
from src.utils.io import ensure_dirs, save_json
from src.utils.seed import set_seed

def load_sparse_npz(path: Path):
    with np.load(path) as f:
        if all(k in f for k in ('data','indices','indptr','shape')):
            return sparse.csr_matrix((f['data'], f['indices'], f['indptr']), shape=f['shape'])
        else:
            return f['data']

def main():
    set_seed(42)
    P = Paths()
    ensure_dirs(P.models_dir, P.reports_dir)

    X_train = load_sparse_npz(P.processed_dir / 'X_train.npz')
    y_train = np.load(P.processed_dir / 'y_train.npy')

    # Estimate positive rate for scale_pos_weight (if 1 is minority class)
    pos_ratio = (y_train == 1).mean() if len(np.unique(y_train)) == 2 else 0.5
    scale_pos_weight = max((1 - pos_ratio) / (pos_ratio + 1e-9), 1.0)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=4,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    # Save model as JSON for portability
    clf.save_model(str(P.model_path))

    # Quick in-sample AUC just as a sanity check (not a valid metric!)
    try:
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
    except Exception:
        train_auc = None

    save_json({'train_auc_hint': train_auc, 'scale_pos_weight': scale_pos_weight}, P.metrics_path)
    print("Model trained. Saved to:", P.model_path)

if __name__ == "__main__":
    main()
