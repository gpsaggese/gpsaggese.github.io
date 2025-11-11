"""
Compute SHAP explanations for the trained XGBoost model.
Generates summary and beeswarm plots; also a dependence plot for a chosen feature.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import shap
import matplotlib.pyplot as plt
from scipy import sparse
from xgboost import XGBClassifier

from src.utils.config import Paths
from src.utils.io import ensure_dirs
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
    ensure_dirs(P.reports_dir)

    pipeline = load(P.pipeline_path)
    clf = XGBClassifier()
    clf.load_model(str(P.model_path))

    X_train = load_sparse_npz(P.processed_dir / 'X_train.npz')
    # SHAP with tree explainer
    explainer = shap.TreeExplainer(clf)
    # Use a small sample for speed in PoC
    if X_train.shape[0] > 300:
        # sparse to dense sample only if needed
        idx = np.random.choice(X_train.shape[0], 300, replace=False)
        X_bg = X_train[idx].toarray() if sparse.issparse(X_train) else X_train[idx]
    else:
        X_bg = X_train.toarray() if sparse.issparse(X_train) else X_train

    shap_values = explainer(X_bg)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_bg, show=False)
    plt.tight_layout()
    plt.savefig(P.shap_summary_png, dpi=160, bbox_inches='tight')
    plt.close()

    # Beeswarm (same as summary for TreeExplainer; kept for clarity)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(P.shap_beeswarm_png, dpi=160, bbox_inches='tight')
    plt.close()

    # Optional dependence plot for the top feature if available
    try:
        # Estimate feature names from OHE
        ohe = None
        for name, trans, cols in pipeline.transformers_:
            if name == 'cat':
                ohe = trans.named_steps['ohe']
        num_cols = []
        for name, trans, cols in pipeline.transformers_:
            if name == 'num':
                num_cols = cols

        feature_names = []
        if ohe is not None:
            ohe_feature_names = ohe.get_feature_names_out()
        else:
            ohe_feature_names = []

        feature_names = list(num_cols) + list(ohe_feature_names)
        top_idx = np.argsort(np.abs(shap_values.values).mean(axis=0))[-1]
        top_name = feature_names[top_idx] if top_idx < len(feature_names) else f"feat_{top_idx}"
        plt.figure()
        shap.dependence_plot(top_idx, shap_values.values, X_bg, show=False, feature_names=feature_names)
        plt.tight_layout()
        plt.savefig(P.reports_dir / f"shap_dependence_{top_name}.png", dpi=160, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Dependence plot skipped:", e)

    print("Saved SHAP plots to:", P.reports_dir)

if __name__ == "__main__":
    main()
