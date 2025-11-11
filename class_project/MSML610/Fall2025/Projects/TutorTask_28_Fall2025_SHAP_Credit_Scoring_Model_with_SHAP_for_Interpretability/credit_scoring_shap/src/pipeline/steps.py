from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap
import itertools

from src.schemas.german_credit_schema import detect_and_binarize_target, basic_feature_types

@dataclass
class Paths:
    root: Path
    raw_csv: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    pipeline_path: Path
    model_path: Path

    @staticmethod
    def from_root(root: Path):
        return Paths(
            root=root,
            raw_csv=root / "data" / "raw" / "german_credit_data.csv",
            interim_dir=root / "data" / "interim",
            processed_dir=root / "data" / "processed",
            models_dir=root / "models",
            reports_dir=root / "reports",
            pipeline_path=root / "models" / "pipeline.joblib",
            model_path=root / "models" / "xgb_model.json",
        )

def ensure_dirs(P: Paths):
    for d in [P.interim_dir, P.processed_dir, P.models_dir, P.reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

def load_raw(P: Paths) -> pd.DataFrame:
    if not P.raw_csv.exists():
        raise FileNotFoundError(f"Missing raw CSV at {P.raw_csv}")
    return pd.read_csv(P.raw_csv)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def split_df(df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target_col])

def build_transformer(train_df: pd.DataFrame, target_col: str):
    num_cols, cat_cols = basic_feature_types(train_df, target_col)
    num_pipe = Pipeline(steps=[('scaler', StandardScaler())])
    cat_pipe = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    return pre, num_cols, cat_cols

def fit_transform(pre: ColumnTransformer, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    X_train = pre.fit_transform(train_df.drop(columns=[target_col]))
    y_train = train_df[target_col].to_numpy()
    X_test = pre.transform(test_df.drop(columns=[target_col]))
    y_test = test_df[target_col].to_numpy()
    return X_train, y_train, X_test, y_test

def train_xgb(X_train, y_train, seed: int = 42) -> XGBClassifier:
    pos_ratio = (y_train == 1).mean()
    spw = max((1 - pos_ratio) / (pos_ratio + 1e-9), 1.0)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective='binary:logistic', eval_metric='logloss',
        n_jobs=4, tree_method='hist', scale_pos_weight=spw, random_state=seed
    )
    clf.fit(X_train, y_train)
    return clf, spw

def evaluate_model(clf: XGBClassifier, X_test, y_test, reports_dir: Path):
    proba = clf.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    # plot confusion
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix (thr=0.5)')
    plt.colorbar(); tick_marks = range(2); classes=['Good(0)','Bad(1)']
    plt.xticks(tick_marks, classes, rotation=45); plt.yticks(tick_marks, classes)
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'), ha="center", color="white" if cm[i,j]>thresh else "black")
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    fig.savefig(reports_dir / 'confusion_matrix.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    metrics = {
        'test_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, preds, output_dict=True)
    }
    with open(reports_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

def shap_explain(clf: XGBClassifier, pre: ColumnTransformer, X_bg, reports_dir: Path):
    import scipy.sparse as sp
    explainer = shap.TreeExplainer(clf)
    if sp.issparse(X_bg):
        import numpy as np
        n_bg = min(400, X_bg.shape[0])
        idx = np.random.choice(X_bg.shape[0], n_bg, replace=False)
        X_plot = X_bg[idx].toarray()
    else:
        X_plot = X_bg
    sv = explainer(X_plot)

    # Recreate feature names
    ohe = None; num_cols = []
    for name, trans, cols in pre.transformers_:
        if name == 'cat':
            ohe = trans.named_steps['ohe']
        elif name == 'num':
            num_cols = cols
    ohe_names = ohe.get_feature_names_out() if ohe else []
    feat_names = list(num_cols) + list(ohe_names)

    shap.summary_plot(sv, X_plot, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout(); plt.savefig(reports_dir / 'shap_summary.png', dpi=160, bbox_inches='tight'); plt.close()

    shap.plots.beeswarm(sv, show=False)
    plt.tight_layout(); plt.savefig(reports_dir / 'shap_beeswarm.png', dpi=160, bbox_inches='tight'); plt.close()

    # Dependence on top feature
    import numpy as np
    top_idx = int(np.argsort(np.abs(sv.values).mean(axis=0))[-1])
    shap.dependence_plot(top_idx, sv.values, X_plot, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig(reports_dir / f'shap_dependence_{feat_names[top_idx] if top_idx < len(feat_names) else top_idx}.png', dpi=160, bbox_inches='tight'); plt.close()
