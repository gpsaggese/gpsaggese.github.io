"""
Evaluate model on held-out test set and save basic metrics + confusion matrix plot.
"""
import json
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
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
    ensure_dirs(P.reports_dir)

    X_test = load_sparse_npz(P.processed_dir / 'X_test.npz')
    y_test = np.load(P.processed_dir / 'y_test.npy')

    clf = XGBClassifier()
    clf.load_model(str(P.model_path))

    proba = clf.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)  # TODO: threshold tuning later

    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    # Save metrics json
    save_json({'test_auc': auc, 'confusion_matrix': cm, 'classification_report': report}, P.metrics_path)

    # Confusion matrix plot
    fig = plt.figure(figsize=(4,4))
    import itertools
    cm_np = np.array(cm)
    plt.imshow(cm_np, interpolation='nearest')
    plt.title('Confusion Matrix (Threshold=0.5)')
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ['Class 0','Class 1']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm_np.max() / 2.
    for i, j in itertools.product(range(cm_np.shape[0]), range(cm_np.shape[1])):
        plt.text(j, i, format(cm_np[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_np[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(P.cm_png, dpi=160, bbox_inches='tight')
    plt.close(fig)

    print(f"AUC: {auc:.3f}. Metrics saved to {P.metrics_path}. Figure: {P.cm_png}")

if __name__ == "__main__":
    main()
