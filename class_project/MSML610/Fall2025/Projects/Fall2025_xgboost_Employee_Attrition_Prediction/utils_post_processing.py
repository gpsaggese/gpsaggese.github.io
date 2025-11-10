"""
Post-processing utilities:
- Model evaluation (Accuracy, F1, ROC-AUC, classification report)
- Feature name extraction
- Feature importance plotting for XGBoost
"""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def evaluate_classifier(model, X_test, y_test, model_name: str = "model") -> Dict[str, object]:
    """
    Evaluate a classifier on test data and return metrics.

    Parameters
    ----------
    model : fitted model
        Must implement predict() and predict_proba().
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    model_name : str, optional
        Name of the model (for logging), by default "model".

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, f1, roc_auc, and full classification_report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Attrition", "Attrition"],
    )

    print(f"\n[{model_name}] Accuracy: {acc:.4f}")
    print(f"[{model_name}] F1-score: {f1:.4f}")
    print(f"[{model_name}] ROC-AUC:  {roc:.4f}")
    print(f"\n[{model_name}] Classification report:\n{report}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc,
        "classification_report": report,
    }


def get_feature_names_from_preprocessor(preprocessor, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
    """
    Extract feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer with 'num' and 'cat' transformers.
    numeric_cols : list of str
        Original numeric column names.
    categorical_cols : list of str
        Original categorical column names.

    Returns
    -------
    feature_names : np.ndarray
        Array of all transformed feature names in correct order.
    """
    # Numeric features keep their original names
    num_features = np.array(numeric_cols)

    # Categorical: need to get names from OneHotEncoder
    cat_transformer = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_transformer.get_feature_names_out(categorical_cols)

    feature_names = np.concatenate([num_features, cat_feature_names])
    return feature_names


def plot_feature_importance(
    model,
    feature_names: np.ndarray,
    top_n: int = 15,
    title: str = "Feature Importances",
):
    """
    Save a bar plot of top N feature importances for a fitted XGBoost model.

    Parameters
    ----------
    model : fitted XGBClassifier
        Must have feature_importances_ attribute.
    feature_names : np.ndarray
        Names of input features in the correct order.
    top_n : int, optional
        Number of top features to show, by default 15.
    title : str, optional
        Plot title, by default "Feature Importances".
    """
    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Length mismatch: got {len(importances)} importances but {len(feature_names)} feature names."
        )

    data = list(zip(feature_names, importances))
    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]

    labels = [d[0] for d in data_sorted]
    scores = [d[1] for d in data_sorted]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=scores, y=labels)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("Saved feature importance plot to xgb_feature_importance.png")
