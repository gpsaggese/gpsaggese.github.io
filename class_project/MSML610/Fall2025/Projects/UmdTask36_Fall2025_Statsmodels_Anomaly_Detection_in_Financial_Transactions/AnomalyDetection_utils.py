# ================================================================
# AnomalyDetection_utils.py
# Advanced Statistical + ML Fraud Detection Utilities
# ================================================================

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from sklearn.ensemble import IsolationForest
import statsmodels.api as sm


# -------------------------------------------------------------
# 0. Small helpers
# -------------------------------------------------------------
def _as_1d_array(x) -> np.ndarray:
    """Convert input to a 1D numpy array."""
    arr = np.asarray(x)
    return arr.ravel()


def _add_constant_df(X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """Always returns a DataFrame with an intercept column added."""
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
    X_df = sm.add_constant(X_df, has_constant="add")
    return X_df


def _safe_std(x: np.ndarray) -> float:
    """Standard deviation with guard against zero variance."""
    s = float(np.nanstd(x))
    return s if s > 0 else 1.0


# -------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load credit card fraud dataset (creditcard.csv)."""
    return pd.read_csv(path)


# -------------------------------------------------------------
# 2. Basic EDA
# -------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> None:
    """Print dataset shape, missing values, and summary statistics."""
    print("=== Basic EDA ===")
    print("Shape:", df.shape)
    print("\nTotal missing values:", int(df.isnull().sum().sum()))
    print("\nSummary statistics:\n", df.describe())


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Print & plot fraud vs non-fraud distribution."""
    counts = df["Class"].value_counts().sort_index()

    print("\n=== Class Distribution ===")
    print(counts)

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return counts


# -------------------------------------------------------------
# 3. Advanced Preprocessing Pipeline
# -------------------------------------------------------------
def prepare_data(df: pd.DataFrame, scale_method: str = "standard"):
    """
    Advanced preprocessing pipeline:
      - Time-aware split (chronological order by 'Time')
      - Scaling (StandardScaler or RobustScaler)
      - SMOTE applied ONLY to training data (for supervised GLM training)

    Args:
        df: DataFrame containing features + 'Class' label
        scale_method: 'standard' or 'robust'

    Returns:
        X_train_res: SMOTE-resampled, scaled training features (for supervised GLM)
        X_test_scaled: scaled test features (original imbalanced test set)
        y_train_res: SMOTE-resampled training labels
        y_test: original test labels (imbalanced)
        scaler: fitted scaler
        X_train_scaled: scaled training features BEFORE SMOTE (for unsupervised models)
        y_train: original training labels BEFORE SMOTE (imbalanced)
    """
    if "Time" not in df.columns or "Class" not in df.columns:
        raise ValueError("Expected columns 'Time' and 'Class' in the dataset.")

    # Sort data chronologically to reduce leakage
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    X = df_sorted.drop(columns=["Class"])
    y = df_sorted["Class"]

    # Time-aware split: 80% train, 20% test
    split_index = int(len(df_sorted) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Choose scaler
    scale_method = (scale_method or "standard").lower().strip()
    if scale_method == "robust":
        scaler = RobustScaler()
    elif scale_method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("scale_method must be either 'standard' or 'robust'")

    # Fit scaler ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on training data only
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Keep y_train_res as a named Series (helps printing/consistency)
    y_train_res = pd.Series(y_train_res, name="Class")

    print("\n=== After SMOTE (Training Set) ===")
    print(y_train_res.value_counts())

    return X_train_res, X_test_scaled, y_train_res, y_test, scaler, X_train_scaled, y_train


# -------------------------------------------------------------
# 4. Statsmodels GLM (Binomial) - Core Statistical Model
# -------------------------------------------------------------
def fit_glm_statsmodels(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
):
    """
    Fit a Binomial GLM (logit link) using statsmodels.

    Returns:
        glm_result: fitted GLMResults object
    """
    X_df = _add_constant_df(X_train)
    y_arr = _as_1d_array(y_train)

    model = sm.GLM(y_arr, X_df, family=sm.families.Binomial())
    glm_result = model.fit(maxiter=100, disp=0)
    return glm_result


def glm_predict_proba(glm_result, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Predict fraud probabilities using a fitted GLM model."""
    X_df = _add_constant_df(X)
    proba = glm_result.predict(X_df)
    return np.asarray(proba)


def glm_predict_labels(
    glm_result,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict binary labels (0/1) from GLM probabilities."""
    proba = glm_predict_proba(glm_result, X)
    labels = (proba >= threshold).astype(int)
    return labels, proba


# -------------------------------------------------------------
# 5. Residuals & Influence for Anomaly Detection
#    (Works on any provided (X, y): train or test)
# -------------------------------------------------------------
def compute_glm_diagnostics(
    glm_result,
    X: Union[np.ndarray, pd.DataFrame],
    y_true: Union[np.ndarray, pd.Series],
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """
    Compute residuals and influence-like diagnostics for a fitted GLM on a given dataset.

    IMPORTANT:
      - For out-of-sample (test) data, statsmodels does not reliably expose hat matrix helpers
        across versions. So we compute:
          * deviance residuals (binomial)
          * pearson residuals (binomial)
          * standardized versions
          * leverage (approx IRLS hat diag using W = mu*(1-mu) for logit)
          * Cook's distance (approx using Pearson residuals + leverage)

    This is a principled, common approximation for GLM diagnostics in practice.

    Returns:
        dict with:
          - deviance (pd.Series)
          - pearson (pd.Series)
          - std_deviance (pd.Series)
          - std_pearson (pd.Series)
          - leverage (np.ndarray)
          - cooks_distance (np.ndarray)
    """
    X_df = _add_constant_df(X)
    idx = X_df.index
    y = _as_1d_array(y_true).astype(float)

    # Predicted mean mu = P(y=1)
    mu = np.clip(np.asarray(glm_result.predict(X_df)), 1e-12, 1 - 1e-12)

    # Binomial deviance residuals:
    # dev_i = sign(y - mu) * sqrt( 2 * [ y*log(y/mu) + (1-y)*log((1-y)/(1-mu)) ] )
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(y == 1.0, np.log(1.0 / mu), 0.0)
        term0 = np.where(y == 0.0, np.log(1.0 / (1.0 - mu)), 0.0)
        dev = np.sign(y - mu) * np.sqrt(2.0 * (term1 + term0))

    # Pearson residuals: (y - mu) / sqrt(mu*(1-mu))
    var = mu * (1.0 - mu)
    pearson = (y - mu) / np.sqrt(np.clip(var, 1e-12, None))

    # Standardized residuals
    std_dev = dev / _safe_std(dev)
    std_pearson = pearson / _safe_std(pearson)

    # Leverage approximation for logit GLM:
    # H = W^{1/2} X (X' W X)^{-1} X' W^{1/2}
    W = np.clip(var, 1e-12, None)
    X_mat = X_df.to_numpy().astype(float)

    # Compute (X' W X)^{-1} safely
    XTWX = X_mat.T @ (W[:, None] * X_mat)
    try:
        XTWX_inv = np.linalg.inv(XTWX)
    except np.linalg.LinAlgError:
        XTWX_inv = np.linalg.pinv(XTWX)

    # Hat diagonal efficiently: diag( W^{1/2} X A X' W^{1/2} ) = W * row_sums( X A X )
    XA = X_mat @ XTWX_inv
    h = W * np.sum(XA * X_mat, axis=1)
    h = np.clip(h, 0.0, 1.0 - 1e-12)

    # Cook's distance approximation for GLM:
    # D_i ≈ (r_i^2 * h_i) / (p * (1 - h_i)^2), using Pearson residual r_i
    p = X_mat.shape[1]  # number of parameters incl. intercept
    cooks = (pearson**2) * h / (p * (1.0 - h) ** 2)

    diagnostics = {
        "deviance": pd.Series(dev, index=idx),
        "pearson": pd.Series(pearson, index=idx),
        "std_deviance": pd.Series(std_dev, index=idx),
        "std_pearson": pd.Series(std_pearson, index=idx),
        "leverage": h,
        "cooks_distance": cooks,
    }
    return diagnostics


def flag_anomalies_from_diagnostics(
    std_dev_resid,
    leverage,
    cooks_d,
    std_threshold: float = 3.0,
    leverage_quantile: float = 0.99,
    cooks_quantile: float = 0.99,
):
    """
    Flag anomalous observations based on:
      - large standardized deviance residuals
      - high leverage
      - high Cook's distance

    Returns:
        flags_combined (np.ndarray of booleans)
        thresholds (dict)
    """
    std_dev_resid = _as_1d_array(std_dev_resid)
    leverage = _as_1d_array(leverage)
    cooks_d = _as_1d_array(cooks_d)

    flags_std = np.abs(std_dev_resid) > float(std_threshold)

    lev_cut = float(np.quantile(leverage, float(leverage_quantile)))
    cooks_cut = float(np.quantile(cooks_d, float(cooks_quantile)))

    flags_influence = (leverage > lev_cut) | (cooks_d > cooks_cut)
    flags_combined = flags_std | flags_influence

    thresholds = {
        "std_threshold": float(std_threshold),
        "leverage_cutoff": lev_cut,
        "cooks_cutoff": cooks_cut,
    }
    return flags_combined, thresholds


# -------------------------------------------------------------
# 6. Supervised Evaluation Metrics (Test Set)
# -------------------------------------------------------------
def evaluate_supervised(y_true, y_proba, threshold: float = 0.5) -> Dict[str, object]:
    """
    Evaluate supervised classification performance from predicted probabilities.
    Returns a dict of metrics + arrays for plotting ROC/PR.
    """
    y_true = _as_1d_array(y_true).astype(int)
    y_proba = _as_1d_array(y_proba).astype(float)

    y_pred = (y_proba >= float(threshold)).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    pr, rc, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rc, pr)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": pr,
        "recall_curve": rc,
    }


def plot_confusion_matrix(cm, labels=("Non-Fraud", "Fraud")) -> None:
    """Plot confusion matrix (matplotlib-only)."""
    cm = np.asarray(cm)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_roc_curve_from_metrics(metrics: Dict[str, object]) -> None:
    """Plot ROC curve using metrics dict from evaluate_supervised/evaluate_isolation_forest."""
    fpr = np.asarray(metrics["fpr"])
    tpr = np.asarray(metrics["tpr"])
    roc_auc = float(metrics["roc_auc"])

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curve_from_metrics(metrics: Dict[str, object]) -> None:
    """Plot Precision-Recall curve using metrics dict."""
    pr = np.asarray(metrics["precision_curve"])
    rc = np.asarray(metrics["recall_curve"])
    pr_auc = float(metrics["pr_auc"])

    plt.figure(figsize=(5, 4))
    plt.plot(rc, pr, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 7. Anomaly Flag Evaluation (Any split)
# -------------------------------------------------------------
def evaluate_anomaly_flags(y_true, anomaly_flags) -> Dict[str, object]:
    """
    Evaluate how well anomaly flags capture actual frauds.

    Treats anomaly_flags == 1 / True as "predicted fraud".
    Returns TP, FP, FN, TN + precision/recall for fraud class.
    """
    y_true = _as_1d_array(y_true).astype(int)
    anomaly_flags = _as_1d_array(anomaly_flags).astype(int)

    tp = int(np.sum((anomaly_flags == 1) & (y_true == 1)))
    fp = int(np.sum((anomaly_flags == 1) & (y_true == 0)))
    fn = int(np.sum((anomaly_flags == 0) & (y_true == 1)))
    tn = int(np.sum((anomaly_flags == 0) & (y_true == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(prec),
        "recall": float(rec),
    }


# -------------------------------------------------------------
# 8. Isolation Forest (Unsupervised Anomaly Detection)
# -------------------------------------------------------------
def fit_isolation_forest(
    X,
    n_estimators: int = 200,
    contamination: Union[str, float] = "auto",
    random_state: int = 42,
):
    """Fit an Isolation Forest model on feature matrix X."""
    iso = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=contamination,
        random_state=int(random_state),
    )
    iso.fit(X)
    return iso


def evaluate_isolation_forest(model, X, y_true) -> Dict[str, object]:
    """
    Evaluate an Isolation Forest model against true labels.

    Returns metrics dict compatible with plotting helpers, plus:
      - anomaly_flags
      - scores
    """
    y_true = _as_1d_array(y_true).astype(int)

    # Lower scores = more anomalous
    scores = model.score_samples(X)

    # model.predict: -1 (anomaly), 1 (normal)
    anomaly_labels = model.predict(X)
    anomaly_flags = (anomaly_labels == -1).astype(int)

    cm = confusion_matrix(y_true, anomaly_flags)
    prec = precision_score(y_true, anomaly_flags, zero_division=0)
    rec = recall_score(y_true, anomaly_flags, zero_division=0)
    f1 = f1_score(y_true, anomaly_flags, zero_division=0)

    # Use -scores so higher means "more likely anomaly/fraud"
    fpr, tpr, _ = roc_curve(y_true, -scores)
    roc_auc = auc(fpr, tpr)

    pr, rc, _ = precision_recall_curve(y_true, -scores)
    pr_auc = auc(rc, pr)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": pr,
        "recall_curve": rc,
        "anomaly_flags": anomaly_flags,
        "scores": scores,
    }


# -------------------------------------------------------------
# 9. Statistical Test Helper (Optional, rubric-friendly)
# -------------------------------------------------------------
def two_proportion_ztest_flag_rate(y_true, flags, alternative: str = "larger") -> Dict[str, float]:
    """
    Two-proportion z-test comparing flag rates between fraud (y=1) and legit (y=0).

    Returns z-statistic, p-value, and the two flag rates.
    """
    from statsmodels.stats.proportion import proportions_ztest

    y_true = _as_1d_array(y_true).astype(int)
    flags = _as_1d_array(flags).astype(int)

    fraud_mask = (y_true == 1)
    legit_mask = (y_true == 0)

    count_fraud = int(flags[fraud_mask].sum())
    n_fraud = int(fraud_mask.sum())

    count_legit = int(flags[legit_mask].sum())
    n_legit = int(legit_mask.sum())

    z, p = proportions_ztest(
        count=[count_fraud, count_legit],
        nobs=[n_fraud, n_legit],
        alternative=alternative,
    )

    rate_fraud = count_fraud / n_fraud if n_fraud > 0 else 0.0
    rate_legit = count_legit / n_legit if n_legit > 0 else 0.0

    return {
        "z": float(z),
        "p_value": float(p),
        "fraud_flag_rate": float(rate_fraud),
        "legit_flag_rate": float(rate_legit),
        "n_fraud": float(n_fraud),
        "n_legit": float(n_legit),
    }
