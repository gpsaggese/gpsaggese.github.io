"""
Unified utilities for the Credit Card Fraud Detection project.

- Data loading, cleaning, feature engineering, scaling, and SMOTE-Tomek balancing
- Anomaly detectors (Isolation Forest, autoencoder)
- Supervised models (LogReg, RandomForest, XGBoost, CatBoost) and soft-voting ensemble
- Evaluation helpers, threshold tuning, feature importance, and WIT integration
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
_libomp_paths = ["/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"]
existing = os.environ.get("DYLD_LIBRARY_PATH", "")
os.environ["DYLD_LIBRARY_PATH"] = ":".join(_libomp_paths + [existing])
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


# Function 1: Set global seeds for reproducibility
def set_global_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Function 2: Load dataset
def load_raw_data(path: str = "data/raw/creditcard.csv", nrows: Optional[int] = None) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found at {path_obj.resolve()}")
    logger.info("Loading dataset from %s", path_obj)
    df = pd.read_csv(path_obj, nrows=nrows)
    logger.info("Loaded %s rows and %s columns", df.shape[0], df.shape[1])
    return df


# Function 3: Clean dataset
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped = before - len(df_clean)
    if dropped:
        logger.info("Dropped %s duplicate rows", dropped)

    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if df_clean[num_cols].isnull().sum().sum() > 0:
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        logger.info("Filled missing numeric values with column medians")

    df_clean["Class"] = df_clean["Class"].astype(int)
    return df_clean


# Function 4: Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["Hour"] = (df_feat["Time"] // 3600) % 24
    df_feat["Amount_log1p"] = np.log1p(df_feat["Amount"])
    df_feat["Amount_per_hour"] = df_feat["Amount"] / (df_feat["Hour"] + 1)
    return df_feat


# Function 5: Train/test split
def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Train/Test split: %s/%s rows (fraud ratio train=%.4f, test=%.4f)",
        len(X_train),
        len(X_test),
        y_train.mean(),
        y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# Function 6: Scale features
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = scaler or StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    logger.info("Scaled features with StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


# Function 7: Balance with SMOTE-Tomek
def balance_with_smote_tomek(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, SMOTETomek]:
    sampler = SMOTETomek(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info("After SMOTE-Tomek: %s rows (fraud ratio=%.4f)", len(X_res), y_res.mean())
    return pd.DataFrame(X_res, columns=X_train.columns), y_res, sampler


# Function 8: Save processed dataset
def save_processed(df: pd.DataFrame, path: str = "data/processed/creditcard_processed.csv") -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed dataset to %s", output_path.resolve())
    return output_path


# Function 9: Evaluate binary classifier
def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_true, y_score) if y_score is not None else None
    pr_auc = average_precision_score(y_true, y_score) if y_score is not None else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


# Function 10: Optimize probability threshold for best metric
def optimize_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    return best_threshold, {
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
    }


# Function 11: Train Isolation Forest
def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: Optional[float] = None,
    n_estimators: int = 300,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    contamination = contamination or 0.00172
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info("Trained Isolation Forest (contamination=%.5f)", contamination)
    return model


# Function 12: Predict with Isolation Forest
def predict_isolation_forest(model: IsolationForest, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = (model.predict(X) == -1).astype(int)
    scores = -model.decision_function(X)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return labels, scores


# Function 13: Train autoencoder anomaly detector
def train_autoencoder(
    X_train: np.ndarray,
    y_train: Optional[pd.Series] = None,
    normal_label: int = 0,
    encoding_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 1024,
    validation_split: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, float, Dict[str, List[float]]]:
    set_global_seed(random_state)
    if y_train is not None:
        X_train = X_train[y_train == normal_label]
        logger.info("Autoencoder training restricted to normal class (n=%s)", len(X_train))

    input_dim = X_train.shape[1]
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(encoding_dim, activation="relu"),
            keras.layers.Dense(encoding_dim // 2, activation="relu"),
            keras.layers.Dense(encoding_dim, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        verbose=0,
    )
    reconstructions = model.predict(X_train, verbose=0)
    errors = np.mean(np.square(reconstructions - X_train), axis=1)
    threshold = float(np.quantile(errors, 0.995))
    logger.info("Autoencoder trained; anomaly threshold set at %.6f", threshold)
    return model, threshold, history.history


# Function 14: Predict with autoencoder anomaly detector
def predict_autoencoder(model: Any, X: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    reconstructions = model.predict(X, verbose=0)
    errors = np.mean(np.square(reconstructions - X), axis=1)
    labels = (errors >= threshold).astype(int)
    return labels, errors


# Function 15: Compute class weights for imbalance
def _compute_class_weights(y_train: pd.Series) -> Dict[int, float]:
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    return {cls: weight for cls, weight in zip(np.unique(y_train), weights)}


# Function 16: Train supervised models
def train_supervised_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    class_weights = _compute_class_weights(y_train)
    models: Dict[str, object] = {}

    log_reg = LogisticRegression(
        max_iter=800,
        class_weight=class_weights,
        n_jobs=-1,
        solver="lbfgs",
    )
    log_reg.fit(X_train, y_train)
    models["log_reg"] = log_reg

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weights,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.06,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        reg_lambda=1.0,
        min_child_weight=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.06,
        depth=8,
        eval_metric="AUC",
        loss_function="Logloss",
        verbose=False,
        random_seed=random_state,
        class_weights=[class_weights[0], class_weights[1]],
    )
    cat.fit(X_train, y_train)
    models["catboost"] = cat

    return models


# Function 17: Build soft-voting ensemble
def build_soft_voting_ensemble(models: Dict[str, object]) -> VotingClassifier:
    estimators = [(name, model) for name, model in models.items() if hasattr(model, "predict_proba")]
    if not estimators:
        raise ValueError("At least one model with predict_proba is required for soft voting.")
    weights = []
    for name, _ in estimators:
        if "xgboost" in name or "catboost" in name:
            weights.append(3)
        elif "random_forest" in name:
            weights.append(2)
        else:
            weights.append(1)
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
    return ensemble


# Function 18: Collect feature importance across models
def collect_feature_importance(models: Dict[str, object], feature_names: List[str]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            rows.append(pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_, "model": name}))
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0])
            rows.append(pd.DataFrame({"feature": feature_names, "importance": coef / coef.sum(), "model": name}))
    if not rows:
        return pd.DataFrame(columns=["feature", "importance", "model"])
    return pd.concat(rows).sort_values(by="importance", ascending=False)


# Function 19: Wrap model for WIT predict interface
def build_predict_fn(model: object, feature_names: List[str]) -> Callable[[pd.DataFrame], np.ndarray]:
    def _predict(df: pd.DataFrame) -> np.ndarray:
        X = df[feature_names]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            proba = np.vstack([1 - scores, scores]).T
        else:
            labels = model.predict(X)
            proba = np.vstack([1 - labels, labels]).T
        return np.asarray(proba)

    return _predict


# Function 20: Build WIT widget
def build_wit_widget(
    sample_df: pd.DataFrame,
    feature_names: List[str],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    target_col: str = "Class",
    label_vocab: Tuple[str, str] = ("legit", "fraud"),
    max_examples: int = 500,
    height: int = 900,
):
    try:
        from witwidget.notebook.visualization import WitConfigBuilder, WitWidget
    except Exception as exc:
        logger.warning(
            "WIT unavailable: %s. Install with `pip install witwidget ipywidgets==7.* ipython<9 tensorflow==2.13.0` "
            "or on Apple Silicon use `tensorflow-macos==2.13.0` with an ARM Python.",
            exc,
        )
        return None

    sample = sample_df.sample(min(max_examples, len(sample_df)), random_state=RANDOM_STATE)
    examples = sample[feature_names + [target_col]].to_dict(orient="records")
    labels = sample[target_col].astype(int).tolist()

    def _predict(examples_json: List[dict]) -> List[List[float]]:
        features_df = pd.DataFrame(examples_json)[feature_names]
        return predict_fn(features_df).tolist()

    config = (
        WitConfigBuilder(examples, feature_names=feature_names)
        .set_target_feature(target_col)
        .set_label_vocab(list(label_vocab))
        .set_custom_predict_fn(_predict)
    )
    if hasattr(config, "set_eval_mode"):
        config = config.set_eval_mode("binary_classification")
    logger.info("WIT widget configured with %s examples", len(examples))
    return WitWidget(config, height=height)
