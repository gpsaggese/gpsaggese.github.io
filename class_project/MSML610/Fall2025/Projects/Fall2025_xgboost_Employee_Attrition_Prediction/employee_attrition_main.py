"""
Employee Attrition Prediction with XGBoost, Logistic Regression, and Random Forest.

This script:
- Loads the IBM HR Employee Attrition dataset
  (downloaded from Kaggle via kagglehub in utils_data_io.load_hr_dataset)
- Preprocesses data (encoding + scaling)
- Trains and compares 3 models:
    * XGBoost
    * Logistic Regression
    * Random Forest
- Evaluates them using Accuracy, F1-score, and ROC-AUC
- Saves feature importance plot for XGBoost
- Saves SHAP summary plot for XGBoost for interpretability
"""

import numpy as np

from utils_data_io import (
    load_hr_dataset,
    build_preprocessor,
    train_test_split_stratified,
)
from utils_post_processing import (
    evaluate_classifier,
    get_feature_names_from_preprocessor,
    plot_feature_importance,
)

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():
    print(">>> DEBUG: multi-model version running (seamless plots)")

    # --------- 1. Dataset loading (from Kaggle via kagglehub in utils_data_io) ----------
    print("Loading dataset directly from Kaggle (via kagglehub)...")
    X, y, categorical_cols, numeric_cols = load_hr_dataset()

    # --------- 2. Train / Test split ----------
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"Train shape: (X={X_train.shape[0]}, features={X_train.shape[1]}), "
        f"Test shape: (X={X_test.shape[0]}, features={X_test.shape[1]})"
    )

    # --------- 3. Class imbalance (for XGBoost) ----------
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos
    print(
        f"\nClass distribution → Neg: {neg}, Pos: {pos}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # --------- 4. Define 3 models with their own pipelines ----------
    models = {}

    # XGBoost (with class weighting via scale_pos_weight)
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    models["XGBoost"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", xgb_clf),
        ]
    )

    # Logistic Regression (with class_weight="balanced")
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    models["LogisticRegression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", log_reg),
        ]
    )

    # Random Forest (with class_weight="balanced")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    models["RandomForest"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", rf_clf),
        ]
    )

    # --------- 5. Train & evaluate all models ----------
    trained_models = {}
    metrics_all = {}

    print("\n=== Training and evaluating models ===")
    for name, pipe in models.items():
        print(f"\n--- {name}: training ---")
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe

        print(f"--- {name}: evaluation on test set ---")
        metrics = evaluate_classifier(
            model=pipe,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
        )
        metrics_all[name] = metrics

    # Compact comparison table
    print("\n=== Model Comparison (Test Set) ===")
    print("Model               Accuracy   F1-score   ROC-AUC")
    print("-------------------------------------------------")
    for name, m in metrics_all.items():
        print(
            f"{name:18} {m['accuracy']:.4f}   {m['f1_score']:.4f}   {m['roc_auc']:.4f}"
        )

    # --------- 6. Feature importance for XGBoost ----------
    print("\n=== XGBoost Feature Importance ===")
    xgb_pipeline = trained_models["XGBoost"]
    xgb_preprocessor = xgb_pipeline.named_steps["preprocess"]
    xgb_model = xgb_pipeline.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(
        xgb_preprocessor, numeric_cols, categorical_cols
    )

    plot_feature_importance(
        model=xgb_model,
        feature_names=feature_names,
        top_n=15,
        title="Top 15 Feature Importances (XGBoost - Employee Attrition)",
    )

    # --------- 7. SHAP values for interpretability (XGBoost) ----------
    print("\n=== SHAP Analysis for XGBoost ===")
    try:
        import shap
        import matplotlib.pyplot as plt

        # For speed, we take a subset of training data after preprocessing
        X_train_processed = xgb_preprocessor.transform(X_train)

        # Sample up to 500 rows for SHAP
        n_samples = min(500, X_train_processed.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train_processed.shape[0], n_samples, replace=False)
        X_sample = X_train_processed[idx]

        print(f"Computing SHAP values on a sample of {n_samples} training points...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot (global feature importance + direction)
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.title("SHAP Summary Plot - XGBoost (Employee Attrition)")
        plt.tight_layout()
        plt.savefig("xgb_shap_summary.png", bbox_inches="tight")
        plt.close()

        print("SHAP summary plot saved as xgb_shap_summary.png")

    except ImportError:
        print("SHAP is not installed. Run `pip install shap` to enable SHAP plots.")
    except Exception as e:
        print(f"Could not compute SHAP values: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
