# SHAP_Credit.API — Design (Skeleton)

> This document explains the small API layer for our SHAP credit scoring tutorial.  
> It will be expanded in later PRs once the modeling is finalized.

## Goals
- Stable functions for data loading, preprocessing, model training, evaluation, and SHAP explanations.
- Keep notebooks minimal by importing from `SHAP_Credit_utils.py`.

## Functions (planned)
- `load_credit_data(path: Optional[str]) -> pd.DataFrame`
- `build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer`
- `train_xgb(X, y) -> xgboost.XGBClassifier`
- `evaluate_model(model, X_test, y_test) -> dict`
- `compute_shap(model, X_sample, preprocessor=None) -> object`

**TODO:** Fill in concrete contracts, types, and invariants after the first real pass on data + model.
