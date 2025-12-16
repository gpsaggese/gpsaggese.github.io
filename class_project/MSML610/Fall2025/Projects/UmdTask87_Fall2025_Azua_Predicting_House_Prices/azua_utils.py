from __future__ import annotations
"""
Provide a lightweight AutoML-style API for tabular regression.

This module focuses on:
- Loading tabular CSV data (optionally deriving date parts).
- Automatically building preprocessing for mixed numeric/categorical features.
- Comparing multiple regression models with K-fold cross-validation.
- Saving/loading deployable sklearn pipelines as artifacts.
"""
import os
import json
import math
import joblib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

DEFAULT_TARGET: str = "Price"

@dataclass
class CVSummary:
    """
    Store cross-validation summary statistics for a regression pipeline.
    """
    rmse: float
    rmse_std: float
    r2: float
    r2_std: float
    rmse_folds: Optional[List[float]] = None
    r2_folds: Optional[List[float]] = None
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this summary into a JSON-serializable dictionary.
        :return: dict with mean/std metrics and optional per-fold lists
        """
        out: Dict[str, Any] = {"rmse": float(self.rmse), "rmse_std": float(self.rmse_std), "r2": float(self.r2), "r2_std": float(self.r2_std)}
        if self.rmse_folds is not None:
            out["rmse_folds"] = [float(x) for x in self.rmse_folds]
        if self.r2_folds is not None:
            out["r2_folds"] = [float(x) for x in self.r2_folds]
        return out
        
class TabularCSVLoader:
    """
    Load a CSV file into a DataFrame with optional target filtering and date feature derivation.
    """
    def load(self,csv_path: str,target: Optional[str] = DEFAULT_TARGET,date_cols: Optional[Sequence[str]] = None,derive_date_parts: bool = False,drop_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Load tabular data from a CSV file and optionally derive date features.
        :param csv_path: path to the CSV file
        :param target: name of the target column; if provided, rows missing target are dropped
        :param date_cols: columns to parse as datetimes (only used if derive_date_parts=True)
        :param derive_date_parts: whether to derive <col>_Year and <col>_Month from date columns
        :param drop_cols: optional columns to drop after loading/feature derivation
        :return: dataframe with optional target filtering and derived date part features
        """
        df = pd.read_csv(csv_path)
        if target is not None and target in df.columns:
            df = df.dropna(subset=[target])
        if derive_date_parts and date_cols is not None:
            for c in date_cols:
                if c not in df.columns:
                    continue
                dt = pd.to_datetime(df[c], errors="coerce")
                df[f"{c}_Year"] = dt.dt.year
                df[f"{c}_Month"] = dt.dt.month
        if drop_cols:
            df = df.drop(columns=list(drop_cols), errors="ignore")
        return df
        
class FeaturePreprocessorBuilder:
    """
    Build a ColumnTransformer preprocessing graph for mixed numeric/categorical data.
    """
    def split_features(self,X: pd.DataFrame,exclude_cols: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Split columns into numeric and categorical lists based on pandas dtypes.
        :param X: feature dataframe
        :param exclude_cols: optional columns to exclude from feature lists
        :return: (num_cols, cat_cols)
        """
        cols = X.columns.tolist()
        if exclude_cols:
            cols = [c for c in cols if c not in set(exclude_cols)]
        X_use = X[cols]
        num_cols = X_use.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_use.columns if c not in num_cols]
        return num_cols, cat_cols
        
    def build(self,num_cols: Sequence[str],cat_cols: Sequence[str]) -> ColumnTransformer:
        """
        Build a preprocessing transformer that imputes missing values and encodes categorical features.
        :param num_cols: numeric feature column names
        :param cat_cols: categorical feature column names
        :return: ColumnTransformer that transforms numeric and categorical subsets
        """
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
        pre = ColumnTransformer(transformers=[("num", num_pipe, list(num_cols)), ("cat", cat_pipe, list(cat_cols))], remainder="drop")
        return pre
        
class ModelFactory:
    """
    Construct candidate regression models for model selection.
    """
    def candidate_models(self, random_state: int = 42) -> Dict[str, Any]:
        """
        Create a dictionary of model name to estimator instances.
        :param random_state: random seed used for stochastic models
        :return: dict mapping model keys to sklearn-compatible regressors
        """
        models: Dict[str, Any] = {}
        models["linreg"] = LinearRegression()
        models["elastic"] = ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=random_state)
        models["rf"] = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state)
        models["xgb"] = XGBRegressor(
            n_estimators=600,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        return models

        
class CrossValidator:
    """
    Evaluate a pipeline with K-fold cross-validation and optional progress bars.
    """
    def evaluate(self,pipe: Pipeline,X: pd.DataFrame,y: Union[pd.Series, np.ndarray],folds: int = 5,random_state: int = 42,fold_progress: bool = True,desc: str = "CV",return_folds: bool = False) -> CVSummary:
        """
        Compute mean/std RMSE and R² across K folds, optionally returning per-fold values.
        :param pipe: sklearn Pipeline to evaluate (must implement fit/predict)
        :param X: feature dataframe
        :param y: target values
        :param folds: number of CV folds
        :param random_state: seed for KFold shuffling
        :param fold_progress: whether to display a tqdm progress bar for folds
        :param desc: label for the fold progress bar
        :param return_folds: whether to include per-fold metric lists in the output
        :return: CVSummary with mean/std metrics (and optionally fold lists)
        """
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(X))
        it = tqdm(splits, desc=desc, total=len(splits), leave=False) if fold_progress else splits
        rmses: List[float] = []
        r2s: List[float] = []
        for tr_idx, va_idx in it:
            pipe_fold = clone(pipe)
            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y_series.iloc[tr_idx]
            y_va = y_series.iloc[va_idx]
            pipe_fold.fit(X_tr, y_tr)
            pred = pipe_fold.predict(X_va)
            rmse = float(math.sqrt(mean_squared_error(y_va, pred)))
            r2 = float(r2_score(y_va, pred))
            rmses.append(rmse)
            r2s.append(r2)
        rmse_mean = float(np.mean(rmses))
        rmse_std = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
        r2_mean = float(np.mean(r2s))
        r2_std = float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0
        out = CVSummary(rmse=rmse_mean, rmse_std=rmse_std, r2=r2_mean, r2_std=r2_std)
        if return_folds:
            out.rmse_folds = rmses
            out.r2_folds = r2s
        return out
        
class ModelSelector:
    """
    Select the best regression pipeline using cross-validated RMSE and fit it on all provided data.
    """
    def __init__(self,random_state: int = 42):
        """
        Initialize the selector with shared builders and configuration.
        :param random_state: seed used for stochastic models and KFold
        :return: None
        """
        self.random_state = int(random_state)
        self.pre_builder = FeaturePreprocessorBuilder()
        self.model_factory = ModelFactory()
        self.cv = CrossValidator()
        
    def select_best(self,df: pd.DataFrame,target: str = DEFAULT_TARGET,folds: int = 5,drop_cols: Optional[Sequence[str]] = None,progress: bool = True,fold_progress: bool = True,return_folds: bool = False,models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train and compare candidate models via K-fold CV, then fit the best pipeline on full data.
        :param df: dataframe containing features and target
        :param target: target column name
        :param folds: number of CV folds
        :param drop_cols: columns to drop from features
        :param progress: whether to display a progress bar over models
        :param fold_progress: whether to display a progress bar over folds for each model
        :param return_folds: whether to store per-fold metric lists in results
        :param models: optional dict of model name -> estimator; if None, uses ModelFactory defaults
        :return: bundle dict containing best pipeline, all results, and feature column lists
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
        X = df.drop(columns=[target])
        if drop_cols:
            X = X.drop(columns=list(drop_cols), errors="ignore")
        y = df[target].astype(float)
        num_cols, cat_cols = self.pre_builder.split_features(X)
        pre = self.pre_builder.build(num_cols, cat_cols)
        use_models = models if models is not None else self.model_factory.candidate_models(random_state=self.random_state)
        results: List[Dict[str, Any]] = []
        best: Dict[str, Any] = {"name": None, "rmse": float("inf"), "r2": -1e18, "pipeline": None}
        items = list(use_models.items())
        it = tqdm(items, desc="Model selection (CV)", total=len(items)) if progress else items
        for name, model in it:
            pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            summary = self.cv.evaluate(pipe, X, y, folds=folds, random_state=self.random_state, fold_progress=fold_progress, desc=f"{name} folds", return_folds=return_folds)
            row: Dict[str, Any] = {"model": str(name), "rmse": summary.rmse, "rmse_std": summary.rmse_std, "r2": summary.r2, "r2_std": summary.r2_std}
            if return_folds and summary.rmse_folds is not None and summary.r2_folds is not None:
                row["rmse_folds"] = summary.rmse_folds
                row["r2_folds"] = summary.r2_folds
            results.append(row)
            if summary.rmse < float(best["rmse"]):
                best.update({"name": str(name), "rmse": float(summary.rmse), "r2": float(summary.r2), "pipeline": pipe})
        if best["pipeline"] is None:
            raise RuntimeError("No candidate models were evaluated; cannot select a best model.")
        best["pipeline"].fit(X, y)
        bundle: Dict[str, Any] = {"best": best, "all_results": results, "num_cols": num_cols, "cat_cols": cat_cols, "drop_cols": list(drop_cols) if drop_cols else [], "folds": int(folds), "random_state": int(self.random_state), "target": str(target)}
        return bundle
        
class ArtifactManager:
    """
    Save and load model artifacts for downstream inference and deployment.
    """
    def save(self,bundle: Dict[str, Any],outdir: str = "artifacts") -> None:
        """
        Save the fitted best pipeline and metadata to an output directory.
        :param bundle: output of ModelSelector.select_best() / train_select_best()
        :param outdir: directory to write artifacts into
        :return: None
        """
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(bundle["best"]["pipeline"], os.path.join(outdir, "model.joblib"))
        meta: Dict[str, Any] = {"model": bundle["best"]["name"], "rmse_cv": float(bundle["best"]["rmse"]), "r2_cv": float(bundle["best"]["r2"]), "num_cols": bundle.get("num_cols", []), "cat_cols": bundle.get("cat_cols", []), "drop_cols": bundle.get("drop_cols", []), "folds": int(bundle.get("folds", 0) or 0), "random_state": int(bundle.get("random_state", 0) or 0), "target": bundle.get("target", None)}
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(meta, f, indent=2)
        try:
            pd.DataFrame(bundle.get("all_results", [])).to_csv(os.path.join(outdir, "cv_results.csv"), index=False)
        except Exception:
            pass
            
    def load_model(self,path: str = "artifacts/model.joblib") -> Any:
        """
        Load a previously saved sklearn pipeline from disk.
        :param path: path to the saved joblib model
        :return: deserialized sklearn Pipeline
        """
        return joblib.load(path)
        
def load_csv(csv_path: str,target: Optional[str] = DEFAULT_TARGET,date_cols: Optional[Sequence[str]] = None,derive_date_parts: bool = False,drop_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Load tabular data from a CSV file with optional target filtering and date feature derivation.
    :param csv_path: path to the CSV file
    :param target: name of the target column; if provided, rows missing target are dropped
    :param date_cols: columns to parse as datetimes (only used if derive_date_parts=True)
    :param derive_date_parts: whether to derive <col>_Year and <col>_Month from date columns
    :param drop_cols: optional columns to drop after loading/feature derivation
    :return: dataframe with optional target filtering and derived date part features
    """
    return TabularCSVLoader().load(csv_path, target=target, date_cols=date_cols, derive_date_parts=derive_date_parts, drop_cols=drop_cols)
    
def split_features(df: pd.DataFrame,target: str = DEFAULT_TARGET,drop_cols: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Split a dataframe into numeric and categorical feature column lists.
    :param df: dataframe containing features and target
    :param target: target column name
    :param drop_cols: columns to exclude from features
    :return: (num_cols, cat_cols)
    """
    X = df.drop(columns=[target], errors="ignore")
    if drop_cols:
        X = X.drop(columns=list(drop_cols), errors="ignore")
    return FeaturePreprocessorBuilder().split_features(X)
    
def build_preprocessor(num_cols: Sequence[str],cat_cols: Sequence[str]) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer for the provided numeric/categorical feature lists.
    :param num_cols: numeric feature column names
    :param cat_cols: categorical feature column names
    :return: ColumnTransformer instance
    """
    return FeaturePreprocessorBuilder().build(num_cols, cat_cols)
    
def candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Return the default candidate model dictionary.
    :param random_state: random seed used for stochastic models
    :return: dict mapping model keys to sklearn-compatible regressors
    """
    return ModelFactory().candidate_models(random_state=random_state)

    
def cv_scores(pipe: Pipeline,X: pd.DataFrame,y: Union[pd.Series, np.ndarray],folds: int = 5,random_state: int = 42,fold_progress: bool = False) -> Tuple[float, float]:
    """
    Evaluate a pipeline with K-fold CV and return (rmse_mean, r2_mean).
    :param pipe: sklearn Pipeline to evaluate
    :param X: feature dataframe
    :param y: target values
    :param folds: number of folds
    :param random_state: seed for KFold shuffling
    :param fold_progress: whether to display a fold progress bar
    :return: (rmse_mean, r2_mean)
    """
    summary = CrossValidator().evaluate(pipe, X, y, folds=folds, random_state=random_state, fold_progress=fold_progress, desc="CV", return_folds=False)
    return float(summary.rmse), float(summary.r2)
    
def train_select_best(df: pd.DataFrame,target: str = DEFAULT_TARGET,folds: int = 5,random_state: int = 42,drop_cols: Optional[Sequence[str]] = None,progress: bool = True,fold_progress: bool = True,return_folds: bool = False,models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run cross-validated model selection and return a bundle containing the fitted best pipeline.
    :param df: dataframe containing features and target
    :param target: target column name
    :param folds: number of CV folds
    :param random_state: seed for model randomness and CV shuffling
    :param drop_cols: columns to exclude from the feature set
    :param progress: whether to display a progress bar over models
    :param fold_progress: whether to display a progress bar over folds per model
    :param return_folds: whether to store per-fold metrics in the result rows
    :param models: optional dict of model name -> estimator; if None, uses default candidates
    :return: dict bundle with keys best, all_results, num_cols, cat_cols, drop_cols
    """
    selector = ModelSelector(random_state=random_state)
    return selector.select_best(df, target=target, folds=folds, drop_cols=drop_cols, progress=progress, fold_progress=fold_progress, return_folds=return_folds, models=models)
    
def save_artifacts(best_bundle: Dict[str, Any],outdir: str = "artifacts") -> None:
    """
    Save the fitted best pipeline and metadata to an output directory.
    :param best_bundle: output of train_select_best()
    :param outdir: directory to write artifacts into
    :return: None
    """
    ArtifactManager().save(best_bundle, outdir=outdir)
    
def load_model(path: str = "artifacts/model.joblib") -> Any:
    """
    Load a previously saved sklearn pipeline from disk.
    :param path: path to the saved joblib model
    :return: deserialized sklearn Pipeline
    """
    return ArtifactManager().load_model(path=path)
