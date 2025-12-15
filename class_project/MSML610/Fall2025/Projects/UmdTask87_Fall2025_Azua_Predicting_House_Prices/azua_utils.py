from __future__ import annotations
"""
Provide utilities for loading the Melbourne housing dataset, building preprocessing pipelines, running cross-validated model selection with progress bars, and saving artifacts for downstream evaluation.
"""
import os,json,joblib,math
from dataclasses import dataclass
from typing import Dict,Any,List,Tuple,Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
TARGET: str = "Price"
DEFAULT_DROP_COLS: List[str] = ["Address"]
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
    
class MelbourneHousingLoader:
    """
    Load and lightly feature-engineer the Melbourne housing dataset from a CSV file.
    """
    def load(self,csv_path: str,target: str = TARGET) -> pd.DataFrame:
        """
        Load the dataset from a CSV and add simple date-derived features if available.
        :param csv_path: path to the CSV file
        :param target: name of the target column
        :return: cleaned dataframe with target present and optionally Year/Month features
        """
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[target])
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"],errors="coerce")
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df = df.drop(columns=["Date"])
        return df
        
class FeaturePreprocessorBuilder:
    """
    Build a ColumnTransformer preprocessing graph for mixed numeric/categorical data.
    """
    def split_features(self,X: pd.DataFrame,target: str = TARGET) -> Tuple[List[str],List[str]]:
        """
        Split columns into numeric and categorical lists based on pandas dtypes.
        :param X: feature dataframe (should not include the target column)
        :param target: target column name (unused if target already excluded)
        :return: (num_cols, cat_cols)
        """
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        return num_cols,cat_cols
        
    def build(self,num_cols: List[str],cat_cols: List[str]) -> ColumnTransformer:
        """
        Build a preprocessing transformer that imputes missing values and encodes categorical features.
        :param num_cols: numeric feature column names
        :param cat_cols: categorical feature column names
        :return: ColumnTransformer that transforms numeric and categorical subsets
        """
        num_pipe = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
        cat_pipe = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))])
        pre = ColumnTransformer(transformers=[("num",num_pipe,num_cols),("cat",cat_pipe,cat_cols)],remainder="drop")
        return pre
        
class ModelFactory:
    """
    Construct candidate regression models for model selection.
    """
    def candidate_models(self,random_state: int = 42) -> Dict[str,Any]:
        """
        Create a dictionary of model name to estimator instances.
        :param random_state: random seed used for stochastic models
        :return: dict mapping model keys to sklearn-compatible regressors
        """
        models: Dict[str,Any] = {}
        models["linreg"] = LinearRegression()
        models["elastic"] = ElasticNet(alpha=0.01,l1_ratio=0.1,random_state=random_state)
        models["rf"] = RandomForestRegressor(n_estimators=400,max_depth=None,n_jobs=-1,random_state=random_state)
        models["xgb"] = XGBRegressor(n_estimators=600,max_depth=8,subsample=0.8,colsample_bytree=0.8,learning_rate=0.05,random_state=random_state,n_jobs=-1,objective="reg:squarederror")
        return models
        
class CrossValidator:
    """
    Evaluate a pipeline with K-fold cross-validation and optional progress bars.
    """
    def evaluate(self,pipe: Pipeline,X: pd.DataFrame,y: pd.Series,folds: int = 5,random_state: int = 42,fold_progress: bool = True,desc: str = "CV",return_folds: bool = False) -> CVSummary:
        """
        Compute mean/std RMSE and R² across K folds, optionally returning per-fold values.
        :param pipe: sklearn Pipeline to evaluate (must implement fit/predict)
        :param X: feature dataframe
        :param y: target series
        :param folds: number of CV folds
        :param random_state: seed for KFold shuffling
        :param fold_progress: whether to display a tqdm progress bar for folds
        :param desc: label for the fold progress bar
        :param return_folds: whether to include per-fold metric lists in the output
        :return: CVSummary with mean/std metrics (and optionally fold lists)
        """
        kf = KFold(n_splits=folds,shuffle=True,random_state=random_state)
        splits = list(kf.split(X))
        it = splits
        if fold_progress:
            it = tqdm(splits,desc=desc,total=len(splits),leave=False)
        rmses: List[float] = []
        r2s: List[float] = []
        for tr_idx,va_idx in it:
            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y.iloc[tr_idx]
            y_va = y.iloc[va_idx]
            pipe.fit(X_tr,y_tr)
            pred = pipe.predict(X_va)
            rmse = float(math.sqrt(mean_squared_error(y_va,pred)))
            r2 = float(r2_score(y_va,pred))
            rmses.append(rmse)
            r2s.append(r2)
        rmse_mean = float(np.mean(rmses))
        rmse_std = float(np.std(rmses,ddof=1)) if len(rmses) > 1 else 0.0
        r2_mean = float(np.mean(r2s))
        r2_std = float(np.std(r2s,ddof=1)) if len(r2s) > 1 else 0.0
        out = CVSummary(rmse=rmse_mean,rmse_std=rmse_std,r2=r2_mean,r2_std=r2_std)
        if return_folds:
            out.rmse_folds = rmses
            out.r2_folds = r2s
        return out
        
class ModelSelector:
    """
    Select the best regression pipeline using cross-validated RMSE and fit it on all training data.
    """
    def __init__(self,random_state: int = 42):
        """
        Initialize the selector with shared builders and configuration.
        :param random_state: seed used for stochastic models and KFold
        :return: None
        """
        self.random_state = random_state
        self.pre_builder = FeaturePreprocessorBuilder()
        self.model_factory = ModelFactory()
        self.cv = CrossValidator()
        
    def select_best(self,df: pd.DataFrame,target: str = TARGET,folds: int = 5,drop_cols: Optional[List[str]] = None,progress: bool = True,fold_progress: bool = True,return_folds: bool = False) -> Dict[str,Any]:
        """
        Train and compare candidate models via K-fold CV, then fit the best pipeline on full data.
        :param df: dataframe containing features and target
        :param target: target column name
        :param folds: number of CV folds
        :param drop_cols: columns to drop from features (defaults to DEFAULT_DROP_COLS)
        :param progress: whether to display a progress bar over models
        :param fold_progress: whether to display a progress bar over folds for each model
        :param return_folds: whether to store per-fold metric lists in results
        :return: bundle dict containing best pipeline, all results, and feature column lists
        """
        if drop_cols is None:
            drop_cols = list(DEFAULT_DROP_COLS)
        X = df.drop(columns=[target])
        if drop_cols:
            X = X.drop(columns=drop_cols,errors="ignore")
        y = df[target].astype(float)
        num_cols,cat_cols = self.pre_builder.split_features(X,target=target)
        pre = self.pre_builder.build(num_cols,cat_cols)
        models = self.model_factory.candidate_models(random_state=self.random_state)
        results: List[Dict[str,Any]] = []
        best: Dict[str,Any] = {"name":None,"rmse":float("inf"),"r2":-1e9,"pipeline":None}
        items = list(models.items())
        it = items
        if progress:
            it = tqdm(items,desc="Model selection (CV)",total=len(items))
        for name,model in it:
            pipe = Pipeline(steps=[("pre",pre),("model",model)])
            summary = self.cv.evaluate(pipe,X,y,folds=folds,random_state=self.random_state,fold_progress=fold_progress,desc=f"{name} folds",return_folds=return_folds)
            row: Dict[str,Any] = {"model":name,"rmse":summary.rmse,"rmse_std":summary.rmse_std,"r2":summary.r2,"r2_std":summary.r2_std}
            if return_folds and summary.rmse_folds is not None and summary.r2_folds is not None:
                row["rmse_folds"] = summary.rmse_folds
                row["r2_folds"] = summary.r2_folds
            results.append(row)
            if summary.rmse < float(best["rmse"]):
                best.update({"name":name,"rmse":summary.rmse,"r2":summary.r2,"pipeline":pipe})
        best["pipeline"].fit(X,y)
        bundle: Dict[str,Any] = {"best":best,"all_results":results,"num_cols":num_cols,"cat_cols":cat_cols,"drop_cols":drop_cols,"folds":folds,"random_state":self.random_state}
        return bundle
        
def load_melbourne(csv_path: str,target: str = TARGET) -> pd.DataFrame:
    """
    Load the Melbourne housing dataset and apply light cleaning/feature engineering.
    :param csv_path: path to melb_data.csv
    :param target: target column name
    :return: cleaned dataframe
    """
    return MelbourneHousingLoader().load(csv_path,target=target)
    
def split_features(df: pd.DataFrame,target: str = TARGET,drop_cols: Optional[List[str]] = None) -> Tuple[List[str],List[str]]:
    """
    Split a dataframe into numeric and categorical feature column lists.
    :param df: dataframe containing features and target
    :param target: target column name
    :param drop_cols: columns to exclude from features
    :return: (num_cols, cat_cols)
    """
    if drop_cols is None:
        drop_cols = list(DEFAULT_DROP_COLS)
    X = df.drop(columns=[target],errors="ignore")
    if drop_cols:
        X = X.drop(columns=drop_cols,errors="ignore")
    return FeaturePreprocessorBuilder().split_features(X,target=target)
    
def build_preprocessor(num_cols: List[str],cat_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer for the provided numeric/categorical feature lists.
    :param num_cols: numeric feature column names
    :param cat_cols: categorical feature column names
    :return: ColumnTransformer instance
    """
    return FeaturePreprocessorBuilder().build(num_cols,cat_cols)
    
def train_select_best(df: pd.DataFrame,target: str = TARGET,folds: int = 5,random_state: int = 42,drop_cols: Optional[List[str]] = None,progress: bool = True,fold_progress: bool = True,return_folds: bool = False) -> Dict[str,Any]:
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
    :return: dict bundle with keys best, all_results, num_cols, cat_cols, drop_cols
    """
    selector = ModelSelector(random_state=random_state)
    return selector.select_best(df,target=target,folds=folds,drop_cols=drop_cols,progress=progress,fold_progress=fold_progress,return_folds=return_folds)
    
def save_artifacts(best_bundle: Dict[str,Any],outdir: str = "artifacts") -> None:
    """
    Save the fitted best pipeline and metadata to an output directory.
    :param best_bundle: output of train_select_best()
    :param outdir: directory to write artifacts into
    :return: None
    """
    os.makedirs(outdir,exist_ok=True)
    joblib.dump(best_bundle["best"]["pipeline"],os.path.join(outdir,"model.joblib"))
    meta: Dict[str,Any] = {"model":best_bundle["best"]["name"],"rmse_cv":float(best_bundle["best"]["rmse"]),"r2_cv":float(best_bundle["best"]["r2"]),"num_cols":best_bundle.get("num_cols",[]),"cat_cols":best_bundle.get("cat_cols",[]),"drop_cols":best_bundle.get("drop_cols",[]),"folds":int(best_bundle.get("folds",0) or 0),"random_state":int(best_bundle.get("random_state",0) or 0)}
    with open(os.path.join(outdir,"metrics.json"),"w") as f:
        json.dump(meta,f,indent=2)
    try:
        pd.DataFrame(best_bundle.get("all_results",[])).to_csv(os.path.join(outdir,"cv_results.csv"),index=False)
    except Exception:
        pass
        
def load_model(path: str = "artifacts/model.joblib") -> Any:
    """
    Load a previously saved sklearn pipeline from disk.
    :param path: path to the saved joblib model
    :return: deserialized sklearn Pipeline
    """
    return joblib.load(path)
