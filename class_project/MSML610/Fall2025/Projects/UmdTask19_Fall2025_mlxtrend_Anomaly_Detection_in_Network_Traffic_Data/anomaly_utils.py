import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def load_unsw_data(paths, columns):
    dfs = [pd.read_csv(p, names=columns, header=None) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess_data(df):
    cat_cols = ['proto', 'service', 'state', 'attack_cat']
    num_cols = [c for c in df.columns if c not in cat_cols + ['label']]
    num_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])
    cat_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_tf, num_cols),
        ('cat', cat_tf, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgb(X_train, y_train, X_test, y_test, scale_pos_weight):
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.12,
        eval_metric='aucpr', random_state=42,
        scale_pos_weight=scale_pos_weight, tree_method='gpu_hist'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model

def evaluate_model(name, y_test, proba):
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    print(f"{name} | ROC: {roc:.3f} | PR: {pr:.3f}")
    return roc, pr
