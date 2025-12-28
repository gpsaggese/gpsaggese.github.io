import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import lakefs_client
from lakefs_client.client import LakeFSClient
from lakefs_client.models import CommitCreation, BranchCreation
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Handle Optional Imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
except ImportError:
    tf = None

class LakeFSDataHandler:
    def __init__(self, host, access_key, secret_key, repo_name):
        self.repo_name = repo_name
        self.configuration = lakefs_client.Configuration()
        self.configuration.host = host
        self.configuration.username = access_key
        self.configuration.password = secret_key
        self.client = LakeFSClient(self.configuration)
        
    def upload_df(self, df, branch, path, message):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        content_io = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        self._upload_and_commit(branch, path, content_io, message)

    def upload_file(self, file_path, branch, dest_path, message):
        with open(file_path, 'rb') as f:
            content_io = io.BytesIO(f.read())
        self._upload_and_commit(branch, dest_path, content_io, message)

    def _upload_and_commit(self, branch, path, content, message):
        print(f"Uploading to branch '{branch}' at path '{path}'...")
        self.client.objects_api.upload_object(
            repository=self.repo_name, branch=branch, path=path, content=content
        )
        print(f"Committing: {message}")
        try:
            self.client.commits_api.commit(
                repository=self.repo_name, branch=branch,
                commit_creation=CommitCreation(message=message)
            )
        except lakefs_client.ApiException as e:
            if "no changes" in str(e):
                print(f"⚠️ No new changes for '{path}' (Already up to date).")
            else:
                raise e

    def load_df(self, branch, path):
        print(f"Downloading from '{branch}/{path}'...")
        try:
            response = self.client.objects_api.get_object(
                repository=self.repo_name, ref=branch, path=path, _preload_content=False
            )
            data = response.read()
            return pd.read_csv(io.BytesIO(data))
        except Exception as e:
            print(f"Error loading: {e}")
            return None

    def create_branch(self, new_branch, source_branch="main"):
        try:
            self.client.branches_api.create_branch(
                repository=self.repo_name,
                branch_creation=BranchCreation(name=new_branch, source=source_branch)
            )
            print(f"Branch '{new_branch}' created.")
        except Exception:
            print(f"Branch '{new_branch}' likely exists. Switching to it.")

# Visualizaion utils

def save_confusion_matrix(y_true, y_pred, algo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {algo}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    fname = f"{algo}_cm.png"
    plt.savefig(fname)
    plt.close()
    return fname

def save_roc_curve(y_true, y_prob, algo):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve: {algo}')
    plt.legend(loc="lower right")
    fname = f"{algo}_roc.png"
    plt.savefig(fname)
    plt.close()
    return fname

def save_pr_curve(y_true, y_prob, algo):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.title(f'Precision-Recall Curve: {algo}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fname = f"{algo}_pr.png"
    plt.savefig(fname)
    plt.close()
    return fname

# Training utils

def preprocess_data_pro(df):
    print("Preprocessing...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['Class'] = y_train_res
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['Class'] = y_test.values
    return train_df, test_df

def train_and_eval(train_df, test_df, algo='rf'):
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    print(f"Training {algo.upper()}...")
    
    if algo == 'rf':
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    elif algo == 'lr':
        model = LogisticRegression(max_iter=1000)
    elif algo == 'xgb' and xgb:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif algo == 'lgbm' and lgb:
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    elif algo == 'nn' and tf:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif algo == 'ensemble':
        est = [('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
               ('lr', LogisticRegression(max_iter=1000))]
        if xgb: est.append(('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
        if lgb: est.append(('lgbm', lgb.LGBMClassifier(random_state=42, verbose=-1)))
        model = VotingClassifier(estimators=est, voting='soft')
        
    # Ensemble (No Logistic regression)
    elif algo == 'power_ensemble':
        est = [('rf', RandomForestClassifier(n_estimators=50, random_state=42))]
        if xgb: est.append(('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
        if lgb: est.append(('lgbm', lgb.LGBMClassifier(random_state=42, verbose=-1)))
        model = VotingClassifier(estimators=est, voting='soft')
        
    # Tuned XGBoost (Grid Search)
    elif algo == 'tuned_xgb' and xgb:
        print("Starting Grid Search for XGBoost (this takes time)...")
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.01]
        }
        base_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        grid_search = GridSearchCV(estimator=base_xgb, param_grid=param_grid, cv=3, scoring='f1', verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best Params: {grid_search.best_params_}")
        model = grid_search.best_estimator_

    else:
        print(f"Algorithm {algo} missing.")
        return None, None, None

    # Train 
    if algo != 'tuned_xgb':
        if algo == 'nn':
            model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=0)
        else:
            model.fit(X_train, y_train)
        
    # Predict
    if algo == 'nn':
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred 

    print(classification_report(y_test, y_pred))
    return y_test, y_pred, y_prob