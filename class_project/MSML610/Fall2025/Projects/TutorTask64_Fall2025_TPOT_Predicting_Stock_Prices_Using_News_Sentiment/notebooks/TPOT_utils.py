"""
TPOT_utils.py
Utility functions for TPOT stock prediction project

Author: Bradley Scott
Date: October - December 2025
"""
'''
[BS12012025] ut_610_000001
[BS12012025] import all necessary modules
'''
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report
import seaborn as sns

'''
[BS12012025] ut_610_000005
[BS12012025] Build out the configuration class to hold constants
'''
class Config:
    # Paths
    DATA_DIR = Path("/workspace/data")
    
    # Data parameters
    MIN_COVERAGE_DAYS = 200
    TOP_N_TICKERS = 25
    
    # Date split
    TRAIN_TEST_CUTOFF = "2023-01-01"
    
    # Feature engineering
    SENTIMENT_EXTREME_THRESHOLD = 0.5
    HIGH_VOLUME_ZSCORE = 2.0
    CONSENSUS_THRESHOLD = 0.7
    BIG_MOVE_THRESHOLD = 0.02
    
    # Model training
    TPOT_GENERATIONS = 5
    TPOT_POPULATION = 12
    TPOT_CV_FOLDS = 3
    TPOT_MAX_TIME_MINS = 60
    TPOT_MAX_EVAL_TIME_MINS = 3
    TPOT_SUBSAMPLE = 0.8

'''
[BS12012025] ut_610_000010
[BS12012025] build out a function to load the data
'''
def load_processed_data(file_path: Path) -> pd.DataFrame:
    """
    Load processed model-ready data.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        DataFrame with processed features
    """
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

'''
[BS12012025] ut_610_000015
[BS12012025] Create high impact filter for trying to look at extreme cases.
'''
def create_high_impact_filter(df: pd.DataFrame, config: Config = None) -> pd.DataFrame:
    """
    Filter dataset to high-impact news events only.
    
    Strategy: Keep days with unusual news volume, extreme sentiment,
    or consensus across articles.
    """
    if config is None:
        config = Config()
    
    print(f"Filtering {len(df):,} rows to high-impact events...")
    
    # Filter 1: High volume days
    df['news_volume_zscore'] = (
        df.groupby('ticker')['news_count']
        .transform(lambda x: zscore(x, nan_policy='omit'))
    )
    high_volume = df['news_volume_zscore'] > config.HIGH_VOLUME_ZSCORE
    
    # Filter 2: Extreme sentiment
    extreme_sentiment = df['avg_sentiment'].abs() > config.SENTIMENT_EXTREME_THRESHOLD
    
    # Filter 3: High consensus
    consensus = (
        (df['news_count'] >= 3) &
        (
            (df['pos_count'] / df['news_count'] > config.CONSENSUS_THRESHOLD) |
            (df['neg_count'] / df['news_count'] > config.CONSENSUS_THRESHOLD)
        )
    )
    
    # Combined filter: high volume OR extreme sentiment OR consensus
    filtered_df = df[
        high_volume | extreme_sentiment | consensus
    ].copy()
    
    pct_kept = len(filtered_df) / len(df) * 100
    print(f"Kept {len(filtered_df):,} rows ({pct_kept:.1f}%)")
    
    return filtered_df
    
'''
[BS12012025] ut_610_000020
[BS12012025] Create features and target
    NB: Have to be careful of not having any data leakage
'''
def prepare_features_and_target(
    df: pd.DataFrame, 
    target_col: str = 'ret_1d',
    binary: bool = True
) -> tuple:
    """
    Prepare feature matrix and target variable.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        binary: If True, convert to binary classification (up/down)
        
    Returns:
        Tuple of (X, y, feature_columns)
    """
    # Define feature candidates
    candidate_feats = [
        # Sentiment Features (from today's news)
        'avg_sentiment',      # Today's average sentiment
        'pos_count',          # Count of positive articles today
        'neg_count',          # Count of negative articles today
        'news_count',         # Total articles today
        
        # Lagged Sentiment Features (from past)
        'sent_lag1',          # Yesterday's sentiment
        'sent_roll5',         # 5-day rolling average sentiment
        'sent_roll10',        # 10-day rolling average sentiment
        'news_count_roll5',   # 5-day rolling news count
        
        # Binary Sentiment Flags
        'sent_pos',           # Is yesterday's sentiment positive?
        'sent_neg',           # Is yesterday's sentiment negative?
        
        #Historical Price Features (from past only)
        'ret_1d_past',        # Yesterday's return
        'ret_5d_past',        # 5-day past return
        'ret_10d_past',       # 10-day past return
        'ret_20d_past',       # 20-day past return (if available)
        
        # Volatility Features (from past)
        'price_vol10',        # 10-day historical volatility
        'price_vol20',        # 20-day historical volatility
        
        # Momentum Features (from past)
        'momentum_5d',        # 5-day momentum
        'momentum_10d',       # 10-day momentum
    ]
    
    # Keep only features that exist
    feat_cols = [c for c in candidate_feats if c in df.columns]
    
    # Build feature matrix
    X = df[['ticker', 'date'] + feat_cols].copy()
    
    # Target variable
    if binary:
        y = (df[target_col] > 0).astype(int)
    else:
        y = df[target_col].astype('float32')
    
    # Remove rows with missing data
    mask = X[feat_cols].notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    
    print(f"Features: {len(feat_cols)}")
    print(f"Samples: {len(X):,}")
    if binary:
        print(f"Class balance: {y.mean():.1%} positive")
    
    return X, y, feat_cols

'''
[BS12012025] ut_610_000025
[BS12012025] Build a function to do the train/test split
'''
def train_test_split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    cutoff_date: str
) -> tuple:
    """
    Split data into train/test using temporal cutoff.
    
    Args:
        X: Feature matrix
        y: Target variable
        cutoff_date: Date string 'YYYY-MM-DD'
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    cutoff = pd.to_datetime(cutoff_date).date()
    
    train_mask = X["date"] < cutoff
    test_mask = X["date"] >= cutoff
    
    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    
    print(f"Train: {len(X_train):,} samples ({X_train['date'].min()} to {X_train['date'].max()})")
    print(f"Test:  {len(X_test):,} samples ({X_test['date'].min()} to {X_test['date'].max()})")
    
    return X_train, X_test, y_train, y_test


'''
[BS12012025] ut_610_000030
[BS12012025] build a function to evaluate the model
'''
def evaluate_classifier(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list
) -> dict:
    """
    Evaluate binary classifier and return metrics.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: Test labels
        feature_cols: List of feature column names
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test[feature_cols])
    y_proba = model.predict_proba(X_test[feature_cols])[:, 1]
    
    # Metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'baseline': 0.5,
        'edge': accuracy_score(y_test, y_pred) - 0.5,
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results

'''
[BS12012025] ut_610_000035
[BS12012025] print the evaluation results
'''
def print_evaluation_summary(results: dict, model_name: str = "Model"):
    """
    Print formatted evaluation results.
    
    Args:
        results: Dictionary from evaluate_classifier
        model_name: Name for display
    """
    print("\n" + "="*60)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:     {results['accuracy']:.2%}")
    print(f"Baseline:     {results['baseline']:.0%}")
    print(f"Edge:         {results['edge']*100:+.2f} percentage points")
    print(f"ROC-AUC:      {results['roc_auc']:.4f}")
    
    if results['accuracy'] > 0.54:
        print("\nExcellent performance!")
    elif results['accuracy'] > 0.52:
        print("\nStatistically significant edge")
    elif results['accuracy'] > 0.51:
        print("\nModest positive result")
    else:
        print("\nNo better than baseline")
    
    print("="*60)


'''
[BS12012025] ut_610_000040
[BS12012025] Build a function to visualize the results
'''
def plot_evaluation_dashboard(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    results: dict,
    save_path: str = None
):
    """
    Create comprehensive evaluation visualization.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        results: Results dictionary from evaluate_classifier
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={results["roc_auc"]:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Prediction Distribution
    axes[0, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='Actual DOWN', color='red')
    axes[0, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Actual UP', color='green')
    axes[0, 1].axvline(0.5, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Predicted Probability of UP')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Prediction Distribution by True Class')
    axes[0, 1].legend()
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xticklabels(['DOWN', 'UP'])
    axes[1, 0].set_yticklabels(['DOWN', 'UP'])
    
    # 4. Calibration
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    for i in range(len(bins) - 1):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(y_test[mask].mean())
        else:
            bin_accs.append(np.nan)
    
    axes[1, 1].plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Actual Fraction Positive')
    axes[1, 1].set_title('Calibration Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()