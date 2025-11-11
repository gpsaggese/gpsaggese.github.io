import pandas as pd
import numpy as np
# ML Model: Using Gradient Boosted Trees for better complexity score
from sklearn.ensemble import GradientBoostingClassifier 
# Fairlearn: The core fairness mitigation library
from fairlearn.reductions import ExponentiatedGradient 
# Fairlearn: Metrics for evaluation
from fairlearn.metrics import MetricFrame, equalized_odds_difference

# --- Data I/O and Feature Engineering ---

def load_and_preprocess_data(filepath: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads, cleans, and engineers features from the Chicago Crime Data, 
    preparing the data for model training.
    
    In the final version, this will include:
    1. Loading Chicago Crime Data (from filepath).
    2. Cleaning and filtering (e.g., focusing on specific crime types).
    3. Geospatial Binning: Aggregating incidents into grid cells (e.g., hexagons or census tracts).
    4. Merging Demographic Data: Linking features (e.g., race, income) to each grid cell.
    5. Creating Target Variable (y): Defining which grid cells are "Hotspots" (e.g., top 10% of incidents).
    """
    
    # --- For MIDTERM CHECK-IN: Placeholder Structure ---
    # We must simulate the output structure for the code to run end-to-end (e.g., in a test notebook).
    N = 100 
    np.random.seed(42)
    
    # 1. Features (X): Geospatial/Crime History features
    X = pd.DataFrame({
        'Crime_Density_Log': np.random.lognormal(mean=0.5, sigma=0.5, size=N),
        'Police_Visits_Avg': np.random.randint(5, 50, size=N)
    })
    
    # 2. Target (y): Hotspot (1) or Not (0)
    y = pd.Series(np.random.randint(0, 2, size=N))
    
    # 3. Sensitive Attributes (A): CRUCIAL FOR INTERSECTIONAL FAIRNESS
    races = np.random.choice(['Black', 'White', 'Hispanic', 'Asian'], size=N, p=[0.4, 0.3, 0.2, 0.1])
    incomes = np.random.choice(['Low', 'Medium', 'High'], size=N, p=[0.4, 0.4, 0.2])
    
    # Create the Intersectional Feature (e.g., 'Black_Low', 'White_High')
    A = pd.Series(races + '_' + incomes, name='Race_Income')
    
    return X, y, A

# --- Model Training and Fairness Mitigation ---

def train_fair_model(X_train, y_train, sensitive_features):
    """
    Trains a Gradient Boosted Tree (GBT) model using Fairlearn's ExponentiatedGradient 
    to mitigate bias based on the sensitive_features (A).
    """
    print("Training Fair Model...")
    
    # 1. Base estimator (The algorithm we are training)
    estimator = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # 2. Mitigation Strategy: Enforce Equalized Odds (the constraint)
    # Equalized Odds aims to equalize True Positive Rate (TPR) and False Positive Rate (FPR)
    mitigator = ExponentiatedGradient(estimator, constraints="equalized_odds")
                                      
    # 3. Fit the model and mitigation wrapper
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    
    print("Fair model training complete.")
    return mitigator

def evaluate_fairness(model, X_test, y_test, A_test):
    """
    Evaluates model performance and fairness metrics across all sensitive groups.
    """
    # 1. Get predictions
    y_pred = model.predict(X_test)
    
    # 2. Define the metrics to track (Performance and Fairness)
    metrics = {
        "Accuracy": np.mean,
        "True Positive Rate (TPR)": lambda y_true, y_pred: np.mean(y_pred[y_true == 1]),
        "False Positive Rate (FPR)": lambda y_true, y_pred: np.mean(y_pred[y_true == 0]),
    }
    
    # 3. Use MetricFrame to calculate metrics for each group
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    
    # 4. Calculate the overall difference in Equalized Odds (Fairness Metric)
    eq_odds_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=A_test)
    
    return metric_frame.by_group, eq_odds_diff