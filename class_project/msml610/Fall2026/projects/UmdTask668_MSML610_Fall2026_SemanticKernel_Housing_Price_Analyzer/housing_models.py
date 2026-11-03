
"""Baseline ML models for the Housing Price Analyzer."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from housing_data import load_housing_data


def train_linear_regression():
    """Train and evaluate a baseline Linear Regression model."""
    X, y = load_housing_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }

    return model, metrics


if __name__ == "__main__":
    model, scores = train_linear_regression()

    print("Linear Regression Results")
    print("-------------------------")

    for name, value in scores.items():
        print(f"{name}: {value:.4f}")

    print("\nModel Coefficients:")
    for feature, coefficient in zip(
        model.feature_names_in_, model.coef_
    ):
        print(f"{feature}: {coefficient:.4f}")
