
"""Model training and evaluation for the Housing Price Analyzer."""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from housing_data import load_housing_data


RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_models():
    """Return the baseline regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Calculate regression evaluation metrics."""
    predictions = model.predict(X_test)

    return {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(
            mean_squared_error(y_test, predictions)
        )),
        "R2": float(r2_score(y_test, predictions)),
    }


def train_models(X=None, y=None):
    """Train and compare models on the same test split."""
    if X is None or y is None:
        if X is not None or y is not None:
            raise ValueError("Provide both X and y, or neither.")
        X, y = load_housing_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    trained_models = {}
    results = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)

        trained_models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)

    comparison = pd.DataFrame.from_dict(
        results, orient="index"
    )
    comparison.index.name = "Model"

    return trained_models, comparison


def train_linear_regression():
    """Maintain compatibility with the original v0.1 baseline."""
    X, y = load_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    models, comparison = train_models()

    print("\nHousing Price Analyzer - Model Comparison")
    print("-----------------------------------------")
    print(comparison.round(4).to_string())

    print("\nLinear Regression Coefficients:")
    linear_model = models["Linear Regression"]
    for feature, coefficient in zip(
        linear_model.feature_names_in_,
        linear_model.coef_,
    ):
        print(f"{feature}: {coefficient:.4f}")

    print("\nRandom Forest Feature Importances:")
    forest_model = models["Random Forest"]
    for feature, importance in zip(
        forest_model.feature_names_in_,
        forest_model.feature_importances_,
    ):
        print(f"{feature}: {importance:.4f}")
