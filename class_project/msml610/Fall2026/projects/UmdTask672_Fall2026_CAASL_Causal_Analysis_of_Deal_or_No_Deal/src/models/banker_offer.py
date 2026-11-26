"""Reusable banker-offer regression models and safe pre-offer features."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

BASE_FEATURES = ["board_ev", "round"]
BOARD_FEATURES = [
    "board_ev", "round", "n_remaining", "board_min", "board_max",
    "prob_above_ev", "board_skew",
]


def model_specs(random_state=42):
    """Return candidate model specifications (fresh estimators each call)."""
    return {
        "Mean baseline": (BASE_FEATURES, make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean"))),
        "EV only": (["board_ev"], make_pipeline(SimpleImputer(strategy="median"), LinearRegression())),
        "EV + Round": (BASE_FEATURES, make_pipeline(SimpleImputer(strategy="median"), LinearRegression())),
        "Full linear": (BOARD_FEATURES, make_pipeline(SimpleImputer(strategy="median"), LinearRegression())),
        "Random forest": (BOARD_FEATURES, make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=1.0,
            random_state=random_state, n_jobs=-1,
        ))),
    }


def train_banker_model(train_df, estimator=None, features=None):
    """Fit a banker-offer model; default to the original two-feature regression."""
    features = BASE_FEATURES if features is None else list(features)
    estimator = estimator if estimator is not None else make_pipeline(
        SimpleImputer(strategy="median"), LinearRegression()
    )
    estimator.fit(train_df[features], train_df["bank_offer"])
    return estimator


def predict_banker_offers(model, df, features=None):
    """Predict offers using the supplied feature columns."""
    features = BASE_FEATURES if features is None else list(features)
    return model.predict(df[features])
