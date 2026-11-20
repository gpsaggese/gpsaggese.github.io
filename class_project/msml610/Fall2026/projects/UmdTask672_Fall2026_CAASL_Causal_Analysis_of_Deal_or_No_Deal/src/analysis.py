"""Baseline association between prior luck and Deal/No Deal decisions."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.formula.api import logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning


# offers_declined == round - 1 in this sample; include round to avoid collinearity.
FEATURE_COLS = [
    "normalized_luck",
    "offer_ev_ratio",
    "log_board_ev",
    "n_remaining",
    "round",
]
FORMULA = "decision ~ " + " + ".join(FEATURE_COLS) + " + C(series_id)"


def _report_luck_effect(result):
    """Pair the log-odds coefficient with the probability-scale derivative."""
    effects = result.get_margeff(at="mean", method="dydx").summary_frame()
    luck_effect = effects.loc[["normalized_luck"]].copy()
    luck_effect.insert(0, "log_odds_coef", result.params["normalized_luck"])
    print("Normalized-luck coefficient and marginal effect at predictor means:")
    print(luck_effect.to_string())


def _fit_clustered(model_df):
    """Fit with game-clustered errors; reject unidentified or unreliable fits."""
    model = logit(FORMULA, data=model_df, missing="raise")
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError(
            "Logit design matrix is rank deficient. Check collinearity among "
            "round, n_remaining, and series indicators."
        )
    if model_df["show_id"].nunique() < 2:
        raise ValueError("Clustered standard errors require at least two shows")
    if model_df["decision"].nunique() != 2:
        raise ValueError("Logit requires both Deal and No Deal outcomes")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            result = model.fit(
                disp=False,
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": model_df["show_id"].to_numpy()},
            )
    except (PerfectSeparationError, PerfectSeparationWarning, np.linalg.LinAlgError) as error:
        raise RuntimeError(
            "Logit fit failed — check for separation or collinearity"
        ) from error
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError(
            "Logit did not converge — check for separation or collinearity"
        )
    if not np.isfinite(result.params).all() or not np.isfinite(result.bse).all():
        raise RuntimeError("Logit returned nonfinite estimates or standard errors")
    return result


def run_analysis(offer_df):
    """Report pooled and per-series associations, not identified causal effects."""
    required = {"show_id", "series_id", "decision", "luck_undefined", *FEATURE_COLS}
    missing = required.difference(offer_df.columns)
    if missing:
        raise ValueError(f"Missing analysis columns: {sorted(missing)}")
    if offer_df["luck_undefined"].isna().any() or not offer_df["luck_undefined"].isin(
        [True, False]
    ).all():
        raise ValueError("luck_undefined must contain nonmissing boolean flags")
    luck = pd.to_numeric(offer_df["normalized_luck"], errors="raise")
    if not offer_df["luck_undefined"].eq(luck.isna()).all():
        raise ValueError("luck_undefined disagrees with normalized_luck missingness")
    model_df = offer_df.loc[offer_df["luck_undefined"].eq(False)].copy()
    if model_df.empty:
        raise ValueError("No offer rows remain after excluding missing-luck rows")
    if model_df[["show_id", "series_id"]].isna().any().any():
        raise ValueError("Model rows require show_id and series_id")
    numeric_cols = ["decision", *FEATURE_COLS]
    for col in [*numeric_cols, "series_id"]:
        model_df[col] = pd.to_numeric(model_df[col], errors="raise").astype(float)
    if model_df.duplicated(["show_id", "round"]).any():
        raise ValueError("Duplicate (show_id, round) pairs in model rows")
    if not np.isfinite(model_df[[*numeric_cols, "series_id"]].to_numpy()).all():
        raise ValueError("Model outcome and predictors must be finite and nonmissing")
    if not model_df["decision"].isin([0, 1]).all():
        raise ValueError("decision must be 0 (No Deal) or 1 (Deal)")

    print(
        f"Input offer rows: {len(offer_df)}; shows: {offer_df['show_id'].nunique()}\n"
        f"Excluded undefined-luck rows: {len(offer_df) - len(model_df)}\n"
        f"Model rows: {len(model_df)}; shows: {model_df['show_id'].nunique()}"
    )
    print("\nModel sample by series:")
    print(model_df.groupby("series_id").agg(
        rows=("show_id", "size"), shows=("show_id", "nunique"),
    ).to_string())
    print(f"Deal rate: {model_df['decision'].mean():.3%}")
    print("\nNormalized luck distribution:")
    print(model_df["normalized_luck"].describe().to_string())

    correlation = model_df["normalized_luck"].corr(model_df["offer_ev_ratio"])
    print(f"\ncorr(normalized_luck, offer_ev_ratio): {correlation:.4f}")
    if not np.isfinite(correlation):
        print("Collinearity note: correlation is undefined; check constant predictors.")
    elif abs(correlation) > 0.5:
        print(
            "Collinearity note: absolute luck–offer-ratio correlation exceeds "
            "0.5; individual coefficients may be difficult to separate."
        )

    # Descriptive screening does not rule out separation.
    print("\nPredictor distributions by decision (separation screening):")
    print(model_df.groupby("decision")[FEATURE_COLS].describe().to_string())

    result = _fit_clustered(model_df)
    print("\nBaseline logit (standard errors clustered by show_id):")
    print(result.summary())
    _report_luck_effect(result)

    # Signs are hypotheses, not validation rules.
    if result.params["offer_ev_ratio"] <= 0:
        warnings.warn(
            "offer_ev_ratio is nonpositive; higher offers were expected to "
            "increase the probability of Deal (decision=1).",
            RuntimeWarning, stacklevel=2,
        )
    if result.params["normalized_luck"] > 0:
        warnings.warn(
            "normalized_luck is positive, opposite to the supplied hypothesis "
            "that negative luck increases the probability of Deal.",
            RuntimeWarning, stacklevel=2,
        )

    for series in [2.00, 3.00, 3.10]:
        sub = model_df.loc[model_df["series_id"].eq(series)].copy()
        if len(sub) < 50:
            print(f"Series {series:.2f}: too few rows ({len(sub)}), skip")
            continue
        try:
            sub_result = _fit_clustered(sub)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            print(f"Series {series:.2f}: fit failed: {error}")
            continue
        print(
            f"Series {series:.2f} — rows: {len(sub)}; "
            f"shows: {sub['show_id'].nunique()}"
        )
        _report_luck_effect(sub_result)

    return result
