"""Build offer-level features from reconstructed round-level game states."""

import ast

import numpy as np
import pandas as pd
from scipy.stats import skew


FEATURE_DTYPES = {
    "board_list": "object",
    "log_board_ev": "float64",
    "board_min": "float64",
    "board_max": "float64",
    "prob_above_ev": "float64",
    "board_skew": "float64",
    "board_ev_prev": "float64",
    "ev_change_prev": "float64",
    "normalized_luck": "float64",
    "offers_declined": "int64",
    "luck_undefined": "bool",
}


def _parse_board(value):
    try:
        values = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError("expected a list of prize amounts")
        prizes = np.asarray(values, dtype=float)
        if prizes.ndim != 1 or not np.isfinite(prizes).all() or (prizes < 0).any():
            raise ValueError("prizes must be finite, nonnegative numbers")
        return prizes.tolist()
    except (TypeError, ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid remaining_board value: {value!r}") from error


def build_features(df):
    """Build lag features before selecting active, nonterminal offer rounds."""
    if df.empty:
        return df.assign(**{
            column: pd.Series(dtype=dtype) for column, dtype in FEATURE_DTYPES.items()
        })
    required = {
        "show_id", "round", "remaining_board", "board_ev", "bank_offer",
        "decision", "offer_ev_ratio",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    df = df.copy()
    if df[["show_id", "round"]].isna().any().any():
        raise ValueError("show_id and round must not be missing")
    rounds = pd.to_numeric(df["round"], errors="coerce")
    if (rounds.isna() | rounds.isin([np.inf, -np.inf])
            | rounds.le(0) | rounds.mod(1).ne(0)).any():
        raise ValueError("round must contain positive, finite integers")
    df["round"] = rounds.astype("Int64")
    if df.duplicated(["show_id", "round"]).any():
        raise ValueError("Duplicate (show_id, round) pairs — lags will be incorrect")

    for column in ["board_ev", "bank_offer", "decision", "offer_ev_ratio"]:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} contains non-numeric values") from error
    if df["board_ev"].dropna().lt(0).any():
        raise ValueError("Negative board_ev values found — check reconstruction")
    if not np.isfinite(df["board_ev"].dropna()).all():
        raise ValueError("Nonfinite board_ev values found — check reconstruction")
    decisions = df["decision"].dropna()
    invalid_decisions = decisions.loc[~decisions.isin([0, 1])].unique()
    if len(invalid_decisions):
        raise ValueError(f"Invalid decision values: {invalid_decisions.tolist()}")
    if np.isinf(df["offer_ev_ratio"].dropna()).any():
        raise ValueError("Infinite offer_ev_ratio — check for zero board_ev")
    if not np.isfinite(df["bank_offer"].dropna()).all():
        raise ValueError("bank_offer contains nonfinite values")
    if df["bank_offer"].dropna().lt(0).any() or df["offer_ev_ratio"].dropna().lt(0).any():
        raise ValueError("bank_offer and offer_ev_ratio must be nonnegative")

    # Board features use every recorded round.
    df["board_list"] = df["remaining_board"].apply(_parse_board)
    sizes = df["board_list"].map(len)
    for index, values, ev in zip(df.index, df["board_list"], df["board_ev"]):
        if values and pd.notna(ev) and not np.isclose(ev, np.mean(values)):
            raise ValueError(f"board_ev disagrees with remaining_board at index {index}")
    if "n_remaining" in df:
        counts = pd.to_numeric(df["n_remaining"], errors="coerce")
        if counts.isna().any() or counts.ne(sizes).any():
            raise ValueError("n_remaining disagrees with remaining_board lengths")
    df["log_board_ev"] = np.log(df["board_ev"].clip(lower=1e-9))
    df["board_min"] = df["board_list"].apply(lambda v: min(v) if v else np.nan)
    df["board_max"] = df["board_list"].apply(lambda v: max(v) if v else np.nan)
    df["prob_above_ev"] = [
        sum(value > ev for value in values) / len(values)
        if values and pd.notna(ev) else np.nan
        for values, ev in zip(df["board_list"], df["board_ev"])
    ]
    df["board_skew"] = df["board_list"].apply(
        lambda v: skew(v) if len(v) >= 3 and np.ptp(v) > 0 else np.nan
    )

    mask = df["bank_offer"].notna() & df["board_ev"].notna() & df["board_ev"].ne(0)
    recomputed = df.loc[mask, "bank_offer"] / df.loc[mask, "board_ev"]
    n_bad = int((df.loc[mask, "offer_ev_ratio"].sub(recomputed).abs() > 1e-6).sum())
    if n_bad:
        raise ValueError(
            f"offer_ev_ratio disagrees with bank_offer/board_ev in {n_bad} rows — "
            f"CSV may be stale or manually edited."
        )

    # Compute luck before dropping non-offer rounds.
    df = df.sort_values(["show_id", "round"])
    df["board_ev_prev"] = df.groupby("show_id")["board_ev"].shift(1)
    df["ev_change_prev"] = df["board_ev"] - df["board_ev_prev"]
    denominator = df["board_ev_prev"].where(df["board_ev_prev"].ne(0))
    df["normalized_luck"] = df["ev_change_prev"] / denominator
    round_gap = df["round"] - df.groupby("show_id")["round"].shift(1)
    df.loc[round_gap.isna() | round_gap.ne(1), "normalized_luck"] = np.nan
    if np.isinf(df["normalized_luck"]).any():
        raise ValueError("Nonfinite normalized_luck — check board EV magnitudes")
    df["luck_undefined"] = df["normalized_luck"].isna()

    # Retain the accepted Deal and all no-Deal games; exclude terminal boards.
    first_deal_round = df.loc[df["decision"].eq(1)].groupby("show_id")["round"].min()
    deal_round = df["show_id"].map(first_deal_round)
    active = deal_round.isna() | df["round"].le(deal_round)
    nonterminal = df["board_list"].map(len).gt(0)
    offer_df = df.loc[active & nonterminal & df["bank_offer"].notna()].copy()
    # Count prior No Deals, excluding the current decision.
    offer_df["offers_declined"] = offer_df.groupby("show_id")["decision"].transform(
        lambda decisions: decisions.eq(0).shift(1, fill_value=False).cumsum()
    )

    if offer_df["decision"].isna().any():
        raise ValueError("Offer rows have missing decisions")
    if offer_df["board_ev"].isna().any():
        raise ValueError("Offer rows have missing board EV")
    if offer_df["board_ev"].le(0).any():
        raise ValueError("Offer rows require positive board_ev")
    if offer_df["offer_ev_ratio"].isna().any():
        raise ValueError("Offer rows have missing offer/EV ratios")
    return offer_df
