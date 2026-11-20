"""Train, select, test, and persist banker-offer prediction models.

Run from project root: python -m src.pipelines.train_banker
"""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.models.banker_offer import model_specs, train_banker_model, predict_banker_offers

SEED = 42


def board_stats(value):
    """Derive board statistics from a serialized list of remaining prizes."""
    try:
        board = ast.literal_eval(value) if isinstance(value, str) else value
        values = np.asarray(board, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return [np.nan] * 5
        mean = values.mean()
        return [len(values), values.min(), values.max(),
                np.mean(values > mean), float(skew(values, bias=False)) if len(values) >= 3 and np.std(values) > 0 else 0.0]
    except (ValueError, SyntaxError, TypeError):
        return [np.nan] * 5


def load_data(root):
    preferred = root / "data/processed/zbw_features.csv"
    fallback = root / "data/processed/zbw_reconstructed.csv"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(f"Neither {preferred} nor {fallback} exists")
    df = pd.read_csv(path)
    print(f"Source: {path.relative_to(root)}")

    board_cols = ["n_remaining", "board_min", "board_max", "prob_above_ev", "board_skew"]
    missing = [col for col in board_cols if col not in df.columns]
    if missing:
        source = "board_list" if "board_list" in df.columns else "remaining_board" if "remaining_board" in df.columns else None
        if source is None:
            raise ValueError(f"Cannot derive missing board features {missing}: no board list column")
        derived = pd.DataFrame(df[source].map(board_stats).tolist(), columns=board_cols, index=df.index)
        for col in missing:
            df[col] = derived[col]

    required = ["show_id", "round", "board_ev", "bank_offer", "decision"] + board_cols
    absent = [col for col in required if col not in df.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    for col in ["round", "board_ev", "bank_offer", "decision"] + board_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Only genuine, observed offer decisions. No target imputation.
    df = df.loc[df["show_id"].notna() & df["round"].notna() &
                df["board_ev"].notna() & df["bank_offer"].notna() &
                df["decision"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No eligible offer decisions remain")
    if df["show_id"].nunique() < 10:
        raise ValueError("Not enough distinct shows for grouped train/validation/test splitting")
    return df


def grouped_split(df):
    # 60% train, 20% validation, 20% final test, grouped by show.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_i, test_i = next(outer.split(df, groups=df["show_id"]))
    dev, test = df.iloc[dev_i].copy(), df.iloc[test_i].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_i, val_i = next(inner.split(dev, groups=dev["show_id"]))
    train, val = dev.iloc[train_i].copy(), dev.iloc[val_i].copy()
    assert not (set(train.show_id) & set(val.show_id) or
                set(train.show_id) & set(test.show_id) or
                set(val.show_id) & set(test.show_id))
    return train, val, test


def scores(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)) if len(y) >= 2 else None}


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "outputs/banker_offer"
    output.mkdir(parents=True, exist_ok=True)
    df = load_data(root)
    train, val, test = grouped_split(df)
    print("\n=== DATA SPLITS ===")
    for label, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"{label}: {part.show_id.nunique()} shows, {len(part)} decisions")

    candidates = model_specs(SEED)
    comparison = []
    for name, (features, estimator) in candidates.items():
        fitted = train_banker_model(train, estimator, features)
        pred = predict_banker_offers(fitted, val, features)
        comparison.append({"Model": name, **scores(val.bank_offer, pred)})

    validation = pd.DataFrame(comparison).sort_values("MAE").reset_index(drop=True)
    validation.to_csv(output / "validation_comparison.csv", index=False)
    print("\n=== VALIDATION MODEL COMPARISON (LOWER MAE IS BETTER) ===")
    print(validation.round(3).to_string(index=False))

    winner = str(validation.iloc[0]["Model"])
    features, estimator = model_specs(SEED)[winner]
    dev = pd.concat([train, val], ignore_index=True)
    final_model = train_banker_model(dev, estimator, features)
    test_pred = predict_banker_offers(final_model, test, features)
    test_scores = scores(test.bank_offer, test_pred)
    print(f"\nSelected model: {winner}")
    print(f"Features: {features}")
    print("\n=== FINAL HELD-OUT TEST METRICS ===")
    for key, value in test_scores.items():
        print(f"{key}: {value:,.4f}" if value is not None else f"{key}: N/A")

    predictions = test[["show_id", "round", "board_ev", "bank_offer"]].copy()
    predictions["predicted_offer"] = test_pred
    predictions["absolute_error"] = abs(predictions.bank_offer - test_pred)
    predictions.to_csv(output / "test_predictions.csv", index=False)

    per_round = []
    for round_number, group in predictions.groupby("round", sort=True):
        result = scores(group.bank_offer, group.predicted_offer)
        per_round.append({"round": round_number, "observations": len(group), **result})
    round_df = pd.DataFrame(per_round)
    round_df.to_csv(output / "test_metrics_by_round.csv", index=False)
    print("\n=== TEST ERRORS BY ROUND ===")
    print(round_df.round(3).to_string(index=False))

    joblib.dump({"model": final_model, "features": features, "model_name": winner},
                output / "banker_offer_model.joblib")
    report = {"source_rows": len(df), "selected_model": winner,
              "selected_features": features, "train_shows": int(train.show_id.nunique()),
              "validation_shows": int(val.show_id.nunique()),
              "test_shows": int(test.show_id.nunique()), "test_metrics": test_scores,
              "selection_metric": "validation MAE", "random_seed": SEED}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved results and model to: {output}")


if __name__ == "__main__":
    main()
