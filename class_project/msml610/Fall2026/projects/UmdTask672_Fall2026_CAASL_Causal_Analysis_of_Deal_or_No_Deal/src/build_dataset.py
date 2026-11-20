"""Reconstruct ZBW prize boards and record show-level mismatches."""

from pathlib import Path

import numpy as np
import pandas as pd

from load_zbw import load_zbw


STANDARD_BOARD = [
    0.5, 1, 2, 5, 10, 25, 50, 75, 100, 150, 250, 500, 750,
    1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 25000,
    50000, 75000, 100000, 200000,
]
# Series 2.11 replaces $1,500 with a second $200,000 prize.
SERIES_2_11_BOARD = [
    0.5, 1, 2, 5, 10, 25, 50, 75, 100, 150, 250, 500, 750,
    1000, 2000, 3000, 5000, 7500, 10000, 15000, 25000,
    50000, 75000, 100000, 200000, 200000,
]
# Series 2.20 uses the prime-time board with a $2 million maximum.
SERIES_2_20_BOARD = [
    0.5, 1, 2, 5, 10, 25, 50, 75, 100, 200, 300, 500, 750,
    1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000,
    100000, 250000, 500000, 1000000, 2000000,
]
BOARDS = {
    2.00: STANDARD_BOARD, 2.10: STANDARD_BOARD,
    2.11: SERIES_2_11_BOARD, 2.20: SERIES_2_20_BOARD,
    3.00: STANDARD_BOARD, 3.10: STANDARD_BOARD,
}
CASE_COLUMNS = [f"case_{i}" for i in range(1, 7)]
DERIVED_COLUMNS = ["remaining_board", "n_remaining", "board_ev", "offer_ev_ratio"]
MISMATCH_COLUMNS = ["show_id", "series_id", "error"]


def _empty_reconstruction(df):
    columns = list(dict.fromkeys([*df.columns, *DERIVED_COLUMNS]))
    return df.iloc[:0].reindex(columns=columns).copy()


def _validate_rounds(df):
    required = {"show_id", "series_id", "round", "bank_offer", *CASE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing reconstruction columns: {sorted(missing)}")
    df = df.copy()
    if df["show_id"].isna().any():
        raise ValueError(f"{int(df['show_id'].isna().sum())} rows have missing show_id")
    for column in ["series_id", "round"]:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or values.isin([np.inf, -np.inf]).any():
            raise ValueError(f"{column} contains null, non-numeric, or nonfinite values")
        df[column] = values
    if df["round"].le(0).any() or df["round"].mod(1).ne(0).any():
        raise ValueError("round must contain positive integers")
    df["round"] = df["round"].astype("Int64")
    if df.duplicated(["show_id", "round"]).any():
        raise ValueError("Duplicate (show_id, round) pairs in input")
    return df


def reconstruct_show(show_df, starting_board):
    """Remove opened prizes and snapshot the board at each bank offer."""
    show_df = show_df.sort_values("round").copy()
    remaining = starting_board.copy()
    boards = []

    for _, row in show_df.iterrows():
        for column in CASE_COLUMNS:
            value = row[column]
            if pd.isna(value):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"show={row['show_id']} series={row['series_id']} "
                    f"round={row['round']}: invalid prize {value!r} in {column}"
                ) from error
            if value not in remaining:
                raise ValueError(
                    f"show={row['show_id']} series={row['series_id']} "
                    f"round={row['round']}: prize {value} not found "
                    f"on remaining board {remaining}"
                )
            remaining.remove(value)
        boards.append(remaining.copy())

    show_df["remaining_board"] = boards
    show_df["n_remaining"] = [len(board) for board in boards]
    show_df["board_ev"] = [
        sum(board) / len(board) if board else np.nan for board in boards
    ]
    show_df["offer_ev_ratio"] = show_df["bank_offer"] / show_df["board_ev"]
    return show_df


def build_dataset(df):
    """Return reconstructed shows and a separate record of failed shows."""
    if df.empty:
        return _empty_reconstruction(df), pd.DataFrame(columns=MISMATCH_COLUMNS)

    df = _validate_rounds(df)
    series_per_show = df.groupby("show_id")["series_id"].nunique()
    mixed = series_per_show[series_per_show > 1].index.tolist()
    if mixed:
        raise ValueError(f"Shows with mixed series_id: {mixed}")
    unknown = set(df["series_id"].unique()) - set(BOARDS)
    if unknown:
        raise ValueError(f"Unknown series_id values: {unknown}")

    games, mismatches = [], []
    for show_id, show_df in df.groupby("show_id"):
        series_id = show_df["series_id"].iloc[0]
        try:
            games.append(reconstruct_show(show_df, BOARDS[series_id]))
        except ValueError as error:
            mismatches.append({
                "show_id": show_id, "series_id": series_id, "error": str(error),
            })

    reconstructed = (
        pd.concat(games, ignore_index=True) if games else _empty_reconstruction(df)
    )
    return reconstructed, pd.DataFrame(mismatches, columns=MISMATCH_COLUMNS)


if __name__ == "__main__":
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_path = output_dir / "zbw_reconstructed.csv"
    mismatches_path = output_dir / "zbw_board_mismatches.csv"
    for path in [reconstructed_path, mismatches_path]:
        path.write_text("")

    raw = load_zbw()
    reconstructed, mismatches = build_dataset(raw)
    print(f"Raw rows: {len(raw)}; shows: {raw['show_id'].nunique()}")
    print(
        f"Reconstructed shows: {reconstructed['show_id'].nunique()}; "
        f"mismatches: {len(mismatches)}"
    )
    save_df = reconstructed.copy()
    save_df["remaining_board"] = save_df["remaining_board"].astype(str)
    save_df.to_csv(reconstructed_path, index=False)
    mismatches.to_csv(mismatches_path, index=False)
    print(f"Saved: {reconstructed_path}")
    if not mismatches.empty:
        print(mismatches.to_string(index=False))
    print(f"Saved: {mismatches_path}")
