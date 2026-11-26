from pathlib import Path
import pandas as pd

ZBW_COLUMNS = [
    "year", "month", "day", "show_id", "series_id", "round",
    "gender", "special_show", "superamount", "contestant_guess",
    "contestant_precision", "bank_offer", "audience_suggestion",
    "decision", "case_1", "case_2", "case_3", "case_4", "case_5", "case_6",
]

def load_zbw(path="data/raw/zbw/deal-data.txt"):
    """Load the ZBW/de Roos & Sarafidis round-level dataset."""
    path = Path(path)
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            n_fields = len(line.split())
            if n_fields and n_fields != len(ZBW_COLUMNS):
                raise ValueError(
                    f"Invalid or incomplete record at line {line_number}: "
                    f"expected {len(ZBW_COLUMNS)} fields, got {n_fields}"
                )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ZBW_COLUMNS)
    df = df.replace(-999, pd.NA)

    rounds = pd.to_numeric(df["round"], errors="coerce")
    invalid_rounds = (
        rounds.isna() | rounds.isin([float("inf"), float("-inf")])
        | rounds.le(0) | rounds.mod(1).ne(0)
    )
    if invalid_rounds.any():
        raise ValueError(f"{int(invalid_rounds.sum())} rows have invalid round values")
    df["round"] = rounds.astype("Int64")

    if df["show_id"].isna().any():
        raise ValueError(f"{int(df['show_id'].isna().sum())} rows have missing show_id")

    series = pd.to_numeric(df["series_id"], errors="coerce")
    if series.isna().any() or series.isin([float("inf"), float("-inf")]).any():
        raise ValueError("series_id contains null, non-numeric, or nonfinite values")
    df["series_id"] = series

    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df

if __name__ == "__main__":
    df = load_zbw()
    print(df.head())
    print("\nShape:", df.shape)
    print("Shows:", df["show_id"].nunique())
    print("\nMissing values:")
    print(df.isna().sum())
