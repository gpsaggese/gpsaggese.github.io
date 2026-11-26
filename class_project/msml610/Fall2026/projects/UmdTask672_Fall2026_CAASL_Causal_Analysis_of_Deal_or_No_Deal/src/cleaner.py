from load_zbw import load_zbw


INCLUDE_SERIES = [2.00, 3.00, 3.10]


def clean(df):
    """Filter eligible series and exclude entire shows with special rounds."""
    missing = {"show_id", "series_id", "special_show"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing cleaner columns: {sorted(missing)}")
    if df["show_id"].isna().any():
        raise ValueError(f"{int(df['show_id'].isna().sum())} rows have missing show_id")

    series_per_show = df.groupby("show_id")["series_id"].nunique()
    mixed = series_per_show[series_per_show > 1].index.tolist()
    if mixed:
        raise ValueError(
            f"Shows with inconsistent series_id across rounds: {mixed}. "
            f"Resolve in source data before cleaning."
        )

    df = df.loc[df["series_id"].isin(INCLUDE_SERIES)].copy()

    # Unknown flags count as clean; any known special round excludes the show.
    contaminated = set(
        df.loc[df["special_show"].isin([1, 2, 3]), "show_id"]
    )
    df = df.loc[~df["show_id"].isin(contaminated)].copy()

    return df


if __name__ == "__main__":
    df = load_zbw()
    clean_df = clean(df)

    print(f"Rows: {len(clean_df)}")
    print(f"Shows: {clean_df['show_id'].nunique()}")
    print("\nBreakdown by series_id:")
    print(
        clean_df.groupby("series_id").agg(
            rows=("show_id", "size"),
            shows=("show_id", "nunique"),
        ).to_string()
    )
