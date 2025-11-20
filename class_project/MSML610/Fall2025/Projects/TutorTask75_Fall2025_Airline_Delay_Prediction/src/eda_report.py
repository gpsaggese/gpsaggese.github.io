#  Pseudocode of what my thought process is in terms of creating this file -> this my next thing now

# EDA for airline delay dataset.
# Loads parquet -> pandas, coerces types, derives simple time features,
# writes summary CSVs to reports/tables/ and figures to reports/figures/.
# --show true will display each plot inline (when run via %run in a notebook)
# and write an HTML gallery at reports/EDA_report.html.

import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/flights_with_weather.parquet"
OUT_DIR   = Path("reports")
FIG_DIR   = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
HTML_PATH = OUT_DIR / "EDA_report.html"

# toggled by --show
SHOW_INLINE = False

# theme knobs
PLOT_PALETTE = "RdPu"  # pinkish palette
def PINK_DIVERGING_CMAP():
    # pink ↔ teal diverging colormap centered at 0 (good for correlations)
    return sns.diverging_palette(330, 10, s=90, l=55, as_cmap=True)

def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path=DATA_PATH) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    return dataset.to_table().to_pandas()

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "DEPARTURE_DELAY","ARRIVAL_DELAY","DISTANCE","AIR_TIME",
        "temp","rhum","prcp","wspd","pres","dwpt","snow","wdir","wpgt","tsun"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "is_delayed" in df.columns:
        df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)

    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    if "dep_hour_rounded" in df.columns:
        df["dep_hour_rounded"] = pd.to_datetime(df["dep_hour_rounded"], errors="coerce")

    if "dep_hour_rounded" in df.columns:
        df["DEP_HOUR"]  = df["dep_hour_rounded"].dt.hour
        df["DEP_MONTH"] = df["dep_hour_rounded"].dt.month
        df["DEP_DOW"]   = df["dep_hour_rounded"].dt.dayofweek
    elif "FL_DATE" in df.columns:
        df["DEP_MONTH"] = df["FL_DATE"].dt.month
        df["DEP_DOW"]   = df["FL_DATE"].dt.dayofweek
    return df

def write_tables(df: pd.DataFrame):
    if "is_delayed" in df.columns:
        df["is_delayed"].value_counts(dropna=False).rename_axis("is_delayed") \
            .to_frame("count").to_csv(TABLE_DIR / "label_counts.csv")

    if {"AIRLINE_NAME","is_delayed"} <= set(df.columns):
        (df.groupby("AIRLINE_NAME", as_index=False)["is_delayed"]
           .agg(delay_rate="mean", n="count")
           .sort_values(["delay_rate","n"], ascending=[False, False])
        ).to_csv(TABLE_DIR / "airline_delay_rates.csv", index=False)

    if {"ORIGIN_AIRPORT","is_delayed"} <= set(df.columns):
        (df.groupby("ORIGIN_AIRPORT", as_index=False)["is_delayed"]
           .agg(delay_rate="mean", n="count")
           .sort_values(["delay_rate","n"], ascending=[False, False])
        ).to_csv(TABLE_DIR / "origin_delay_rates.csv", index=False)

    key_cols = [
        "DEPARTURE_DELAY","ARRIVAL_DELAY","DISTANCE","AIR_TIME",
        "temp","rhum","prcp","wspd","pres","station_id",
        "AIRLINE_NAME","ORIGIN_AIRPORT","DESTINATION_AIRPORT",
        "dep_hour_rounded","FL_DATE"
    ]
    have_cols = [c for c in key_cols if c in df.columns]
    df[have_cols].isna().mean().rename("null_rate").to_frame() \
        .to_csv(TABLE_DIR / "missingness_key_columns.csv")

def _style():
    sns.set_theme(style="whitegrid", palette=PLOT_PALETTE)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"

def inline_show():
    if SHOW_INLINE:
        try: plt.show()
        except Exception: pass

def plot_distribution_delay(df: pd.DataFrame):
    if "ARRIVAL_DELAY" not in df.columns: return
    _style()
    plt.figure(figsize=(8,4))
    sns.histplot(df["ARRIVAL_DELAY"].dropna(), bins=100, kde=True,
                 color=sns.color_palette(PLOT_PALETTE)[-1])
    plt.title("Distribution of Arrival Delay (minutes)")
    plt.xlabel("Arrival delay (min)"); plt.ylabel("Count")
    plt.axvline(15, ls="--", color="black")
    plt.savefig(FIG_DIR / "dist_arrival_delay.png"); inline_show(); plt.close()

def plot_label_balance(df: pd.DataFrame):
    if "is_delayed" not in df.columns: return
    _style()
    plt.figure(figsize=(5,4))
    sns.countplot(x="is_delayed", data=df, palette=PLOT_PALETTE)
    plt.title("Label distribution (0=on-time, 1=delayed)")
    plt.xlabel("is_delayed"); plt.ylabel("Count")
    plt.savefig(FIG_DIR / "label_distribution.png"); inline_show(); plt.close()

def plot_airline_mean_delay(df: pd.DataFrame, top=20):
    if {"AIRLINE_NAME","ARRIVAL_DELAY"} - set(df.columns): return
    _style()
    by_airline = (df.groupby("AIRLINE_NAME", as_index=False)["ARRIVAL_DELAY"].mean()
                    .sort_values("ARRIVAL_DELAY", ascending=False).head(top))
    plt.figure(figsize=(9, max(4, top*0.35)))
    sns.barplot(data=by_airline, x="ARRIVAL_DELAY", y="AIRLINE_NAME",
                palette=PLOT_PALETTE)
    plt.title("Average Arrival Delay by Airline")
    plt.xlabel("Mean arrival delay (min)"); plt.ylabel("Airline")
    plt.savefig(FIG_DIR / "airline_mean_arrival_delay.png"); inline_show(); plt.close()

def plot_origin_delay_rate(df: pd.DataFrame, min_count=5000):
    if {"ORIGIN_AIRPORT","is_delayed"} - set(df.columns): return
    _style()
    tmp = (df.groupby("ORIGIN_AIRPORT", as_index=False)
             .agg(delay_rate=("is_delayed","mean"), n=("is_delayed","count")))
    tmp = tmp[tmp["n"] >= min_count].sort_values("delay_rate", ascending=False).head(30)
    plt.figure(figsize=(9,6))
    sns.barplot(data=tmp, x="delay_rate", y="ORIGIN_AIRPORT", palette=PLOT_PALETTE)
    plt.title(f"Delay Rate by Origin Airport (n ≥ {min_count})")
    plt.xlabel("Delay rate"); plt.ylabel("Origin")
    plt.savefig(FIG_DIR / "origin_delay_rates.png"); inline_show(); plt.close()

def plot_hourly_pattern(df: pd.DataFrame):
    if {"DEP_HOUR","is_delayed"} - set(df.columns): return
    _style()
    by_hour = df.groupby("DEP_HOUR", as_index=False)["is_delayed"].mean()
    plt.figure(figsize=(7,4))
    sns.lineplot(data=by_hour, x="DEP_HOUR", y="is_delayed",
                 marker="o", color=sns.color_palette(PLOT_PALETTE)[-2])
    plt.title("Delay Rate by Scheduled Departure Hour")
    plt.xlabel("Hour of day"); plt.ylabel("Delay rate")
    plt.xticks(range(0,24,2))
    plt.savefig(FIG_DIR / "delay_rate_by_hour.png"); inline_show(); plt.close()

def plot_weather_bins(df: pd.DataFrame):
    if {"prcp","is_delayed"} - set(df.columns): return
    _style()
    bins = [-0.01, 0.0, 0.5, 2.0, 5.0, 10.0, df["prcp"].max()]
    labels = ["0", "0–0.5", "0.5–2", "2–5", "5–10", f">10"]
    tmp = df.copy()
    tmp["prcp_bin"] = pd.cut(tmp["prcp"].fillna(0), bins=bins, labels=labels, include_lowest=True)
    by_bin = tmp.groupby("prcp_bin", observed=False)["is_delayed"].mean().reset_index()
    plt.figure(figsize=(7,4))
    sns.barplot(data=by_bin, x="prcp_bin", y="is_delayed", palette=PLOT_PALETTE)
    plt.title("Delay Rate vs Precipitation (mm/hr)")
    plt.xlabel("Precip bin"); plt.ylabel("Delay rate")
    plt.savefig(FIG_DIR / "delay_vs_precip.png"); inline_show(); plt.close()

def plot_corr_heatmap(df: pd.DataFrame):
    cols = [c for c in ["DEPARTURE_DELAY","ARRIVAL_DELAY","DISTANCE","AIR_TIME",
                        "temp","rhum","prcp","wspd","pres","is_delayed"] if c in df.columns]
    if not cols: return
    _style()
    corr = df[cols].corr(numeric_only=True)

    # triangular (upper) heatmap, diagonal shown (like your example)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # keep lower triangle + diagonal
    cmap = PINK_DIVERGING_CMAP()

    plt.figure(figsize=(9,7))
    sns.heatmap(
        corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
        vmin=-1, vmax=1, square=True, linewidths=.5, cbar=True
    )
    plt.title("Correlation heatmap (upper triangle)")
    plt.savefig(FIG_DIR / "correlation_heatmap.png"); inline_show(); plt.close()

def build_html():
    fig_rows = [
        ("dist_arrival_delay.png", "Distribution of Arrival Delay (minutes)", None),
        ("label_distribution.png", "Label distribution (0=on-time, 1=delayed)", "label_counts.csv"),
        ("airline_mean_arrival_delay.png", "Average Arrival Delay by Airline", "airline_delay_rates.csv"),
        ("origin_delay_rates.png", "Delay Rate by Origin Airport (n ≥ 5k)", "origin_delay_rates.csv"),
        ("delay_rate_by_hour.png", "Delay Rate by Scheduled Departure Hour", None),
        ("delay_vs_precip.png", "Delay Rate vs Precipitation (mm/hr)", None),
        ("correlation_heatmap.png", "Correlation heatmap (upper triangle)", "missingness_key_columns.csv"),
    ]
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EDA Report</title>",
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:32px auto;padding:0 16px;} img{max-width:100%;height:auto;border:1px solid #eee;box-shadow:0 2px 8px rgba(0,0,0,.06);} h2{margin-top:28px} .cap{color:#444;margin:8px 0 18px} a{color:#0b5fff;text-decoration:none}</style>",
        "</head><body>",
        "<h1>EDA Report</h1>",
        "<p>This page bundles the figures generated by <code>eda_report.py</code> and links the summary tables.</p>",
        f"<p><strong>Figures dir:</strong> {FIG_DIR}</p>",
        f"<p><strong>Tables dir:</strong> {TABLE_DIR}</p>",
    ]
    for fname, caption, table in fig_rows:
        fpath = FIG_DIR / fname
        if fpath.exists():
            lines.append(f"<h2>{caption}</h2>")
            lines.append(f"<img src='{os.path.relpath(fpath, OUT_DIR)}' alt='{caption}'>")
            if table:
                tpath = TABLE_DIR / table
                if tpath.exists():
                    rel = os.path.relpath(tpath, OUT_DIR)
                    lines.append(f"<div class='cap'>Related table: <a href='{rel}'>{table}</a></div>")
            else:
                lines.append("<div class='cap'>&nbsp;</div>")
    lines.append("</body></html>")
    HTML_PATH.write_text("\n".join(lines), encoding="utf-8")
    return HTML_PATH

def try_inline_display(html_path: Path):
    try:
        from IPython.display import HTML, display  # type: ignore
        with open(html_path, "r", encoding="utf-8") as f:
            display(HTML(f.read()))
    except Exception:
        pass

def main(argv=None):
    global SHOW_INLINE
    parser = argparse.ArgumentParser(description="Generate EDA figures/tables; optional inline display/HTML.")
    parser.add_argument("--show", type=str, default="false",
                        help="true/false: display plots inline and write an HTML gallery")
    args = parser.parse_args(argv)
    SHOW_INLINE = str(args.show).strip().lower() in ("1","true","yes","y")

    ensure_dirs()
    df = load_data()
    df = coerce_types(df)
    write_tables(df)

    plot_distribution_delay(df)
    plot_label_balance(df)
    plot_airline_mean_delay(df, top=20)
    plot_origin_delay_rate(df, min_count=5000)
    plot_hourly_pattern(df)
    plot_weather_bins(df)
    plot_corr_heatmap(df)

    print("EDA wrote:")
    print(f"  figures → {FIG_DIR.resolve()}")
    print(f"  tables  → {TABLE_DIR.resolve()}")

    if SHOW_INLINE:
        html_path = build_html()
        print(f"Wrote HTML report: {html_path.resolve()}")
        try_inline_display(html_path)

if __name__ == "__main__":
    main()
