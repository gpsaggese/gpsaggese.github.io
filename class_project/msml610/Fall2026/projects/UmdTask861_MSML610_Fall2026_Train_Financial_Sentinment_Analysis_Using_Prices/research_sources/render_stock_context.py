"""Render an exportable figure from the public stock-context snapshot only."""

import argparse
from datetime import datetime, time
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def render(snapshot, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    data = json.loads(Path(snapshot).read_text())
    if data.get("schema_version") != "context-1.0":
        raise ValueError("Unsupported context snapshot")
    ny = ZoneInfo("America/New_York")
    stamp = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ny)
    cutoff = stamp(data["analysis_cutoff"])
    summary, community = data["summary"], data["community"]
    colors = {"SNDK": "#008b81", "SPY": "#7d5bc5", "QQQ": "#3074bd", "SOXX": "#c28719", "MU": "#c15c8b"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "axes.edgecolor": "#cdd5df", "axes.labelcolor": "#35465c",
                        "xtick.color": "#516076", "ytick.color": "#516076", "text.color": "#14273c"})
    fig = plt.figure(figsize=(14, 9), facecolor="#f5f8fb")
    grid = fig.add_gridspec(2, 2, left=.065, right=.96, bottom=.14, top=.75, wspace=.22, hspace=.46)
    fig.text(.065, .94, f"{data['symbol']} | Multi-source stock context", size=23, weight="bold")
    fig.text(.065, .901, f"Observed through {cutoff:%b %d, %Y at %I:%M %p %Z}  /  Retrospective research snapshot", color="#516076")
    value = lambda x, fmt: fmt.format(x) if x is not None else "Unavailable"
    metrics = [("Observed price", value(summary["last_price"], "${:,.2f}")),
               ("From previous close", value(summary["change_pct"], "{:+.2f}%")),
               ("Difference vs SPY", value(summary["excess_vs_spy_pp"], "{:+.2f} pp")),
               ("Prediction model", data["prediction_status"])]
    for index, (label, number) in enumerate(metrics):
        x = .065 + index * .235
        fig.text(x, .852, label, size=9, color="#516076")
        fig.text(x, .823, number, size=18, weight="bold")

    ax = fig.add_subplot(grid[0, 0], facecolor="white")
    for symbol, color in colors.items():
        rows = [p for p in data["intraday"] if p["symbol"] == symbol]
        if rows:
            ax.plot([stamp(p["timestamp"]) for p in rows], [p["return_pct"] for p in rows], label=symbol, color=color, lw=1.9)
    ax.axhline(0, color="#9aa8b6", lw=.8)
    ax.axvline(datetime.combine(cutoff.date(), time(9, 30), ny), color="#9aa8b6", lw=.8, ls="--")
    ax.set_title("Price paths vs previous close", loc="left", weight="bold", pad=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ny))
    ax.set_xlabel("New York time · completed 5-minute bars")
    ax.grid(axis="y", color="#e7edf2", lw=.7)
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="lower left")

    ax = fig.add_subplot(grid[0, 1], facecolor="white")
    peers = sorted([p for p in data["benchmarks"] if p["change_pct"] is not None], key=lambda p:p["change_pct"])
    ax.barh([p["symbol"] for p in peers], [p["change_pct"] for p in peers], color=["#008b81" if p["symbol"] == data["symbol"] else "#bccbdc" for p in peers], height=.65)
    ax.axvline(0, color="#9aa8b6", lw=.8)
    for i, row in enumerate(peers):
        ax.text(row["change_pct"] + .06, i, f"{row['change_pct']:+.2f}%", va="center", size=9)
    ax.set_xlim(min(p["change_pct"] for p in peers)-.4, max(1, max(p["change_pct"] for p in peers)+1))
    ax.set_title("Stock, sector and broad market", loc="left", weight="bold", pad=12)
    ax.set_xlabel("Observed change from previous close (%)")

    ax = fig.add_subplot(grid[1, 0], facecolor="white")
    stock = [p for p in data["intraday"] if p["symbol"] == data["symbol"]]
    ax.plot([stamp(p["timestamp"]) for p in stock], [p["price"] for p in stock], color="#008b81", lw=2)
    selected = [p for p in data["levels"] if p["id"] in ("previous_low", "rolling_low_20")]
    for level, color in zip(selected, ("#c28719", "#7d5bc5")):
        ax.axhline(level["price"], color=color, lw=1.2, ls="--", label=f"{level['label']}: ${level['price']:,.2f}")
    ax.set_title("Price and prior-session reference levels", loc="left", weight="bold", pad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ny))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("New York time · selected reference levels")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color="#e7edf2", lw=.7)

    ax = fig.add_subplot(grid[1, 1], facecolor="white")
    labels = ["Bullish", "Bearish", "Mixed", "Unknown"]
    counts = [community[name.lower()] for name in labels]
    ax.barh(labels, counts, color=["#008b81", "#ca6969", "#c28719", "#bbc4cf"], height=.6)
    for i, number in enumerate(counts): ax.text(number+.35, i, str(number), va="center", weight="bold")
    ax.set_xlim(0, max(counts+[1])+4)
    ax.invert_yaxis()
    ax.set_title("Futu post stance · keyword baseline", loc="left", weight="bold", pad=12)
    ax.set_xlabel(f"{community['unique_posts']} unique posts; {community['available_at_cutoff']} available at the price cutoff")
    fig.text(.065, .080, "Reference levels use completed prior-day data, reconstructed after the event; they are not guaranteed barriers.", size=9, color="#516076")
    received = stamp(community["received_at"]) if community["received_at"] else None
    fig.text(.065, .058, f"Futu sample received {received:%H:%M %Z}. Labels describe sampled text, not all investors or a return probability." if received else "Community receipt time unavailable.", size=9, color="#516076")
    fig.text(.065, .033, "Sources: Alpaca SIP price archive and Futu public sample. News evidence and source availability are in the interactive explorer.", size=8, color="#6d7b8c")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.snapshot, args.output)
