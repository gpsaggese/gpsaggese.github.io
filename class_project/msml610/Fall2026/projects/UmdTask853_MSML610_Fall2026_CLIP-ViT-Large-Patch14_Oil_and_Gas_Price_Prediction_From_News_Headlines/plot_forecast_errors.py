"""Draw held-out cumulative squared-error comparisons from benchmark predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, parse_dates=["date"])
    frame = frame.loc[frame["split"].eq(args.split)].copy()
    if frame.empty:
        raise ValueError(f"No {args.split} predictions found")
    _, ax = plt.subplots(figsize=(9, 4.5))
    for label, group in frame.groupby("model"):
        group = group.sort_values("date")
        ax.plot(group.date, (group.actual_return - group.predicted_return).pow(2).cumsum(), label=label)
    ax.set(xlabel="Forecast origin date", ylabel="Cumulative squared return error",
           title=f"{args.split.title()} cumulative forecasting error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
