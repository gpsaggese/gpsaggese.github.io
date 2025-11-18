#!/usr/bin/env python3
"""
Generate a static HTML training report from a structured run directory.

Usage:
  python scripts/generate_report.py --run_dir runs/structured/20250101_120000_run

Produces:
  report.html and PNG plots in the run directory
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric, out_png, title, phase_filter=None):
    if phase_filter is not None:
        data = df[df["phase"] == phase_filter]
    else:
        data = df
    if data.empty:
        return False
    x = data["global_step"].values
    y = data[metric].values
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("Global Step")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate training report from run directory")
    ap.add_argument("--run_dir", required=True, help="Path to structured run directory")
    args = ap.parse_args()

    metrics_csv = os.path.join(args.run_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"metrics.csv not found in {args.run_dir}")

    df = pd.read_csv(metrics_csv)

    # Ensure numeric types where appropriate
    for col in ["global_step", "epoch", "update", "lr", "loss", "perplexity", "accuracy", "throughput"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outputs = []
    plots = [
        ("loss", "Training Loss", "train"),
        ("perplexity", "Training Perplexity", "train"),
        ("accuracy", "Training Accuracy", "train"),
        ("lr", "Learning Rate", "train"),
        ("throughput", "Training Throughput (updates/s)", "train"),
        ("loss", "Validation Loss", "val"),
        ("perplexity", "Validation Perplexity", "val"),
    ]
    for metric, title, phase in plots:
        out_png = os.path.join(args.run_dir, f"plot_{phase}_{metric}.png")
        ok = plot_metric(df, metric, out_png, title, phase_filter=phase)
        if ok:
            outputs.append((title, os.path.basename(out_png)))

    # Simple HTML
    html_path = os.path.join(args.run_dir, "report.html")
    with open(html_path, "w") as f:
        f.write("<html><head><title>Training Report</title></head><body>\n")
        f.write(f"<h1>Training Report</h1>\n")
        f.write("<ul>\n")
        for title, png in outputs:
            f.write(f"  <li>{title}: <br/><img src=\"{png}\" style=\"max-width:800px\"/></li>\n")
        f.write("</ul>\n")
        f.write("</body></html>\n")

    print(f"Report written to: {html_path}")


if __name__ == "__main__":
    main()

