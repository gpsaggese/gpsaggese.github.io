"""
Generate the figures referenced from `paper.md`.

Run with:
> python3 figures/make_figures.py

The numbers come from `run_benchmarks.py --summary` on the run of 2026-11-16
(`research/agentic_data_science/bakeoff/`).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
    }
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_SERIES = "#2a78d6"

# #############################################################################
# Figure 1: Claude Code accuracy per benchmark with 95% bootstrap confidence
# intervals (Results section).
# #############################################################################

# benchmark, tasks, accuracy, CI low, CI high
RESULTS = [
    ("DABench\n", 50, 0.78, 0.66, 0.88),
    ("QRData\n", 50, 0.70, 0.56, 0.82),
    ("DABstep\n(dev)", 10, 0.60, 0.30, 0.90),
    ("DA-Code\n(text)", 50, 0.52, 0.38, 0.66),
]

names = [f"{b}\nn={n}" for b, n, *_ in RESULTS]
acc = [r[2] for r in RESULTS]
err = [[r[2] - r[3] for r in RESULTS], [r[4] - r[2] for r in RESULTS]]

fig, ax = plt.subplots(figsize=(3.4, 2.4))
xs = range(len(RESULTS))
ax.bar(xs, acc, width=0.55, color=_SERIES, zorder=2)
ax.errorbar(xs, acc, yerr=err, fmt="none", ecolor=_INK, elinewidth=1, capsize=3, zorder=3)
for x, a in zip(xs, acc):
    ax.text(x + 0.31, a, f"{a:.0%}", va="center", ha="left", fontsize=8, color=_INK)
ax.set_xticks(list(xs))
ax.set_xticklabels(names)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy (95% CI)", color=_INK_2)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#b5b4ae")
ax.tick_params(colors=_INK_2)
fig.tight_layout()
fig.savefig(os.path.join(_HERE, "accuracy_by_benchmark.png"))
print("wrote figures/accuracy_by_benchmark.png")
