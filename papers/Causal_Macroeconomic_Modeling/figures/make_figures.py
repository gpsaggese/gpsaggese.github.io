"""
Generate the figures referenced from `paper.tex`.

Run with:
> python3 figures/make_figures.py

Add one block per figure below, following the pattern of Figure 1: build
the plot with matplotlib, save it under `figures/`, and reference it from
`paper.tex` with an `includegraphics` command on `figures/<name>.png`. Update
the Makefile's `$(OUT)` prerequisite list to match.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
    }
)

# #############################################################################
# Figure 1: <one-line description of what this figure shows and which
# section of paper.md references it>.
# #############################################################################

x = np.linspace(0, 10, 200)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(x, y, color="#1a5276", linewidth=1.8)
ax.set_xlabel("<x label>")
ax.set_ylabel("<y label>")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/example_figure.png", dpi=300)
plt.close(fig)

print("Wrote figures/example_figure.png")
