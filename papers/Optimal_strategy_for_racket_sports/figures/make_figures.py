"""
Generate the four figures used in `paper.md`.

Figures 3 and 4 read the CSV files next to this script, which the
`racket_sports_05_variable_sensitivity` notebook writes with
`RUN_FULL_SIMS = True`.

Run with:
> python3 figures/make_figures.py
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
# Figure 1: 1D trajectory-feasibility schematic (Section III/IV).
# #############################################################################

g = 9.81
h0 = 1.0
theta_deg = 8.0
theta = np.radians(theta_deg)
v0 = 22.9
d_net = 12.0
h_net = 0.914
d_target = 20.0

t_land = d_target / (v0 * np.cos(theta))
t = np.linspace(0, t_land, 200)
x = v0 * np.cos(theta) * t
z = h0 + v0 * np.sin(theta) * t - 0.5 * g * t**2

fig, ax = plt.subplots(figsize=(3.4, 2.3))

# Nominal trajectory.
ax.plot(x, z, color="#1a5276", linewidth=1.8, label="Nominal trajectory")

# Perturbed trajectories (angle and speed error).
rng = np.random.default_rng(0)
spread_x = []
for dtheta_deg, dv in [(-3, -0.08), (-1.5, 0.05), (1.5, -0.05), (3, 0.08)]:
    th = np.radians(theta_deg + dtheta_deg)
    v = v0 * (1 + dv)
    # Solve landing time for this perturbed shot (z=0).
    a = -0.5 * g
    b = v * np.sin(th)
    c = h0
    disc = b**2 - 4 * a * c
    tl = (-b - np.sqrt(disc)) / (2 * a)
    tt = np.linspace(0, tl, 100)
    xx = v * np.cos(th) * tt
    zz = h0 + v * np.sin(th) * tt - 0.5 * g * tt**2
    ax.plot(xx, zz, color="#aeb6bf", linewidth=0.9, linestyle="--", zorder=1)
    spread_x.append(xx[-1])

# Net.
ax.plot([d_net, d_net], [0, h_net], color="#000000", linewidth=2.0)
ax.annotate(
    "Net",
    xy=(d_net, h_net),
    xytext=(d_net + 0.4, h_net + 0.55),
    fontsize=8,
)

# Landing-point spread (error footprint) near the target.
lo, hi = min(spread_x + [d_target]), max(spread_x + [d_target])
ax.plot([lo, hi], [-0.05, -0.05], color="#c0392b", linewidth=3.0, solid_capstyle="butt")
ax.annotate("Landing\nspread", xy=((lo + hi) / 2, -0.05), xytext=((lo + hi) / 2 - 3.6, -1.1), fontsize=8)

# Source and target markers.
ax.scatter([0], [h0], color="#000000", zorder=5, s=18)
ax.annotate("S", xy=(0, h0), xytext=(-0.9, h0 + 0.35), fontsize=9)
ax.scatter([d_target], [0], color="#000000", zorder=5, s=18)
ax.annotate("T", xy=(d_target, 0), xytext=(d_target + 0.3, 0.35), fontsize=9)

ax.axhline(0, color="#7f8c8d", linewidth=0.8)
ax.set_xlim(-1.5, 22)
ax.set_ylim(-1.4, 3.2)
ax.set_xlabel("Horizontal distance (m)")
ax.set_ylabel("Height (m)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/trajectory_schematic.png", dpi=300)
plt.close(fig)

# #############################################################################
# Figure 2: Toy-example grid scoring (Section VI), tennis vs. pickleball.
# #############################################################################

cells = np.array([0.3, 0.7, 1.1, 1.5, 1.9])
p_in = np.array([0.95, 0.91, 0.85, 0.77, 0.65])
labels = [f"$c_{i+1}$\n({d:.1f} m)" for i, d in enumerate(cells)]

scenarios = [
    ("Tennis groundstroke\n($T_f=0.8$ s)", 0.9, "#1a5276"),
    ("Pickleball net exchange\n($T_f=0.2$ s)", 0.0, "#a04000"),
]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
width = 0.35
x_pos = np.arange(len(cells))
handles = []

for ax, (title, reach_radius, color) in zip(axes, scenarios):
    reachable = cells <= reach_radius
    score = p_in * (1 - reachable.astype(float))

    bars1 = ax.bar(x_pos - width / 2, p_in, width, label=r"$P_{\mathrm{in}}$", color="#aeb6bf")
    bars2 = ax.bar(x_pos + width / 2, score, width, label=r"$S$", color=color)
    handles = [bars1, bars2]

    for i, r in enumerate(reachable):
        color_r = "#c0392b" if r else "#1e8449"
        ax.text(x_pos[i], p_in[i] + 0.05, f"R={int(r)}", ha="center", fontsize=7, color=color_r)

    best = int(np.argmax(score))
    ax.scatter([x_pos[best] + width / 2], [score[best] + 0.12], marker="*", color="black", s=60, zorder=5)

    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.3)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Probability / score")
fig.legend(handles, [r"$P_{\mathrm{in}}$", r"$S$"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("figures/toy_example_scores.png", dpi=300)
plt.close(fig)

print("Wrote figures/trajectory_schematic.png and figures/toy_example_scores.png")

# #############################################################################
# Figure 3: Which variables matter, one at a time (Section VII-A).
# #############################################################################

# The CSV files come from the `racket_sports_05_variable_sensitivity` notebook.
sweeps = np.genfromtxt(
    "figures/variable_sensitivity_sweeps.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
sports = [("Tennis", "#1a5276", "o"), ("Pickleball", "#a04000", "s")]
variables = [
    ("ball_speed", r"$v_b$ (m/s)"),
    ("v_p", r"$v_p$ (m/s)"),
    ("t_r", r"$t_r$ (s)"),
    ("std_x2", r"$\sigma_x$ (m)"),
    ("std_y2", r"$\sigma_y$ (m)"),
]

fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.5), sharey=True)
for ax, (variable, label) in zip(axes.flat, variables):
    for sport, color, marker in sports:
        rows = sweeps[(sweeps["setting"] == variable) & (sweeps["sport"] == sport)]
        rows = rows[np.argsort(rows["value"])]
        ax.plot(
            rows["value"],
            rows["max_p_winner"],
            color=color,
            marker=marker,
            markersize=2,
            linewidth=1.0,
            label=sport,
        )
    ax.set_xlabel(label, labelpad=1)
    ax.set_ylim(0.6, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel(r"Highest $P_{\mathrm{win}}$")
# The sixth panel holds the legend.
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[1, 2].axis("off")
axes[1, 2].legend(handles, labels, loc="center", frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig("figures/variable_sensitivity_sweeps.png", dpi=300)
plt.close(fig)

# #############################################################################
# Figure 4: Serve depth, slow ball and fast server (Section VII-A).
# #############################################################################

depths = np.genfromtxt(
    "figures/variable_sensitivity_serve_depth.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
for ax, (sport, color, _) in zip(axes, sports):
    rows = depths[depths["sport"] == sport]
    # The serve across the box: at the T, in the middle, and wide.
    x1_values = np.unique(rows["x1"])
    serves = [
        (x1_values[-1], "T", "-"),
        (x1_values[len(x1_values) // 2], "Middle", "--"),
        (x1_values[0], "Wide", ":"),
    ]
    for x1, label, linestyle in serves:
        line = rows[rows["x1"] == x1]
        line = line[np.argsort(line["y1"])]
        ax.plot(
            line["y1"],
            line["max_p_winner"],
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    ax.set_title(sport)
    ax.set_xlabel(r"Serve depth $y_1$ (m)")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
axes[0].set_ylabel(r"Highest $P_{\mathrm{win}}$")
fig.tight_layout(pad=0.6)
fig.savefig("figures/variable_sensitivity_serve_depth.png", dpi=300)
plt.close(fig)

print(
    "Wrote figures/variable_sensitivity_sweeps.png and"
    " figures/variable_sensitivity_serve_depth.png"
)
