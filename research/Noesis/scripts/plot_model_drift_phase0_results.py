#!/usr/bin/env python
"""Create paper-ready Phase 0 capability and Hamming-MMD figures."""

import argparse
import csv
import json
import math
import os
from typing import Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_WARMUP_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_controlled_warmup_live/"
    "grade_summary_v2.json"
)
DEFAULT_COMPARISON_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_method_comparison/"
    "comparison_summary.json"
)
DEFAULT_STABILITY_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_fingerprint_stability_live/"
    "stability_summary.json"
)
DEFAULT_CAPABILITY_SCENARIOS = (
    "research/Noesis/docs/findings/model_drift_capability/"
    "scenario_summary.csv"
)
DEFAULT_CUSUM_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_cusum_pilot/summary.json"
)
DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_method_comparison/figures"
)


BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#059669"
PURPLE = "#7C3AED"
INK = "#172033"
MUTED = "#5B6475"
GRID = "#D7DCE5"
PALE_BLUE = "#DBEAFE"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup_summary", default=DEFAULT_WARMUP_SUMMARY)
    parser.add_argument("--comparison_summary", default=DEFAULT_COMPARISON_SUMMARY)
    parser.add_argument("--stability_summary", default=DEFAULT_STABILITY_SUMMARY)
    parser.add_argument(
        "--capability_scenarios", default=DEFAULT_CAPABILITY_SCENARIOS
    )
    parser.add_argument("--cusum_summary", default=DEFAULT_CUSUM_SUMMARY)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    return parser.parse_args()


def _read_json(path: str) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as file_:
        value = json.load(file_)
    if not isinstance(value, dict):
        raise ValueError("expected an object in %s" % path)
    return value


def _read_csv(path: str) -> Sequence[Mapping[str, str]]:
    with open(path, encoding="utf-8", newline="") as file_:
        return list(csv.DictReader(file_))


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - radius, center + radius


def _style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=10)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, dst_dir: str, stem: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    figure.savefig(
        os.path.join(dst_dir, stem + ".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        os.path.join(dst_dir, stem + ".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_live_capability(summary: Mapping[str, object], dst_dir: str) -> None:
    route_order = (
        "cloudflare-llama-3.2-3b",
        "cloudflare-llama-3.1-8b",
        "cloudflare-llama-3.3-70b",
        "coreweave-llama-3.3-70b",
    )
    labels = ("Cloudflare\n3B", "Cloudflare\n8B", "Cloudflare\n70B", "CoreWeave\n70B")
    by_route = {row["route_name"]: row for row in summary["routes"]}
    rows = [by_route[name] for name in route_order]
    values = np.asarray([row["substantive_accuracy"] for row in rows])
    intervals = [
        _wilson_interval(row["substantive_correct"], row["graded"])
        for row in rows
    ]
    errors = np.asarray(
        [[value - low for value, (low, _) in zip(values, intervals)],
         [high - value for value, (_, high) in zip(values, intervals)]]
    )
    colors = [PURPLE, ORANGE, BLUE, GREEN]
    figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    positions = np.arange(len(rows))
    bars = axis.bar(positions, values * 100, width=0.62, color=colors)
    axis.errorbar(
        positions,
        values * 100,
        yerr=errors * 100,
        fmt="none",
        ecolor=INK,
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
    )
    for bar, value, row in zip(bars, values, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.3,
            "%s/%s\n%.1f%%" % (
                row["substantive_correct"],
                row["graded"],
                100 * value,
            ),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
            fontweight="semibold",
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 82)
    axis.set_yticks(np.arange(0, 81, 20))
    axis.set_ylabel("Substantive accuracy (%)", color=INK, fontsize=11)
    axis.set_title(
        "Live capability benchmark by served route",
        loc="left",
        fontsize=15,
        color=INK,
        fontweight="semibold",
        pad=15,
    )
    axis.text(
        0,
        1.01,
        "Bars show one completion per route and item; whiskers are 95% Wilson intervals.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    _style_axes(axis)
    _save(figure, dst_dir, "live-capability-by-route")


def _plot_method_comparison(summary: Mapping[str, object], dst_dir: str) -> None:
    rows = list(summary["scenario_comparison"])
    fraction_by_name = {
        "partial_substitution_10": 10,
        "partial_substitution_25": 25,
        "partial_substitution_50": 50,
        "full_swap": 100,
    }
    rows.sort(key=lambda row: fraction_by_name[row["scenario"]])
    fractions = np.asarray([fraction_by_name[row["scenario"]] for row in rows])
    capability = np.asarray([row["capability_detection_rate"] for row in rows]) * 100
    mmd = np.asarray([row["mmd_detection_rate"] for row in rows]) * 100
    figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    axis.plot(
        fractions,
        capability,
        color=BLUE,
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Capability detector",
    )
    axis.plot(
        fractions,
        mmd,
        color=ORANGE,
        marker="s",
        markersize=7,
        linewidth=2.4,
        label="Hamming-MMD",
    )
    for x_value, y_value in zip(fractions, capability):
        axis.annotate(
            "%.1f%%" % y_value,
            (x_value, y_value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=BLUE,
        )
    for x_value, y_value in zip(fractions, mmd):
        axis.annotate(
            "%.1f%%" % y_value,
            (x_value, y_value),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=ORANGE,
        )
    axis.set_xlim(5, 105)
    axis.set_ylim(-7, 112)
    axis.set_xticks(fractions, ["10%", "25%", "50%", "100%"])
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_xlabel("Substituted-response fraction", color=INK, fontsize=11)
    axis.set_ylabel("Detection rate (%)", color=INK, fontsize=11)
    axis.set_title(
        "Directional substitution sensitivity",
        loc="left",
        fontsize=15,
        color=INK,
        fontweight="semibold",
        pad=15,
    )
    axis.text(
        0,
        1.01,
        "Capability uses independent synthetic trials; MMD uses fixed live pools. Rates are not a head-to-head ranking.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    axis.legend(frameon=False, loc="upper left", fontsize=10)
    _style_axes(axis)
    _save(figure, dst_dir, "substitution-sensitivity-comparison")


def _plot_stability(summary: Mapping[str, object], dst_dir: str) -> None:
    rows = list(summary["comparisons"])
    rows.sort(key=lambda row: row["p_value"])
    labels = [
        "%s vs %s" % (
            row["left_pool"].replace("reference-", "Pool ").title(),
            row["right_pool"].replace("reference-", "Pool ").title(),
        )
        for row in rows
    ]
    values = np.asarray([row["p_value"] for row in rows])
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    axis.axvspan(0, 0.05, color="#FEE2E2", alpha=0.85, zorder=0)
    axis.axvline(0.05, color="#DC2626", linewidth=1.3, linestyle="--")
    axis.hlines(positions, 0, values, color=PALE_BLUE, linewidth=6, zorder=1)
    axis.scatter(values, positions, color=BLUE, s=52, zorder=2)
    for x_value, y_value in zip(values, positions):
        axis.text(
            x_value + 0.018,
            y_value,
            "%.3f" % x_value,
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    axis.text(
        0.055,
        -0.65,
        "α = 0.05",
        fontsize=9,
        color="#B91C1C",
        va="bottom",
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 0.9)
    axis.set_xticks(np.arange(0, 0.91, 0.2))
    axis.set_xlabel("Hamming-MMD permutation p-value", color=INK, fontsize=11)
    axis.set_title(
        "Five-pool 70B fingerprint stability",
        loc="left",
        fontsize=15,
        color=INK,
        fontweight="semibold",
        pad=15,
    )
    axis.text(
        0,
        1.01,
        "All 10 same-model comparisons remained above the alert threshold; Holm-adjusted alerts: 0/10.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    _style_axes(axis)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    axis.yaxis.grid(False)
    _save(figure, dst_dir, "mmd-stability-pvalues")


def _plot_synthetic_detection(
    rows: Sequence[Mapping[str, str]], dst_dir: str
) -> None:
    selected = (
        ("healthy_a_a", "Healthy A/A"),
        ("full_weak_model_swap", "Full weak-model swap"),
        ("small_immaterial_drop", "Small immaterial drop"),
        ("threshold_level_drop", "Threshold-level drop"),
        ("gradual_degradation", "Gradual degradation"),
        ("partial_substitution_10", "Partial substitution, 10%"),
        ("partial_substitution_25", "Partial substitution, 25%"),
        ("partial_substitution_50", "Partial substitution, 50%"),
    )
    by_name = {row["scenario_name"]: row for row in rows}
    values = np.asarray(
        [float(by_name[name]["detection_rate"]) * 100 for name, _ in selected]
    )
    labels = [label for _, label in selected]
    colors = [
        MUTED,
        BLUE,
        MUTED,
        ORANGE,
        BLUE,
        ORANGE,
        ORANGE,
        BLUE,
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    positions = np.arange(len(selected))
    bars = axis.barh(positions, values, height=0.66, color=colors)
    for bar, value in zip(bars, values):
        axis.text(
            max(value + 1.5, 1.5),
            bar.get_y() + bar.get_height() / 2,
            "%.1f%%" % value,
            ha="left",
            va="center",
            fontsize=8.5,
            color=INK,
            fontweight="semibold",
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 112)
    axis.set_xticks(np.arange(0, 101, 20))
    axis.set_xlabel("Detection rate (%)", color=INK, fontsize=11)
    axis.set_title(
        "Synthetic capability-detector sensitivity",
        loc="left",
        fontsize=15,
        color=INK,
        fontweight="semibold",
        pad=15,
    )
    axis.text(
        0,
        1.01,
        "Each scenario contains 1,000 trials; large and persistent losses were detected most reliably.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    _style_axes(axis)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    axis.yaxis.grid(False)
    _save(figure, dst_dir, "synthetic-capability-detection")


def _plot_cusum_delay(summary: Mapping[str, object], dst_dir: str) -> None:
    scenario_order = (
        ("full_weak_model_swap", "Full weak-model\nswap"),
        ("threshold_level_drop", "Threshold-level\ndrop"),
        ("gradual_degradation", "Gradual\ndegradation"),
        ("partial_substitution_25", "Partial\n25%"),
        ("partial_substitution_50", "Partial\n50%"),
    )
    by_name = {row["scenario_name"]: row for row in summary["scenarios"]}
    rows = [by_name[name] for name, _ in scenario_order]
    labels = [label for _, label in scenario_order]
    cusum = np.asarray([row["cusum_median_query"] for row in rows])
    fisher = np.asarray([row["fisher_median_query"] for row in rows])
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    cusum_bars = axis.bar(
        positions - width / 2,
        cusum,
        width=width,
        color=BLUE,
        label="CUSUM",
    )
    fisher_bars = axis.bar(
        positions + width / 2,
        fisher,
        width=width,
        color=ORANGE,
        label="Fisher",
    )
    for bars, values in ((cusum_bars, cusum), (fisher_bars, fisher)):
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 6,
                "%d" % value,
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
                fontweight="semibold",
            )
    for position, row in zip(positions, rows):
        rate = 100 * row["fisher_detection_rate"]
        if rate < 100:
            axis.text(
                position + width / 2,
                row["fisher_median_query"] + 28,
                "%.1f%% detected" % rate,
                ha="center",
                va="bottom",
                fontsize=8,
                color=ORANGE,
            )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 350)
    axis.set_yticks(np.arange(0, 351, 50))
    axis.set_ylabel("Median post-change grades to alarm", color=INK, fontsize=11)
    axis.set_title(
        "CUSUM detected changes with fewer grades",
        loc="left",
        fontsize=15,
        color=INK,
        fontweight="semibold",
        pad=15,
    )
    axis.text(
        0,
        1.01,
        "Medians are among detected trials; CUSUM was given the frozen healthy success probability.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    axis.legend(frameon=False, loc="upper left", fontsize=10, ncol=2)
    _style_axes(axis)
    _save(figure, dst_dir, "cusum-fisher-detection-delay")


def _main() -> None:
    args = _parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    )
    _plot_live_capability(_read_json(args.warmup_summary), args.dst_dir)
    _plot_method_comparison(_read_json(args.comparison_summary), args.dst_dir)
    _plot_stability(_read_json(args.stability_summary), args.dst_dir)
    _plot_synthetic_detection(
        _read_csv(args.capability_scenarios), args.dst_dir
    )
    _plot_cusum_delay(_read_json(args.cusum_summary), args.dst_dir)
    print("Figures: %s" % args.dst_dir)


if __name__ == "__main__":
    _main()
