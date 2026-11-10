"""
Generate the data figures referenced from `paper.tex`.

All values are exact computations (no simulation and no live provider calls):

- the detection probabilities enumerate the binomial distribution and apply the
  verdict bound of `research/Noesis/checker/checker_statistics.py`;
- the calibration acceptance probabilities evaluate the closed-form
  expressions derived in Section "Calibration".

Run with:
> python3 figures/make_figures.py
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

import research.Noesis.checker.checker_statistics as rnochsta  # noqa: E402

_OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Palette shared with the diagrams (`.claude/skills/figure.rules.md`).
_RED = "#D64545"
_ORANGE = "#D98E2B"
_BLUE = "#3C6FB0"
_PURPLE = "#9B7DB1"
_GREEN = "#4F9A5C"
_GRAY = "#8C8C8C"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Verdict-rule parameters (`WindowPolicy` and the MVP defaults).
_Q_MIN = 0.85
_CONFIDENCE = 0.95
_EXACT_BELOW = 10


def flag_probability(n: int, accuracy: float, q_min: float = _Q_MIN) -> float:
    """
    Return P(conclusive quality failure) for `n` canaries at a true accuracy.

    :param n: number of canaries with usable evidence
    :param accuracy: true per-canary success probability of the seller
    :param q_min: promised canary accuracy
    :return: probability that the one-sided upper bound falls below `q_min`
    """
    total = 0.0
    for successes in range(n + 1):
        bound = rnochsta.upper_confidence_bound(
            successes, n, _CONFIDENCE, _EXACT_BELOW
        )
        if rnochsta.is_conclusive_failure(bound, q_min):
            total += (
                math.comb(n, successes)
                * accuracy**successes
                * (1 - accuracy) ** (n - successes)
            )
    return total


# #############################################################################
# Figure: detection power of the quality verdict.
# #############################################################################


def make_detection_power() -> None:
    """
    Plot the quality-verdict detection probability against `n`.
    """
    sizes = list(range(8, 41))
    curves = [
        (0.5, _RED, "-", "o"),
        (0.6, _ORANGE, "-", "s"),
        (0.7, _BLUE, "-", "^"),
        (0.8, _PURPLE, "-", "D"),
        (0.95, _GREEN, "--", None),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    for accuracy, color, style, marker in curves:
        values = [flag_probability(n, accuracy) for n in sizes]
        ax.plot(
            sizes,
            values,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=3,
            linewidth=1.4,
            label="a = %.2f" % accuracy,
        )
    ax.axhline(0.8, color=_GRAY, linestyle=":", linewidth=0.9)
    ax.text(40.3, 0.8, "0.8", va="center", fontsize=7, color=_GRAY)
    ax.axvline(10, color=_GRAY, linestyle=":", linewidth=0.9)
    ax.text(10.2, 1.02, "MVP default n = 10", fontsize=7, color=_GRAY)
    ax.set_xlabel("Canaries with usable evidence, n")
    ax.set_ylabel("P(quality verdict is FAILED)")
    ax.set_xlim(8, 40)
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT_DIR, "detection_power.png"), dpi=300)
    plt.close(fig)


# #############################################################################
# Figure: probability that a candidate canary survives calibration.
# #############################################################################


def healthy_acceptance(p: float, healthy_targets: int, repetitions: int = 3) -> float:
    """
    Return P(all attempts on all healthy targets are correct).

    Attempts are assumed independent.

    :param p: per-attempt accuracy of a healthy target
    :param healthy_targets: number of healthy provider/model targets
    :param repetitions: attempts per canary and target
    :return: acceptance probability under the healthy criterion
    """
    return p ** (repetitions * healthy_targets)


def weak_acceptance(w: float, repetitions: int = 3, maximum: float = 0.6) -> float:
    """
    Return P(weak-model accuracy <= `maximum`).

    Attempts are assumed independent with per-attempt accuracy `w`.

    :param w: per-attempt accuracy of the weak model
    :param repetitions: attempts per canary
    :param maximum: largest accepted weak-model accuracy
    :return: acceptance probability under the weak criterion
    """
    limit = math.floor(maximum * repetitions + 1e-9)
    return sum(
        math.comb(repetitions, k) * w**k * (1 - w) ** (repetitions - k)
        for k in range(limit + 1)
    )


def make_calibration_acceptance() -> None:
    """
    Plot the calibration acceptance probabilities.
    """
    grid = [i / 200 for i in range(0, 201)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for targets, color in ((1, _BLUE), (2, _GREEN), (3, _PURPLE)):
        left.plot(
            grid,
            [healthy_acceptance(p, targets) for p in grid],
            color=color,
            linewidth=1.4,
            label="%d healthy target%s" % (targets, "" if targets == 1 else "s"),
        )
    left.set_xlim(0.8, 1.0)
    left.set_ylim(0, 1.0)
    left.set_xlabel("Healthy per-attempt accuracy, p")
    left.set_ylabel("P(canary passes healthy test)")
    left.legend(loc="lower right", frameon=False)
    right.plot(
        grid,
        [weak_acceptance(w) for w in grid],
        color=_RED,
        linewidth=1.4,
    )
    right.axvline(0.6, color=_GRAY, linestyle=":", linewidth=0.9)
    right.text(0.61, 0.92, "w = 0.6", fontsize=7, color=_GRAY)
    right.set_xlim(0, 1.0)
    right.set_ylim(0, 1.0)
    right.set_xlabel("Weak-model per-attempt accuracy, w")
    right.set_ylabel("P(canary passes weak test)")
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT_DIR, "calibration_acceptance.png"), dpi=300)
    plt.close(fig)


def print_tables() -> None:
    """
    Print the numeric values quoted in the text.
    """
    print("P(FAILED), q_min = %.2f" % _Q_MIN)
    for n in (8, 10, 12, 16, 20, 30, 40):
        row = [flag_probability(n, a) for a in (0.95, 0.8, 0.7, 0.6, 0.5)]
        print(n, " ".join("%.4f" % value for value in row))
    print("Smallest n with P(FAILED) >= 0.8")
    for accuracy in (0.8, 0.75, 0.7, 0.6, 0.5, 0.4):
        for n in range(8, 400):
            if flag_probability(n, accuracy) >= 0.8:
                print(accuracy, n)
                break
    print("Weak acceptance (R = 3, maximum = 0.6)")
    for w in (0.2, 0.4, 0.5, 0.6, 0.8):
        print(w, "%.4f" % weak_acceptance(w))
    print("Healthy acceptance, p^(3 H)")
    for p in (0.99, 0.98, 0.95):
        print(p, " ".join("%.4f" % healthy_acceptance(p, h) for h in (1, 2, 3)))


if __name__ == "__main__":
    make_detection_power()
    make_calibration_acceptance()
    print_tables()
