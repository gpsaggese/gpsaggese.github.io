"""
Exact false-alarm / detection probabilities for the Noesis violation rules.

Run: python3 scripts/detection_math.py
"""

from math import comb, sqrt

Z = 1.645  # one-sided 95%


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def flag_prob(n: int, p_true: float, promise: float, *, rule: str) -> float:
    """P(contract flagged) when the true per-request success rate is `p_true`."""
    total = 0.0
    for k in range(n + 1):
        lo, hi = wilson(k, n)
        flagged = hi < promise if rule == "upper" else lo < promise
        if flagged:
            total += comb(n, k) * p_true**k * (1 - p_true) ** (n - k)
    return total


if __name__ == "__main__":
    rows = [
        ("Healthy latency 0.97 vs 0.90, n=50", 50, 0.97, 0.90),
        ("Healthy latency 0.95 vs 0.90, n=50", 50, 0.95, 0.90),
        ("Slowed, latency 0.20 vs 0.90, n=50", 50, 0.20, 0.90),
        ("Healthy canaries 0.98 vs 0.85, n=10", 10, 0.98, 0.85),
        ("Healthy canaries 0.95 vs 0.85, n=10", 10, 0.95, 0.85),
        ("Weak model canaries 0.50 vs 0.85, n=10", 10, 0.50, 0.85),
        ("Weak model canaries 0.70 vs 0.85, n=10", 10, 0.70, 0.85),
        # v0.1 merged metric: 50 buyer requests (all pass) + 10 canaries at 50%.
        (
            "v0.1 merged: weak model, 60 req, p=0.917 vs 0.90",
            60,
            1 - (10 / 60) * 0.5,
            0.90,
        ),
    ]
    print(f"{'scenario':52s} {'paper(lower<R)':>15s} {'ours(upper<R)':>14s}")
    for name, n, p, r in rows:
        print(
            f"{name:52s} {flag_prob(n, p, r, rule='lower'):15.4%} "
            f"{flag_prob(n, p, r, rule='upper'):14.4%}"
        )
    rho = 0.8
    print("\nReputation (rho0=0.8, lambda=0.3), consecutive failures:")
    for i in range(4):
        rho *= 0.7
        print(f"  after {i + 1}: {rho:.3f}")
