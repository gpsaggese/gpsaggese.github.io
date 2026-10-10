"""
Compute one-sided confidence bounds used by the checker verdict rule.

Import as:

import research.Noesis.checker_statistics as rnochsta
"""

import math
import statistics
from typing import Optional


def _validate_counts(successes: int, total: int) -> None:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise TypeError("successes and total must be integers")
    if total < 0:
        raise ValueError("total must not be negative")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")


def _validate_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")


def wilson_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    """
    Return the one-sided Wilson upper confidence bound.
    """
    _validate_counts(successes, total)
    _validate_confidence(confidence)
    if total == 0:
        raise ValueError("a confidence bound needs at least one observation")
    observed = successes / total
    z_score = statistics.NormalDist().inv_cdf(confidence)
    z_squared = z_score * z_score
    denominator = 1 + z_squared / total
    center = observed + z_squared / (2 * total)
    radius = z_score * math.sqrt(
        observed * (1 - observed) / total + z_squared / (4 * total * total)
    )
    return min(1.0, (center + radius) / denominator)


def _binomial_cdf(successes: int, total: int, probability: float) -> float:
    return sum(
        math.comb(total, value)
        * probability**value
        * (1 - probability) ** (total - value)
        for value in range(successes + 1)
    )


def clopper_pearson_upper(
    successes: int, total: int, confidence: float = 0.95
) -> float:
    """
    Return the exact one-sided Clopper-Pearson upper bound.
    """
    _validate_counts(successes, total)
    _validate_confidence(confidence)
    if total == 0:
        raise ValueError("a confidence bound needs at least one observation")
    if successes == total:
        return 1.0
    alpha = 1 - confidence
    low = successes / total
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2
        if _binomial_cdf(successes, total, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def upper_confidence_bound(
    successes: int,
    total: int,
    confidence: float = 0.95,
    exact_below: int = 10,
) -> Optional[float]:
    """
    Use an exact small-sample bound and Wilson otherwise.
    """
    _validate_counts(successes, total)
    _validate_confidence(confidence)
    if exact_below < 1:
        raise ValueError("exact_below must be positive")
    if total == 0:
        return None
    if total < exact_below:
        return clopper_pearson_upper(successes, total, confidence)
    return wilson_upper(successes, total, confidence)


def is_conclusive_failure(upper_bound: Optional[float], promised_rate: float) -> bool:
    """
    Return whether even the optimistic bound misses the promise.
    """
    if not 0 <= promised_rate <= 1:
        raise ValueError("promised_rate must be between zero and one")
    return upper_bound is not None and upper_bound < promised_rate
