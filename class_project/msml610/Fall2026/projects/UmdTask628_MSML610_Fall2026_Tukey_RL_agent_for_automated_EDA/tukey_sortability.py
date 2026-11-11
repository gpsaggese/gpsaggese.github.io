"""
Sortability of simulated data: how far a simple score reveals the causal order.

A sortability compares a per-variable score along the true graph: 1 means
sorting by the score gives a causal order, 1/2 a random guess, 0 the reverse.
All functions come from CausalDisco 0.2.4 (BSD); this module fixes which
definition each number uses and always passes the binary adjacency:
- `paths`: every directed path counted with multiplicity (Reisach et al. 2021,
  2023), the definition of var- and R2-sortability in those papers
- `edges`: every edge counted once, ties 1/2 (Reisach et al. 2026, Eq. 4), the
  definition of relatives-sortability
- scores: marginal variance; R2 of each variable on all the others; the number
  of relatives `|desc(anc(v))|`, from the true graph (oracle) or from
  significant sample correlations at level 0.05 (Reisach et al. 2026,
  Algorithm 1)

CausalDisco's `edges` measure multiplies score differences by the adjacency
entries, so a weighted matrix with negative weights would flip those edges;
hence the binary adjacency. Scores closer than 1e-9 count as ties, so the equal
variances of standardized data give exactly 1/2 instead of a coin flip decided
by rounding.

Import as:

import tukey_sortability as tsortab
"""

import logging
from typing import Dict

import CausalDisco.analytics as cdanaly
import numpy as np

_LOG = logging.getLogger(__name__)

_MEASURES = {"paths": "paths", "edges": "adjacent_pairs"}
_TOL = 1e-9


def sortabilities(
    x: np.ndarray, B: np.ndarray, *, alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute var-, R2- and relatives-sortability under both definitions.

    :param x: data, shape (n_samples, n_nodes)
    :param B: true DAG (nonzero entries are edges)
    :param alpha: test level that defines a significant correlation
    :return: `<score>_<definition>` values; empty when the graph has no edge
    """
    B = (B != 0).astype(float)
    if B.sum() == 0:
        return {}
    values = {}
    for tag, measure in _MEASURES.items():
        values[f"var_{tag}"] = cdanaly.var_sortability(
            x, B, tol=_TOL, measure=measure
        )
        values[f"r2_{tag}"] = cdanaly.r2_sortability(
            x, B, tol=_TOL, measure=measure
        )
        values[f"rel_emp_{tag}"] = cdanaly.relatives_sortability(
            x, B, tol=_TOL, measure=measure, alpha=alpha
        )
        values[f"rel_oracle_{tag}"] = cdanaly.true_relatives_sortability(
            B, tol=_TOL, measure=measure
        )
    values = {key: float(value) for key, value in values.items()}
    return values
