"""
Synthetic causal worlds with a known answer, and the Gym-style environment.

An instance is a hidden random DAG G*, data sampled from a structural equation
model on it, and a train/test split. The environment exposes D_train and D_test
to the experimenter (as the spec's Milestone 1 asks) but an agent only sees
`Evidence`: the free correlation matrix of D_train and the results of its own
tool calls. G* and the test rows only reach the scorer.

Families in this module:
- `proto`: the prototype's generator, bit-identical (exponential noise, raw
  scale, sequential split), to reproduce its numbers
- `linear_exp`, `linear_uniform`, `linear_gauss`: linear SEMs with one noise
  family; standardized with D_train statistics by default
- `linear_gauss_eqvar_raw`: Gaussian noise with equal variances on the raw
  scale, where the DAG is identifiable (Peters and Buhlmann 2014)
- `mixed`: one of the three noise families drawn per graph
- `mixed_node`: a noise family drawn per node (target: the ngDAG pattern of
  Hoyer et al. 2008 when Gaussian and non-Gaussian nodes mix)
- `nonlinear`: additive noise, one mechanism per graph (`tanh(3z)` or the
  bounded quadratic `clip(z^2, 0, 9)` of the standardized parent)
- `iscm`: linear SEM standardized inside the model (Ormaniec et al. 2025);
  with `GraphConfig.noise_share` set, every node with parents gets exactly that
  noise share, which standardization cannot undo (scaling every noise by the
  same factor only rescales the data)
- `pgmpy_lg`: pgmpy's random linear-Gaussian Bayesian network (the spec's
  named generator), kept as pgmpy draws it and filtered by rejection
- `sklearn_reg`: scikit-learn `make_regression`, informative features -> y
- `sklearn_confounder`: `make_regression` with two targets sharing one
  informative feature (an observed common cause)
- `hidden_confounder`: two targets with an unobserved common cause, so causal
  sufficiency fails and the target is a PAG (variants in
  `GraphConfig.confounder`)
- Topologies (`GraphConfig.topology`): Erdos-Renyi by edge count (default),
  scale-free hubs, chain, star and layered graphs

Contracts of the linear families (checked in `test/test_tukey_env.py`):
- every node's noise explains at least 10% of its variance: incoming weights
  of a node are scaled down when needed (`cap_signal_share()`), since raw
  variance otherwise grows with depth
- every true edge is visible once the child's other parents are controlled for:
  population `|pcorr(i, j | Pa(j) - {i})| >= 0.1` (`passes_edge_margin()`);
  weights are redrawn until it holds
- other conditioning sets are not filtered: in dense graphs some set of size 2
  almost always makes a true edge look absent (Uhler et al. 2013), so the
  smallest such partial correlation is reported as a difficulty covariate
  (`min_partial_corr()`)

Import as:

import tukey_env as tenv
"""

import dataclasses
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pgmpy.models as pgmodel
import sklearn.datasets as skdata

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Noise family of each single-family linear generator.
_FAMILY_NOISE = {
    "linear_exp": "exponential",
    "linear_uniform": "uniform",
    "linear_gauss": "gaussian",
    "linear_gauss_eqvar_raw": "gaussian",
}
_MIXED_NOISES = ("exponential", "uniform", "gaussian")
_NONLINEAR_MECHANISMS = ("tanh", "quadratic")
FAMILIES = (
    ("proto", "mixed")
    + tuple(_FAMILY_NOISE)
    + (
        "mixed_node",
        "nonlinear",
        "iscm",
        "pgmpy_lg",
        "sklearn_reg",
        "sklearn_confounder",
        "hidden_confounder",
    )
)
TOPOLOGIES = ("er", "scale_free", "chain", "star", "layered")

# Noise-share floor and edge-strength margin of the linear families.
_MIN_NOISE_SHARE = 0.1
_MIN_EDGE_PCORR = 0.1
_MAX_TRIES = 500
# Nonlinear family: per-edge signal share floor, linearity ceiling, pilot size.
_MIN_EDGE_SIGNAL_SHARE = 0.05
_MAX_LINEAR_R2 = 0.9
_N_PILOT = 20_000


# #############################################################################
# Configuration and instance
# #############################################################################


@dataclasses.dataclass(frozen=True)
class GraphConfig:
    """
    Settings of the instance generator.
    """

    # Number of variables.
    n_nodes: int = 6
    # Number of edges (capped at n_nodes * (n_nodes - 1) / 2).
    n_edges: int = 6
    # Generator family, one of `FAMILIES`.
    family: str = "mixed"
    # Noise scales: "random" draws U(0.5, 1.5) per node, "equal" uses 1.
    noise_scale: str = "random"
    # "post" standardizes with D_train statistics, "none" keeps the raw scale,
    # "rescale" multiplies each raw column by a random factor in [0.1, 10].
    standardize: str = "post"
    # Absolute edge weights are drawn uniformly from this range.
    w_range: Tuple[float, float] = (0.5, 2.0)
    # Number of rows, split into train and test.
    n_samples: int = 3000
    test_frac: float = 0.33
    # Graph shape, one of `TOPOLOGIES`.
    topology: str = "er"
    # `sklearn_confounder`: "observed" common cause; `hidden_confounder`:
    # "hidden_a", "hidden_b" or "hidden_c" (see `_sample_hidden_confounder()`).
    confounder: str = "none"
    # `iscm` only: the exact noise share (1 - R2 on the parents) of every node
    # with parents, and the levels at which a weight draw must pass the edge
    # margin (so one seed gives the same graph and draws at every level; see
    # `_sample_iscm_noise_share()`). None keeps the historical iSCM.
    noise_share: Optional[float] = None
    noise_grid: Tuple[float, ...] = ()


# The prototype's setting: 6 nodes, 6 edges, exponential noise, raw scale.
PROTO_CONFIG = GraphConfig(
    family="proto", noise_scale="equal", standardize="none"
)
# Gaussian noise with equal variances on the raw scale: the DAG is identifiable.
EQVAR_RAW_CONFIG = GraphConfig(
    family="linear_gauss_eqvar_raw", noise_scale="equal", standardize="none"
)


@dataclasses.dataclass(frozen=True)
class Instance:
    """
    One sampled world.

    Everything except `cfg`, `d_train` and `d_test` is private to the
    environment, the scorer and the reports.
    """

    cfg: GraphConfig
    seed: int
    d_train: pd.DataFrame
    d_test: pd.DataFrame
    truth: tscore.Truth
    # Weighted adjacency: `W[i, j]` is the effect of i on j.
    W: np.ndarray
    # Noise standard deviation and family of each node.
    noise_std: np.ndarray
    noise_families: Tuple[str, ...]
    # Why the target class was chosen, for the logs and the report.
    target_reason: str
    # Number of weight draws rejected by the edge-strength margin.
    n_rejected: int
    # Edge mechanism: "linear", or the nonlinear family's "tanh"/"quadratic".
    mechanism: str = "linear"


# #############################################################################
# Generators
# #############################################################################


def _topological_order(B: np.ndarray) -> List[int]:
    """
    Return a topological order of a DAG.

    :param B: DAG adjacency
    :return: nodes with every parent before its children
    """
    in_degree = B.sum(axis=0).astype(int)
    order: List[int] = []
    ready = sorted(np.flatnonzero(in_degree == 0).tolist())
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in np.flatnonzero(B[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(int(child))
    hdbg.dassert_eq(len(order), B.shape[0], "Graph has a cycle")
    return order


def _sample_er_dag(
    rng: np.random.Generator, n_nodes: int, n_edges: int
) -> np.ndarray:
    """
    Sample a DAG with exactly `n_edges` edges, as the prototype does.

    A random order is drawn and `n_edges` of the order-consistent pairs are
    chosen uniformly without replacement.

    :param rng: random generator
    :param n_nodes: number of nodes
    :param n_edges: number of edges
    :return: DAG adjacency
    """
    order = rng.permutation(n_nodes)
    pairs = [
        (order[i], order[j])
        for i, j in itertools.combinations(range(n_nodes), 2)
    ]
    n_edges = min(n_edges, len(pairs))
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    for idx in rng.choice(len(pairs), size=n_edges, replace=False):
        i, j = pairs[idx]
        B[i, j] = 1
    return B


def _sample_topology(
    rng: np.random.Generator, topology: str, n_nodes: int, n_edges: int
) -> np.ndarray:
    """
    Sample a DAG of a non-ER shape along a random order.

    :param rng: random generator
    :param topology: "scale_free", "chain", "star" or "layered"
    :param n_nodes: number of nodes
    :param n_edges: number of edges (`n_nodes - 1` for chain and star)
    :return: DAG adjacency
    """
    order = rng.permutation(n_nodes)
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    if topology in ("chain", "star"):
        hdbg.dassert_eq(n_edges, n_nodes - 1, "%s has d - 1 edges", topology)
        for k in range(1, n_nodes):
            parent = order[k - 1] if topology == "chain" else order[0]
            B[parent, order[k]] = 1
        return B
    if topology == "layered":
        n_layers = max(2, int(round(np.sqrt(n_nodes))))
        layer = np.array_split(order, n_layers)
        pairs = [
            (int(i), int(j))
            for upper, lower in zip(layer[:-1], layer[1:])
            for i in upper
            for j in lower
        ]
        probs = None
    elif topology == "scale_free":
        pairs = [
            (order[i], order[j])
            for i, j in itertools.combinations(range(n_nodes), 2)
        ]
        # Heavy-tailed parent weights make a few nodes hubs.
        hub_weight = rng.pareto(1.5, size=n_nodes) + 1.0
        probs = np.array([hub_weight[i] for i, _ in pairs])
        probs = probs / probs.sum()
    else:
        raise ValueError(f"Unknown topology '{topology}'")
    hdbg.dassert_lte(n_edges, len(pairs), "Too many edges for %s", topology)
    for idx in rng.choice(len(pairs), size=n_edges, replace=False, p=probs):
        i, j = pairs[idx]
        B[i, j] = 1
    return B


def _sample_dag(rng: np.random.Generator, cfg: GraphConfig) -> np.ndarray:
    """
    Sample the hidden DAG with the configured topology.

    :param rng: random generator
    :param cfg: generator settings
    :return: DAG adjacency
    """
    hdbg.dassert_in(cfg.topology, TOPOLOGIES)
    if cfg.topology == "er":
        B = _sample_er_dag(rng, cfg.n_nodes, cfg.n_edges)
    else:
        B = _sample_topology(rng, cfg.topology, cfg.n_nodes, cfg.n_edges)
    return B


def population_covariance(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute the covariance of a linear SEM `x = W.T x + e`.

    :param W: weighted adjacency (`W[i, j]` is the effect of i on j)
    :param noise_var: noise variance of each node
    :return: covariance matrix of x
    """
    mixing = np.linalg.inv(np.eye(W.shape[0]) - W.T)
    cov = mixing @ np.diag(noise_var) @ mixing.T
    return cov


def _partial_corr(corr: np.ndarray, i: int, j: int, cond: List[int]) -> float:
    """
    Compute a partial correlation from a correlation matrix.

    :param corr: correlation matrix
    :param i: first variable
    :param j: second variable
    :param cond: conditioning set
    :return: partial correlation of i and j given `cond`
    """
    idx = [i, j] + cond
    precision = np.linalg.inv(corr[np.ix_(idx, idx)])
    pcorr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    return float(pcorr)


def _population_corr(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix of a linear SEM.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: correlation matrix of x
    """
    cov = population_covariance(W, noise_var)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr


def cap_signal_share(
    W: np.ndarray,
    noise_var: np.ndarray,
    *,
    min_noise_share: float = _MIN_NOISE_SHARE,
) -> np.ndarray:
    """
    Scale down the incoming weights of nodes whose noise share is too small.

    Nodes are processed in topological order. When a node's noise would
    explain less than `min_noise_share` of its variance, all its incoming
    weights are multiplied by the factor that brings the share to the floor,
    so every node's R2 on its parents is at most `1 - min_noise_share`.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :param min_noise_share: floor on `noise_var / Var(x)`
    :return: capped copy of `W`
    """
    W = W.copy()
    n_nodes = W.shape[0]
    cov = np.zeros((n_nodes, n_nodes))
    max_ratio = (1 - min_noise_share) / min_noise_share
    for j in _topological_order((W != 0).astype(int)):
        signal_var = W[:, j] @ cov @ W[:, j]
        max_signal_var = noise_var[j] * max_ratio
        if signal_var > max_signal_var:
            W[:, j] *= np.sqrt(max_signal_var / signal_var)
            signal_var = W[:, j] @ cov @ W[:, j]
        # Parents come first, so their covariances are final.
        cov[:, j] = cov @ W[:, j]
        cov[j, :] = cov[:, j]
        cov[j, j] = signal_var + noise_var[j]
    return W


def edge_strengths(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute each true edge's partial correlation given the child's other parents.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: `|pcorr(i, j | Pa(j) - {i})|` per edge, in `np.argwhere(W)` order
    """
    corr = _population_corr(W, noise_var)
    strengths = []
    for i, j in np.argwhere(W != 0):
        others = [int(k) for k in np.flatnonzero(W[:, j]) if k != i]
        strengths.append(abs(_partial_corr(corr, i, j, others)))
    return np.array(strengths)


def passes_edge_margin(W: np.ndarray, noise_var: np.ndarray) -> bool:
    """
    Check that every true edge is visible given the child's other parents.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: True if every edge strength is at least 0.1
    """
    passes = bool(np.all(edge_strengths(W, noise_var) >= _MIN_EDGE_PCORR))
    return passes


def min_partial_corr(
    W: np.ndarray, noise_var: np.ndarray, *, max_cond: int = 2
) -> float:
    """
    Find the smallest partial correlation of a true edge over small sets.

    A value near 0 means some test with at most `max_cond` conditioning
    variables makes a true edge look absent (a near-unfaithful set).

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :param max_cond: largest conditioning set
    :return: `min |pcorr(i, j | S)|` over edges i -> j and `|S| <= max_cond`
    """
    corr = _population_corr(W, noise_var)
    n_nodes = W.shape[0]
    smallest = 1.0
    for i, j in np.argwhere(W != 0):
        others = [k for k in range(n_nodes) if k not in (i, j)]
        for size in range(max_cond + 1):
            for cond in itertools.combinations(others, size):
                pcorr = _partial_corr(corr, i, j, list(cond))
                smallest = min(smallest, abs(pcorr))
    return smallest


def _sample_noise(
    rng: np.random.Generator, family: str, n_samples: int, scale: float
) -> np.ndarray:
    """
    Sample centred noise with standard deviation `scale`.

    :param rng: random generator
    :param family: "exponential", "uniform" or "gaussian"
    :param n_samples: number of draws
    :param scale: standard deviation
    :return: noise vector
    """
    if family == "exponential":
        noise = rng.exponential(1.0, size=n_samples) - 1.0
    elif family == "uniform":
        noise = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n_samples)
    elif family == "gaussian":
        noise = rng.normal(size=n_samples)
    else:
        raise ValueError(f"Unknown noise family '{family}'")
    return scale * noise


def _simulate_linear(
    rng: np.random.Generator,
    W: np.ndarray,
    families: Tuple[str, ...],
    noise_std: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Sample rows of a linear SEM in topological order.

    :param rng: random generator
    :param W: weighted adjacency
    :param families: noise family per node
    :param noise_std: noise standard deviation per node
    :param n_samples: number of rows
    :return: data, shape (n_samples, n_nodes)
    """
    n_nodes = W.shape[0]
    x = np.zeros((n_samples, n_nodes))
    for j in _topological_order((W != 0).astype(int)):
        noise = _sample_noise(rng, families[j], n_samples, noise_std[j])
        x[:, j] = x @ W[:, j] + noise
    return x


def _target_for_linear(
    families: Tuple[str, ...],
    cfg: GraphConfig,
    *,
    equal_noise_kept: bool = True,
) -> Tuple[str, str]:
    """
    Decide what is identifiable for a linear SEM instance.

    :param families: noise family per node
    :param cfg: generator settings
    :param equal_noise_kept: whether `noise_scale="equal"` leaves the noise
        variances of the simulated SEM equal; False for iSCM, whose internal
        standardization divides each node's noise by its own standard
        deviation, so equal raw scales do not give equal effective noise
    :return: target class and the reason
    """
    gaussian = [f == "gaussian" for f in families]
    if not any(gaussian):
        target = (tscore.TARGET_DAG, "linear, every noise non-Gaussian (LiNGAM)")
    elif all(gaussian):
        equal_raw = (
            equal_noise_kept
            and cfg.noise_scale == "equal"
            and cfg.standardize == "none"
        )
        if equal_raw:
            target = (
                tscore.TARGET_DAG,
                "linear Gaussian, equal noise variances, raw scale",
            )
        else:
            target = (
                tscore.TARGET_CPDAG,
                "linear Gaussian: equivalence class only",
            )
    else:
        target = (tscore.TARGET_NGDAG, "mixed Gaussian and non-Gaussian noise")
    return target


def _split_and_scale(
    x: np.ndarray,
    cfg: GraphConfig,
    rng_split: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split rows into train and test, then standardize with train statistics.

    :param x: data
    :param cfg: generator settings
    :param rng_split: random generator of the split; None keeps the row order
        (the prototype's sequential split)
    :return: training rows and test rows
    """
    n_samples = x.shape[0]
    cut = int(n_samples * (1 - cfg.test_frac))
    if rng_split is None:
        rows = np.arange(n_samples)
    else:
        rows = rng_split.permutation(n_samples)
    x_train, x_test = x[rows[:cut]], x[rows[cut:]]
    if cfg.standardize == "post":
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    elif cfg.standardize == "rescale":
        # One log-uniform factor per column: the raw variances no longer
        # follow the causal order.
        hdbg.dassert_is_not(rng_split, None)
        factors = np.exp(
            rng_split.uniform(np.log(0.1), np.log(10.0), size=x.shape[1])
        )
        x_train, x_test = x_train * factors, x_test * factors
    else:
        hdbg.dassert_eq(cfg.standardize, "none")
    return x_train, x_test


def _sample_proto(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample an instance exactly as the prototype's `Environment` does.

    :param cfg: generator settings (family "proto", raw scale)
    :param seed: the prototype's seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    hdbg.dassert_eq(cfg.standardize, "none", "The prototype's data is raw")
    W = rlproto.random_dag(cfg.n_nodes, cfg.n_edges, seed, w_range=cfg.w_range)
    x = rlproto.simulate_lingam(W, cfg.n_samples, seed)
    x_train, x_test = _split_and_scale(x, cfg, None)
    sample = {
        "W": W,
        "x_train": x_train,
        "x_test": x_test,
        # Centred Exp(1) noise has standard deviation 1.
        "noise_std": np.ones(cfg.n_nodes),
        "noise_families": ("exponential",) * cfg.n_nodes,
        "target_class": tscore.TARGET_DAG,
        "target_reason": "prototype (LiNGAM)",
        "n_rejected": 0,
    }
    return sample


def _sample_linear(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample a linear-SEM instance that meets the family contracts.

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    if cfg.family == "linear_gauss_eqvar_raw":
        # Equal noise variances; the raw scale (EQVAR_RAW_CONFIG) identifies
        # the DAG, while "post" and "rescale" give the same instances rescaled
        # (every random draw happens before scaling), which only identify the
        # equivalence class.
        hdbg.dassert_eq(cfg.noise_scale, "equal", "Use EQVAR_RAW_CONFIG")
    # The same seed gives the same graph and weights in every linear family,
    # so families can be compared on paired instances.
    rng_graph, rng_weights, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    B = _sample_dag(rng_graph, cfg)
    if cfg.family == "mixed_node":
        draws = rng_graph.integers(len(_MIXED_NOISES), size=cfg.n_nodes)
        families = tuple(_MIXED_NOISES[k] for k in draws)
    else:
        if cfg.family == "mixed":
            family = _MIXED_NOISES[rng_graph.integers(len(_MIXED_NOISES))]
        else:
            family = _FAMILY_NOISE[cfg.family]
        families = (family,) * cfg.n_nodes
    # Draw weights and noise scales, cap the signal shares and redraw until
    # every edge passes the margin.
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        signs = rng_weights.choice([-1.0, 1.0], size=B.shape)
        W = B * signs * rng_weights.uniform(*cfg.w_range, size=B.shape)
        if cfg.noise_scale == "equal":
            noise_std = np.ones(cfg.n_nodes)
        else:
            noise_std = rng_weights.uniform(0.5, 1.5, size=cfg.n_nodes)
        W = cap_signal_share(W, noise_std**2)
        if passes_edge_margin(W, noise_std**2):
            break
        n_rejected += 1
    else:
        raise RuntimeError(f"No weights passed the margin in {_MAX_TRIES} tries")
    x = _simulate_linear(rng_noise, W, families, noise_std, cfg.n_samples)
    # Permute the columns so that column order cannot leak the causal order.
    perm = rng_perm.permutation(cfg.n_nodes)
    x = x[:, perm]
    x_train, x_test = _split_and_scale(x, cfg, rng_split)
    target_class, target_reason = _target_for_linear(families, cfg)
    sample = {
        "W": W[np.ix_(perm, perm)],
        "x_train": x_train,
        "x_test": x_test,
        "noise_std": noise_std[perm],
        "noise_families": tuple(families[k] for k in perm),
        "target_class": target_class,
        "target_reason": target_reason,
        "n_rejected": n_rejected,
    }
    return sample


def _permuted_sample(
    x: np.ndarray,
    W: np.ndarray,
    noise_std: np.ndarray,
    families: Tuple[str, ...],
    cfg: GraphConfig,
    rng_perm: np.random.Generator,
    rng_split: np.random.Generator,
) -> Dict[str, Any]:
    """
    Permute the columns, split the rows and package an instance's fields.

    :param x: data in generation order
    :param W: weighted adjacency in generation order
    :param noise_std: noise standard deviation per node
    :param families: noise family per node
    :param cfg: generator settings
    :param rng_perm: random generator of the column permutation
    :param rng_split: random generator of the row split
    :return: fields of `Instance` (without the target)
    """
    perm = rng_perm.permutation(cfg.n_nodes)
    x_train, x_test = _split_and_scale(x[:, perm], cfg, rng_split)
    sample = {
        "W": W[np.ix_(perm, perm)],
        "x_train": x_train,
        "x_test": x_test,
        "noise_std": noise_std[perm],
        "noise_families": tuple(families[k] for k in perm),
        # Column c of the data is generation column `perm[c]`.
        "perm": perm,
    }
    return sample


def _mechanism(name: str, z: np.ndarray) -> np.ndarray:
    """
    Apply a nonlinear mechanism to a standardized parent.

    :param name: "tanh" or "quadratic"
    :param z: standardized parent values
    :return: `tanh(3 z)` or `clip(z^2, 0, 9)`
    """
    if name == "tanh":
        return np.tanh(3.0 * z)
    hdbg.dassert_eq(name, "quadratic")
    return np.clip(z**2, 0.0, 9.0)


def _simulate_nonlinear(
    rng: np.random.Generator,
    W: np.ndarray,
    families: Tuple[str, ...],
    noise_std: np.ndarray,
    mechanism: str,
    center: np.ndarray,
    scale: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Sample `x_j = sum_i W_ij f((x_i - center_i) / scale_i) + e_j`.

    :param rng: random generator
    :param W: weighted adjacency
    :param families: noise family per node
    :param noise_std: noise standard deviation per node
    :param mechanism: nonlinear mechanism
    :param center: standardization constants of each node (from a pilot)
    :param scale: standardization constants of each node (from a pilot)
    :param n_samples: number of rows
    :return: data
    """
    n_nodes = W.shape[0]
    x = np.zeros((n_samples, n_nodes))
    for j in _topological_order((W != 0).astype(int)):
        noise = _sample_noise(rng, families[j], n_samples, noise_std[j])
        signal = np.zeros(n_samples)
        for i in np.flatnonzero(W[:, j]):
            z = (x[:, i] - center[i]) / scale[i]
            signal += W[i, j] * _mechanism(mechanism, z)
        x[:, j] = signal + noise
    return x


def _calibrate_nonlinear(
    rng: np.random.Generator,
    W: np.ndarray,
    families: Tuple[str, ...],
    noise_std: np.ndarray,
    mechanism: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Fix the standardization constants on a pilot sample and check each edge.

    Nodes are processed in topological order on a 20,000-row pilot: a node's
    incoming weights are scaled down if its noise would explain less than 10%
    of its variance; each edge must carry at least 5% of the child's variance;
    and with `tanh` each edge's mechanism must be visibly nonlinear (linear
    R^2 of `f(z)` on `z` at most 0.9).

    :param rng: random generator of the pilot
    :param W: weighted adjacency
    :param families: noise family per node
    :param noise_std: noise standard deviation per node
    :param mechanism: nonlinear mechanism
    :return: capped weights, centers, scales, and whether every edge passes
    """
    W = W.copy()
    n_nodes = W.shape[0]
    x = np.zeros((_N_PILOT, n_nodes))
    center = np.zeros(n_nodes)
    scale = np.ones(n_nodes)
    passes = True
    for j in _topological_order((W != 0).astype(int)):
        noise = _sample_noise(rng, families[j], _N_PILOT, noise_std[j])
        parents = np.flatnonzero(W[:, j])
        terms = {}
        for i in parents:
            z = (x[:, i] - center[i]) / scale[i]
            f_z = _mechanism(mechanism, z)
            if mechanism == "tanh":
                passes &= np.corrcoef(z, f_z)[0, 1] ** 2 <= _MAX_LINEAR_R2
            terms[i] = f_z
        signal = (
            sum(W[i, j] * terms[i] for i in parents) if len(parents) else 0.0
        )
        signal_var = float(np.var(signal)) if len(parents) else 0.0
        max_signal_var = (
            noise_std[j] ** 2 * (1 - _MIN_NOISE_SHARE) / _MIN_NOISE_SHARE
        )
        if signal_var > max_signal_var:
            W[:, j] *= np.sqrt(max_signal_var / signal_var)
            signal = sum(W[i, j] * terms[i] for i in parents)
        x[:, j] = signal + noise
        center[j] = x[:, j].mean()
        scale[j] = x[:, j].std()
        for i in parents:
            share = np.var(W[i, j] * terms[i]) / scale[j] ** 2
            passes &= share >= _MIN_EDGE_SIGNAL_SHARE
    return W, center, scale, bool(passes)


def _random_weights(
    rng: np.random.Generator, B: np.ndarray, cfg: GraphConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw signed edge weights and noise scales as the linear families do.

    :param rng: random generator
    :param B: DAG adjacency
    :param cfg: generator settings
    :return: weights and noise standard deviations
    """
    signs = rng.choice([-1.0, 1.0], size=B.shape)
    W = B * signs * rng.uniform(*cfg.w_range, size=B.shape)
    if cfg.noise_scale == "equal":
        noise_std = np.ones(cfg.n_nodes)
    else:
        noise_std = rng.uniform(0.5, 1.5, size=cfg.n_nodes)
    return W, noise_std


def _sample_nonlinear(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample an additive-noise model with one nonlinear mechanism per graph.

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    rng_graph, rng_weights, rng_noise, rng_split, rng_perm, rng_pilot = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)
    ]
    B = _sample_dag(rng_graph, cfg)
    family = _MIXED_NOISES[rng_graph.integers(len(_MIXED_NOISES))]
    mechanism = _NONLINEAR_MECHANISMS[rng_graph.integers(2)]
    families = (family,) * cfg.n_nodes
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        W, noise_std = _random_weights(rng_weights, B, cfg)
        W, center, scale, passes = _calibrate_nonlinear(
            rng_pilot, W, families, noise_std, mechanism
        )
        if passes:
            break
        n_rejected += 1
    else:
        raise RuntimeError(f"No nonlinear weights passed in {_MAX_TRIES} tries")
    x = _simulate_nonlinear(
        rng_noise,
        W,
        families,
        noise_std,
        mechanism,
        center,
        scale,
        cfg.n_samples,
    )
    sample = _permuted_sample(
        x, W, noise_std, families, cfg, rng_perm, rng_split
    )
    sample["target_class"] = tscore.TARGET_DAG
    sample["target_reason"] = "nonlinear additive noise (Hoyer et al. 2009)"
    sample["n_rejected"] = n_rejected
    sample["mechanism"] = mechanism
    return sample


def _noise_share_weights(
    W: np.ndarray, B: np.ndarray, noise_share: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Scale each node's incoming weights to give it an exact noise share.

    Nodes are processed in topological order on unit-variance parents: a node
    j with parents gets weights `w * sqrt((1 - q) / v)`, where `v = w' C w` is
    the variance its raw weights give on the standardized parents (`C` their
    correlation), and noise standard deviation `sqrt(q)`; a root gets unit
    noise. Every node then has unit variance and `R2 = 1 - q` on its parents.

    :param W: raw weighted adjacency (only the ratios of a node's incoming
        weights matter)
    :param B: DAG adjacency
    :param noise_share: q, the noise share of every node with parents
    :return: effective weights and noise standard deviations, or None when a
        node's raw signal variance is degenerate
    """
    n_nodes = B.shape[0]
    W_eff = np.zeros_like(W)
    noise_eff = np.ones(n_nodes)
    corr = np.zeros((n_nodes, n_nodes))
    for j in _topological_order(B):
        if B[:, j].any():
            weights = W[:, j]
            v = float(weights @ corr @ weights)
            if not np.isfinite(v) or v <= 1e-12:
                return None
            W_eff[:, j] = weights * np.sqrt((1 - noise_share) / v)
            noise_eff[j] = np.sqrt(noise_share)
        corr[:, j] = corr @ W_eff[:, j]
        corr[j, :] = corr[:, j]
        corr[j, j] = 1.0
    return W_eff, noise_eff


def _sample_iscm_noise_share(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample an iSCM instance whose nodes have an exact noise share.

    The draws are those of `_sample_iscm()` (same streams, graph, noise family,
    raw weights and raw noise), so the instances of one seed at different
    levels differ only in their noise share. A weight draw is accepted only if
    it passes the edge margin at every level of `cfg.noise_grid`, so the
    accepted draw does not depend on the level.

    :param cfg: generator settings with `noise_share` in `noise_grid`
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    hdbg.dassert_in(cfg.noise_share, cfg.noise_grid)
    for level in cfg.noise_grid:
        # Above the 10% floor, so the cap of the other families never binds.
        hdbg.dassert_lt(_MIN_NOISE_SHARE, level)
        hdbg.dassert_lt(level, 1.0)
    rng_graph, rng_weights, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    B = _sample_dag(rng_graph, cfg)
    family = _MIXED_NOISES[rng_graph.integers(len(_MIXED_NOISES))]
    families = (family,) * cfg.n_nodes
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        # The noise scales are drawn (as in `_sample_iscm()`) but unused.
        W, _ = _random_weights(rng_weights, B, cfg)
        scaled = {q: _noise_share_weights(W, B, q) for q in cfg.noise_grid}
        if all(
            v is not None and passes_edge_margin(v[0], v[1] ** 2)
            for v in scaled.values()
        ):
            break
        n_rejected += 1
    else:
        raise RuntimeError(
            f"No noise-share weights passed in {_MAX_TRIES} tries"
        )
    W_eff, noise_eff = scaled[cfg.noise_share]
    x = _simulate_linear(rng_noise, W_eff, families, noise_eff, cfg.n_samples)
    sample = _permuted_sample(
        x, W_eff, noise_eff, families, cfg, rng_perm, rng_split
    )
    target_class, target_reason = _target_for_linear(
        families, cfg, equal_noise_kept=False
    )
    sample["target_class"] = target_class
    sample["target_reason"] = (
        f"iSCM, noise share {cfg.noise_share}: " + target_reason
    )
    sample["n_rejected"] = n_rejected
    return sample


def _sample_iscm(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample a linear SEM standardized inside the model (iSCM).

    Each variable is divided by its exact population standard deviation as
    soon as it is generated, so every variable has unit variance and weights
    act on standardized parents (Ormaniec et al., ICLR 2025). The result is a
    linear SEM with effective weights `W_ij / sd_j` and noise `e_j / sd_j`.

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    if cfg.noise_share is not None:
        return _sample_iscm_noise_share(cfg, seed)
    rng_graph, rng_weights, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    B = _sample_dag(rng_graph, cfg)
    family = _MIXED_NOISES[rng_graph.integers(len(_MIXED_NOISES))]
    families = (family,) * cfg.n_nodes
    n_rejected = 0
    max_ratio = (1 - _MIN_NOISE_SHARE) / _MIN_NOISE_SHARE
    for _ in range(_MAX_TRIES):
        W, noise_std = _random_weights(rng_weights, B, cfg)
        # Standardize in topological order (parents already have unit
        # variance), capping each node's signal share on the standardized
        # parents so that its noise explains at least 10% of its variance.
        W_eff = np.zeros_like(W)
        noise_eff = np.zeros(cfg.n_nodes)
        corr = np.zeros((cfg.n_nodes, cfg.n_nodes))
        for j in _topological_order(B):
            weights = W[:, j].copy()
            signal_var = weights @ corr @ weights
            if signal_var > noise_std[j] ** 2 * max_ratio:
                weights *= np.sqrt(noise_std[j] ** 2 * max_ratio / signal_var)
                signal_var = weights @ corr @ weights
            sd_j = np.sqrt(signal_var + noise_std[j] ** 2)
            W_eff[:, j] = weights / sd_j
            noise_eff[j] = noise_std[j] / sd_j
            corr[:, j] = corr @ W_eff[:, j]
            corr[j, :] = corr[:, j]
            corr[j, j] = 1.0
        if passes_edge_margin(W_eff, noise_eff**2):
            break
        n_rejected += 1
    else:
        raise RuntimeError(f"No iSCM weights passed in {_MAX_TRIES} tries")
    x = _simulate_linear(rng_noise, W_eff, families, noise_eff, cfg.n_samples)
    sample = _permuted_sample(
        x, W_eff, noise_eff, families, cfg, rng_perm, rng_split
    )
    target_class, target_reason = _target_for_linear(
        families, cfg, equal_noise_kept=False
    )
    sample["target_class"] = target_class
    sample["target_reason"] = "iSCM: " + target_reason
    sample["n_rejected"] = n_rejected
    return sample


def _sample_pgmpy(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample pgmpy's random linear-Gaussian Bayesian network, filtered.

    pgmpy draws intercepts and weights from N(0, 1) and noise deviations of
    any size; networks with no edge, a node whose noise explains less than 10%
    of its variance, or an edge below the margin are redrawn (never modified).

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    rng_graph, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    n_nodes = cfg.n_nodes
    names = [f"X_{k}" for k in range(n_nodes)]
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        model = pgmodel.LinearGaussianBayesianNetwork.get_random(
            n_nodes=n_nodes,
            n_edges=cfg.n_edges,
            loc=0.0,
            scale=1.0,
            seed=int(rng_graph.integers(2**31)),
        )
        W = np.zeros((n_nodes, n_nodes))
        noise_std = np.zeros(n_nodes)
        for cpd in model.get_cpds():
            j = names.index(cpd.variable)
            noise_std[j] = float(cpd.std)
            for parent, beta in zip(cpd.evidence, cpd.beta[1:]):
                W[names.index(parent), j] = beta
        noise_var = noise_std**2
        cov = population_covariance(W, noise_var)
        passes = (
            np.count_nonzero(W) > 0
            and np.all(noise_var / np.diag(cov) >= _MIN_NOISE_SHARE)
            and passes_edge_margin(W, noise_var)
        )
        if passes:
            break
        n_rejected += 1
    else:
        raise RuntimeError(f"No pgmpy network passed in {_MAX_TRIES} tries")
    data = model.simulate(
        n_samples=cfg.n_samples, seed=int(rng_graph.integers(2**31))
    )
    x = data[names].to_numpy()
    families = ("gaussian",) * n_nodes
    sample = _permuted_sample(
        x, W, noise_std, families, cfg, rng_perm, rng_split
    )
    sample["target_class"] = tscore.TARGET_CPDAG
    sample["target_reason"] = "pgmpy linear Gaussian: equivalence class only"
    sample["n_rejected"] = n_rejected
    return sample


def _add_noise_for_r2(
    rng: np.random.Generator, y: np.ndarray, coef: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Add Gaussian noise so that the target's population R^2 is U(0.3, 0.8).

    The features of `make_regression` are independent standard normals, so
    the signal variance is `sum(coef^2)`.

    :param rng: random generator
    :param y: noiseless target
    :param coef: coefficients of the target
    :return: noisy target and the noise standard deviation
    """
    r2 = rng.uniform(0.3, 0.8)
    signal_var = float(np.sum(coef**2))
    noise_sd = np.sqrt(signal_var * (1 - r2) / r2)
    return y + noise_sd * rng.normal(size=y.shape[0]), noise_sd


def _sample_sklearn_reg(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample `make_regression`: `n_edges` informative features cause y.

    There are `n_nodes - 1` features; y is a collider of its informative
    features, so all its edges are compelled (CPDAG = DAG).

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    n_features = cfg.n_nodes - 1
    hdbg.dassert_lte(2, cfg.n_edges)
    hdbg.dassert_lte(cfg.n_edges, n_features)
    rng_graph, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        features, y, coef = skdata.make_regression(
            n_samples=cfg.n_samples,
            n_features=n_features,
            n_informative=cfg.n_edges,
            noise=0.0,
            shuffle=True,
            coef=True,
            random_state=int(rng_graph.integers(2**31)),
        )
        y, noise_sd = _add_noise_for_r2(rng_noise, y, coef)
        W = np.zeros((cfg.n_nodes, cfg.n_nodes))
        W[:n_features, n_features] = coef
        noise_std = np.ones(cfg.n_nodes)
        noise_std[n_features] = noise_sd
        if passes_edge_margin(W, noise_std**2):
            break
        n_rejected += 1
    else:
        raise RuntimeError(
            f"No make_regression draw passed in {_MAX_TRIES} tries"
        )
    x = np.column_stack([features, y])
    families = ("gaussian",) * cfg.n_nodes
    sample = _permuted_sample(
        x, W, noise_std, families, cfg, rng_perm, rng_split
    )
    sample["target_class"] = tscore.TARGET_CPDAG
    sample["target_reason"] = "make_regression: y is a collider (CPDAG = DAG)"
    sample["n_rejected"] = n_rejected
    return sample


def _sample_sklearn_confounder(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample two targets driven by one observed informative feature U.

    `make_regression` with `n_targets=2` and `n_informative=1`: U -> T1 and
    U -> T2, no T1-T2 edge; the other `n_nodes - 3` features are isolated.
    Each target's R^2 on U is drawn from U(0.3, 0.8). With U observed a CI test
    given U separates T1 and T2 (the spec's Key Example 2).

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    hdbg.dassert_eq(cfg.confounder, "observed", "Only an observed U for now")
    hdbg.dassert_eq(cfg.n_edges, 2, "U -> T1 and U -> T2")
    n_features = cfg.n_nodes - 2
    hdbg.dassert_lte(1, n_features)
    rng_graph, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    features, targets, coef = skdata.make_regression(
        n_samples=cfg.n_samples,
        n_features=n_features,
        n_informative=1,
        n_targets=2,
        noise=0.0,
        shuffle=True,
        coef=True,
        random_state=int(rng_graph.integers(2**31)),
    )
    source = int(np.flatnonzero(coef[:, 0])[0])
    W = np.zeros((cfg.n_nodes, cfg.n_nodes))
    noise_std = np.ones(cfg.n_nodes)
    columns = [features]
    for t in range(2):
        target, noise_sd = _add_noise_for_r2(
            rng_noise, targets[:, t], coef[:, t]
        )
        columns.append(target[:, None])
        W[source, n_features + t] = coef[source, t]
        noise_std[n_features + t] = noise_sd
    x = np.column_stack(columns)
    families = ("gaussian",) * cfg.n_nodes
    sample = _permuted_sample(
        x, W, noise_std, families, cfg, rng_perm, rng_split
    )
    sample["target_class"] = tscore.TARGET_CPDAG
    sample["target_reason"] = "observed common cause (fork): CPDAG"
    sample["n_rejected"] = 0
    return sample


def _sample_hidden_confounder(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample two targets with a hidden common cause U.

    U -> T1 and U -> T2 with U unobserved: causal sufficiency fails, and the
    observed data identify a PAG that keeps the T1-T2 adjacency (no observed
    set separates them):
    - "hidden_a": the other observed variables are isolated (PAG T1 o-o T2)
    - "hidden_b": plus observed private causes A -> T1 and B -> T2, which make
      the confounding identifiable (PAG A o-> T1 <-> T2 <-o B)
    - "hidden_c": as "hidden_a" with U, features and noises drawn from one
      non-Gaussian family, where direction tests are tempted to orient T1-T2

    Built directly: `make_regression` cannot give each target its own cause
    or draw non-Gaussian inputs. Every input has unit variance; each target's
    R^2 on its causes is drawn from U(0.3, 0.8); with private causes, U's
    share of a target's signal variance is in [0.25, 0.75].

    :param cfg: generator settings (`n_edges` counts U's two edges and, for
        "hidden_b", the two private edges)
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    variant = cfg.confounder
    hdbg.dassert_in(variant, ("hidden_a", "hidden_b", "hidden_c"))
    n_private = 2 if variant == "hidden_b" else 0
    hdbg.dassert_eq(cfg.n_edges, 2 + n_private)
    n_isolated = cfg.n_nodes - 2 - n_private
    hdbg.dassert_lte(0, n_isolated)
    rng_graph, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    family = "gaussian"
    if variant == "hidden_c":
        family = str(rng_graph.choice(["exponential", "uniform"]))
    n_samples = cfg.n_samples
    u = _sample_noise(rng_noise, family, n_samples, 1.0)
    private = [
        _sample_noise(rng_noise, family, n_samples, 1.0)
        for _ in range(n_private)
    ]
    isolated = [
        _sample_noise(rng_noise, family, n_samples, 1.0)
        for _ in range(n_isolated)
    ]
    # Generation order of the observed columns: private causes, T1, T2,
    # isolated features; U gets the last index of the full graph.
    targets_idx = [n_private, n_private + 1]
    u_idx = cfg.n_nodes
    W_full = np.zeros((cfg.n_nodes + 1, cfg.n_nodes + 1))
    noise_std = np.ones(cfg.n_nodes)
    targets = []
    low, high = cfg.w_range
    for t, target_idx in enumerate(targets_idx):
        u_coef = rng_graph.choice([-1.0, 1.0]) * rng_graph.uniform(low, high)
        W_full[u_idx, target_idx] = u_coef
        signal = u_coef * u
        signal_var = u_coef**2
        if n_private:
            # Redraw the private weight until U's share is in [0.25, 0.75].
            while True:
                p_coef = rng_graph.choice([-1.0, 1.0]) * rng_graph.uniform(
                    low, high
                )
                if 0.25 <= u_coef**2 / (u_coef**2 + p_coef**2) <= 0.75:
                    break
            W_full[t, target_idx] = p_coef
            signal = signal + p_coef * private[t]
            signal_var += p_coef**2
        r2 = rng_noise.uniform(0.3, 0.8)
        noise_sd = np.sqrt(signal_var * (1 - r2) / r2)
        targets.append(
            signal + _sample_noise(rng_noise, family, n_samples, noise_sd)
        )
        noise_std[target_idx] = noise_sd
    x = np.column_stack(private + targets + isolated)
    W = W_full[: cfg.n_nodes, : cfg.n_nodes]
    families = (family,) * cfg.n_nodes
    sample = _permuted_sample(
        x, W, noise_std, families, cfg, rng_perm, rng_split
    )
    perm = sample["perm"]
    pag = tscore.pag_from_dag((W_full != 0).astype(int), [u_idx])
    sample["pag"] = pag[np.ix_(perm, perm)]
    position = np.argsort(perm)
    pair = sorted(int(position[k]) for k in targets_idx)
    sample["confounded"] = ((pair[0], pair[1]),)
    sample["target_class"] = tscore.TARGET_PAG
    sample["target_reason"] = "hidden common cause: PAG (Zhang 2008)"
    sample["n_rejected"] = 0
    return sample


_SAMPLERS = {
    "proto": _sample_proto,
    "nonlinear": _sample_nonlinear,
    "iscm": _sample_iscm,
    "pgmpy_lg": _sample_pgmpy,
    "sklearn_reg": _sample_sklearn_reg,
    "sklearn_confounder": _sample_sklearn_confounder,
    "hidden_confounder": _sample_hidden_confounder,
}


def sample_instance(cfg: GraphConfig, seed: int) -> Instance:
    """
    Sample a hidden graph, its data and the train/test split.

    :param cfg: generator settings
    :param seed: instance seed (same seed and settings give the same instance)
    :return: the instance
    """
    hdbg.dassert_in(cfg.family, FAMILIES)
    if cfg.noise_share is not None or cfg.noise_grid:
        hdbg.dassert_eq(cfg.family, "iscm", "Noise shares exist only for iSCM")
        hdbg.dassert_is_not(cfg.noise_share, None)
    sampler = _SAMPLERS.get(cfg.family, _sample_linear)
    sample = sampler(cfg, seed)
    B = (sample["W"] != 0).astype(int)
    nongauss = np.array([f != "gaussian" for f in sample["noise_families"]])
    truth = tscore.make_truth(
        B,
        sample["target_class"],
        nongauss,
        sample["x_train"],
        sample["x_test"],
        pag=sample.get("pag"),
        confounded=sample.get("confounded", ()),
    )
    columns = [f"x{k}" for k in range(cfg.n_nodes)]
    instance = Instance(
        cfg=cfg,
        seed=seed,
        d_train=pd.DataFrame(sample["x_train"], columns=columns),
        d_test=pd.DataFrame(sample["x_test"], columns=columns),
        truth=truth,
        W=sample["W"],
        noise_std=sample["noise_std"],
        noise_families=sample["noise_families"],
        target_reason=sample["target_reason"],
        n_rejected=sample["n_rejected"],
        mechanism=sample.get("mechanism", "linear"),
    )
    return instance


# #############################################################################
# TukeyEnv
# #############################################################################


class TukeyEnv:
    """
    Gym-style environment: hidden graph, tool calls on D_train, scored emit.

    `reset()` returns `(evidence, info)`; `step(action)` runs one tool call and
    returns `(evidence, reward, terminated, truncated, info)`, where the reward
    is the call's training cost (negative). `submit_graph(edges)` ends the
    episode and returns the score. Signatures follow Gymnasium; the
    `gymnasium` package itself is not used because observations have a
    variable size.
    """

    def __init__(
        self,
        cfg: GraphConfig,
        *,
        budget: Optional[int] = None,
        toolset: str = "m3",
    ) -> None:
        """
        Create an environment for one generator configuration.

        :param cfg: generator settings
        :param budget: tool-call budget per episode (0 allows no tool call)
            - Default: None means `3 * n_nodes`
        :param toolset: tools an agent may call (key of `ttools.TOOLSETS`)
        """
        self.cfg = cfg
        self.budget = 3 * cfg.n_nodes if budget is None else budget
        hdbg.dassert_lte(0, self.budget)
        self.toolset = toolset
        self._instance: Optional[Instance] = None
        self._evidence: Optional[ttools.Evidence] = None
        self._all_actions = ttools.enumerate_tool_actions(
            cfg.n_nodes, toolset=toolset
        )
        self._done = True
        self._submitted = False

    @property
    def d_train(self) -> pd.DataFrame:
        """
        Training rows of the current instance (the experimenter's view).
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        return self._instance.d_train

    @property
    def d_test(self) -> pd.DataFrame:
        """
        Test rows of the current instance (the scorer's data, never an agent's).
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        return self._instance.d_test

    @property
    def instance(self) -> Instance:
        """
        The current instance with its truth (the experimenter's view, e.g., to
        check it against a set manifest; never an agent's).
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        return self._instance

    def reset(self, *, seed: int) -> Tuple[ttools.Evidence, Dict[str, Any]]:
        """
        Sample a new instance and start an episode.

        :param seed: instance seed
        :return: the agent's evidence and `info` (sizes and costs only)
        """
        self._instance = sample_instance(self.cfg, seed)
        cache = ttools.ToolCache(self._instance.d_train.to_numpy())
        self._evidence = ttools.Evidence(
            cache, self.budget, toolset=self.toolset
        )
        self._done = False
        self._submitted = False
        info = {
            "n_nodes": self.cfg.n_nodes,
            "n_train": cache.n_train,
            "budget": self.budget,
            "tool_costs": dict(ttools.TOOL_COSTS),
        }
        return self._evidence, info

    def legal_actions(self) -> List[ttools.Action]:
        """
        List the tool calls still allowed, plus STOP.

        :return: legal actions
        """
        hdbg.dassert(not self._done, "Episode is over: call reset()")
        legal = self._evidence.legal_tool_actions(self._all_actions)
        legal.append(ttools.Action(ttools.STOP))
        return legal

    def step(
        self, action: ttools.Action
    ) -> Tuple[ttools.Evidence, float, bool, bool, Dict[str, Any]]:
        """
        Run one tool call, or STOP.

        :param action: a legal action
        :return: evidence, reward (training cost of the call), terminated
            (True after STOP), truncated (True once the budget is spent), info
        """
        hdbg.dassert(not self._done, "Episode is over: call reset()")
        if action.kind == ttools.STOP:
            self._done = True
            return self._evidence, 0.0, True, False, {}
        self._evidence.record(action)
        reward = -ttools.TOOL_COSTS[action.kind]
        truncated = self._evidence.budget_left == 0
        return self._evidence, reward, False, truncated, {}

    def reward_oracle(
        self, *, lam: float = 0.5
    ) -> Callable[[np.ndarray], float]:
        """
        Return a function that scores any DAG on the current instance.

        For trainers only (it reads the hidden graph): reinforcement learning
        scores many sampled graphs per state without ending the episode. Scores
        are cost-free and cached per graph.

        :param lam: weight of the structural term in the combined score
            - Default: 0.5 (1 is structure only, 0 prediction only)
        :return: function from a DAG adjacency to its combined score
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        truth = self._instance.truth
        cache: Dict[bytes, float] = {}

        def _score(B: np.ndarray) -> float:
            key = B.astype(np.int8).tobytes()
            if key not in cache:
                scores = tscore.score_graph(B, truth, lam=lam, legacy=False)
                cache[key] = scores["score"]
            return cache[key]

        return _score

    def submit_graph(self, edges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Score a discovered graph (the prototype's `graph.json` contract).

        :param edges: `[(parent, child), ...]`
        :return: score dictionary of `tukey_scoring.score_graph()`, plus the
            number of tool calls used and of dropped edges
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        # One submission per episode: repeated scores would leak the graph.
        hdbg.dassert(not self._submitted, "Graph already submitted: call reset()")
        B_hat, n_dropped = tscore.edges_to_adjacency(edges, self.cfg.n_nodes)
        scores = tscore.score_graph(B_hat, self._instance.truth)
        scores["n_dropped_edges"] = n_dropped
        scores["calls_used"] = self._evidence.calls_used
        self._done = True
        self._submitted = True
        return scores
