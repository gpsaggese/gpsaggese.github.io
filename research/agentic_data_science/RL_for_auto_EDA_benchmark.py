"""
Benchmark harness for the RL-for-AutoEDA project (issue #506).

Measures a scripted baseline against standard structure-learning algorithms
across many synthetic graphs, as the first stage of the project's benchmark
milestone. Structured so the Claude Code agent can be added as an additional
method once run in an environment with SDK access.

Generator: pgmpy's LinearGaussianBayesianNetwork.get_random() + .simulate(),
matching the generator named in the paper (Section III) and replacing the
prototype's self-contained NumPy generator.

Identifiability and scoring: pgmpy linear-Gaussian networks produce Gaussian
data, under which a DAG is identifiable only up to its CPDAG (Markov equivalence
class). Whether the benchmark should score raw DAGs or CPDAGs depends on this and
is an open decision for GP, so it is exposed as a parameter (`cpdag`) rather than
fixed. CPDAG scoring is the default here because the current generator is
Gaussian; a non-Gaussian/LiNGAM generator would make raw-DAG scoring appropriate.

Methods compared:
  random    - random-guess floor
  scripted  - the prototype's correlation-skeleton + partial-correlation heuristic
  PC        - constraint-based structure learning (causal-learn)
  GES       - score-based structure learning (causal-learn)

Usage:
  python RL_for_auto_EDA_benchmark.py
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges


# ---------------------------------------------------------------------------
# Environment: pgmpy linear-Gaussian generator
# ---------------------------------------------------------------------------
class Environment:
    """One synthetic problem: a random linear-Gaussian DAG (retained as the
    held-out ground truth) and data simulated from it."""

    def __init__(self, n_nodes: int = 6, n_edges: int = 6,
                 n_samples: int = 2000, seed: int = 0) -> None:
        self.n_nodes = n_nodes
        self.seed = seed
        model = LGBN.get_random(n_nodes=n_nodes, n_edges=n_edges, seed=seed)
        self.nodes = list(model.nodes())
        idx = {name: i for i, name in enumerate(self.nodes)}
        self.B_true = np.zeros((n_nodes, n_nodes), dtype=int)
        for u, v in model.edges():
            self.B_true[idx[u], idx[v]] = 1
        df = model.simulate(n_samples=n_samples, seed=seed)
        self.X = df[self.nodes].to_numpy()


# ---------------------------------------------------------------------------
# CPDAG conversion and Structural Hamming Distance
# ---------------------------------------------------------------------------
def to_cpdag(B: np.ndarray) -> np.ndarray:
    """Reduce a DAG adjacency to a CPDAG representation: the undirected skeleton
    with v-structures (i -> k <- j, i and j non-adjacent) oriented. This is the
    standard equivalence-class starting point (Meek's first rule); it is not a
    full Meek propagation and should be replaced with a vetted DAG-to-CPDAG
    routine before results are reported as final.
    """
    n = B.shape[0]
    skeleton = ((B + B.T) > 0).astype(int)
    C = skeleton.copy()
    for k in range(n):
        parents = [p for p in range(n) if B[p, k] == 1]
        for i, j in itertools.combinations(parents, 2):
            if skeleton[i, j] == 0:
                C[i, k], C[k, i] = 1, 0
                C[j, k], C[k, j] = 1, 0
    return C


def raw_shd(B_hat: np.ndarray, B_true: np.ndarray) -> int:
    """SHD between two DAGs directly, counting one error per differing node-pair
    (add, delete, or reverse). Appropriate when the generator makes edge
    direction identifiable."""
    n = B_true.shape[0]
    return int(sum(
        (B_true[i, j], B_true[j, i]) != (B_hat[i, j], B_hat[j, i])
        for i, j in itertools.combinations(range(n), 2)
    ))


def shd(B_hat: np.ndarray, B_true: np.ndarray, cpdag: bool = True) -> int:
    """Structural Hamming Distance.

    With cpdag=True both graphs are reduced to their CPDAG before comparison, so
    a method is not charged for orientations that are unrecoverable from
    observational Gaussian data. With cpdag=False the raw DAGs are compared. The
    choice is an open decision for GP and is therefore parameterised.
    """
    if not cpdag:
        return raw_shd(B_hat, B_true)
    A, Bc = to_cpdag(B_hat), to_cpdag(B_true)
    n = B_true.shape[0]
    return int(sum(
        (A[i, j], A[j, i]) != (Bc[i, j], Bc[j, i])
        for i, j in itertools.combinations(range(n), 2)
    ))


# ---------------------------------------------------------------------------
# Methods (each returns a binary DAG adjacency matrix)
# ---------------------------------------------------------------------------
def random_agent(env: Environment, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k = env.n_nodes, int(env.B_true.sum())
    B = np.zeros((n, n), dtype=int)
    pairs = list(itertools.permutations(range(n), 2))
    for i in rng.choice(len(pairs), size=min(k, len(pairs)), replace=False):
        B[pairs[i]] = 1
    return B


def scripted_agent(env: Environment, thresh: float = 0.15) -> np.ndarray:
    """Correlation skeleton pruned by single-variable partial correlation. Edges
    are oriented by node index; equivalence-class scoring makes that convention
    harmless where direction is not identifiable."""
    X, n = env.X, env.n_nodes

    def corr(i: int, j: int) -> float:
        return abs(np.corrcoef(X[:, i], X[:, j])[0, 1])

    def partial_corr(i: int, j: int, k: int) -> float:
        ri = X[:, i] - LinearRegression().fit(X[:, [k]], X[:, i]).predict(X[:, [k]])
        rj = X[:, j] - LinearRegression().fit(X[:, [k]], X[:, j]).predict(X[:, [k]])
        return abs(np.corrcoef(ri, rj)[0, 1])

    B = np.zeros((n, n), dtype=int)
    for i, j in itertools.combinations(range(n), 2):
        if corr(i, j) >= thresh and not any(
            partial_corr(i, j, k) < thresh for k in range(n) if k not in (i, j)
        ):
            B[i, j] = 1
    return B


def pc_agent(env: Environment) -> np.ndarray:
    cg = pc(env.X, alpha=0.05, indep_test="fisherz", show_progress=False)
    return _from_causallearn(cg.G.graph, env.n_nodes)


def ges_agent(env: Environment) -> np.ndarray:
    record = ges(env.X)
    return _from_causallearn(record["G"].graph, env.n_nodes)


def _from_causallearn(G: np.ndarray, n: int) -> np.ndarray:
    """Convert a causal-learn adjacency (G[i,j] == -1 and G[j,i] == 1 means
    i -> j) to a binary DAG adjacency."""
    B = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if G[i, j] == -1 and G[j, i] == 1:
                B[i, j] = 1
    return B


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------
def run(configs, n_seeds: int = 10, cpdag: bool = True) -> None:
    methods = {
        "random": lambda e: random_agent(e, e.seed),
        "scripted": scripted_agent,
        "PC": pc_agent,
        "GES": ges_agent,
    }
    scoring = "CPDAG" if cpdag else "raw-DAG"
    print(f"\nMean {scoring} SHD over {n_seeds} graphs per row (lower is better)\n")
    header = f"{'nodes':>5} {'edges':>5} " + " ".join(f"{m:>9}" for m in methods)
    print(header)
    print("-" * len(header))
    for n_nodes, n_edges in configs:
        results = {m: [] for m in methods}
        for seed in range(n_seeds):
            env = Environment(n_nodes=n_nodes, n_edges=n_edges, seed=seed)
            for name, fn in methods.items():
                try:
                    results[name].append(shd(fn(env), env.B_true, cpdag=cpdag))
                except Exception:
                    results[name].append(np.nan)
        row = f"{n_nodes:>5} {n_edges:>5} " + " ".join(
            f"{np.nanmean(results[m]):>9.2f}" for m in methods
        )
        print(row)


if __name__ == "__main__":
    run(configs=[(5, 4), (6, 6), (8, 8), (10, 10)], n_seeds=10, cpdag=True)
