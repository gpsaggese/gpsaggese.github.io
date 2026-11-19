"""
Statistical tools an agent can call on the training rows, and the evidence log.

Every tool is a pure function of D_train and its arguments, so results are
cached per dataset. Outputs are scale-invariant (correlations, standardized
coefficients, t-statistics, moments of standardized variables) except
`variance`, which is the only channel through which the raw scale of a column
can reach the agent; this keeps the variance-shortcut experiment clean.

The correlation matrix is free: it is observed at reset, as the prototype's
scripted agent computes it first. Partial correlations are functions of it, so
`ci_test` is a calculator; tools that add new information are `lingam_dir`,
`nongauss`, `regress` (residual shape) and, in the Milestone 4 tool set,
`mutual_info`, which sees nonlinear dependence that correlations miss.

Import as:

import tukey_tools as ttools
"""

import dataclasses
import functools
import itertools
import logging
from typing import Dict, List, Tuple

import numpy as np
import scipy.special
import scipy.stats
import sklearn.feature_selection as skfeat

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# Tool names.
CI_TEST = "ci_test"
REGRESS = "regress"
LINGAM_DIR = "lingam_dir"
NONGAUSS = "nongauss"
VARIANCE = "variance"
MUTUAL_INFO = "mutual_info"
STOP = "stop"
# Tool sets: Milestone 3 policies use the first five tools; Milestone 4 adds
# mutual information. A policy and its evidence always share one tool set.
TOOLSETS = {
    "m3": (CI_TEST, REGRESS, LINGAM_DIR, NONGAUSS, VARIANCE),
    "m4": (CI_TEST, REGRESS, LINGAM_DIR, NONGAUSS, VARIANCE, MUTUAL_INFO),
}

# Training-only cost of one call; evaluation scores are cost-free.
TOOL_COSTS = {
    CI_TEST: 0.001,
    REGRESS: 0.001,
    LINGAM_DIR: 0.001,
    NONGAUSS: 0.001,
    VARIANCE: 0.001,
    MUTUAL_INFO: 0.001,
    STOP: 0.0,
}

# Constants of the maximum-entropy approximation of differential entropy
# (Hyvarinen 1998), as used by the pairwise likelihood-ratio measure.
_ENTROPY_K1 = 79.047
_ENTROPY_K2 = 7.4129
_ENTROPY_GAMMA = 0.37457

# Fewest training rows a `ToolCache` accepts: the largest CI test conditions on
# 2 variables, and its Fisher z needs `n - 2 - 3 > 0` degrees of freedom.
_MIN_TRAIN_ROWS = 6
# A correlation matrix whose smallest eigenvalue is at most this is not
# positive definite in floating point (duplicated or linearly dependent
# columns, or fewer rows than variables): partial correlations and regressions
# on it are undefined.
_MIN_CORR_EIGENVALUE = 1e-10


# #############################################################################
# Action
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Action:
    """
    One agent action: a tool call with its arguments, or STOP.

    Arguments are canonical so equal calls compare equal:
    - `ci_test`: `(i, j, *S)` with `i < j` and `S` sorted, `1 <= |S| <= 2`
    - `regress`: `(target, *predictors)` with predictors sorted, `1 <= |P| <= 2`
    - `lingam_dir`: `(i, j)` with `i < j`; the sign of the result orients
    - `mutual_info`: `(i, j)` with `i < j`
    - `nongauss`, `variance`: `(i,)`
    - `stop`: `()`
    """

    kind: str
    args: Tuple[int, ...] = ()


def ci_test_action(i: int, j: int, cond: Tuple[int, ...]) -> Action:
    """
    Build a canonical conditional-independence test action.

    :param i: first variable
    :param j: second variable
    :param cond: conditioning set (1 or 2 variables)
    :return: action with sorted arguments
    """
    a, b = min(i, j), max(i, j)
    action = Action(CI_TEST, (a, b) + tuple(sorted(cond)))
    return action


def regress_action(target: int, predictors: Tuple[int, ...]) -> Action:
    """
    Build a canonical regression action.

    :param target: regressed variable
    :param predictors: 1 or 2 predictors
    :return: action with sorted predictors
    """
    action = Action(REGRESS, (target,) + tuple(sorted(predictors)))
    return action


def enumerate_tool_actions(n_nodes: int, *, toolset: str = "m3") -> List[Action]:
    """
    List every tool call available on a graph of `n_nodes` variables.

    :param n_nodes: number of variables
    :param toolset: key of `TOOLSETS`
    :return: all canonical tool actions (STOP excluded)
    """
    hdbg.dassert_in(toolset, TOOLSETS, "Unknown tool set")
    nodes = range(n_nodes)
    actions = []
    for i, j in itertools.combinations(nodes, 2):
        others = [k for k in nodes if k not in (i, j)]
        for size in (1, 2):
            for cond in itertools.combinations(others, size):
                actions.append(ci_test_action(i, j, cond))
        actions.append(Action(LINGAM_DIR, (i, j)))
    for target in nodes:
        others = [k for k in nodes if k != target]
        for size in (1, 2):
            for preds in itertools.combinations(others, size):
                actions.append(regress_action(target, preds))
        actions.append(Action(NONGAUSS, (target,)))
        actions.append(Action(VARIANCE, (target,)))
    if MUTUAL_INFO in TOOLSETS[toolset]:
        # Appended, so the Milestone 3 calls keep their order.
        for i, j in itertools.combinations(nodes, 2):
            actions.append(Action(MUTUAL_INFO, (i, j)))
    return actions


@functools.lru_cache(maxsize=None)
def _canonical_actions(n_nodes: int, toolset: str) -> frozenset:
    """
    Return every canonical tool action of a graph size, as a set.

    :param n_nodes: number of variables
    :param toolset: key of `TOOLSETS`
    :return: the actions of `enumerate_tool_actions()`
    """
    actions = frozenset(enumerate_tool_actions(n_nodes, toolset=toolset))
    return actions


# #############################################################################
# ToolCache
# #############################################################################


def _entropy(u: np.ndarray) -> float:
    """
    Approximate the differential entropy of a standardized variable.

    :param u: standardized sample
    :return: entropy estimate (Hyvarinen 1998)
    """
    term1 = np.mean(np.log(np.cosh(u))) - _ENTROPY_GAMMA
    term2 = np.mean(u * np.exp(-(u**2) / 2))
    entropy = (
        (1 + np.log(2 * np.pi)) / 2
        - _ENTROPY_K1 * term1**2
        - _ENTROPY_K2 * term2**2
    )
    return float(entropy)


class ToolCache:
    """
    Compute and cache tool results on one dataset's training rows.
    """

    def __init__(self, x_train: np.ndarray) -> None:
        """
        Precompute the statistics every tool reuses.

        The data are checked once here, not at each tool call: at least 2
        variables, `_MIN_TRAIN_ROWS` rows, no constant column and a positive
        definite correlation matrix.

        :param x_train: training rows, shape (n_train, n_nodes)
        """
        self.n_train, self.n_nodes = x_train.shape
        hdbg.dassert_lte(
            2, self.n_nodes, "The tools compare variables: pass at least 2"
        )
        hdbg.dassert_lte(
            _MIN_TRAIN_ROWS,
            self.n_train,
            "Too few training rows for a CI test given 2 variables",
        )
        self._x = x_train
        std = x_train.std(axis=0)
        hdbg.dassert(np.all(std > 0), "A constant column cannot be analysed")
        self._z = (x_train - x_train.mean(axis=0)) / std
        self.corr = np.corrcoef(x_train, rowvar=False)
        hdbg.dassert_lt(
            _MIN_CORR_EIGENVALUE,
            np.linalg.eigvalsh(self.corr).min(),
            "The correlation matrix of the training rows is not positive "
            "definite (duplicated or linearly dependent columns)",
        )
        self._cache: Dict[Action, Dict[str, float]] = {}

    def _ci_test(
        self, i: int, j: int, cond: Tuple[int, ...]
    ) -> Dict[str, float]:
        """
        Test i independent of j given `cond` with the partial correlation.

        :param i: first variable
        :param j: second variable
        :param cond: conditioning set
        :return: `pcorr`, Fisher `z`, `p`
        """
        idx = [i, j] + list(cond)
        precision = np.linalg.inv(self.corr[np.ix_(idx, idx)])
        pcorr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
        z, p = self.fisher_z(pcorr, len(cond))
        result = {"pcorr": float(pcorr), "z": z, "p": p}
        return result

    def _regress(
        self, target: int, predictors: Tuple[int, ...]
    ) -> Dict[str, float]:
        """
        Regress a standardized variable on 1 or 2 standardized predictors.

        :param target: regressed variable
        :param predictors: predictor variables
        :return: per-predictor standardized `coef_k`, `t_k`, `p_k`; `r2`;
            residual Anderson-Darling statistic `resid_ad`; for one predictor,
            `resid_dep = |corr(x_pred, |resid|)|`, the prototype's direction
            statistic
        """
        preds = list(predictors)
        r_pp = self.corr[np.ix_(preds, preds)]
        r_py = self.corr[preds, target]
        coef = np.linalg.solve(r_pp, r_py)
        r2 = float(coef @ r_py)
        dof = self.n_train - len(preds) - 1
        cov_inv_diag = np.diag(np.linalg.inv(r_pp))
        se = np.sqrt(max(1e-12, 1 - r2) / dof * cov_inv_diag)
        t_stats = coef / se
        # `scipy.stats.t.sf(x, dof)` is `stdtr(dof, -x)`; calling it directly
        # gives the same bits without the distribution object's overhead.
        p_values = 2 * scipy.special.stdtr(dof, -np.abs(t_stats))
        resid = self._z[:, target] - self._z[:, preds] @ coef
        result = {"r2": r2}
        for k, pred in enumerate(preds):
            result[f"coef_{pred}"] = float(coef[k])
            result[f"t_{pred}"] = float(t_stats[k])
            result[f"p_{pred}"] = float(p_values[k])
        result["resid_ad"] = float(
            scipy.stats.anderson(resid, method="interpolate").statistic
        )
        if len(preds) == 1:
            dep = np.corrcoef(self._z[:, preds[0]], np.abs(resid))[0, 1]
            result["resid_dep"] = float(abs(dep))
        return result

    def _lingam_dir(self, i: int, j: int) -> Dict[str, float]:
        """
        Compute the pairwise likelihood ratio of i -> j against j -> i.

        Hyvarinen and Smith (2013): `R = [H(x_j) + H(r_i|j)] - [H(x_i) +
        H(r_j|i)]` with entropies of standardized variables; `R > 0` favours
        i -> j. It is the measure DirectLiNGAM uses to order variables.

        :param i: first variable
        :param j: second variable
        :return: `stat` (positive favours i -> j)
        """
        zi, zj = self._z[:, i], self._z[:, j]
        rho = self.corr[i, j]
        resid_i_on_j = zi - rho * zj
        resid_j_on_i = zj - rho * zi
        h_resid_i = _entropy(resid_i_on_j / resid_i_on_j.std())
        h_resid_j = _entropy(resid_j_on_i / resid_j_on_i.std())
        stat = (_entropy(zj) + h_resid_i) - (_entropy(zi) + h_resid_j)
        result = {"stat": float(stat)}
        return result

    def _nongauss(self, i: int) -> Dict[str, float]:
        """
        Measure how non-Gaussian a variable's marginal distribution is.

        :param i: variable
        :return: `skew`, excess `kurt`, Jarque-Bera p-value `jb_p`
        """
        zi = self._z[:, i]
        result = {
            "skew": float(scipy.stats.skew(zi)),
            "kurt": float(scipy.stats.kurtosis(zi)),
            "jb_p": float(scipy.stats.jarque_bera(zi).pvalue),
        }
        return result

    def _mutual_info(self, i: int, j: int) -> Dict[str, float]:
        """
        Estimate the mutual information of two variables (Kraskov et al. 2004).

        `mi_excess` subtracts the mutual information of a Gaussian pair with
        the same correlation, `-log(1 - r^2) / 2`: it is near 0 for
        linear-Gaussian dependence. It is an estimated excess over that
        Gaussian reference, not a test of nonlinearity: non-Gaussian noise can
        make it positive for a linear pair, and some nonlinear pairs give a
        negative value.

        :param i: first variable
        :param j: second variable
        :return: `mi` and `mi_excess`, in nats
        """
        mi = skfeat.mutual_info_regression(
            self._z[:, [i]], self._z[:, j], n_neighbors=3, random_state=0
        )[0]
        r = float(np.clip(self.corr[i, j], -0.999999, 0.999999))
        gaussian_mi = -0.5 * np.log(1 - r**2)
        result = {"mi": float(mi), "mi_excess": float(mi - gaussian_mi)}
        return result

    def _compute(self, action: Action) -> Dict[str, float]:
        """
        Dispatch an action to its tool.

        :param action: tool action
        :return: result dictionary
        """
        if action.kind == CI_TEST:
            result = self._ci_test(
                action.args[0], action.args[1], action.args[2:]
            )
        elif action.kind == REGRESS:
            result = self._regress(action.args[0], action.args[1:])
        elif action.kind == LINGAM_DIR:
            result = self._lingam_dir(action.args[0], action.args[1])
        elif action.kind == NONGAUSS:
            result = self._nongauss(action.args[0])
        elif action.kind == VARIANCE:
            result = {"log_var": float(np.log(self._x[:, action.args[0]].var()))}
        elif action.kind == MUTUAL_INFO:
            result = self._mutual_info(action.args[0], action.args[1])
        else:
            raise ValueError(f"Unknown tool '{action.kind}'")
        return result

    def run(self, action: Action) -> Dict[str, float]:
        """
        Run a tool, from the cache when possible.

        :param action: tool action (not STOP)
        :return: result dictionary of the tool
        """
        if action not in self._cache:
            self._cache[action] = self._compute(action)
        result = self._cache[action]
        return result

    def fisher_z(self, r: float, n_cond: int) -> Tuple[float, float]:
        """
        Compute the Fisher z statistic and two-sided p-value of a correlation.

        :param r: (partial) correlation
        :param n_cond: size of the conditioning set
        :return: z statistic and p-value
        """
        r = float(np.clip(r, -0.999999, 0.999999))
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(self.n_train - n_cond - 3)
        # `scipy.stats.norm.sf(x)` is `ndtr(-x)`; calling it directly gives the
        # same bits about 100 times faster (checked on SciPy 1.18.1).
        p = 2 * scipy.special.ndtr(-abs(z))
        return float(z), float(p)


# #############################################################################
# Evidence
# #############################################################################


class Evidence:
    """
    What an agent has observed in an episode: the only input of a policy.

    Holds the free correlation matrix, the results of the tool calls made so
    far, and the budget. It never holds the hidden graph or the test rows.
    The boundary is a convention, not enforced isolation: the private
    `_cache` holds the training rows and can run a tool without counting a
    call. The supplied agents read only the public fields, and their calls go
    through `TukeyEnv.step()`, which calls `record()`.
    """

    def __init__(
        self, cache: ToolCache, budget: int, *, toolset: str = "m3"
    ) -> None:
        """
        Start an episode's evidence log.

        :param cache: tool cache of the episode's training rows
        :param budget: maximum number of tool calls
        :param toolset: tools the agent may call (key of `TOOLSETS`)
        """
        hdbg.dassert_in(toolset, TOOLSETS, "Unknown tool set")
        self.toolset = toolset
        self._cache = cache
        self.n_nodes = cache.n_nodes
        self.n_train = cache.n_train
        self.corr = cache.corr.copy()
        self.budget = budget
        self.results: Dict[Action, Dict[str, float]] = {}
        self.history: List[Action] = []

    @property
    def calls_used(self) -> int:
        """
        Number of tool calls made so far.
        """
        return len(self.history)

    @property
    def budget_left(self) -> int:
        """
        Number of tool calls still allowed.
        """
        return self.budget - len(self.history)

    def record(self, action: Action) -> Dict[str, float]:
        """
        Run a tool call and add its result to the evidence.

        :param action: canonical tool action (e.g., from `ci_test_action()`)
            not yet taken, within budget
        :return: the tool's result
        """
        hdbg.dassert_lt(0, self.budget_left, "Tool budget exhausted")
        hdbg.dassert_not_in(action, self.results, "Repeated tool call")
        # Only canonical calls of this toolset on existing variables run, so a
        # reordered call cannot pass as a new one.
        hdbg.dassert(
            action in _canonical_actions(self.n_nodes, self.toolset),
            "Not a canonical tool call of toolset '%s': %s",
            self.toolset,
            action,
        )
        result = self._cache.run(action)
        self.results[action] = result
        self.history.append(action)
        return result

    def legal_tool_actions(self, all_actions: List[Action]) -> List[Action]:
        """
        Filter tool actions to those still allowed.

        :param all_actions: `enumerate_tool_actions(n_nodes)`
        :return: actions not yet taken, or an empty list if the budget is spent
        """
        if self.budget_left <= 0:
            legal: List[Action] = []
        else:
            legal = [a for a in all_actions if a not in self.results]
        return legal
