"""
Generate synthetic benchmark outcomes for K agent configs on N tasks and S
seeds.

Model:
- Config k has a true skill theta_k ~ Normal(mu_theta, sigma_theta)
- Task i has a difficulty b_i ~ Normal(mu_b, sigma_b)
- Config k and task i have an interaction u_ki ~ Normal(0, sigma_u), so that
  configs are not shifted copies of each other
- P(resolve) = sigmoid(theta_k - b_i + u_ki)
- Each seed is an independent Bernoulli draw given that probability

Mapping to the paper's notation (proposed, not confirmed by GP). This is the
only place in the code that uses it:
- S_A: one config k, i.e. one fixed agent + scaffold
- Y: the `resolved` outcome of config k on one task and one seed
- Ybar_A = E[Y | S_A, I_A]: `ybar[k]`, the expected outcome of config k over
  the task population and the seeds (see `compute_true_resolve_rate()`)
- L_A(Y) = Y - Ybar_A: everything else, i.e. which tasks were drawn, the
  interaction u_ki of those tasks, and the seed

Tables use the generic columns `config`, `task`, `seed`, `resolved`.

Import as:

import skill_luck_sim.generate as slsigene
"""

import dataclasses
import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

# Number of Gauss-Hermite nodes used to integrate over the task population.
# 101 nodes give an error below 1e-12 for the logit spreads used here.
_N_QUADRATURE_NODES = 101


# #############################################################################
# Parameters
# #############################################################################


@dataclasses.dataclass(frozen=True)
class SimParams:
    """
    Parameters of one simulated benchmark.

    The defaults are assumptions, not estimates from real data:
    - `sigma_theta=0.2` makes configs close together, which is the realistic
      case for scaffold variants of the same agent
    - `mu_theta=1.0`, `mu_b=0.0`, `sigma_b=2.0`, `sigma_u=1.0` give true
      resolve rates around 64%, with many tasks almost always or almost
      never solved
    """

    n_configs: int = 10
    n_tasks: int = 100
    n_seeds: int = 3
    mu_theta: float = 1.0
    sigma_theta: float = 0.2
    mu_b: float = 0.0
    sigma_b: float = 2.0
    sigma_u: float = 1.0

    def __post_init__(self) -> None:
        if self.n_configs < 1 or self.n_tasks < 1 or self.n_seeds < 1:
            raise ValueError(
                "n_configs, n_tasks, n_seeds must be >= 1, got "
                f"{self.n_configs}, {self.n_tasks}, {self.n_seeds}"
            )
        if min(self.sigma_theta, self.sigma_b, self.sigma_u) < 0:
            raise ValueError("Standard deviations must be >= 0")


# #############################################################################
# Model pieces
# #############################################################################


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the logistic function elementwise.

    :param x: logits
    :return: probabilities in (0, 1)
    """
    p = 1.0 / (1.0 + np.exp(-x))
    return p


def draw_skills(params: SimParams, rng: np.random.Generator) -> np.ndarray:
    """
    Draw the true skill of each config.

    :param params: simulation parameters
    :param rng: random generator
    :return: array of shape (n_configs,)
    """
    theta = rng.normal(
        params.mu_theta, params.sigma_theta, size=params.n_configs
    )
    return theta


def draw_difficulties(
    n_tasks: int, params: SimParams, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw task difficulties from the task population.

    :param n_tasks: number of tasks to draw
    :param params: simulation parameters
    :param rng: random generator
    :return: array of shape (n_tasks,)
    """
    b = rng.normal(params.mu_b, params.sigma_b, size=n_tasks)
    return b


def draw_interactions(
    n_configs: int, n_tasks: int, params: SimParams, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw the config-by-task interaction u_ki.

    :param n_configs: number of configs
    :param n_tasks: number of tasks
    :param params: simulation parameters
    :param rng: random generator
    :return: array of shape (n_configs, n_tasks)
    """
    u = rng.normal(0.0, params.sigma_u, size=(n_configs, n_tasks))
    return u


def draw_outcomes(
    p: np.ndarray, n_seeds: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw one Bernoulli outcome per seed for each solve probability.

    :param p: solve probabilities of any shape
    :param n_seeds: number of seeds
    :param rng: random generator
    :return: 0/1 array of shape `p.shape + (n_seeds,)`
    """
    y = (rng.random(p.shape + (n_seeds,)) < p[..., None]).astype(np.int8)
    return y


def compute_true_resolve_rate(
    theta: np.ndarray, params: SimParams
) -> np.ndarray:
    """
    Compute Ybar_k, the expected resolve rate of each config over the task
    population.

    For a task drawn from the population, the logit theta_k - b + u is
    Normal(theta_k - mu_b, sigma_b^2 + sigma_u^2), so
    Ybar_k = E[sigmoid(theta_k - mu_b + s Z)] with s^2 = sigma_b^2 + sigma_u^2
    and Z ~ Normal(0, 1). The expectation is computed by Gauss-Hermite
    numerical integration, not by Monte Carlo, so it has no sampling noise.

    E.g., theta_k = mu_b gives Ybar_k = 0.5 for any s, by symmetry.

    :param theta: skills, shape (n_configs,)
    :param params: simulation parameters
    :return: array of shape (n_configs,)
    """
    # `hermegauss` uses the weight exp(-z^2 / 2), so the weights sum to
    # sqrt(2 pi) and dividing by that gives an expectation under Normal(0, 1).
    nodes, weights = np.polynomial.hermite_e.hermegauss(_N_QUADRATURE_NODES)
    weights = weights / np.sqrt(2.0 * np.pi)
    scale = np.sqrt(params.sigma_b**2 + params.sigma_u**2)
    logits = (np.asarray(theta)[:, None] - params.mu_b) + scale * nodes[None, :]
    ybar = sigmoid(logits) @ weights
    return ybar


# #############################################################################
# World
# #############################################################################


@dataclasses.dataclass(frozen=True)
class World:
    """
    One draw of configs, tasks, and outcomes.

    - `theta`: skills, shape (K,)
    - `ybar`: true resolve rates over the task population, shape (K,)
    - `b`: difficulties of the sampled tasks, shape (N,)
    - `u`: interactions, shape (K, N)
    - `y`: outcomes, shape (K, N, S)
    """

    theta: np.ndarray
    ybar: np.ndarray
    b: np.ndarray
    u: np.ndarray
    y: np.ndarray


def draw_world(params: SimParams, rng: np.random.Generator) -> World:
    """
    Draw configs, tasks, interactions, and outcomes for one benchmark run.

    :param params: simulation parameters
    :param rng: random generator
    :return: the drawn world
    """
    theta = draw_skills(params, rng)
    ybar = compute_true_resolve_rate(theta, params)
    b = draw_difficulties(params.n_tasks, params, rng)
    u = draw_interactions(params.n_configs, params.n_tasks, params, rng)
    p = sigmoid(theta[:, None] - b[None, :] + u)
    y = draw_outcomes(p, params.n_seeds, rng)
    world = World(theta=theta, ybar=ybar, b=b, u=u, y=y)
    return world


def outcomes_to_frame(y: np.ndarray) -> pd.DataFrame:
    """
    Convert an outcome array into a long table.

    :param y: outcomes of shape (K, N, S)
    :return: table with columns `config`, `task`, `seed`, `resolved`, one row
        per (config, task, seed)
    """
    n_configs, n_tasks, n_seeds = y.shape
    config, task, seed = np.meshgrid(
        np.arange(n_configs),
        np.arange(n_tasks),
        np.arange(n_seeds),
        indexing="ij",
    )
    df = pd.DataFrame(
        {
            "config": config.ravel(),
            "task": task.ravel(),
            "seed": seed.ravel(),
            "resolved": y.ravel().astype(int),
        }
    )
    return df
