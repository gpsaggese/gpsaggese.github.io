"""
Simulate how talent and luck drive success, and estimate the effect of luck.

The simulation follows "Simulation Design" in the experimental evaluation of
the white paper: agents with talents receive lucky and unlucky events that
change their capital multiplicatively.
The Bayesian model regresses log-capital on luck and talents.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_utils as rcaoasalcsu
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import tqdm

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint

# Import PyMC and ArviZ only for the type hints, since they are imported inside
# the functions that use them.
if TYPE_CHECKING:
    import arviz as az
    import pymc as pm

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    _LOG.debug("notebook_log=%s", notebook_log.name)
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# __all__ = [
#     "Agent",
#     "create_population",
#     "calculate_gini",
#     "get_results_dataframe",
#     "generate_summary_statistics",
#     "validate_simulation_results",
#     "run_simulation",
#     "run_policy_simulation",
#     "fit_bayesian_luck_model",
#     "summarize_bayesian_fit",
#     "posterior_predictive_check",
# ]


# #############################################################################
# Agent
# #############################################################################


class Agent:
    """
    Agent representing an individual in the simulation.

    Each agent has four characteristics that define their position in the system:

    1. Intensity (0-1): Activity level and effort.
       - How active the agent is in seeking opportunities and experiences.
       - Higher intensity -> higher probability of encountering events (both good and bad).
       - Think of it as "surface area for luck": more active people encounter more events.
       - Influences event exposure probability via sigmoid function.

    2. IQ (0-1): Ability to capitalize on opportunities.
       - When a lucky event occurs, IQ determines if the agent successfully exploits it.
       - Does NOT create opportunities, only gates whether they can be converted to gains.
       - Unlucky events always apply (no IQ gate).
       - Used as probability of capitalizing on beneficial events.

    3. Networking (0-1): Social connectivity and spillover.
       - Represents social connections and access to network effects.
       - When an agent benefits from a lucky event, there's a chance (10%) that
         a connected agent also benefits (at reduced impact: 50% of original).
       - Spillover amount weighted by networking score.

    4. Initial Capital: Starting wealth.
       - Set to 1.0 for all agents in baseline simulation.
       - This ensures inequality EMERGES from dynamics, not inherited advantages.
       - Minimum enforced: 0.01 (prevent collapse to zero).
    """

    def __init__(
        self,
        agent_id: int,
        intensity: float,
        iq: float,
        networking: float,
        *,
        initial_capital: float = 1.0,
    ) -> None:
        """
        Initialize the Agent with talents and initial capital.

        :param agent_id: Unique agent identifier (typically agent's index)
        :param intensity: Intensity talent, affects event exposure (0-1)
        :param iq: IQ talent, affects ability to capitalize on luck (0-1)
        :param networking: Networking talent, affects spillover effects (0-1)
        :param initial_capital: Starting wealth level
            - Default: 1.0
        """
        _LOG.debug(
            "agent_id=%s intensity=%s iq=%s networking=%s initial_capital=%s",
            agent_id,
            intensity,
            iq,
            networking,
            initial_capital,
        )
        self.id = int(agent_id)
        # Enforce bounds and safe floor for capital.
        self.talent = {
            "intensity": float(np.clip(intensity, 0.0, 1.0)),
            "iq": float(np.clip(iq, 0.0, 1.0)),
            "networking": float(np.clip(networking, 0.0, 1.0)),
            "initial_capital": float(max(0.01, initial_capital)),
        }
        self.capital = float(self.talent["initial_capital"])
        self.capital_history: List[float] = [self.capital]
        self.lucky_events: int = 0
        self.unlucky_events: int = 0

    @property
    def talent_norm(self) -> float:
        """
        Euclidean norm of the 4D talent vector.

        :return: L2 norm of talent dimensions
        """
        values = np.array(
            [
                self.talent["intensity"],
                self.talent["iq"],
                self.talent["networking"],
                self.talent["initial_capital"],
            ],
            dtype=float,
        )
        talent_norm = float(np.linalg.norm(values))
        _LOG.debug("return=%s", talent_norm)
        return talent_norm

    def get_event_probability(self) -> float:
        """
        Probability of encountering an event based on intensity.

        Uses a sigmoid centered at 0.5. Higher intensity = higher exposure.

        :return: Event probability in [0, 1]
        """
        alpha = 2.0
        # No debug log here: this runs for every agent at every event, and a
        # log call would slow the simulation down noticeably.
        probability = float(
            1.0 / (1.0 + np.exp(-alpha * (self.talent["intensity"] - 0.5)))
        )
        return probability

    def apply_event(self, event_type: str, impact: float) -> None:
        """
        Apply an event to capital using multiplicative dynamics.

        :param event_type: kind of event
            - `"lucky"`: multiply capital by `1 + impact`
            - `"unlucky"`: multiply capital by `1 - impact`, floored at 0.01
        :param impact: magnitude as a decimal (e.g., 0.25 = 25%)
        """
        _LOG.debug("event_type=%s impact=%s", event_type, impact)
        impact = float(abs(impact))
        if event_type == "lucky":
            self.capital *= 1.0 + impact
            self.lucky_events += 1
        elif event_type == "unlucky":
            self.capital *= 1.0 - impact
            self.capital = max(0.01, self.capital)
            self.unlucky_events += 1
        else:
            raise ValueError(f"Unknown event type: {event_type}")
        self.capital_history.append(self.capital)


# #############################################################################
# Population
# #############################################################################


def create_population(*, n_agents: int = 100, seed: int = 42) -> List[Agent]:
    """
    Create a population of agents with normally distributed talents.

    :param n_agents: number of agents to create
        - Default: 100
    :param seed: RNG seed for reproducibility
        - Default: 42
    :return: `Agent` objects, each with random talents and capital 1.0
    """
    _LOG.debug(hprint.to_str("n_agents seed"))
    hdbg.dassert_lt(0, n_agents, "n_agents must be positive")
    rng = np.random.default_rng(seed)
    # Draw the three talents of each agent, clipped to [0, 1].
    agents: List[Agent] = []
    for i in range(n_agents):
        intensity = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        iq = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        networking = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        agents.append(Agent(i, intensity, iq, networking, initial_capital=1.0))
    return agents


# #############################################################################
# Results and metrics
# #############################################################################


# TODO(ai_gp): Rename to _calculate_gini() and update callers, as this is only used internally (coding.rules.md:## Mark Private Functions)
# Not done: `causal_success_API.py` and `causal_success_example.py` call
# `calculate_gini()` directly, so it is not internal and stays public.
def calculate_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for non-negative values.

    The Gini coefficient measures inequality in a distribution (e.g., wealth).

    :param values: 1D array of non-negative values (e.g., capital amounts)
    :return: Gini coefficient in [0, 1]
    """
    _LOG.debug("values.shape=%s", np.shape(values))
    x = np.asarray(values, dtype=float)
    hdbg.dassert_lt(
        0, x.size, "Cannot calculate Gini coefficient for empty array"
    )
    hdbg.dassert(
        not np.any(x < 0),
        "Gini coefficient requires non-negative values",
    )
    if np.all(x == 0):
        # All the values are zero, so there is no inequality.
        gini = 0.0
    else:
        # Use the closed form on sorted values:
        # `G = 2 sum_i i x_(i) / (n sum_i x_i) - (n + 1) / n`.
        x_sorted = np.sort(x)
        n = x_sorted.size
        index = np.arange(1, n + 1, dtype=float)
        gini = (2.0 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (
            n + 1.0
        ) / n
        gini = float(np.clip(gini, 0.0, 1.0))
    _LOG.debug("return=%s", gini)
    return gini


# TODO(ai_gp): Rename to _get_results_dataframe() and update callers, as this is only used internally (coding.rules.md:## Mark Private Functions)
# Not done: `causal_success_API.py` and `causal_success_example.py` call
# `get_results_dataframe()` directly, so it is not internal and stays public.
def get_results_dataframe(agents: List[Agent]) -> pd.DataFrame:
    """
    Convert a list of agents to a DataFrame for analysis.

    :param agents: List of Agent objects
    :return: DataFrame with agent attributes
    """
    _LOG.debug("len(agents)=%s", len(agents))
    if not agents:
        df = pd.DataFrame()
    else:
        # Build one row per agent.
        rows: List[Dict[str, Any]] = []
        for a in agents:
            rows.append(
                {
                    "id": a.id,
                    "talent_intensity": a.talent["intensity"],
                    "talent_iq": a.talent["iq"],
                    "talent_networking": a.talent["networking"],
                    "initial_capital": a.talent["initial_capital"],
                    "talent_norm": a.talent_norm,
                    "capital": a.capital,
                    "lucky_events": a.lucky_events,
                    "unlucky_events": a.unlucky_events,
                    "net_events": a.lucky_events - a.unlucky_events,
                }
            )
        df = pd.DataFrame(rows)
    _LOG.debug("df.shape=%s", df.shape)
    return df


def generate_summary_statistics(agents: List[Agent]) -> Dict[str, float]:
    """
    Compute summary statistics of the simulation output.

    :param agents: List of Agent objects (after simulation)
    :return: Dictionary mapping metric names to float values. Example output:
        {
            'n_agents': 100.0,
            'mean_capital': 2.15,
            'median_capital': 1.85,
            'std_capital': 1.42,
            'min_capital': 0.01,
            'max_capital': 8.50,
            'capital_range': 850.0,
            'gini_coefficient': 0.38,
            'top_10_pct_share': 0.35,
            'top_20_pct_share': 0.52,
            'bottom_50_pct_share': 0.15,
            'mean_lucky_events': 4.2,
            'mean_unlucky_events': 4.1,
            'mean_talent_norm': 1.95,
        }
    """
    _LOG.debug("len(agents)=%s", len(agents))
    df = get_results_dataframe(agents)
    if df.empty:
        stats: Dict[str, float] = {"n_agents": 0}
    else:
        # Compute the capital totals used by the shares.
        capital = df["capital"].to_numpy(dtype=float)
        gini = calculate_gini(capital)
        min_cap = float(np.min(capital))
        max_cap = float(np.max(capital))
        total_cap = float(np.sum(capital))
        n = len(df)
        # Guard against division by zero (should not happen due to floor).
        cap_range = max_cap / max(min_cap, 1e-12)
        # Use at least one agent in each group, even for small populations.
        top_10_n = max(1, n // 10)
        top_20_n = max(1, n // 5)
        bottom_50_n = max(1, n // 2)
        stats = {
            "n_agents": float(n),
            "mean_capital": float(np.mean(capital)),
            "median_capital": float(np.median(capital)),
            "std_capital": float(np.std(capital)),
            "min_capital": min_cap,
            "max_capital": max_cap,
            "capital_range": float(cap_range),
            "gini_coefficient": float(gini),
            "top_10_pct_share": float(
                df.nlargest(top_10_n, "capital")["capital"].sum() / total_cap
            ),
            "top_20_pct_share": float(
                df.nlargest(top_20_n, "capital")["capital"].sum() / total_cap
            ),
            "bottom_50_pct_share": float(
                df.nsmallest(bottom_50_n, "capital")["capital"].sum() / total_cap
            ),
            "mean_lucky_events": float(
                df["lucky_events"].to_numpy(dtype=float).mean()
            ),
            "mean_unlucky_events": float(
                df["unlucky_events"].to_numpy(dtype=float).mean()
            ),
            "mean_talent_norm": float(
                df["talent_norm"].to_numpy(dtype=float).mean()
            ),
        }
    _LOG.debug("return=%s", stats)
    return stats


def validate_simulation_results(agents: List[Agent]) -> bool:
    """
    Validate simulation results for basic correctness.

    Raise `AssertionError` if anything looks inconsistent.

    :param agents: List of Agent objects to validate
    :return: True if validation passes
    """
    _LOG.debug("len(agents)=%s", len(agents))
    df = get_results_dataframe(agents)
    hdbg.dassert(not df.empty, "No agents provided to validate")
    hdbg.dassert(not (df["capital"] < 0).any(), "Negative capital detected")
    hdbg.dassert(not df.isnull().any().any(), "NaN values detected")
    hdbg.dassert(
        not ((df["lucky_events"] < 0).any() or (df["unlucky_events"] < 0).any()),
        "Negative event counts detected",
    )
    # Each event appends one entry to the history of the initial capital.
    for a in agents:
        expected = 1 + a.lucky_events + a.unlucky_events
        hdbg.dassert_eq(
            len(a.capital_history),
            expected,
            "Agent has inconsistent capital history length",
        )
    is_valid = True
    _LOG.debug("return=%s", is_valid)
    return is_valid


# #############################################################################
# Simulation
# #############################################################################


def _select_exposed_agent(agents: List[Agent], rng: np.random.Generator) -> int:
    """
    Pick the agent hit by one event, with probability proportional to exposure.

    The exposure of an agent is `Agent.get_event_probability()`. If all
    exposures are zero, every agent is equally likely.

    :param agents: population of agents
    :param rng: random generator of the simulation
    :return: index of the selected agent
    """
    _LOG.debug("len(agents)=%s", len(agents))
    n_agents = len(agents)
    exposure = np.array([a.get_event_probability() for a in agents], dtype=float)
    exposure = (
        exposure / exposure.sum()
        if exposure.sum() > 0
        else np.ones(n_agents) / n_agents
    )
    selected_idx = int(rng.choice(n_agents, p=exposure))
    _LOG.debug("return=%s", selected_idx)
    return selected_idx


def run_simulation(
    agents: List[Agent],
    *,
    n_periods: int = 80,
    n_lucky_events_per_period: int = 5,
    n_unlucky_events_per_period: int = 5,
    lucky_mean: float = 0.25,
    lucky_std: float = 0.08,
    unlucky_mean: float = 0.15,
    unlucky_std: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> List[Agent]:
    """
    Execute the agent-based simulation over multiple periods.

    :param agents: List of Agent objects to simulate (modified in-place)
    :param n_periods: Number of time periods to simulate
        - Default: 80
    :param n_lucky_events_per_period: Lucky events per period
        - Default: 5
    :param n_unlucky_events_per_period: Unlucky events per period
        - Default: 5
    :param lucky_mean: Mean impact of lucky events
        - Default: 0.25, i.e., 25%
    :param lucky_std: Std dev of lucky event impacts
        - Default: 0.08
    :param unlucky_mean: Mean impact of unlucky events
        - Default: 0.15, i.e., 15%
    :param unlucky_std: Std dev of unlucky event impacts
        - Default: 0.05
    :param seed: RNG seed for reproducibility
        - Default: 42
    :param verbose: Show a progress bar if True
        - Default: False
    :return: Same agents list with updated capital and event histories
    """
    _LOG.debug(
        hprint.to_str(
            "n_periods n_lucky_events_per_period n_unlucky_events_per_period"
            " lucky_mean lucky_std unlucky_mean unlucky_std seed verbose"
        )
    )
    hdbg.dassert_lt(0, n_periods, "n_periods must be positive")
    hdbg.dassert_lt(0, len(agents), "agents list cannot be empty")
    hdbg.dassert_lte(
        0,
        n_lucky_events_per_period,
        "event counts per period must be non-negative",
    )
    hdbg.dassert_lte(
        0,
        n_unlucky_events_per_period,
        "event counts per period must be non-negative",
    )
    # Set up the random generator and the optional progress bar.
    rng = np.random.default_rng(seed)
    n_agents = len(agents)
    if verbose:
        periods_iter = tqdm.tqdm(
            range(n_periods), desc="Running simulation", unit="period"
        )
    else:
        periods_iter = range(n_periods)
    for _ in periods_iter:
        # Lucky events.
        for _ in range(n_lucky_events_per_period):
            selected_idx = _select_exposed_agent(agents, rng)
            selected = agents[selected_idx]
            impact = float(
                np.clip(rng.normal(lucky_mean, lucky_std), 0.05, 0.50)
            )
            # IQ gates whether a lucky event can be capitalized on.
            if rng.random() < selected.talent["iq"]:
                selected.apply_event("lucky", impact)
            # Networking spillover (10%).
            if rng.random() < 0.1:
                net = np.array(
                    [a.talent["networking"] for a in agents], dtype=float
                )
                if net.sum() > 0:
                    net = net / net.sum()
                    inherited_idx = int(rng.choice(n_agents, p=net))
                    if (
                        inherited_idx != selected_idx
                        and rng.random() < agents[inherited_idx].talent["iq"]
                    ):
                        agents[inherited_idx].apply_event("lucky", impact * 0.5)
        # Unlucky events.
        for _ in range(n_unlucky_events_per_period):
            selected_idx = _select_exposed_agent(agents, rng)
            selected = agents[selected_idx]
            impact = float(
                np.clip(rng.normal(unlucky_mean, unlucky_std), 0.05, 0.30)
            )
            selected.apply_event("unlucky", impact)
    _LOG.debug("len(agents)=%s", len(agents))
    return agents


def run_policy_simulation(
    agents: List[Agent],
    *,
    policy: str = "egalitarian",
    resource_amount: float = 100.0,
    # TODO(ai_gp): Avoid None default for cate_values; consider alternative design or use empty array default (coding.rules.md:## Minimize Default Values of None in Function Interfaces)
    # Not done: only the "cate_optimal" policy uses `cate_values`, and an empty
    # array default would hide a missing input instead of failing on it.
    cate_values: Optional[np.ndarray] = None,
    **simulation_kwargs: Any,
) -> List[Agent]:
    """
    Allocate initial resources under a policy, then run the standard simulation.

    1. "egalitarian"
       - Every agent gets: resource_amount / n_agents
       - Rationale: Reduce initial inequality, give everyone equal chance
       - Typical outcome: Lowest final Gini (most equitable)
       - Typical outcome: Moderate total welfare

    2. "meritocratic"
       - Allocation proportional to `talent_norm` (total ability)
       - Rationale: Reward potentially capable people
       - Typical outcome: Moderate final Gini
       - Typical outcome: High total welfare (resources go to productive people)

    3. "performance"
       - Allocation proportional to current capital (rich get richer)
       - Rationale: Compound success (controversial, tested for comparison)
       - Typical outcome: Highest final Gini (most unequal)
       - Typical outcome: Lowest total welfare (resources wasted on already-rich)

    4. "random"
       - One randomly chosen agent gets ALL resources
       - Rationale: Extreme luck-based allocation
       - Typical outcome: Very high Gini
       - Typical outcome: Highest possible total welfare (concentrated resources)

    5. "cate_optimal"
       - Allocation proportional to CATE estimates (heterogeneous treatment
         effects)
       - Rationale: Give resources to agents who benefit most from them
       - Requires: `cate_values` array with one value per agent
       - Typical outcome: High total welfare, moderate Gini
       - Note: Only allocates to agents with non-negative CATE

    :param agents: List of Agent objects (capital modified in-place)
    :param policy: Allocation rule: `"egalitarian"`, `"meritocratic"`,
        `"performance"`, `"random"`, or `"cate_optimal"`
        - Default: `"egalitarian"`
    :param resource_amount: Total budget to distribute at t=0
        - Default: 100.0
    :param cate_values: 1D array of CATE estimates, one per agent
        - Required if `policy="cate_optimal"`, ignored otherwise
    :param simulation_kwargs: Additional arguments forwarded to
        `run_simulation()` (e.g., `n_periods=80, seed=42, verbose=True`)
    :return: Same agents list after resource allocation and full simulation
    """
    _LOG.debug(hprint.to_str("policy resource_amount simulation_kwargs"))
    hdbg.dassert_lt(0, len(agents), "agents list cannot be empty")
    hdbg.dassert_lt(
        -0.0001, resource_amount, "resource_amount must be non-negative"
    )
    n = len(agents)
    rng = np.random.default_rng(simulation_kwargs.get("seed", None))
    # Handle random policy separately (single winner).
    if policy == "random":
        winner_idx = int(rng.integers(n))
        _LOG.debug("winner_idx=%s", winner_idx)
        agents[winner_idx].capital += resource_amount
        agents[winner_idx].capital_history[0] = agents[winner_idx].capital
    else:
        # For all other policies, we compute weights and allocate
        # proportionally.
        weights = np.zeros(n, dtype=float)
        if policy == "egalitarian":
            weights[:] = 1.0
        elif policy == "meritocratic":
            weights = np.array([a.talent_norm for a in agents], dtype=float)
        elif policy == "performance":
            weights = np.array([a.capital for a in agents], dtype=float)
        elif policy == "cate_optimal":
            hdbg.dassert_is_not(
                cate_values,
                None,
                "cate_values must be provided when policy='cate_optimal'.",
            )
            cate_array = np.asarray(cate_values, dtype=float)
            hdbg.dassert_eq(
                cate_array.shape[0],
                n,
                "cate_values must have one value per agent",
            )
            # Use only non-negative CATEs; negative values are clamped to zero.
            weights = np.maximum(cate_array, 0.0)
        else:
            raise ValueError(
                f"Unknown policy: {policy}. Must be one of: "
                f"egalitarian, meritocratic, performance, random, cate_optimal"
            )
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            # Fallback: if everything is zero, allocate equally.
            weights = np.ones(n, dtype=float)
            total_weight = float(n)
        shares = weights / total_weight
        allocations = shares * float(resource_amount)
        _LOG.debug("allocations[:5]=%s", allocations[:5])
        for a, alloc in zip(agents, allocations):
            a.capital += float(alloc)
            # Keep history consistent at t=0.
            a.capital_history[0] = a.capital
    # Run the standard simulation from the allocated capital.
    agents = run_simulation(agents, **simulation_kwargs)
    return agents


# #############################################################################
# Bayesian model
# #############################################################################


def fit_bayesian_luck_model(
    df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> Tuple["pm.Model", "az.InferenceData"]:
    """
    Fit a Bayesian regression model to estimate causal effect of luck on capital.

    :param df: DataFrame from `get_results_dataframe(agents)`, with the
        columns `capital`, `lucky_events`, `talent_intensity`, `talent_iq`,
        `talent_networking`
    :param draws: Number of posterior draws per chain
        - Higher means a more accurate posterior and a longer runtime
        - Default: 1000
    :param tune: Number of NUTS tuning (burn-in) iterations, which are
        discarded and not used in inference
        - Default: 1000
    :param target_accept: NUTS sampler target acceptance rate
        - Valid range: (0.5, 1.0), higher is slower but more stable
        - Default: 0.9
    :param random_seed: RNG seed for reproducibility
        - Default: 42
    :return: tuple `(model, idata)`
        - `model`: PyMC `Model` (for diagnostics, re-sampling, etc.)
        - `idata`: ArviZ `InferenceData` with the posterior samples, to use
          with `summarize_bayesian_fit()` or `posterior_predictive_check()`
    """
    _LOG.debug(hprint.to_str("df.shape draws tune target_accept random_seed"))
    # Import PyMC only here, so that the simulation works without it.
    import pymc as pm  # type: ignore

    required_cols = [
        "capital",
        "lucky_events",
        "talent_intensity",
        "talent_iq",
        "talent_networking",
    ]
    hdbg.dassert_is_subset(
        required_cols, df.columns, "DataFrame is missing required columns"
    )
    capital = df["capital"].to_numpy(dtype=float)
    # Log-capital is more stable and closer to normal.
    y = np.log(capital)
    lucky = df["lucky_events"].to_numpy(dtype=float)
    intensity = df["talent_intensity"].to_numpy(dtype=float)
    iq = df["talent_iq"].to_numpy(dtype=float)
    networking = df["talent_networking"].to_numpy(dtype=float)
    # THE QUESTION
    # ============
    # Does luck causally affect outcomes, even after controlling for talent?
    # This model answers that by regressing log(capital) on both luck and talent.
    #
    # THE MODEL
    # =========
    # Linear Bayesian regression:
    #
    #     log(capital_i) = alpha
    #                      + beta_luck * lucky_events_i
    #                      + beta_intensity * talent_intensity_i
    #                      + beta_iq * talent_iq_i
    #                      + beta_networking * talent_networking_i
    #                      + epsilon_i
    #
    # Where:
    #     - log(capital) is the outcome (log-scale for stability)
    #     - lucky_events is the treatment (how many beneficial events occurred)
    #     - talent_* are confounders (control for inherent ability)
    #     - epsilon ~ N(0, sigma) is residual error
    #
    # PRIORS
    # ======
    # All coefficients use weakly informative N(0, 1) priors (centered at 0).
    # This allows the data to dominate the inference without strong prior beliefs.
    with pm.Model() as model:
        # Priors: fairly weakly informative, centered at 0.
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta_luck = pm.Normal("beta_luck", mu=0.0, sigma=1.0)
        beta_intensity = pm.Normal("beta_intensity", mu=0.0, sigma=1.0)
        beta_iq = pm.Normal("beta_iq", mu=0.0, sigma=1.0)
        beta_networking = pm.Normal("beta_networking", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu = (
            alpha
            + beta_luck * lucky
            + beta_intensity * intensity
            + beta_iq * iq
            + beta_networking * networking
        )
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    _LOG.debug("Sampled draws=%s tune=%s", draws, tune)
    return model, idata


# Parameters of the Bayesian model summarized by default: the main coefficients
# and sigma.
_DEFAULT_SUMMARY_VAR_NAMES = (
    "alpha",
    "beta_luck",
    "beta_intensity",
    "beta_iq",
    "beta_networking",
    "sigma",
)


def summarize_bayesian_fit(
    idata: "az.InferenceData",
    *,
    var_names: Tuple[str, ...] = _DEFAULT_SUMMARY_VAR_NAMES,
) -> pd.DataFrame:
    """
    Summarize the posterior of the Bayesian model parameters.

    The summary has the posterior mean, sd, and credible intervals of each
    parameter.

    :param idata: ArviZ `InferenceData` returned by `fit_bayesian_luck_model()`
    :param var_names: parameter names to summarize
        - Default: the coefficients and sigma
    :return: pandas DataFrame with summary statistics (mean, sd, hdi, etc.)
    """
    _LOG.debug(hprint.to_str("var_names"))
    # Import ArviZ only here, so that the simulation works without it.
    import arviz as az  # type: ignore

    # `az.summary()` returns a DataFrame with its default `fmt="wide"`, while
    # its type hint also allows an xarray Dataset.
    summary = cast(pd.DataFrame, az.summary(idata, var_names=list(var_names)))
    _LOG.debug("summary.shape=%s", summary.shape)
    return summary


def posterior_predictive_check(
    model: "pm.Model",
    idata: "az.InferenceData",
    df: pd.DataFrame,
    *,
    random_seed: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Simple posterior predictive check (PPC).

    This function draws from the posterior predictive distribution and compares
    simulated log-capital to the observed log-capital.

    :param model: PyMC model returned by `fit_bayesian_luck_model()`
    :param idata: ArviZ `InferenceData` with posterior draws
    :param df: same DataFrame used for fitting
    :param random_seed: RNG seed for reproducibility
        - Default: 123
    :return: dict with:
        - `"y_obs"`: observed log-capital
        - `"y_pred_mean"`: posterior predictive mean log-capital per agent
        - `"y_pred_std"`: posterior predictive std-dev per agent
    """
    _LOG.debug(hprint.to_str("df.shape random_seed"))
    # Import PyMC only here, so that the simulation works without it.
    import pymc as pm  # type: ignore

    capital = df["capital"].to_numpy(dtype=float)
    y_obs = np.log(capital)
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=random_seed,
            progressbar=False,
        )
    # `ppc` is an `arviz.InferenceData`, and `y_obs` lives inside the
    # `posterior_predictive` group and has shape (chains, draws, n) or
    # (draws, n) depending on PyMC version.
    y_sim = np.asarray(ppc["posterior_predictive"]["y_obs"])
    if y_sim.ndim == 3:
        # (chains, draws, n) -> (chains * draws, n).
        y_sim = y_sim.reshape(-1, y_sim.shape[-1])
    elif y_sim.ndim == 2:
        # (draws, n) -> OK.
        pass
    else:
        raise ValueError(f"Unexpected PPC shape for y_obs: {y_sim.shape}")
    y_pred_mean = y_sim.mean(axis=0)
    y_pred_std = y_sim.std(axis=0)
    _LOG.debug("y_sim.shape=%s", y_sim.shape)
    ppc_results = {
        "y_obs": y_obs,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std,
    }
    return ppc_results


# #############################################################################
# Plotting
# #############################################################################


def plot_talent_capital_luck(
    df: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the talent distribution, the capital distribution, and luck vs capital.

    The three panels show that talent is roughly symmetric while capital is
    skewed, and how much of log-capital lines up with lucky events.

    :param df: output of `get_results_dataframe()`
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug("df.shape=%s figsize=%s", df.shape, figsize)
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    plot_df = df.assign(log_capital=np.log(df["capital"]))
    gini = calculate_gini(df["capital"].to_numpy())
    lucky_events = plot_df["lucky_events"].to_numpy(dtype=float)
    log_capital = plot_df["log_capital"].to_numpy(dtype=float)
    corr = float(np.corrcoef(lucky_events, log_capital)[0, 1])
    n_bins = 20
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # Plot the talent distribution with its mean.
    sns.histplot(data=plot_df, x="talent_norm", bins=n_bins, ax=axes[0])
    talent_mean = df["talent_norm"].mean()
    axes[0].axvline(
        talent_mean,
        color="red",
        linestyle="--",
        label=f"Mean: {talent_mean:.3f}",
    )
    axes[0].set_xlabel("Talent norm")
    axes[0].set_title(
        f"Talent (skewness {scipy.stats.skew(df['talent_norm']):.2f})"
    )
    axes[0].legend()
    # Plot the capital distribution with its mean and Gini.
    sns.histplot(
        data=plot_df, x="capital", bins=n_bins, color="orange", ax=axes[1]
    )
    capital_mean = df["capital"].mean()
    axes[1].axvline(
        capital_mean,
        color="red",
        linestyle="--",
        label=f"Mean: {capital_mean:.2f}",
    )
    axes[1].set_xlabel("Capital ($)")
    axes[1].set_title(
        f"Capital (Gini {gini:.3f}, skewness "
        f"{scipy.stats.skew(df['capital']):.2f})"
    )
    axes[1].legend()
    # Plot lucky events against log-capital with a linear fit.
    sns.regplot(
        data=plot_df,
        x="lucky_events",
        y="log_capital",
        ci=None,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red", "linestyle": "--"},
        ax=axes[2],
    )
    axes[2].set_xlabel("Lucky events")
    axes[2].set_ylabel("Log(capital)")
    axes[2].set_title(f"Luck vs log-capital (r = {corr:.3f})")
    fig.suptitle(f"{len(df)} agents after the simulation")
    fig.tight_layout()
    plt.show()
