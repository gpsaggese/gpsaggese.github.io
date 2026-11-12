"""
Causal Success Analysis - Simulation and Inference Utilities.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.causal_success_utils as rcaoasalcsu
"""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

import helpers.hdbg as hdbg

# Optional Bayesian dependencies (simulation works without these).
# try:
import pymc as pm  # type: ignore
import arviz as az  # type: ignore
# except Exception:  # pragma: no cover - optional import.
#     pm = None
#     az = None

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
       - Higher intensity → higher probability of encountering events (both good and bad).
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
    ):
        """
        Initialize the Agent with talents and initial capital.

        :param agent_id: Unique agent identifier (typically agent's index)
        :param intensity: Intensity talent, affects event exposure (0-1)
        :param iq: IQ talent, affects ability to capitalize on luck (0-1)
        :param networking: Networking talent, affects spillover effects (0-1)
        :param initial_capital: Starting wealth level (default 1.0)
        """
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
        return float(np.linalg.norm(values))

    def get_event_probability(self) -> float:
        """
        Probability of encountering an event based on intensity.

        Uses a sigmoid centered at 0.5. Higher intensity = higher exposure.

        :return: Event probability in [0, 1]
        """
        alpha = 2.0
        return float(
            1.0 / (1.0 + np.exp(-alpha * (self.talent["intensity"] - 0.5)))
        )

    def apply_event(self, event_type: str, impact: float) -> None:
        """
        Apply an event to capital using multiplicative dynamics.

        :param event_type: "lucky" or "unlucky"
        :param impact: magnitude as a decimal (e.g., 0.25 = 25%)
        """
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


def create_population(n_agents: int = 100, *, seed: int = 42) -> List[Agent]:
    """
    Create a population of agents with normally distributed talents.

    :param n_agents: number of agents to create (default 100)
    :param seed: RNG seed for reproducibility (default 42)
    :return: List of Agent objects, each with random talents and capital=1.0
    """
    hdbg.dassert_lt(0, n_agents, "n_agents must be positive")
    rng = np.random.default_rng(seed)
    agents: List[Agent] = []
    for i in range(n_agents):
        intensity = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        iq = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        networking = float(np.clip(rng.normal(0.5, 0.15), 0.0, 1.0))
        agents.append(Agent(i, intensity, iq, networking, initial_capital=1.0))
    return agents


def calculate_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for non-negative values.

    The Gini coefficient measures inequality in a distribution (e.g., wealth).

    :param values: 1D array of non-negative values (e.g., capital amounts)
    :return: Gini coefficient in [0, 1]
    """
    x = np.asarray(values, dtype=float)
    hdbg.dassert_lt(
        0, x.size, "Cannot calculate Gini coefficient for empty array"
    )
    hdbg.dassert(
        not np.any(x < 0),
        "Gini coefficient requires non-negative values",
    )
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (
        n + 1.0
    ) / n
    return float(np.clip(gini, 0.0, 1.0))


def get_results_dataframe(agents: List[Agent]) -> pd.DataFrame:
    """
    Convert a list of agents to a DataFrame for analysis.

    :param agents: List of Agent objects
    :return: DataFrame with agent attributes
    """
    if not agents:
        return pd.DataFrame()
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
    return pd.DataFrame(rows)


def generate_summary_statistics(agents: List[Agent]) -> Dict[str, float]:
    """
    Generate comprehensive summary statistics for the simulation output.

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
    df = get_results_dataframe(agents)
    if df.empty:
        return {"n_agents": 0}
    capital = df["capital"].to_numpy(dtype=float)
    gini = calculate_gini(capital)
    min_cap = float(np.min(capital))
    max_cap = float(np.max(capital))
    total_cap = float(np.sum(capital))
    n = len(df)
    # Guard against division by zero (should not happen due to floor).
    cap_range = max_cap / max(min_cap, 1e-12)
    top_10_n = max(1, n // 10)
    top_20_n = max(1, n // 5)
    bottom_50_n = max(1, n // 2)
    return {
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
        "mean_lucky_events": float(df["lucky_events"].mean()),
        "mean_unlucky_events": float(df["unlucky_events"].mean()),
        "mean_talent_norm": float(df["talent_norm"].mean()),
    }


def validate_simulation_results(agents: List[Agent]) -> bool:
    """
    Validate simulation results for basic correctness.

    Raises AssertionError if anything looks inconsistent.

    :param agents: List of Agent objects to validate
    :return: True if validation passes
    """
    df = get_results_dataframe(agents)
    hdbg.dassert(not df.empty, "No agents provided to validate")
    hdbg.dassert(not (df["capital"] < 0).any(), "Negative capital detected")
    hdbg.dassert(not df.isnull().any().any(), "NaN values detected")
    hdbg.dassert(
        not ((df["lucky_events"] < 0).any() or (df["unlucky_events"] < 0).any()),
        "Negative event counts detected",
    )
    for a in agents:
        expected = 1 + a.lucky_events + a.unlucky_events
        hdbg.dassert_eq(
            len(a.capital_history),
            expected,
            "Agent has inconsistent capital history length (expected, got):",
            expected,
            len(a.capital_history),
        )
    return True


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
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> List[Agent]:
    """
    Execute the agent-based simulation over multiple periods.

    :param agents: List of Agent objects to simulate (modified in-place)
    :param n_periods: Number of time periods to simulate (default 80)
    :param n_lucky_events_per_period: Lucky events per period (default 5)
    :param n_unlucky_events_per_period: Unlucky events per period (default 5)
    :param lucky_mean: Mean impact of lucky events (default 0.25 = 25%)
    :param lucky_std: Std dev of lucky event impacts (default 0.08)
    :param unlucky_mean: Mean impact of unlucky events (default 0.15 = 15%)
    :param unlucky_std: Std dev of unlucky event impacts (default 0.05)
    :param seed: RNG seed for reproducibility (default 42)
    :param verbose: Show progress bar if True (requires tqdm, default False)
    :return: Same agents list with updated capital and event histories
    """
    hdbg.dassert_lt(0, n_periods, "n_periods must be positive")
    hdbg.dassert(agents, "agents list cannot be empty")
    hdbg.dassert(
        n_lucky_events_per_period >= 0 and n_unlucky_events_per_period >= 0,
        "event counts per period must be non-negative",
    )

    rng = np.random.default_rng(seed)
    n_agents = len(agents)
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore

            periods_iter = tqdm(
                range(n_periods), desc="Running simulation", unit="period"
            )
        except Exception:
            periods_iter = range(n_periods)
    else:
        periods_iter = range(n_periods)
    for _ in periods_iter:
        # Lucky events.
        for _ in range(n_lucky_events_per_period):
            exposure = np.array(
                [a.get_event_probability() for a in agents], dtype=float
            )
            exposure = (
                exposure / exposure.sum()
                if exposure.sum() > 0
                else np.ones(n_agents) / n_agents
            )
            selected_idx = int(rng.choice(n_agents, p=exposure))
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
            exposure = np.array(
                [a.get_event_probability() for a in agents], dtype=float
            )
            exposure = (
                exposure / exposure.sum()
                if exposure.sum() > 0
                else np.ones(n_agents) / n_agents
            )
            selected_idx = int(rng.choice(n_agents, p=exposure))
            selected = agents[selected_idx]
            impact = float(
                np.clip(rng.normal(unlucky_mean, unlucky_std), 0.05, 0.30)
            )
            selected.apply_event("unlucky", impact)
    return agents


def run_policy_simulation(
    agents: List[Agent],
    *,
    policy: str = "egalitarian",
    resource_amount: float = 100.0,
    cate_values: Optional[np.ndarray] = None,
    **simulation_kwargs,
) -> List[Agent]:
    """
    Allocate initial resources under a policy, then run the standard simulation.

    1. "egalitarian"
       - Every agent gets: resource_amount / n_agents
       - Rationale: Reduce initial inequality, give everyone equal chance
       - Typical outcome: Lowest final Gini (most equitable)
       - Typical outcome: Moderate total welfare

    2. "meritocratic"
       - Allocation ∝ talent_norm (total ability)
       - Rationale: Reward potentially capable people
       - Typical outcome: Moderate final Gini
       - Typical outcome: High total welfare (resources go to productive people)

    3. "performance"
       - Allocation ∝ current capital (rich get richer)
       - Rationale: Compound success (controversial, tested for comparison)
       - Typical outcome: Highest final Gini (most unequal)
       - Typical outcome: Lowest total welfare (resources wasted on already-rich)

    4. "random"
       - One randomly chosen agent gets ALL resources
       - Rationale: Extreme luck-based allocation
       - Typical outcome: Very high Gini
       - Typical outcome: Highest possible total welfare (concentrated resources)

    5. "cate_optimal"
       - Allocation ∝ CATE estimates (heterogeneous treatment effects)
       - Rationale: Give resources to agents who benefit most from them
       - Requires: cate_values array with one value per agent
       - Typical outcome: High total welfare, moderate Gini
       - Note: Only allocates to agents with non-negative CATE

    :param agents: List of Agent objects (capital modified in-place)
    :param policy: Allocation rule: "egalitarian", "meritocratic", "performance",
                   "random", or "cate_optimal" (default "egalitarian")
    :param resource_amount: Total budget to distribute at t=0 (default 100.0)
    :param cate_values: 1D array of CATE estimates, one per agent.
                        Required if policy="cate_optimal", ignored otherwise.
    :param simulation_kwargs: Additional arguments forwarded to run_simulation()
                              (e.g., n_periods=80, seed=42, verbose=True)
    :return: Same agents list after resource allocation and full simulation
    """
    hdbg.dassert(agents, "agents list cannot be empty")
    hdbg.dassert_lt(
        -0.0001, resource_amount, "resource_amount must be non-negative"
    )
    n = len(agents)
    rng = np.random.default_rng(simulation_kwargs.get("seed", None))
    # Handle random policy separately (single winner).
    if policy == "random":
        winner_idx = int(rng.integers(n))
        agents[winner_idx].capital += resource_amount
        agents[winner_idx].capital_history[0] = agents[winner_idx].capital
        return run_simulation(agents, **simulation_kwargs)
    # For all other policies, we compute weights and allocate proportionally.
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
            "cate_values must have length (expected, got):",
            n,
            cate_array.shape[0],
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
    for a, alloc in zip(agents, allocations):
        a.capital += float(alloc)
        # Keep history consistent at t=0.
        a.capital_history[0] = a.capital
    return run_simulation(agents, **simulation_kwargs)


# #############################################################################
# Bayesian model
# #############################################################################
# -------------------------------------------------------------------


def fit_bayesian_luck_model(
    df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42,
):
    """
    Fit a Bayesian regression model to estimate causal effect of luck on capital.

    :param df: DataFrame from get_results_dataframe(agents), must have:
               'capital', 'lucky_events', 'talent_intensity', 'talent_iq', 'talent_networking'
    :param draws: Number of posterior draws per chain (default 1000)
               Higher = more accurate posterior, longer runtime
    :param tune: Number of NUTS tuning/burn-in iterations (default 1000)
             These are discarded and not used in inference
    :param target_accept: NUTS sampler target acceptance rate (default 0.9)
                 Valid range: (0.5, 1.0), higher = slower but more stable
    :param random_seed: RNG seed for reproducibility (default 42)

    :return: Tuple (model, idata):
        - model: PyMC Model object (for diagnostics, re-sampling, etc.)
        - idata: ArviZ InferenceData object containing posterior samples
                Use with summarize_bayesian_fit() or posterior_predictive_check()
    """
    required_cols = [
        "capital",
        "lucky_events",
        "talent_intensity",
        "talent_iq",
        "talent_networking",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    hdbg.dassert(not missing, "DataFrame is missing required columns:", missing)
    capital = df["capital"].to_numpy(dtype=float)
    y = np.log(capital)  # log-capital is more stable and closer to normal.
    lucky = df["lucky_events"].to_numpy(dtype=float)
    intensity = df["talent_intensity"].to_numpy(dtype=float)
    iq = df["talent_iq"].to_numpy(dtype=float)
    networking = df["talent_networking"].to_numpy(dtype=float)
    # THE QUESTION
    # ============
    # Does luck causally affect outcomes, even after controlling for talent?
    # This model answers that by regressing log(capital) on both luck and talent.

    # THE MODEL
    # =========
    # Linear Bayesian regression:

    #     log(capital_i) = alpha
    #                      + beta_luck * lucky_events_i
    #                      + beta_intensity * talent_intensity_i
    #                      + beta_iq * talent_iq_i
    #                      + beta_networking * talent_networking_i
    #                      + epsilon_i

    # Where:
    #     - log(capital) is the outcome (log-scale for stability)
    #     - lucky_events is the treatment (how many beneficial events occurred)
    #     - talent_* are confounders (control for inherent ability)
    #     - epsilon ~ N(0, sigma) is residual error

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
    return model, idata


def summarize_bayesian_fit(
    idata, *, var_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Return a tidy summary table (posterior mean, sd, and credible intervals).

    For the Bayesian model parameters.

    :param idata: ArviZ InferenceData returned by fit_bayesian_luck_model
    :param var_names: optional subset of parameter names to summarize
    :return: pandas DataFrame with summary statistics (mean, sd, hdi, etc.)
    """
    if var_names is None:
        # By default, summarize the main coefficients and sigma.
        var_names = [
            "alpha",
            "beta_luck",
            "beta_intensity",
            "beta_iq",
            "beta_networking",
            "sigma",
        ]
    summary = az.summary(idata, var_names=var_names)
    return summary


def posterior_predictive_check(
    model,
    idata,
    df: pd.DataFrame,
    *,
    random_seed: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Simple posterior predictive check (PPC).

    This function draws from the posterior predictive distribution and compares
    simulated log-capital to the observed log-capital.

    :param model: PyMC model returned by fit_bayesian_luck_model
    :param idata: ArviZ InferenceData with posterior draws
    :param df: same DataFrame used for fitting
    :param random_seed: RNG seed for reproducibility
    :return: dict with:
        - "y_obs": observed log-capital
        - "y_pred_mean": posterior predictive mean log-capital per agent
        - "y_pred_std": posterior predictive std-dev per agent
    """
    capital = df["capital"].to_numpy(dtype=float)
    y_obs = np.log(capital)
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=random_seed,
            progressbar=False,
        )
    # `ppc` is an `arviz.InferenceData`; `y_obs` lives inside the
    # `posterior_predictive` group and has shape (chains, draws, n) or
    # (draws, n) depending on PyMC version.
    y_sim = np.asarray(ppc.posterior_predictive["y_obs"])
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
    return {
        "y_obs": y_obs,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std,
    }
