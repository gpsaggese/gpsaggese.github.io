"""
Causal Success Analysis - Simulation Utilities (with Bayesian inference).

This module is the main toolbox for the project. It provides:

1. An agent-based simulation where:
   - Agents have talents (intensity, IQ, networking, initial capital)
   - Wealth evolves through multiplicative lucky and unlucky events

2. Analysis helpers:
   - Inequality metrics (Gini, top shares)
   - Summary statistics
   - Basic validation checks

3. A simple Bayesian model (using PyMC, if available) that:
   - Regresses log(final capital) on:
       * number of lucky events (treatment)
       * talent dimensions (controls)
   - Returns a posterior over the "luck effect" and other coefficients

The Bayesian part is optional: if PyMC/ArviZ are not installed,
all simulation and summary functions still work as before.

Import as:

import research.A_Causal_Analysis_of_Success_in_Modern_Society.causal_success_utils as racaosimscsu
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# Optional Bayesian dependencies (simulation works without these).
try:
    import pymc as pm  # type: ignore
    import arviz as az  # type: ignore
except Exception:  # pragma: no cover - optional import.
    pm = None
    az = None

__all__ = [
    "Agent",
    "create_population",
    "calculate_gini",
    "get_results_dataframe",
    "generate_summary_statistics",
    "validate_simulation_results",
    "run_simulation",
    "run_policy_simulation",
    "fit_bayesian_luck_model",
    "summarize_bayesian_fit",
    "posterior_predictive_check",
]

# #############################################################################
# Agent
# #############################################################################

class Agent:
    """
    Agent representing an individual in the simulation.

    :param agent_id: Unique agent identifier
    :param intensity: Intensity talent dimension (0-1)
    :param iq: IQ talent dimension (0-1)
    :param networking: Networking talent dimension (0-1)
    :param initial_capital: Starting wealth level

    :ivar id: Unique agent identifier
    :ivar talent: dict of talent dimensions (intensity, iq, networking, initial_capital)
    :ivar capital: Current wealth level
    :ivar capital_history: List of capital values over time
    :ivar lucky_events: Count of beneficial events received
    :ivar unlucky_events: Count of detrimental events received
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


def create_population(n_agents: int = 100, *, seed: int = 42) -> List[Agent]:
    """
    Create a population of agents with normally distributed talents.

    intensity, iq, networking ~ N(0.5, 0.15) clipped to [0, 1].
    initial_capital defaults to 1.0 for all agents.

    :param n_agents: number of agents
    :param seed: RNG seed for reproducibility
    :return: List of Agent objects
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
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

    :param values: Array of non-negative values
    :return: Gini coefficient in [0, 1]
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot calculate Gini coefficient for empty array")
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values")
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
    Generate summary statistics for the simulation output.

    :param agents: List of Agent objects
    :return: Dictionary of summary statistics
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

    Raises ValueError if anything looks inconsistent.

    :param agents: List of Agent objects to validate
    :return: True if validation passes
    """
    df = get_results_dataframe(agents)
    if df.empty:
        raise ValueError("No agents provided to validate")
    if (df["capital"] < 0).any():
        raise ValueError("Negative capital detected")
    if df.isnull().any().any():
        raise ValueError("NaN values detected")
    if (df["lucky_events"] < 0).any() or (df["unlucky_events"] < 0).any():
        raise ValueError("Negative event counts detected")
    for a in agents:
        expected = 1 + a.lucky_events + a.unlucky_events
        if len(a.capital_history) != expected:
            raise ValueError(
                f"Agent {a.id} has inconsistent capital history length "
                f"(expected {expected}, got {len(a.capital_history)})"
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

    Notes:
        - lucky impact clipped to [0.05, 0.50]
        - unlucky impact clipped to [0.05, 0.30]
        - capital floored at 0.01
        - networking spillover: 10% chance, 50% impact

    :param agents: List of Agent objects to simulate
    :param n_periods: Number of simulation periods
    :param n_lucky_events_per_period: Lucky events per period
    :param n_unlucky_events_per_period: Unlucky events per period
    :param lucky_mean: Mean lucky event impact
    :param lucky_std: Std dev lucky event impact
    :param unlucky_mean: Mean unlucky event impact
    :param unlucky_std: Std dev unlucky event impact
    :param seed: RNG seed for reproducibility
    :param verbose: Enable progress bar if True
    :return: List of Agent objects after simulation
    """
    if n_periods <= 0:
        raise ValueError(f"n_periods must be positive, got {n_periods}")
    if not agents:
        raise ValueError("agents list cannot be empty")
    if n_lucky_events_per_period < 0 or n_unlucky_events_per_period < 0:
        raise ValueError("event counts per period must be non-negative")

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

    Policies:
        - egalitarian: equal distribution
        - meritocratic: proportional to talent_norm
        - performance: proportional to current capital
        - random: one random winner gets all
        - cate_optimal: proportional to non-negative CATE estimates (cate_values)

    :param agents: list of Agent objects
    :param policy: allocation rule
    :param resource_amount: total budget to allocate at t=0
    :param cate_values: array of CATE estimates (len = n_agents), required for "cate_optimal"
    :param simulation_kwargs: forwarded to run_simulation (e.g., n_periods, seed, etc.)
    :return: List of Agent objects after simulation
    """
    if not agents:
        raise ValueError("agents list cannot be empty")
    if resource_amount < 0:
        raise ValueError("resource_amount must be non-negative")
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
        if cate_values is None:
            raise ValueError(
                "cate_values must be provided when policy='cate_optimal'."
            )
        cate_array = np.asarray(cate_values, dtype=float)
        if cate_array.shape[0] != n:
            raise ValueError(
                f"cate_values must have length {n}, got {cate_array.shape[0]}."
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


# -------------------------------------------------------------------
# Bayesian model: effect of luck on log-capital, controlling for talent.
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
    Fit a simple Bayesian regression model:

        log(capital) ~ alpha
                       + beta_luck * lucky_events
                       + beta_intensity * talent_intensity
                       + beta_iq * talent_iq
                       + beta_networking * talent_networking
                       + epsilon

    The main quantity of interest is beta_luck: the (log-scale) effect of
    one additional lucky event on final capital, controlling for talent.

    :param df: DataFrame from get_results_dataframe(agents)
    :param draws: number of posterior draws per chain
    :param tune: number of warmup/burn-in iterations
    :param target_accept: NUTS target acceptance rate
    :param random_seed: RNG seed for reproducibility
    :return: (model, idata) where:
        - model is the PyMC model object
        - idata is an ArviZ InferenceData with posterior samples
    """
    if pm is None or az is None:
        raise ImportError(
            "PyMC / ArviZ are not available. Install them to use the Bayesian model, "
            "or skip this step if you only need the simulation."
        )
    required_cols = [
        "capital",
        "lucky_events",
        "talent_intensity",
        "talent_iq",
        "talent_networking",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    capital = df["capital"].to_numpy(dtype=float)
    y = np.log(capital)  # log-capital is more stable and closer to normal.
    lucky = df["lucky_events"].to_numpy(dtype=float)
    intensity = df["talent_intensity"].to_numpy(dtype=float)
    iq = df["talent_iq"].to_numpy(dtype=float)
    networking = df["talent_networking"].to_numpy(dtype=float)
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
    if az is None:
        raise ImportError(
            "ArviZ is not available. Install it to summarize Bayesian results."
        )
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
    if pm is None:
        raise ImportError(
            "PyMC is not available. Install it to run posterior predictive checks."
        )
    capital = df["capital"].to_numpy(dtype=float)
    y_obs = np.log(capital)
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=random_seed,
            progressbar=False,
        )
    # ppc["y_obs"] has shape (chains, draws, n) or (draws, n) depending on PyMC version.
    y_sim = np.asarray(ppc["y_obs"])
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
