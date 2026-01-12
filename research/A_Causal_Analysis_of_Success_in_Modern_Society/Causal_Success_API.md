# Causal Success Analysis: API Documentation

## Overview

This document explains the core functions and classes in
`causal_success_utils.py`. If you're new to the project, this is where you'll
learn what each function does and how to use it. The code implements an
agent-based simulation where a population of individuals (agents) experience
random events over time, and we measure how inequality emerges from these
events.

Think of this as the technical manual. For a step-by-step walkthrough of the
complete analysis, check out `causal_success_example.md` instead. For a
higher-level narrative and motivation, see `causal_success_tutorial.md`.

## The Agent Class

Every simulation needs agents. In our case, an agent represents a person with
certain abilities and resources.

### What an Agent Contains

Each agent has four main attributes that define their talents:

- **Intensity**: how active someone is. Higher intensity means more exposure to
  opportunities and risks. Think of it like surface area for luck: the more
  you're out there doing things, the more chances you have to encounter events
  (both good and bad).
- **IQ**: ability to capitalize on opportunities. When a lucky event happens,
  this determines whether the agent can actually take advantage of it. Not every
  opportunity works out, even when it appears.
- **Networking**: social connections. This lets beneficial events sometimes
  spill over to others. If you know the right people, you might catch
  opportunities that weren't directly aimed at you.
- **Initial Capital**: starting wealth. We usually set this to `1.0` for
  everyone so that inequality emerges from the simulation dynamics, not from
  inherited advantages.

Beyond these talents, each agent tracks:

- `capital`: current wealth
- `capital_history`: list of capital values over time
- `lucky_events` and `unlucky_events`: event counts

### Creating a Single Agent

  ```python
  from causal_success_utils import Agent

  # Create an Agent with Specific Talents
  person = Agent(
      agent_id=0,
      intensity=0.6,     # Pretty active (range: 0 to 1)
      iq=0.7,            # Good at seizing opportunities
      networking=0.5,    # Average connections
      initial_capital=1.0
  )

  print(person.talent)
  # {'intensity': 0.6, 'iq': 0.7, 'networking': 0.5, 'initial_capital': 1.0}

  prob = person.get_event_probability()
  print(f"Event exposure probability: {prob:.3f}")
  # Event Exposure Probability: 0.648
  ```

Event exposure probability is computed via a sigmoid centered at 0.5, so
intensity values above 0.5 lead to above-average exposure and vice versa.

### Applying Events to Agents

Events change an agent's capital through **multiplication**, not addition. This
is what creates compounding effects.

  ```python
  # Apply a lucky event (25% gain)
  person.apply_event("lucky", 0.25)
  print(f"Capital after lucky event: ${person.capital:.2f}")
  # Capital after lucky event: $1.25

  # Apply an unlucky event (15% loss)
  person.apply_event("unlucky", 0.15)
  print(f"Capital after unlucky event: ${person.capital:.2f}")
  # Capital after unlucky event: $1.06

  print(person.lucky_events, person.unlucky_events)
  # 1 1
  ```

A 25% gain on $1 is $0.25, but a 25% gain on $100 is $25. This multiplicative
process is what drives inequality over time.

## Population Functions

### `create_population`

  ```python
  from causal_success_utils import create_population

  agents = create_population(n_agents=100, seed=42)
  print(f"Population size: {len(agents)}")
  # Population size: 100
  ```

**Behavior:**

- Generates `n_agents` agents
- `intensity`, `iq`, and `networking` are drawn from `N(0.5, 0.15)` and clipped
  to `[0, 1]`
- `initial_capital` defaults to `1.0` for everyone

**Parameters:**

- `n_agents: int`: number of agents
- `seed: int`: RNG seed for reproducibility

## Running the Simulation

### `run_simulation`

This is the core engine that moves the system forward over time.

```python
from causal_success_utils import run_simulation, create_population

agents = create_population(n_agents=100, seed=42)

agents = run_simulation(
    agents=agents,
    n_periods=80,
    n_lucky_events_per_period=5,
    n_unlucky_events_per_period=5,
    lucky_mean=0.25,
    lucky_std=0.08,
    unlucky_mean=0.15,
    unlucky_std=0.05,
    seed=42,
    verbose=True
)

print("Simulation complete!")
print(f"Final capital range: ${min(a.capital for a in agents):.2f} "
      f"to ${max(a.capital for a in agents):.2f}")
```

**High-level logic per period:**

- For each lucky event:
  1. Compute exposure probabilities from `get_event_probability()`
     (intensity-based)
  2. Select an agent at random, weighted by exposure
  3. Draw an impact (percent gain), clipped between 5% and 50%
  4. With probability equal to the agent's `iq`, apply the multiplicative gain
  5. With 10% probability, spill over to another agent, weighted by `networking`
     and gated by their `iq`

- For each unlucky event:
  1. Same exposure mechanism
  2. Draw impact (percent loss), clipped between 5% and 30%
  3. Apply multiplicative loss to capital, floored at `0.01`

**Key parameters:**

- `n_periods`: number of time periods
- `n_lucky_events_per_period`, `n_unlucky_events_per_period`
- `lucky_mean`, `lucky_std`, `unlucky_mean`, `unlucky_std`: mean and std of
  event magnitudes
- `seed`: RNG seed
- `verbose`: if `True`, uses `tqdm` to show a progress bar (if available)

### `run_policy_simulation`

Adds an initial resource allocation step before running the standard simulation.

```python
from causal_success_utils import create_population, run_policy_simulation

agents = create_population(n_agents=100, seed=42)

agents = run_policy_simulation(
    agents=agents,
    policy="egalitarian",
    resource_amount=100.0,
    n_periods=80,
    seed=42
)
```

**Policies:**

- `"egalitarian"`: equal split of `resource_amount` across agents
- `"meritocratic"`: resources proportional to `talent_norm`
- `"performance"`: resources proportional to current capital (rich-get-richer)
- `"random"`: one randomly chosen agent receives all the resources
- `"cate_optimal"`: resources proportional to (non-negative) CATE estimates
  passed in cate_values
- **Arguments:**

- `agents: List[Agent]`: population of agents to allocate resources to and then
  simulate
- `policy: str`: one of `"egalitarian"`, `"meritocratic"`, `"performance"`,
  `"random"`, or `"cate_optimal"`
- `resource_amount: float`: total resource budget to distribute at time 0
- `cate_values: Optional[np.ndarray]`: required when `policy="cate_optimal"`;
  1D array of CATE estimates (one per agent)
- `**simulation_kwargs`: additional keyword arguments passed through to
  `run_simulation` (e.g., `n_periods`, `seed`, etc.)

## Analysis Functions

### `calculate_gini`

Compute the Gini coefficient of a non-negative 1D array.

  ```python
  from causal_success_utils import calculate_gini
  import numpy as np

  capital_values = np.array([1.5, 2.0, 8.5, 12.3, 25.7])
  gini = calculate_gini(capital_values)
  print(f"Gini coefficient: {gini:.3f}")
  ```

Returns a float in `[0, 1]`.

### `get_results_dataframe`

Convert a list of agents into a tidy `pandas.DataFrame`:

  ```python
  from causal_success_utils import get_results_dataframe

  df = get_results_dataframe(agents)
  print(df.head())
  print(df.columns)
  ```

Columns include:

- `id`
- `talent_intensity`, `talent_iq`, `talent_networking`
- `initial_capital`
- `talent_norm`
- `capital`
- `lucky_events`, `unlucky_events`, `net_events`

### `generate_summary_statistics`

Compute key summary statistics from the simulation:

  ```python
  from causal_success_utils import generate_summary_statistics

  stats = generate_summary_statistics(agents)
  for key, value in stats.items():
      print(f"{key:25s}: {value:.4f}")
  ```

Includes:

- `n_agents`
- `mean_capital`, `median_capital`, `std_capital`
- `min_capital`, `max_capital`, `capital_range`
- `gini_coefficient`
- `top_10_pct_share`, `top_20_pct_share`, `bottom_50_pct_share`
- `mean_lucky_events`, `mean_unlucky_events`
- `mean_talent_norm`

### `validate_simulation_results`

Basic sanity checks on the simulation output:

```python
from causal_success_utils import validate_simulation_results

try:
    validate_simulation_results(agents)
    print("All checks passed.")
except ValueError as e:
    print("Validation failed:", e)
```

Checks:

- No negative capital
- No NaN values
- Non-negative event counts
- Capital history length is consistent with total events per agent

## Complete Example Workflow

  ```python
  import numpy as np
  import pandas as pd
  from causal_success_utils import (
      create_population,
      run_simulation,
      get_results_dataframe,
      calculate_gini,
      generate_summary_statistics,
      validate_simulation_results,
  )

  agents = create_population(n_agents=100, seed=42)
  agents = run_simulation(agents, n_periods=80, seed=42, verbose=True)

  validate_simulation_results(agents)

  df = get_results_dataframe(agents)
  gini = calculate_gini(df["capital"].values)
  stats = generate_summary_statistics(agents)

  print("Gini:", gini)
  print("Top 10% share:", stats["top_10_pct_share"])
  ```

## Bayesian Inference API

In addition to the simulation and descriptive statistics, the project includes a
**Bayesian regression layer** that estimates the effect of luck on log-capital
while controlling for talent.

All Bayesian functions live in `causal_success_utils.py` and rely on **PyMC**
and **ArviZ**. They are **optional**: if these libraries are not installed, you
can still run the simulation and summary functions.

### Overview of the Model

The Bayesian model is a linear regression on the log of final capital:

\[ \log(\text{capital}\_i) = \alpha

- \beta\_{\text{luck}} \cdot \text{lucky_events}\_i
- \beta\_{\text{intensity}} \cdot \text{talent_intensity}\_i
- \beta\_{\text{iq}} \cdot \text{talent_iq}\_i
- \beta\_{\text{networking}} \cdot \text{talent_networking}\_i
- \varepsilon_i \]

- `beta_luck` Is the Primary Quantity of Interest: the (Log-Scale) Effect of
  One additional lucky event, holding talent constant.
- Priors Are Weakly Informative, Centered at 0.

### `fit_bayesian_luck_model`

  ```python
  from causal_success_utils import (
      create_population,
      run_simulation,
      get_results_dataframe,
      fit_bayesian_luck_model,
  )

  # Simulate Data
  agents = create_population(n_agents=100, seed=42)
  agents = run_simulation(agents, n_periods=80, seed=42)
  df = get_results_dataframe(agents)

  # Fit Bayesian Model
  model, idata = fit_bayesian_luck_model(
      df,
      draws=1000,
      tune=1000,
      target_accept=0.9,
      random_seed=42,
  )
  ```

**Arguments:**

- `df: pd.DataFrame`: output from `get_results_dataframe`
- `draws: int`: posterior draws per chain (default 1000)
- `tune: int`: warm-up / burn-in iterations (default 1000)
- `target_accept: float`: NUTS target acceptance rate (default 0.9)
- `random_seed: int`: random seed

**Returns:**

- `model`: PyMC model object
- `idata`: ArviZ `InferenceData` with posterior samples

> Note: If PyMC or ArviZ are not installed, this function will raise an
> `ImportError`.

### `summarize_bayesian_fit`

Get a tidy summary table (posterior mean, standard deviation, and credible
intervals) for the main parameters.

```python
from causal_success_utils import summarize_bayesian_fit

summary = summarize_bayesian_fit(idata)
print(summary)
```

By default, it summarizes:

- `alpha`
- `beta_luck`
- `beta_intensity`
- `beta_iq`
- `beta_networking`
- `sigma`

You can also pass a custom `var_names` list if you want to restrict the summary.

### `posterior_predictive_check`

Run a basic posterior predictive check (PPC) by simulating log-capital from the
model and comparing to the observed log-capital.

  ```python
  from causal_success_utils import posterior_predictive_check

  ppc_results = posterior_predictive_check(model, idata, df)

  y_obs = ppc_results["y_obs"]
  y_pred_mean = ppc_results["y_pred_mean"]
  y_pred_std = ppc_results["y_pred_std"]

  print("Observed log-capital (first 5):", y_obs[:5])
  print("Predicted mean log-capital (first 5):", y_pred_mean[:5])
  ```

**Returns a dictionary with:**

- `"y_obs"`: observed log-capital
- `"y_pred_mean"`: posterior predictive mean log-capital per agent
- `"y_pred_std"`: posterior predictive standard deviation per agent

This is useful for checking how well the model captures the distribution of
outcomes.

## When to Use What

- Use the **simulation functions** (`create_population`, `run_simulation`,
  `run_policy_simulation`) to generate data and explore how talent and luck
  interact in the model world.
- Use the **summary and inequality functions** (`get_results_dataframe`,
  `generate_summary_statistics`, `calculate_gini`) to quantify emergent patterns
  (inequality, top shares, etc.).
- Use the **Bayesian functions** (`fit_bayesian_luck_model`,
  `summarize_bayesian_fit`, `posterior_predictive_check`) when you want:
  - A principled posterior over the effect of luck,
  - Credible intervals around effect sizes,
  - And basic model checking via posterior predictive simulations.

Together, these APIs give you a complete pipeline from simulation → descriptive
analysis → causal and Bayesian inference.
