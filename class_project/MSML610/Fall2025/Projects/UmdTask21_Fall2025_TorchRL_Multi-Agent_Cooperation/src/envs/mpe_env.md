# `mpe_env.py` — PettingZoo MPE Factory

## Purpose
Creates a **parallel PettingZoo MPE environment** via `make_mpe_env(...)`, configured for `simple_spread_v3`. It centralizes scenario choice and hyperparameters like agent count, local reward ratio, and horizon. See the project goals and CTDE design in the README.

## Why it matters
- Single source of truth for **environment config**, ensuring reproducible experiments.
- Encapsulates **render mode**, `continuous_actions=False` for discrete control, and a fixed seed.

## Key API
- `make_mpe_env(render_mode=None, N=3, local_ratio=0.5, max_cycles=25) -> env`  Returns a **parallel_env** with `.reset()` and `.step(actions_dict)`.
- (Helpers) `run_random_episode()`, `print_env_specs()` — useful for quick sanity checks.

## I/O shapes & contracts
- PettingZoo parallel env returns **dicts** keyed by agent names, which are later **tensorized** by `MPEWrapper`.
- Discrete action space configured for compatibility with `SharedMLPPolicy` (Categorical).

## How it connects
- Imported by **`MPEWrapper`** to obtain the actual env instance.
- Changing the scenario (e.g., `simple_reference`) or parameters happens here in one place.

## Common extensions
- Toggle `continuous_actions=True` to try continuous control (then adjust policy).
- Add wrappers for **reward shaping**, **noisy observations**, or **normalization**.
- Parameterize seed and scenario from a **config file / CLI**.