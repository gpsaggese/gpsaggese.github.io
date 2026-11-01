# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gymnasium Registry API
#
# - This notebook explores the `gymnasium` registry, from the global catalog of env
#   IDs to registering custom envs
# - The pedagogical arc:
#   - `gym.envs.registry` and `EnvSpec`: what the catalog stores
#   - `gym.spec()`, `gym.make()`, and `gym.register()`: read, build, and add
#   - `gym.pprint_registry()` and the ID naming convention
#   - Custom `kwargs`, versions, and the full lifecycle of an env
# - References:
#   - API: https://gymnasium.farama.org/api/registry
#   - GitHub: https://github.com/Farama-Foundation/Gymnasium

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete

# %%
import gymnasium_utils as utils
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**:
#   - Every environment must be registered before `gym.make()` can create it
#   - The registry is the central catalog that maps string IDs (e.g.,
#     `"CartPole-v1"`) to their implementations
#
# - **Key abstractions**:
#   - `gym.envs.registry`: a plain `dict[str, EnvSpec]` holding all registered
#     environments
#   - `EnvSpec`: a metadata object containing the entry point, version, kwargs,
#     and configuration for one environment
#   - `gym.make()`: the factory that reads a spec, instantiates the env, and
#     wraps it with standard wrappers
#   - `gym.register()`: the function that adds a new entry to the registry
#
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `gym.register("id")` | Add an env to the catalog | Stores `registry["id"] = EnvSpec(...)` |
# | `gym.make("id")` | Build an env from its spec | Calls `entry_point(**kwargs)`, then adds wrappers |
# | `gym.spec("id")` | Look up a spec | Returns `registry["id"]` |
# | `gym.envs.registry` | Global catalog | `dict[str, EnvSpec]` |
#
# - **Key classes**:
#   - `EnvSpec`: holds id, entry_point, version, kwargs, and metadata
#   - `WrapperSpec`: describes an additional wrapper to apply
#   - `gym.envs.registry`: the global registry dict

# %% [markdown]
# # Part 1: The registry and `EnvSpec`

# %% [markdown]
# ## Cell 1.1: `gym.envs.registry`: the global catalog
#
# - **Mental model**:
#   - A plain Python `dict` that maps environment string IDs to `EnvSpec` objects
#   - Every env that `gym.make()` can create must have an entry here
#
# - **Key insight**:
#   - This is the single source of truth for available environments

# %%
# The registry is a regular dict[str, EnvSpec].
reg = gym.envs.registry
print("type(reg):", type(reg))
print("len(reg):", len(reg))

# %%
# List the first 10 environment IDs to see naming conventions.
all_ids = sorted(reg.keys())
sample_ids = all_ids[:10]
display(pd.DataFrame({"registered_id": sample_ids}))

# %%
# The naming convention is: [namespace/]EnvName-v[version]
# E.g., "CartPole-v1", "phys2d/CartPole-v0"
# Show a few namespaced envs.
namespaced = [env_id for env_id in all_ids if "/" in env_id]
display(pd.DataFrame({"namespaced_ids": namespaced}))

# %% [markdown]
# ## Cell 1.2: `EnvSpec`: the metadata object
#
# - **Mental model**:
#   - Each entry in the registry is an `EnvSpec`, a container of metadata that
#     tells `gym.make()` how to construct the environment
#
# - An `EnvSpec` stores:
#   - `id`: the string identifier
#   - `entry_point`: the class path or callable to instantiate
#   - `version`: integer version number parsed from the id
#   - `kwargs`: default keyword arguments passed to the constructor
#   - `max_episode_steps`: optional time limit
#   - `reward_threshold`: performance target for solving the env
#   - `nondeterministic`: whether the env is stochastic
#   - `namespace`: optional prefix namespace
#   - `order_enforce`: whether to enforce call ordering
#   - `disable_env_checker`: whether to skip the passive checker wrapper
#   - `additional_wrappers`: extra wrappers applied on top
#   - `vector_entry_point`: optional vectorized version

# %%
# Get an EnvSpec and inspect its type and attributes.
spec = gym.spec("CartPole-v1")
print("type(spec):", type(spec))
print()

# List all public methods.
hintros.print_obj_info(gym.envs.registration.EnvSpec)

# %%
# Build a DataFrame comparing multiple EnvSpecs.
specs = [
    gym.spec("CartPole-v1"),
    gym.spec("MountainCar-v0"),
    gym.spec("FrozenLake-v1"),
]
fields = [
    "id",
    "entry_point",
    "max_episode_steps",
    "reward_threshold",
    "version",
]
data = {f: [getattr(s, f) for s in specs] for f in fields}
data["id"] = [s.id for s in specs]
display(pd.DataFrame(data))

# %%
# EnvSpec has a .pprint() method for quick inspection.
spec.pprint()

# %% [markdown]
# ## Cell 1.3: `gym.spec()`: look up by ID
#
# - **Mental model**:
#   - A convenience function that retrieves an `EnvSpec` from the registry by its
#     string ID
#   - Equivalent to `registry[id]` but with error handling and parsing

# %%
# Look up a spec from its string ID.
spec = gym.spec("CartPole-v1")
print("id:", spec.id)
print("entry_point:", spec.entry_point)
print("max_episode_steps:", spec.max_episode_steps)
print("version:", spec.version)

# %% [markdown]
# # Part 2: Building and registering envs

# %% [markdown]
# ## Cell 2.1: `gym.make()`: from registry to environment
#
# - **Mental model**: `gym.make()` is the factory that:
#   1. Looks up the ID in the registry (via `gym.spec()`)
#   2. Instantiates the env from `entry_point`
#   3. Applies wrappers (`TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`)
#   4. Returns the fully wrapped environment
#
# - The kwargs passed to `gym.make()` are forwarded to the env constructor,
#   merged with the `EnvSpec.kwargs`

# %%
# Simplest usage: just pass an ID.
env = gym.make("CartPole-v1")
print("type(env):", type(env))
print("env:", env)
print("env.spec:", env.spec)
env.close()

# %%
# Pass keyword arguments that override the spec defaults.
# CartPole doesn't have spec kwargs, but we can show the mechanism.
env = gym.make("CartPole-v1", render_mode=None)
print("env.spec.kwargs:", env.spec.kwargs)
env.close()

# %%
# Customize max_episode_steps via gym.make().
env = gym.make("CartPole-v1", max_episode_steps=100)
print("env.spec.max_episode_steps:", env.spec.max_episode_steps)
# Verify the TimeLimit wrapper uses 100 steps.
print("env (wrapped):", env)
env.close()

# %%
# gym.make() can also accept an EnvSpec directly instead of a string ID.
spec = gym.spec("CartPole-v1")
env = gym.make(spec)
print("type(env):", type(env))
print("env.spec.id:", env.spec.id)
env.close()

# %%
# gym.make() with a module prefix: "module:Env-v0" form.
# This is useful for environments defined in external packages.
# The module is imported first, then the env is created.
# (Example with a built-in module to demonstrate the syntax.)
env = gym.make("gymnasium.envs.classic_control.cartpole:CartPole-v1")
print("env created via module prefix:", env)
env.close()

# %% [markdown]
# ## Cell 2.2: `gym.register()`: add your own environment
#
# - **Mental model**: registers a new environment in the global registry so it
#   can be created with `gym.make()`
#
# - Required: `id` (string) and `entry_point` (class or string path)
# - Optional: `max_episode_steps`, `reward_threshold`, `kwargs`, etc.


# %%
# Define a minimal custom environment.
class SimpleGrid(Env):
    """
    A 1D grid where the agent must reach the rightmost cell.
    """

    def __init__(self, size: int = 5) -> None:
        super().__init__()
        self._size = size
        self._pos = 0
        self.observation_space = Discrete(size)
        self.action_space = Discrete(2)  # 0 = left, 1 = right

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0
        return self._pos, {}

    def step(self, action):
        # Move right (1) or left (0), clamped to bounds.
        self._pos += 1 if action == 1 else -1
        self._pos = int(np.clip(self._pos, 0, self._size - 1))
        terminated = self._pos == self._size - 1
        reward = 1.0 if terminated else 0.0
        return self._pos, reward, terminated, False, {}


# %%
# Register the custom environment.
gym.register(id="SimpleGrid-v0", entry_point=SimpleGrid, max_episode_steps=20)

# Verify it appears in the registry.
print("SimpleGrid-v0 in registry:", "SimpleGrid-v0" in gym.envs.registry)

# %%
# Create the registered env via gym.make().
env = gym.make("SimpleGrid-v0", size=5)
obs, _ = env.reset(seed=0)
print("initial obs:", obs)
# Walk right until done.
done = False
while not done:
    obs, reward, terminated, truncated, _ = env.step(1)
    print(f"obs={obs}, reward={reward}, terminated={terminated}")
    done = terminated or truncated
env.close()

# %% [markdown]
# # Part 3: Browsing the registry

# %% [markdown]
# ## Cell 3.1: `gym.pprint_registry()`: pretty-print the catalog
#
# - **Mental model**: prints a formatted, columnar view of all registered
#   environments, organized by namespace

# %%
# By default, pprint_registry prints to console and returns None.
result = gym.pprint_registry(num_cols=4)
print("return value:", result)

# %%
# Use disable_print=True to get the string without printing it.
registry_text = gym.pprint_registry(num_cols=4, disable_print=True)
print("type(registry_text):", type(registry_text))
# Show the first 1000 characters.
print("registry_text[:1000]:\n", registry_text[:1000])

# %%
# Exclude specific namespaces (e.g., phys2d, tabular) to focus on core envs.
registry_text = gym.pprint_registry(
    num_cols=4,
    exclude_namespaces=["phys2d", "tabular"],
    disable_print=True,
)
print("registry_text:\n", registry_text)

# %%
# Print a custom subset of the registry.
subset = {k: v for k, v in gym.envs.registry.items() if "CartPole" in k}
subset_text = gym.pprint_registry(
    print_registry=subset, num_cols=2, disable_print=True
)
print("subset_text:\n", subset_text)

# %% [markdown]
# ## Cell 3.2: Namespace and versioning convention
#
# - **Mental model**: environment IDs follow the pattern
#   `[namespace/]EnvName-v[version]`
#   - `namespace`: optional prefix for grouping (e.g., `phys2d/`, `tabular/`)
#   - `EnvName`: CamelCase environment name
#   - `-vN`: version number suffix starting from 0
#
# - Multiple versions of the same env can coexist (e.g., `CartPole-v0` and
#   `CartPole-v1`) with different reward thresholds, episode lengths, etc.

# %%
# Show all versions of CartPole.
cartpole_specs = [
    gym.spec(k) for k in gym.envs.registry if "CartPole" in k and "/" not in k
]
data = {
    "id": [s.id for s in cartpole_specs],
    "version": [s.version for s in cartpole_specs],
    "max_episode_steps": [s.max_episode_steps for s in cartpole_specs],
    "reward_threshold": [s.reward_threshold for s in cartpole_specs],
}
display(pd.DataFrame(data))

# %%
# Show namespace-prefixed envs.
ns_envs = [k for k in gym.envs.registry if "phys2d" in k]
ns_specs = [gym.spec(k) for k in ns_envs]
data = {
    "id": [s.id for s in ns_specs],
    "namespace": [s.namespace for s in ns_specs],
    "entry_point": [s.entry_point for s in ns_specs],
}
display(pd.DataFrame(data))

# %% [markdown]
# # Part 4: Composition

# %% [markdown]
# ## Cell 4.1: Register with custom kwargs
#
# - The `kwargs` parameter in `gym.register()` sets default constructor
#   arguments, merged with any kwargs passed to `gym.make()` (make kwargs take
#   precedence)


# %%
class ConfigurableGrid(Env):
    """
    A grid env whose size and reward are configurable.
    """

    def __init__(self, size: int = 5, goal_reward: float = 1.0) -> None:
        super().__init__()
        self._size = size
        self._goal_reward = goal_reward
        self._pos = 0
        self.observation_space = Discrete(size)
        self.action_space = Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0
        return self._pos, {}

    def step(self, action):
        self._pos += 1 if action == 1 else -1
        self._pos = int(np.clip(self._pos, 0, self._size - 1))
        terminated = self._pos == self._size - 1
        reward = self._goal_reward if terminated else 0.0
        return self._pos, reward, terminated, False, {}


# Register with default kwargs: default size=5, goal_reward=10.0.
gym.register(
    id="ConfigurableGrid-v0",
    entry_point=ConfigurableGrid,
    kwargs={"size": 5, "goal_reward": 10.0},
    max_episode_steps=10,
)

# %%
# Create with default kwargs from registration.
env = gym.make("ConfigurableGrid-v0")
print("spec.kwargs:", env.spec.kwargs)
obs, _ = env.reset(seed=0)
print("initial obs:", obs)
# Walk right to the goal (needs 4 steps from position 0 to 4).
for _ in range(5):
    obs, reward, terminated, truncated, _ = env.step(1)
    if terminated:
        print(f"Goal! obs={obs}, reward={reward}")
        break
env.close()

# %%
# Override kwargs at make() time.
env = gym.make("ConfigurableGrid-v0", size=3, goal_reward=5.0)
print("spec.kwargs:", env.spec.kwargs)
steps, total_reward = utils.run_episode(env, seed=0, policy=lambda _obs: 1)
print(f"Goal! obs={steps[-1].obs}, total_reward={total_reward}")
env.close()

# %% [markdown]
# ## Cell 4.2: Multiple versions of the same env
#
# - Register two versions of the same environment with different settings,
#   following the `-vN` convention

# %%
gym.register(
    id="SimpleGrid-v1",
    entry_point=SimpleGrid,
    kwargs={"size": 10},
    max_episode_steps=100,
    reward_threshold=1.0,
)
gym.register(
    id="SimpleGrid-v2",
    entry_point=SimpleGrid,
    kwargs={"size": 3},
    max_episode_steps=5,
    reward_threshold=1.0,
)

# Compare specs.
v1 = gym.spec("SimpleGrid-v1")
v2 = gym.spec("SimpleGrid-v2")
data = {
    "field": ["id", "kwargs", "max_episode_steps", "reward_threshold"],
    "v1": [v1.id, v1.kwargs, v1.max_episode_steps, v1.reward_threshold],
    "v2": [v2.id, v2.kwargs, v2.max_episode_steps, v2.reward_threshold],
}
display(pd.DataFrame(data))

# %% [markdown]
# ## Cell 4.3: Full workflow: register, list, spec, make, use
#
# - The complete lifecycle of an environment through the registry


# %%
class WindyGrid(Env):
    """
    A 1D grid where a random wind pushes the agent left sometimes.
    """

    def __init__(self, size: int = 5, wind_p: float = 0.2) -> None:
        super().__init__()
        self._size = size
        self._wind_p = wind_p
        self._pos = 0
        self.observation_space = Discrete(size)
        self.action_space = Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 0
        return self._pos, {}

    def step(self, action):
        # Action moves the agent.
        self._pos += 1 if action == 1 else -1
        # Wind occasionally pushes left.
        if self.np_random.random() < self._wind_p:
            self._pos -= 1
        self._pos = int(np.clip(self._pos, 0, self._size - 1))
        terminated = self._pos == self._size - 1
        reward = 1.0 if terminated else 0.0
        return self._pos, reward, terminated, False, {}


# %%
# Step 1: Register.
gym.register(
    id="WindyGrid-v0",
    entry_point=WindyGrid,
    kwargs={"size": 6, "wind_p": 0.3},
    max_episode_steps=50,
    nondeterministic=True,
)

# Step 2: Verify it appears in the list.
print("WindyGrid-v0 in registry:", "WindyGrid-v0" in gym.envs.registry)

# Step 3: Inspect the spec.
spec = gym.spec("WindyGrid-v0")
print("spec.id:", spec.id)
print("spec.nondeterministic:", spec.nondeterministic)

# Step 4: Make the env.
env = gym.make("WindyGrid-v0", size=6)

# Step 5: Use the env.
steps, total_reward = utils.run_episode(env, seed=42, policy=lambda _obs: 1)
outcome = "Reached goal" if steps[-1].terminated else "Truncated"
print(f"{outcome} after {len(steps)} steps, total reward: {total_reward}")
env.close()

# %% [markdown]
# # Part 5: API patterns

# %% [markdown]
# ## Cell 5.1: Factory and spec patterns
#
# - **Factory Pattern**: `gym.register()` / `gym.make()` follow classic factory:
#   - Register associates a name with a construction recipe (entry_point + kwargs)
#   - Make uses the recipe to create instances
#
# - **Spec Pattern**: `EnvSpec` is a configuration object that:
#   - Holds all metadata separate from the env instance
#   - Can be serialized with `to_json()` / `from_json()`
#   - Multiple envs can share the same spec structure

# %%
# EnvSpec serialization.
spec = gym.spec("CartPole-v1")
json_str = spec.to_json()
print("JSON:", json_str[:200])

# Restore from JSON.
restored = gym.envs.registration.EnvSpec.from_json(json_str)
print("restored.id:", restored.id)
print("restored.max_episode_steps:", restored.max_episode_steps)

# %%
# EnvSpec.make() is a shortcut equivalent to gym.make(spec).
spec = gym.spec("CartPole-v1")
env = spec.make()
print("type(env):", type(env))
print("env.spec.id:", env.spec.id)
env.close()

# %% [markdown]
# # Part 6: Interactive exploration

# %% [markdown]
# ## Cell 6.1: Explore the registry
#
# - Try modifying the cells below to explore the registry

# %%
# Explore: how many envs are in each namespace?
reg = gym.envs.registry
namespaces = {}
for env_id in reg:
    ns = env_id.split("/")[0] if "/" in env_id else "(root)"
    namespaces[ns] = namespaces.get(ns, 0) + 1
ns_df = pd.DataFrame(
    {"namespace": list(namespaces.keys()), "count": list(namespaces.values())}
)
display(ns_df.sort_values("count", ascending=False))

# %%
# Explore: what happens if you register an already-registered ID?
# Gymnasium does not raise: it warns and replaces the spec. Use a new ID so that
# `CartPole-v1` stays intact for the next cells.
gym.register(
    id="ShortCartPole-v0",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=50,
)
print("first register:", gym.spec("ShortCartPole-v0").max_episode_steps)
# Register the same ID again without `max_episode_steps`: the old value is lost.
gym.register(
    id="ShortCartPole-v0",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
)
print("second register:", gym.spec("ShortCartPole-v0").max_episode_steps)

# %%
# Explore: what fields does an EnvSpec of a toy-text env look like?
spec = gym.spec("FrozenLake-v1")
spec.pprint(print_all=True)
print("spec.kwargs:", spec.kwargs)

# %%
# Explore: what is the difference between CartPole-v0 and CartPole-v1?
specs = [gym.spec("CartPole-v0"), gym.spec("CartPole-v1")]
data = {}
for attr_name in ["id", "version", "max_episode_steps", "reward_threshold"]:
    data[attr_name] = [getattr(s, attr_name) for s in specs]
display(pd.DataFrame(data, index=["v0", "v1"]))

# %% [markdown]
# ## Summary: The Mental Model
#
# - The Gymnasium Registry is a central `dict[str, EnvSpec]` that maps
#   environment IDs to their metadata and constructors
# - `gym.register()` adds entries (by ID + entry_point), `gym.spec()` looks them
#   up, and `gym.make()` consumes them to produce fully wrapped environment
#   instances
# - `EnvSpec` holds all configuration (kwargs, reward threshold, episode limit,
#   version, namespace) separate from the env object itself, enabling
#   serialization, inspection, and reuse
# - The naming convention `[namespace/]EnvName-v[version]` organizes envs into
#   groups and allows multiple versions of the same environment to coexist
