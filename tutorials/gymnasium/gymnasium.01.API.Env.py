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
# # Gymnasium Env API
#
# - This notebook explores the `gymnasium.Env` API, building up from the smallest
#   working example on `CartPole-v1`
# - The pedagogical arc:
#   - `Env`, `reset()`, `step()`, and a full episode
#   - `Space`: a first look, in depth in notebook 03
#   - `render()`, `close()`, and the registry, in depth in notebook 02
#   - A custom `Env`
#   - Wrappers and vector envs: a first look, in depth in notebooks 04 and 05
# - References:
#   - API: https://gymnasium.farama.org/api/env
#   - GitHub: https://github.com/Farama-Foundation/Gymnasium

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

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
# - **What problem it solves**: RL algorithms need an environment (a world the
#   agent acts in). Without a standard interface, every environment has a different
#   API and agent code is not portable
#
# - **Key abstraction**: `gymnasium.Env`: a single class every environment
#   implements. It exposes four methods (`reset`, `step`, `render`, `close`) and
#   two attributes (`observation_space`, `action_space`)
#
# - **Mental model**: the agent calls `reset()` once, then loops calling `step()`
#   until `terminated or truncated`
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `gym.make(id)` | Build a registered env | Adds standard wrappers such as `TimeLimit` |
# | `env.reset(seed=...)` | Start an episode | Returns `(observation, info)` |
# | `env.step(action)` | Advance one timestep | Returns `(observation, reward, terminated, truncated, info)` |
# | `env.observation_space`, `env.action_space` | Describe valid data | `Space` objects |
#
# - **Key classes**:
#   - `gymnasium.Env`: base environment class
#   - `gymnasium.spaces.*`: describe valid observations and actions
#   - `gymnasium.Wrapper`: wrap an env to add behavior
#   - `gymnasium.vector.VectorEnv`: run many envs in parallel

# %% [markdown]
# # Part 1: The `gymnasium.Env` base class

# %% [markdown]
# ## Cell 1.1: Inspect `gymnasium.Env`
#
# - **Mental model**: an object that holds the state of a world. You send it
#   actions and it returns observations and rewards
#
# - Every builtin and custom environment is a subclass of `gymnasium.Env`

# %%
# Inspect what Env exposes as a markdown list.
hintros.print_obj_info(gym.Env)

# %% [markdown]
# ## Cell 1.2: Construct an `Env` with `gym.make()`
#
# - `gym.make(id)` is the standard factory: it looks up a registered environment
#   by string ID and returns a fully wrapped `Env`
# - Use `render_mode=None` (default) to skip visual rendering (safe in notebooks)

# %%
# Create the simplest classic-control environment.
env = gym.make("CartPole-v1", render_mode=None)
# `reset(seed=...)` does not seed the action space, so seed it here.
env.action_space.seed(0)
print("type(env):", type(env))
print("env:", env)

# %% [markdown]
# ## Cell 1.3: Inspect the env object
#
# - Two key attributes describe the interface contract:
#   - `observation_space`: what the env returns
#   - `action_space`: what the env accepts

# %%
# Print type and content of env attributes.
print("observation_space type:", type(env.observation_space))
print("observation_space:", env.observation_space)

print("\naction_space type:", type(env.action_space))
print("action_space:", env.action_space)

print("\nspec:", env.spec)

print("\nunwrapped type:", type(env.unwrapped))
print("unwrapped:", env.unwrapped)

# %% [markdown]
# # Part 2: The episode loop

# %% [markdown]
# ## Cell 2.1: `reset()`: start an episode
#
# - **Mental model**: puts the environment into an initial state and returns the
#   first observation. Must be called before the first `step()`
#
# - Signature:
#   ```
#   reset(*, seed=None, options=None) -> (observation, info)
#   ```
#   - `seed`: optional integer for reproducible randomness
#   - `options`: optional dict of env-specific configuration
#   - Returns `(observation, info)`: always a 2-tuple

# %%
# Reset and inspect the return values.
observation, info = env.reset(seed=42)
print("observation:", observation)
print("observation.shape:", observation.shape)
print("observation.dtype:", observation.dtype)
print("info:", info)

# %%
# Show that reset is reproducible with the same seed.
obs1, _ = env.reset(seed=0)
obs2, _ = env.reset(seed=0)
print("Same seed -> same obs:", np.array_equal(obs1, obs2))

# %% [markdown]
# ## Cell 2.2: `step()`: advance one timestep
#
# - **Mental model**: applies one action, advances the world by one timestep, and
#   returns what the agent sees next
#
# - Signature:
#   ```
#   step(action) -> (observation, reward, terminated, truncated, info)
#   ```
#   - `terminated`: episode ended naturally (goal / failure)
#   - `truncated`: episode cut off externally (time limit)
#   - The distinction matters for value bootstrapping in RL algorithms

# %%
env.reset(seed=42)
# Take one random action and inspect all five return values.
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
utils.print_step(obs, reward, terminated, truncated, info)

# %% [markdown]
# ## Cell 2.3: Run a full episode
#
# - The canonical episode loop: reset once, then step until done

# %%
# Seed the env and the action space, so that a rerun repeats the episode.
env.reset(seed=42)
env.action_space.seed(42)
total_reward = 0.0
step_count = 0
done = False
# Loop until the episode ends (either terminated or truncated).
while not done:
    # Sample a random action from the action space.
    action = env.action_space.sample()
    # Advance the environment by one timestep.
    _obs, reward, terminated, truncated, _info = env.step(action)
    total_reward += reward
    step_count += 1
    # An episode ends when terminated (goal/failure) or truncated (time limit).
    done = terminated or truncated
print(f"Steps: {step_count}  Total reward: {total_reward}")

# %% [markdown]
# # Part 3: Spaces: describing valid data
#
# - **Mental model**:
#     - A `Space` is a typed contract
#     - It knows the shape, dtype, and bounds of valid data
#     - It can sample random valid values and check membership
#
# - Every env has exactly two spaces:
#   - `env.observation_space`: type of observations returned by `step`/`reset`
#   - `env.action_space`: type of actions accepted by `step`

# %% [markdown]
# ## Cell 3.1: `Discrete(n)`: integers 0...n-1

# %%
# `Discrete(n)` represents a set of n integer values 0, 1, ..., n-1.
# It is used for action spaces with a finite number of discrete choices
# (e.g., move left/right in CartPole has n=2).
d = spaces.Discrete(3)
# Seed the space so that reruns print the same samples.
d.seed(0)
print("type:", type(d))
print("n:", d.n)
print("dtype:", d.dtype)


# %%
# Sample several times to see the range.
samples = [int(d.sample()) for _ in range(10)]
print("samples:", samples)

# %%
# Membership check.
print("2 in d:", d.contains(2))
print("5 in d:", d.contains(5))

# %% [markdown]
# ## Cell 3.2: `Box(low, high, shape)`: continuous tensor

# %%
# `Box(low, high, shape)` represents a continuous tensor where each element is
# bounded in [low, high]. It is used for observation spaces with continuous
# values (e.g., position, velocity in CartPole has shape (4,) with bounds).
b = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
b.seed(0)

print("type:", type(b))
print("low:", b.low)
print("high:", b.high)
print("shape:", b.shape)
print("dtype:", b.dtype)
print("sample:", b.sample())
print(
    "contains [0.5, -0.5, 0.0]:",
    b.contains(np.array([0.5, -0.5, 0.0], dtype=np.float32)),
)
print(
    "contains [2.0, 0.0, 0.0]:",
    b.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32)),
)

# %% [markdown]
# ## Cell 3.3: Space comparison: CartPole vs MountainCar

# %%
cart = gym.make("CartPole-v1")
mountain = gym.make("MountainCar-v0")
# Compare observation and action spaces of two classic-control envs.
for name, e in [("CartPole-v1", cart), ("MountainCar-v0", mountain)]:
    print(f"{name}:")
    print(f"  obs_space: {e.observation_space}")
    print(f"  obs_shape: {e.observation_space.shape}")
    print(f"  act_space: {e.action_space}")
    print()
cart.close()
mountain.close()

# %% [markdown]
# ## Cell 3.4: Other space types

# %%
# MultiDiscrete: vector of independent discrete dims.
md = spaces.MultiDiscrete([3, 4, 2])
md.seed(0)
print("MultiDiscrete sample:", md.sample())

# %%
# MultiBinary: vector of n binary flags.
mb = spaces.MultiBinary(5)
mb.seed(0)
print("MultiBinary sample:", mb.sample())

# %%
# Dict: named collection of spaces.
dict_space = spaces.Dict(
    {
        "pos": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        "speed": spaces.Discrete(5),
    }
)
dict_space.seed(0)
print("Dict sample:", dict_space.sample())

# %%
# Tuple: ordered collection of spaces.
tup = spaces.Tuple((spaces.Discrete(3), spaces.Box(0.0, 1.0, shape=(2,))))
tup.seed(0)
print("Tuple sample:", tup.sample())

# %% [markdown]
# # Part 4: Rendering and the registry

# %% [markdown]
# ## Cell 4.1: `render()` and `close()`
#
# - `render()`: returns a visual representation of the current state
#   - `render_mode="rgb_array"` -> numpy uint8 array `(H, W, 3)`
#   - `render_mode="human"` -> opens a display window
# - `close()`: releases resources (windows, file handles, networking)

# %%
# Use rgb_array mode to capture a frame without opening a window.
# Note: requires `pip install "gymnasium[classic-control]"` (pygame).
env_rgb = gym.make("CartPole-v1", render_mode="rgb_array")
env_rgb.reset(seed=0)
frame = env_rgb.render()
print("frame type:", type(frame))
print("frame shape:", frame.shape)
print("frame dtype:", frame.dtype)
env_rgb.close()

# %%
# Show the frame: this is what `render_mode="rgb_array"` returns.
utils.plot_frame(frame, "CartPole-v1 frame from render_mode='rgb_array'")

# %%
# Always close when finished.
env.close()
print("env closed")

# %% [markdown]
# ## Cell 4.2: `gym.make()`: the environment registry
#
# - `gym.make(id, **kwargs)` looks up the ID in a global registry and wraps the
#   result in standard wrappers (e.g. `TimeLimit`)
# - `gym.envs.registry` is a dict of `EnvSpec` objects

# %%
# Show a sample of registered environments.
all_ids = sorted(gym.envs.registry.keys())
print(f"Total registered environments: {len(all_ids)}")
sample_ids = all_ids[:10]
display(pd.DataFrame({"id": sample_ids}))

# %%
# Inspect an EnvSpec.
spec = gym.spec("CartPole-v1")
data = {
    "field": ["id", "entry_point", "max_episode_steps", "reward_threshold"],
    "value": [
        spec.id,
        spec.entry_point,
        spec.max_episode_steps,
        spec.reward_threshold,
    ],
}
display(pd.DataFrame(data))

# %% [markdown]
# # Part 5: Composition

# %% [markdown]
# ## Cell 5.1: Build a custom `Env`
#
# - Subclass `gym.Env`, declare the two spaces, and implement `reset` and `step`
# - Minimum required: `__init__`, `reset`, `step`
# - `action_space` / `observation_space` must be set in `__init__`


# %%
class OneDGridEnv(gym.Env):
    """
    Walk a 1-D grid to reach position (size-1).
    """

    def __init__(self, size: int = 5) -> None:
        super().__init__()
        # Grid length and starting position (leftmost cell).
        self._size = size
        self._pos = 0
        # Observation: current integer position (0 to size-1).
        self.observation_space = spaces.Discrete(size)
        # Actions: 0 = left, 1 = right.
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        # Reset the RNG and place the agent at the leftmost cell.
        super().reset(seed=seed)
        self._pos = 0
        return self._pos, {}

    def step(self, action):
        # Move right (action=1) or left (action=0), clamped to grid bounds.
        self._pos += 1 if action == 1 else -1
        self._pos = int(np.clip(self._pos, 0, self._size - 1))
        # Episode terminates when the agent reaches the rightmost cell.
        terminated = self._pos == self._size - 1
        # Reward is 1 only on reaching the goal, 0 otherwise.
        reward = 1.0 if terminated else 0.0
        return self._pos, reward, terminated, False, {}


# %%
# Verify the custom env behaves like any other Gym env.
custom_env = OneDGridEnv(size=5)
# Walk right until done.
steps, _ = utils.run_episode(custom_env, seed=7, policy=lambda _obs: 1)
for idx, step in enumerate(steps):
    utils.print_step(*step, label=f"Step {idx}", compact=True)

# %% [markdown]
# ## Cell 5.2: Register and use a custom env with `gym.make()`

# %%
# Register the custom env so it can be created via gym.make().
gym.register(id="OneDGrid-v0", entry_point=OneDGridEnv, max_episode_steps=20)
# Create the registered env with a custom size (overriding the default).
env2 = gym.make("OneDGrid-v0", size=4)
# Reset returns the starting position (0 for leftmost cell).
obs, _ = env2.reset(seed=0)
print("obs after reset:", obs)
# Step right: action=1 moves one cell to the right.
obs, reward, terminated, truncated, info = env2.step(1)
print("obs after step right:", obs, "reward:", reward)
# Clean up the env.
env2.close()

# %% [markdown]
# ## Cell 5.3: Wrappers
#
# - **Mental model**:
#     - A `Wrapper` wraps an `Env` and intercepts its calls
#     - The wrapped object is still an `Env` and can be wrapped again
#     - Wrappers avoid modifying environment code while adding behavior

# %%
base_env = gym.make("CartPole-v1")
# Stack two wrappers.
env_w = TimeLimit(base_env, max_episode_steps=50)
env_w = RecordEpisodeStatistics(env_w)
print("type of wrapped env:", type(env_w))
print("is still Env?", isinstance(env_w, gym.Env))
# unwrapped reaches through all wrapper layers.
print("unwrapped type:", type(env_w.unwrapped))

# %%
# Run an episode and inspect the statistics injected into `info`.
steps, _ = utils.run_episode(env_w, seed=0)
for step in steps:
    # Use compact=True to print all five values on a single line.
    utils.print_step(*step, compact=True)
# RecordEpisodeStatistics adds 'episode' key at episode end.
print("episode stats:", steps[-1].info["episode"])
env_w.close()

# %% [markdown]
# ## Cell 5.4: Vectorized environments
#
# - **Mental model**: a `VectorEnv` steps a batch of envs simultaneously.
#   Observations, rewards, and flags become arrays with a leading batch dimension
# - Auto-resets sub-envs when their episodes end, so the batch keeps stepping

# %%
# Run 3 CartPole envs in parallel.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
vec_env.action_space.seed(0)
print("type:", type(vec_env))
obs_batch, info_batch = vec_env.reset(seed=0)
print("obs_batch shape:", obs_batch.shape)  # (3, 4)
# Step all 3 envs simultaneously.
actions = vec_env.action_space.sample()
print("actions batch:", actions)
obs_batch, rewards, terminations, truncations, info_batch = vec_env.step(actions)
data = {
    "field": ["obs_batch.shape", "rewards", "terminations", "truncations"],
    "value": [
        str(obs_batch.shape),
        str(rewards),
        str(terminations),
        str(truncations),
    ],
}
display(pd.DataFrame(data))
vec_env.close()

# %% [markdown]
# # Part 6: Interactive exploration

# %% [markdown]
# ## Cell 6.1: Explore any env's interface

# %%
# Explore any env's interface.
env6 = gym.make("MountainCar-v0")
env6.action_space.seed(0)
hintros.print_obj_info(type(env6))

# %%
# What does reset return?
obs, info = env6.reset(seed=0)
print("obs:", obs, "  info:", info)

# %%
# What does a random step look like?
obs, reward, terminated, truncated, info = env6.step(env6.action_space.sample())
print("obs:", obs, "reward:", reward, "terminated:", terminated)

# %%
# What is the observation space bound?
print("obs low:", env6.observation_space.low)
print("obs high:", env6.observation_space.high)
env6.close()

# %% [markdown]
# ## Summary: The Mental Model
#
# - `gymnasium.Env` is the single interface for all RL environments: call
#   `reset()` to start an episode and `step(action)` in a loop to advance it
# - Two `Space` objects (`observation_space`, `action_space`) encode the contract:
#   they define valid data types, shapes, and bounds and can sample random valid
#   values
# - Wrappers extend behavior by composition without modifying environment code,
#   and `VectorEnv` parallelizes a batch of envs under the same interface
# - A custom environment is just a subclass of `gym.Env` that declares the two
#   spaces and implements `reset` and `step`
