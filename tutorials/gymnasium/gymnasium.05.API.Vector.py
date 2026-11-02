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
# # Gymnasium Vector API
#
# - This notebook explores the `gymnasium.vector` API for running many copies of an
#   env as one batch
# - The pedagogical arc:
#   - `VectorEnv`, `SyncVectorEnv`, and `gym.make_vec()`
#   - Batched `reset()` and `step()`, and the autoreset modes
#   - Batched spaces and `AsyncVectorEnv`
#   - Vector wrappers and utility functions
#   - Composition examples: custom wrappers, mixed envs, and selective reset
# - References:
#   - API: https://gymnasium.farama.org/api/vector
#   - GitHub: https://github.com/Farama-Foundation/Gymnasium

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import multiprocessing as mp
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.vector import (
    AsyncVectorEnv,
    AutoresetMode,
    SyncVectorEnv,
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
)
from gymnasium.vector.utils import batch_space, create_empty_array, iterate
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from gymnasium.wrappers.vector import ClipReward

# On macOS, the default spawn start method prevents AsyncVectorEnv from working
# when run as a script. Force fork to make it work in both script and notebook
# contexts.
mp.set_start_method("fork", force=True)

# %%
import gymnasium_utils as utils
import helpers.hdbg as hdbg
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
#   - Running many RL environments sequentially wastes wall-clock time
#   - Vector environments run `N` independent copies of the same env in parallel
#     (or serially but batched) and return observations, rewards, and flags as
#     arrays with a leading batch dimension
#   - RL algorithms sample a batch of transitions per step, making vector envs
#     essential for efficient training
#
# - **Key abstraction**: `gymnasium.vector.VectorEnv`: the abstract base class
#   for all vector environments. It mirrors the single-env API (`reset`, `step`,
#   `render`, `close`) but operates on batches
#
# - **Two concrete implementations**:
#   - `SyncVectorEnv`: runs envs sequentially in the same process (simple,
#     low-overhead)
#   - `AsyncVectorEnv`: runs envs in separate `multiprocessing` processes (true
#     parallelism, useful for I/O-bound envs)
#
# - **Mental model**:
#   - Every value gains a batch dimension of size `N = num_envs`
#   - Sub-envs that finish an episode are **auto-reset** on the next step,
#     keeping the batch full without manual intervention
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `reset(seed)` | Reset every sub-env | Returns `(observations[N], info)` |
# | `step(actions[N])` | Step every sub-env | Returns `(observations[N], rewards[N], terminations[N], truncations[N], info)` |
# | `num_envs` | Batch size | `N` |
# | `single_observation_space`, `observation_space` | Space of one sub-env, and of the batch | Same pair for actions |
#
# - **Key additional attributes**:
#   - `num_envs`: the batch size
#   - `single_observation_space` / `single_action_space`: the space of one
#     sub-environment (before batching)
#   - `observation_space` / `action_space`: the batched space

# %% [markdown]
# # Part 1: Creating vector envs

# %% [markdown]
# ## Cell 1.1: `VectorEnv`: the abstract base class
#
# - **Mental model**: an interface that wraps `N` independent `gym.Env`
#   instances and exposes batch-oriented `reset()` and `step()` methods
#
# - You never instantiate `VectorEnv` directly; you use `SyncVectorEnv` or
#   `AsyncVectorEnv`

# %%
# VectorEnv is the abstract base class: all methods raise NotImplementedError.
print("VectorEnv MRO:", [c.__name__ for c in VectorEnv.__mro__])

# %% [markdown]
# ## Cell 1.2: Create a `SyncVectorEnv` manually
#
# - **Mental model**: `SyncVectorEnv` takes a sequence of callables (env
#   factories), calls each one, and steps them all sequentially inside a single
#   `for` loop
#   - Simple and debuggable (no multiprocessing)
#   - Wall-clock = `sum(step_time_per_env)`, not `max(step_time)`
#
# - The two key "single" vs "batched" attributes:
#   - `single_observation_space`: what one sub-env sees
#   - `observation_space`: the batched version (adds a leading dim)

# %%
# Manually create a SyncVectorEnv with 3 CartPole instances.
env_fns = [lambda: gym.make("CartPole-v1") for _ in range(3)]
sync_env = SyncVectorEnv(env_fns)

print("type:", type(sync_env).__name__)
print("num_envs:", sync_env.num_envs)
print("single_observation_space:", sync_env.single_observation_space)
print("observation_space:", sync_env.observation_space)
print("single_action_space:", sync_env.single_action_space)
print("action_space:", sync_env.action_space)

# Clean up.
sync_env.close()

# %% [markdown]
# ## Cell 1.3: `gym.make_vec()`: the standard factory
#
# - **Mental model**: `gym.make_vec()` is the vector equivalent of
#   `gym.make()`: it creates `N` copies of a registered environment using a
#   single ID string and returns a vector environment
#   - `vectorization_mode="sync"` -> `SyncVectorEnv`
#   - `vectorization_mode="async"` -> `AsyncVectorEnv`
#   - `vectorization_mode=None` (default) -> uses `vector_entry_point` if
#     available, otherwise falls back to `"sync"`

# %%
# Create 4 CartPole envs using make_vec (the standard factory).
vec_env = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")
print("type:", type(vec_env).__name__)
print("num_envs:", vec_env.num_envs)
print("action_space:", vec_env.action_space)

# Single vs batched shape.
obs_shape = vec_env.single_observation_space.shape  # one env
batch_obs_shape = vec_env.observation_space.shape  # batched
data = {
    "property": ["single_observation_space.shape", "observation_space.shape"],
    "value": [str(obs_shape), str(batch_obs_shape)],
}
display(pd.DataFrame(data))

vec_env.close()

# %% [markdown]
# # Part 2: Batch reset and step

# %% [markdown]
# ## Cell 2.1: `reset()`: batch reset
#
# - **Mental model**: calls `reset()` on all sub-environments and concatenates
#   the observations into a single array
#   - Returns `(observations, info)` where `observations.shape` is
#     `(num_envs,) + single_observation_space.shape`
#   - `seed` can be:
#     - `None`: each env gets a random seed
#     - `int`: seeds are `[seed, seed+1, ..., seed+N-1]`
#     - `list[int]`: explicit per-env seeds

# %%
# Create 3 envs and reset with an integer seed.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")

obs, info = vec_env.reset(seed=42)
print("obs shape:", obs.shape)
print("obs dtype:", obs.dtype)
print("obs:\n", obs)
print("info:", info)

vec_env.close()

# %%
# Show that resetting with the same seed reproduces the same batch of obs.
vec_env2 = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")

obs0, _ = vec_env2.reset(seed=0)
obs1, _ = vec_env2.reset(seed=0)
print("Same seed -> same first env obs:", np.array_equal(obs0[0], obs1[0]))
print("Same seed -> all env obs identical:", np.array_equal(obs0, obs1))

vec_env2.close()

# %% [markdown]
# ## Cell 2.2: `step()`: batch step
#
# - **Mental model**: takes a batch of `N` actions (one per sub-env) and
#   returns `(observations, rewards, terminations, truncations, info)` where
#   every numeric value is an array of shape `(N,)` or `(N,) + space.shape`
#
# - Key behaviour: **autoreset**. When a sub-env returns `terminated=True` or
#   `truncated=True`, it is automatically reset on the **next** `step()` call
#   (the default `AutoresetMode.NEXT_STEP`)
#   - This means the batch never shrinks; the agent always gets `N`
#     observations
#   - The step that ends an episode returns its final observation as usual, and
#     the reset observation arrives on the next step, so `info` has no final keys

# %%
# Step all 3 envs with random actions and inspect the batch returns.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
vec_env.reset(seed=42)
# Seed the action space so that reruns take the same actions.
vec_env.action_space.seed(0)

# Sample one action per env.
actions = vec_env.action_space.sample()
print("actions:", actions, "dtype:", actions.dtype)

obs, rewards, terminated, truncated, info = vec_env.step(actions)

data = {
    "field": [
        "observations.shape",
        "observations[0]",
        "rewards",
        "terminated",
        "truncated",
        "info",
    ],
    "value": [
        str(obs.shape),
        str(obs[0]),
        str(rewards),
        str(terminated),
        str(truncated),
        str(info),
    ],
}
display(pd.DataFrame(data))

vec_env.close()

# %%
# Run up to 100 steps and track per-env episode length.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
vec_env.reset(seed=42)
vec_env.action_space.seed(0)

steps_to_done = [0, 0, 0]

for t in range(100):
    actions = vec_env.action_space.sample()
    obs, rewards, terminated, truncated, info = vec_env.step(actions)
    # Check which envs just terminated.
    just_done = np.logical_or(terminated, truncated)
    for i in range(3):
        if just_done[i] and steps_to_done[i] == 0:
            steps_to_done[i] = t + 1
    if all(s > 0 for s in steps_to_done):
        break

print("Steps until done per env:", steps_to_done)

vec_env.close()

# %% [markdown]
# ## Cell 2.3: `AutoresetMode`: controlling auto-reset behaviour
#
# - **Mental model**: the autoreset mode determines **when** a sub-env that
#   finished an episode gets reset
#
# - Three modes via `AutoresetMode` enum:
#   - `NEXT_STEP` (default): the sub-env is reset on the **next** `step()` call
#     after `terminated or truncated`. The agent sees one extra transition with
#     `reward=0, terminated=False, truncated=False` while the env resets
#   - `SAME_STEP`: the sub-env is reset **immediately** in the same `step()`
#     call. The step returns the reset observation, and the final one moves to
#     `info["final_obs"]`, next to `info["final_info"]`
#   - `DISABLED`: no autoreset; the agent must manually call `reset()` with a
#     `reset_mask` option

# %%
# Show the AutoresetMode values.
display(
    pd.DataFrame(
        {
            "mode": [m.name for m in AutoresetMode],
            "value": [m.value for m in AutoresetMode],
        }
    )
)

# %%
# Compare NEXT_STEP vs SAME_STEP behaviour on a short-lived env.


class ShortFrozenLake(gym.Env):
    """
    Toy FrozenLake that always falls into a hole in a few steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self._step_count = 0
        # The observation is the step count, so that the reset obs is visible.
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self._step_count, {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= 3  # always done after 3 steps
        return (
            self._step_count,
            0.0,
            terminated,
            False,
            {},
        )


# Register the toy env.
gym.register(id="ShortFL-v0", entry_point=ShortFrozenLake)

# Test with NEXT_STEP: the final obs is in `obs`, and `info` has no final keys.
print("NEXT_STEP")
vec_next = gym.make_vec("ShortFL-v0", num_envs=2, vectorization_mode="sync")
vec_next.reset(seed=0)
for t in range(6):
    obs, rewards, term, trunc, info = vec_next.step(
        vec_next.action_space.sample()
    )
    print(f"  step={t}: term={term}, obs={obs}, keys={sorted(info)}")
    hdbg.dassert_not_in("final_obs", info, "NEXT_STEP must not set final keys")
vec_next.close()

# Test with SAME_STEP: `obs` is the reset obs, and the final obs is in `info`.
print("SAME_STEP")
vec_same = SyncVectorEnv(
    [lambda: ShortFrozenLake() for _ in range(2)],
    autoreset_mode=AutoresetMode.SAME_STEP,
)
vec_same.reset(seed=0)
for t in range(6):
    obs, rewards, term, trunc, info = vec_same.step(
        vec_same.action_space.sample()
    )
    final_obs = info.get("final_obs", "N/A")
    print(f"  step={t}: term={term}, obs={obs}, final_obs={final_obs}")
    hdbg.dassert_eq(
        "final_obs" in info, bool(term.any()), "Final keys appear when done"
    )
vec_same.close()

# %%
# With DISABLED, terminated envs stay terminated and must be manually reset.
vec_disabled = SyncVectorEnv(
    [lambda: ShortFrozenLake() for _ in range(2)],
    autoreset_mode=AutoresetMode.DISABLED,
)
vec_disabled.reset(seed=0)
for t in range(6):
    obs, rewards, term, trunc, info = vec_disabled.step(
        vec_disabled.action_space.sample()
    )
    print(f"  step={t}: term={term}, obs[0]={obs[0]}")
    # Check which envs are done and reset them manually using reset_mask.
    just_done = np.where(term | trunc)[0]
    if len(just_done) > 0:
        print(f"    -> resetting envs {just_done}")
        mask = np.zeros(2, dtype=np.bool_)
        mask[just_done] = True
        obs, info = vec_disabled.reset(seed=0, options={"reset_mask": mask})
        print(f"    -> obs after reset: {obs}")
vec_disabled.close()

# %% [markdown]
# # Part 3: Spaces and parallelism

# %% [markdown]
# ## Cell 3.1: Spaces in vector environments
#
# - **Mental model**: a vector environment has **four** space attributes:
#   - `single_observation_space`: what one sub-env returns (e.g., `Box(4,)`)
#   - `observation_space`: the batched version (e.g., `Box(N, 4)`)
#   - `single_action_space`: what one sub-env accepts
#   - `action_space`: the batched version
#
# - The batched action space knows how to **sample** `N` actions at once and
#   how to **check membership** for a batch of actions

# %%
# Inspect the four spaces.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")

print("single_observation_space:", vec_env.single_observation_space)
print("observation_space:       ", vec_env.observation_space)
print("single_action_space:     ", vec_env.single_action_space)
print("action_space:            ", vec_env.action_space)

# %%
# The batched action_space.sample() returns N actions in one call.
vec_env.action_space.seed(0)
sample = vec_env.action_space.sample()
print("action_space.sample():", sample, "shape:", sample.shape)

# Membership check works on batches.
valid = vec_env.action_space.contains(sample)
print("batch contains(N valid actions):", valid)

# Per-env membership: each action in the batch is checked individually.
invalid_actions = np.array([10, 20, 5], dtype=sample.dtype)
valid2 = vec_env.action_space.contains(invalid_actions)
print("batch contains(N invalid actions):", valid2)

# %%
# MultiDiscrete action space: each env gets its own Discrete.
# The batched action space becomes MultiDiscrete([2, 2, 2]) for 3 envs.
hdbg.dassert(not vec_env.closed, "The env must still be open here")
print("batched action space type:", type(vec_env.action_space).__name__)
print("batched action space nvec:", vec_env.action_space.nvec)

vec_env.close()

# %% [markdown]
# ## Cell 3.2: `AsyncVectorEnv`: multiprocess parallelism
#
# - **Mental model**: each sub-env runs in its own `multiprocessing` process.
#   Observations are communicated back via shared memory or pipes
#   - True parallelism: wall-clock time ~= `max(step_time_per_env)` not `sum`
#   - Use when envs are I/O-bound or CPU-heavy (e.g., image rendering, physics
#     simulators)
#   - Use `shared_memory=True` (default) for large observations (e.g., images)
#   - Use `daemon=True` (default) so child processes die when the parent exits

# %%
# Create an AsyncVectorEnv with 4 envs.
async_env = AsyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(4)],
    shared_memory=True,
    daemon=True,
)

print("type:", type(async_env).__name__)
print("num_envs:", async_env.num_envs)
print("single_observation_space:", async_env.single_observation_space)

# Reset and step work identically to SyncVectorEnv.
obs, info = async_env.reset(seed=42)
print("obs.shape:", obs.shape)

actions = async_env.action_space.sample()
obs, rewards, term, trunc, info = async_env.step(actions)
print("rewards:", rewards)

async_env.close()


# %%
# Compare sync vs async wall-clock time (3 envs, 100 steps each).
def time_vector_env(vec_env_class, num_envs=3, num_steps=100):
    """
    Return the seconds taken by `num_steps` steps of random actions.

    :param vec_env_class: `SyncVectorEnv` or `AsyncVectorEnv`
    :param num_envs: number of sub-envs
    :param num_steps: number of steps
    :return: elapsed wall-clock seconds
    """
    env = vec_env_class(
        [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
    )
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(num_steps):
        env.step(env.action_space.sample())
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


# Single env for baseline.
single = gym.make("CartPole-v1")
single.reset(seed=0)
start = time.perf_counter()
for _ in range(100):
    _, _, terminated, truncated, _ = single.step(single.action_space.sample())
    # Reset when an episode ends, as the vector envs do on their own.
    if terminated or truncated:
        single.reset()
single_time = time.perf_counter() - start
single.close()

sync_time = time_vector_env(SyncVectorEnv, num_envs=3, num_steps=100)
async_time = time_vector_env(AsyncVectorEnv, num_envs=3, num_steps=100)

data = {
    "mode": [
        "1 env (sequential)",
        "SyncVectorEnv (3 envs)",
        "AsyncVectorEnv (3 envs)",
    ],
    "time_100_steps (s)": [
        f"{single_time:.4f}",
        f"{sync_time:.4f}",
        f"{async_time:.4f}",
    ],
}
display(pd.DataFrame(data))

# %% [markdown]
# # Part 4: Wrappers and utilities

# %% [markdown]
# ## Cell 4.1: `VectorWrapper`: transforming vector environments
#
# - **Mental model**: mirrors `gym.Wrapper` but operates on batches
#   - `VectorActionWrapper`: transforms the action batch before passing to
#     `step()`
#   - `VectorObservationWrapper`: transforms the observation batch from
#     `step()` and `reset()`
#   - `VectorRewardWrapper`: transforms the reward batch from `step()`
#
# - Pre-built vector wrappers like `ClipReward`, `NormalizeObservation`,
#   `RecordEpisodeStatistics` exist in `gymnasium.wrappers.vector`


# %%
# Build a custom VectorObservationWrapper that flattens observations.
class FlattenObsWrapper(VectorObservationWrapper):
    """
    Flatten batched observations into a 2D array.
    """

    def __init__(self, env):
        super().__init__(env)
        # Compute the flattened single-observation space.
        single_shape = env.single_observation_space.shape
        flat_size = int(np.prod(single_shape))
        self.single_observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_size,),
            dtype=np.float32,
        )
        # Batch the flattened space too, so that both spaces match the output.
        self.observation_space = batch_space(
            self.single_observation_space, env.num_envs
        )

    def observations(self, obs):
        # obs shape: (N, *single_shape) -> (N, flat_size)
        return obs.reshape(obs.shape[0], -1)


# Test the wrapper. CartPole obs are already flat, so reshape them to (2, 2).
base_vec = SyncVectorEnv(
    [
        lambda: gym.wrappers.ReshapeObservation(gym.make("CartPole-v1"), (2, 2))
        for _ in range(2)
    ]
)
print("base single_observation_space:", base_vec.single_observation_space)
wrapped = FlattenObsWrapper(base_vec)
print("single_observation_space (flattened):", wrapped.single_observation_space)
print("observation_space (flattened):", wrapped.observation_space)

obs, _ = wrapped.reset(seed=42)
print("obs.shape (flattened):", obs.shape)
print("obs[0]:", obs[0])
hdbg.dassert_eq(obs.shape, (2, 4), "Each (2, 2) obs must become 4 values")
hdbg.dassert(wrapped.observation_space.contains(obs), "Obs must fit the space")

wrapped.close()

# %%
# Use a pre-built vector wrapper (ClipReward).
vec_env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
clipped = ClipReward(vec_env, min_reward=-1.0, max_reward=1.0)
print("wrapped type:", type(clipped).__name__)

clipped.reset(seed=0)
obs, rewards, term, trunc, info = clipped.step(clipped.action_space.sample())
print("clipped rewards:", rewards)

clipped.close()

# %% [markdown]
# ## Cell 4.2: Utility functions
#
# - The `gymnasium.vector.utils` module provides helpers for working with
#   batched spaces
#
# - Key functions:
#   - `batch_space(space, n)`: batch a single space into N copies
#   - `create_empty_array(space, n)`: create an empty array matching the
#     batched space structure
#   - `concatenate(space, items, out)`: concatenate samples into a batched
#     array
#   - `iterate(space, batch)`: iterate over the first dimension of a batch

# %%
# batch_space: replicate a space N times.
single = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
batched = batch_space(single, n=3)
print("single:", single)
print("batched:", batched)
print("batched shape:", batched.shape)

# %%
# batch_space works with any space type.
d = Discrete(5)
batched_d = batch_space(d, n=3)
print("Discrete single:", d)
print("Discrete batched:", batched_d)

# %%
# create_empty_array: pre-allocate arrays for batched observations.
empty = create_empty_array(single, n=3, fn=np.zeros)
print("empty array shape:", empty.shape)
print("empty array:\n", empty)

# %%
# iterate: loop over the batch dimension.
actions = np.array([0, 1, 0], dtype=np.int64)
for i, action in enumerate(iterate(batched_d, actions)):
    print(f"  env {i}: action={action}")

# %% [markdown]
# # Part 5: Composition

# %% [markdown]
# ## Cell 5.1: Minimal vector environment
#
# - Create 2 envs, reset once, step a few times, inspect all return values

# %%
# Minimal vector env workflow.
vec_env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
obs, info = vec_env.reset(seed=0)
vec_env.action_space.seed(0)
print("Step  0 obs:\n", obs)
for t in range(3):
    obs, rewards, term, trunc, info = vec_env.step(vec_env.action_space.sample())
    print(
        f"Step {t + 1}: rewards={rewards}, term={term}, obs[0, :3]={obs[0, :3]}"
    )
vec_env.close()

# %% [markdown]
# ## Cell 5.2: Use a custom wrapper
#
# - Apply a `VectorRewardWrapper` that scales rewards by a factor


# %%
class ScaleReward(VectorRewardWrapper):
    """
    Scale all rewards by a constant factor.
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self._scale = scale

    def rewards(self, rewards):
        return rewards * self._scale


# Build two separate envs, so that stepping one does not advance the other.
vec_env = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
scaled = ScaleReward(
    gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync"), scale=0.1
)

# Reset both with the same seed and send both the same actions.
scaled.reset(seed=0)
vec_env.reset(seed=0)
scaled.action_space.seed(0)
actions = scaled.action_space.sample()
obs_scaled, rewards_scaled, term_scaled, trunc_scaled, info_scaled = scaled.step(
    actions
)
print("scaled rewards:", rewards_scaled)
# Compare with the original (unscaled) rewards from the base env.
obs_orig, rewards_orig, term_orig, trunc_orig, info_orig = vec_env.step(actions)
print("original rewards:", rewards_orig)
hdbg.dassert(np.array_equal(obs_scaled, obs_orig), "Both envs must stay in step")
hdbg.dassert(
    np.array_equal(rewards_scaled, 0.1 * rewards_orig), "Rewards must be scaled"
)
scaled.close()
vec_env.close()

# %% [markdown]
# ## Cell 5.3: Async with custom env configurations
#
# - `AsyncVectorEnv` can create sub-envs with different parameters by passing
#   different factories

# %%
# Two FrozenLake envs with different map sizes via lambda factories.
# Use observation_mode="different" to allow different obs spaces.
async_mixed = AsyncVectorEnv(
    [
        lambda: gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False),
        lambda: gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False),
    ],
    shared_memory=False,
    daemon=True,
    observation_mode="different",
)
print("num_envs:", async_mixed.num_envs)
print("single_observation_space:", async_mixed.single_observation_space)
print("observation_space:", async_mixed.observation_space)

obs, info = async_mixed.reset(seed=42)
print("obs:", obs)

async_mixed.close()

# %% [markdown]
# ## Cell 5.4: `reset_mask`: selective reset
#
# - When `AutoresetMode` is not what you need, pass `reset_mask` in `options`
#   to selectively reset specific sub-envs

# %%
# Reset only env 0 and 2 out of 3.
vec_env = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
vec_env.reset(seed=0)
vec_env.action_space.seed(0)
# Step a few times to change state.
for _ in range(5):
    vec_env.step(vec_env.action_space.sample())

# Selective reset using reset_mask.
obs, info = vec_env.reset(
    seed=99,
    options={"reset_mask": np.array([True, False, True])},
)
print("obs after selective reset:\n", obs)
# Env 1 (not reset) has advanced 5 more steps; envs 0 and 2 are fresh.

vec_env.close()

# %% [markdown]
# ## Cell 5.5: Nest single-env wrappers inside a vector env
#
# - Standard single-env wrappers can be applied **inside** each sub-env factory


# %%
def make_wrapped_cartpole():
    """
    Create a CartPole with a short time limit.
    """
    env = gym.make("CartPole-v1")
    env = TimeLimit(env, max_episode_steps=50)
    env = RecordEpisodeStatistics(env)
    return env


vec_env = SyncVectorEnv([make_wrapped_cartpole for _ in range(2)])
obs, info = vec_env.reset(seed=0)
vec_env.action_space.seed(0)
for t in range(60):
    obs, rewards, term, trunc, info = vec_env.step(vec_env.action_space.sample())
    if np.any(term | trunc):
        print(f"step {t}: envs {np.where(term | trunc)[0]} terminated")
        break
vec_env.close()

# %% [markdown]
# # Part 6: API patterns

# %% [markdown]
# ## Cell 6.1: Vector env patterns
#
# - **Factory pattern**: `gym.make_vec(id, num_envs, vectorization_mode)` is
#   the standard way to create vector envs. Avoid constructing `SyncVectorEnv`
#   / `AsyncVectorEnv` directly unless you need custom per-env parameters
# - **Autoreset pattern**: the default `NEXT_STEP` mode auto-resets finished
#   envs on the next call. The batch never shrinks; the agent always sees `N`
#   observations
# - **Wrapper stacking**: single-env wrappers apply inside the env factory;
#   vector wrappers (e.g., `ClipReward`) wrap the whole `VectorEnv` after
#   construction
# - **Four-space contract**: `single_observation_space` vs `observation_space`
#   and `single_action_space` vs `action_space` is the key mental model for
#   understanding shapes
# - **Selective reset**: use `options={"reset_mask": mask}` to reset only a
#   subset of sub-envs without affecting the others

# %% [markdown]
# # Part 7: Interactive exploration

# %% [markdown]
# ## Cell 7.1: Explore vector envs
#
# - Try modifying the examples above:
#   - What shape does `batch_space` produce for a `Dict` space?
#   - How does `AsyncVectorEnv` handle an environment with a long per-step
#     computation?

# %%
# Explore the SyncVectorEnv interface.
sync_env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
print("Public attributes/methods:")
for m in sorted([m for m in dir(sync_env) if not m.startswith("_")]):
    print(f"  {m}")
sync_env.close()

# %%
# Explore what batch_space does with a Dict space.

dict_space = Dict(
    {
        "pos": Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "goal": Discrete(3),
    }
)
batched = batch_space(dict_space, n=4)
print("batched Dict space:", batched)

# %% [markdown]
# ## Summary: The Mental Model
#
# - `VectorEnv` is the batched version of `gym.Env`: `reset()` and `step()`
#   return arrays with a leading batch dimension of size `num_envs`
#   - Create one with `gym.make_vec()`, or build `SyncVectorEnv` (sequential) or
#     `AsyncVectorEnv` (multiprocess) directly for per-env parameters
# - Sub-envs that finish are auto-reset by default (`AutoresetMode.NEXT_STEP`),
#   keeping the batch full without manual intervention
# - The four-space contract (`single_observation_space`, `observation_space`,
#   `single_action_space`, `action_space`) is the key to understanding the
#   shapes
# - Wrappers work at two levels: standard single-env wrappers inside each
#   factory, and vector wrappers (`VectorObservationWrapper`,
#   `VectorRewardWrapper`) around the whole `VectorEnv`
