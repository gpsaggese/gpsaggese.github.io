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
# # Description
#
# - Teach the `gymnasium.Wrapper` API surface by progressively building up from the
#   smallest possible working example
# - Focus on primitives: what a wrapper is, how it intercepts environment calls,
#   and how wrappers compose into layered transformations
#
# - References:
#   - API: https://gymnasium.farama.org/api/wrappers/
#   - GitHub: https://github.com/Farama-Foundation/Gymnasium

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging

# Third-party libraries.
import numpy as np
import pandas as pd

# %%
import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

hdbg.init_logger(verbosity=logging.INFO)
# hnotebook.config_notebook()

import tutorials.gymnasium.gymnasium_utils as gymutils

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**: RL agents interact with an environment via
#   `reset()` and `step()` .  When you need to transform observations, actions, or
#   rewards (e.g., normalize pixels, clip actions, scale rewards), modifying the
#   environment's source code is impractical.  Wrappers solve this by wrapping an
#   `Env` and intercepting its calls
#
# - **Key abstraction**: `gymnasium.Wrapper`: a class that holds an inner `Env` and
#   delegates every method ( `reset` , `step` , `render` , `close` ) to it.
#   Subclasses override specific methods to inject transformations
#
# - **Mental model**:
#   ```
#   Wrapper
#     └── env (the inner Env, which may itself be a Wrapper)
#           └── ... (further wrappers)
#                 └── base Env (leaf)
#   ```
#   - Calling `wrapper.step(action)` flows through each wrapper layer from
#     outermost to innermost, then the result flows back out
#   - `wrapper.unwrapped` strips all layers and returns the bare `Env`
#
# - **Key classes**:
#   - `gymnasium.Wrapper`: base class, delegates everything to `self.env`
#   - `gymnasium.ObservationWrapper`: override `observation()` to transform
#     observations
#   - `gymnasium.ActionWrapper`: override `action()` to transform actions
#   - `gymnasium.RewardWrapper`: override `reward()` to transform rewards
#   - `TransformObservation`, `TransformAction`, `TransformReward`: lambda-based
#     wrappers that take a function argument
#   - Builtin utility wrappers: `TimeLimit`, `RecordEpisodeStatistics`,
#     `Autoreset`, etc.

# %% [markdown]
# ## Primitive 1: `Wrapper`: The Base Class
#
# - **Mental model**: a `Wrapper` is an environment that contains another
#   environment.  Every method call is forwarded to the inner env by default, so a
#   wrapper can selectively intercept calls without reimplementing the full API
#
# - The constructor takes a single argument: the environment to wrap

# %%
import gymnasium as gym

# Create a bare CartPole environment.
base_env = gym.make("CartPole-v1", render_mode=None)
print("base_env type:", type(base_env))

# Wrap it with the base Wrapper class.
from gymnasium import Wrapper

wrapper = Wrapper(base_env)
print("wrapper type:", type(wrapper))

# The wrapper IS still an Env (isinstance check).
print("isinstance(wrapper, gym.Env):", isinstance(wrapper, gym.Env))

# %% [markdown]
# ### Delegation: method calls pass through to the inner env
#
# - By default, `Wrapper` does nothing: all calls fall through to `self.env`

# %%
# Reset is delegated.
obs, info = wrapper.reset(seed=42)
print("obs (delegated):", obs)

# Step is delegated. Seed the action space so that reruns take the same action.
wrapper.action_space.seed(0)
obs2, reward, terminated, truncated, _info = wrapper.step(
    wrapper.action_space.sample()
)
gymutils.print_step(
    obs=obs2,
    reward=reward,
    terminated=terminated,
    truncated=truncated,
    info={},
    compact=True,
)

# %%
# The wrapper exposes the same spaces as the base env.
print("action_space:", wrapper.action_space)
print("observation_space:", wrapper.observation_space)

# %% [markdown]
# ### Inspecting the wrapper environment
#
# - `env.env`: the directly wrapped environment
# - `env.unwrapped`: the innermost bare environment (recursively unwraps all layers)

# %%
# Show the wrapper chain.
print("wrapper.env is base_env:", wrapper.env is base_env)
print("wrapper.unwrapped:", type(wrapper.unwrapped))
print(
    "wrapper.unwrapped is base_env.unwrapped:",
    wrapper.unwrapped is base_env.unwrapped,
)

# Clean up.
wrapper.close()

# %% [markdown]
# ## Primitive 2: `ObservationWrapper`: Transform Observations
#
# - **Mental model**: An `ObservationWrapper` intercepts the observation that
#   `step()` and `reset()` return, passes it through `self.observation()`, and
#   returns the modified observation
#
# - The `observation()` method is the only method you need to override
#   - Called on every observation: both from `reset()` and from `step()`
# - The `step()` / `reset()` flow:
#   ```
#   inner_env.step(action) → (obs, reward, terminated, truncated, info)
#   wrapper.observation(obs) → modified_obs
#   return (modified_obs, reward, terminated, truncated, info)
#   ```

# %%
from gymnasium import ObservationWrapper


class FlipObservation(ObservationWrapper):
    """
    Negate every element of the observation vector.
    """

    def observation(self, observation):
        return -observation


# Build two separate envs, so that stepping one does not advance the other.
base = gym.make("CartPole-v1")
flip_env = FlipObservation(gym.make("CartPole-v1"))

# Reset both with the same seed and inspect: the observation is negated.
obs, _ = flip_env.reset(seed=42)
obs_base, _ = base.reset(seed=42)
print("original obs:", obs_base)
print("flipped obs: ", obs)
hdbg.dassert(np.array_equal(obs, -obs_base), "Reset obs must be negated")

# %%
# The observation() method is called on every step.
base.reset(seed=42)
flip_env.reset(seed=42)
flip_env.action_space.seed(0)
action = flip_env.action_space.sample()
orig_obs, *_ = base.step(action)
flip_obs, *_ = flip_env.step(action)
print("original step obs:", orig_obs)
print("flipped step obs: ", flip_obs)
hdbg.dassert(np.array_equal(flip_obs, -orig_obs), "Step obs must be negated")

# %% [markdown]
# ## Primitive 3: `ActionWrapper`: Transform Actions
#
# - **Mental model**: An `ActionWrapper` intercepts the action that `step()`
#   receives and passes it through `self.action()` before forwarding it to the
#   inner environment
#
# - The `action()` method is the only method you need to override
#   - Called on every action before `step()` on the inner env
# - The `step()` flow:
#   ```
#   wrapper.action(action) → modified_action
#   inner_env.step(modified_action) → (obs, reward, ...)
#   return (obs, reward, ...)
#   ```

# %%
from gymnasium import ActionWrapper


class SwapAction(ActionWrapper):
    """
    Swap the two discrete actions: 0 ↔ 1.

    A minimal action transformation on a 2-action env (CartPole).
    """

    def action(self, action):
        # Swap 0 and 1: action 0 becomes 1, action 1 becomes 0.
        return 1 - action


# %%
# Create a wrapped env and compare action behavior.
base2 = gym.make("CartPole-v1")
action_swap_env = SwapAction(gym.make("CartPole-v1"))

base2.reset(seed=0)
action_swap_env.reset(seed=0)

# Step with action=0 → the wrapper swaps to action=1 before passing to inner env.
obs_a, *_ = base2.step(0)
obs_b, *_ = action_swap_env.step(0)
print("action=0 on base env -> obs:", obs_a)
print("action=0 on swap wrapper (→ action=1) -> obs:", obs_b)
print(
    "Different because a different action was taken:",
    not np.array_equal(obs_a, obs_b),
)
hdbg.dassert(not np.array_equal(obs_a, obs_b), "The action must be swapped")

# %% [markdown]
# ## Primitive 4: `RewardWrapper`: Transform Rewards
#
# - **Mental model**: A `RewardWrapper` intercepts the reward that `step()` returns
#   and passes it through `self.reward()` before returning it to the caller
#
# - The `reward()` method is the only method you need to override
# - The `step()` flow:
#   ```
#   inner_env.step(action) → (obs, reward, terminated, truncated, info)
#   wrapper.reward(reward) → modified_reward
#   return (obs, modified_reward, terminated, truncated, info)
#   ```

# %%
from gymnasium import RewardWrapper


class DoubleReward(RewardWrapper):
    """
    Multiply every reward by a factor.
    """

    def __init__(self, env, factor=2.0):
        super().__init__(env)
        self._factor = factor

    def reward(self, reward):
        return reward * self._factor


# %%
# Build the wrapped env and compare rewards.
base3 = gym.make("CartPole-v1")
reward_env = DoubleReward(gym.make("CartPole-v1"), factor=3.0)

base3.reset(seed=0)
reward_env.reset(seed=0)

action = 1

# Compare step rewards.
_, rew_a, *_ = base3.step(action)
_, rew_b, *_ = reward_env.step(action)
print(f"base reward: {rew_a}   scaled reward: {rew_b}")
print(f"ratio: {rew_b / rew_a}")
hdbg.dassert_eq(rew_b, 3.0 * rew_a, "The reward must be scaled by the factor")

# %% [markdown]
# ## Primitive 5: Lambda Wrappers (Function-Based)
#
# - **Mental model**: Instead of writing a full subclass, you can pass a function
#   to `TransformObservation`, `TransformAction`, or `TransformReward`.  The
#   function is applied to every observation/action/reward
#
# - These are the "quick and clean" way for simple transformations

# %%
from gymnasium.wrappers import (
    TransformObservation,
    TransformAction,
    TransformReward,
)

# %%
# TransformObservation: add Gaussian noise to every observation.
base4 = gym.make("CartPole-v1")
# Create the generator once, so that each call draws new noise.
noise_rng = np.random.default_rng(42)
noisy_env = TransformObservation(
    gym.make("CartPole-v1"),
    func=lambda obs: obs + 0.1 * noise_rng.normal(size=obs.shape),
    observation_space=base4.observation_space,
)

obs_clean, *_ = base4.reset(seed=0)
obs_noisy, *_ = noisy_env.reset(seed=0)
data = {
    "obs_source": ["clean", "noisy"],
    "observation": [str(obs_clean), str(obs_noisy)],
}
display(pd.DataFrame(data))

# Step both envs twice: the noise is their difference, and it changes each step.
noise_1 = noisy_env.step(0)[0] - base4.step(0)[0]
noise_2 = noisy_env.step(0)[0] - base4.step(0)[0]
print("noise_1=", noise_1)
print("noise_2=", noise_2)
hdbg.dassert(
    not np.array_equal(noise_1, noise_2), "Each step must draw new noise"
)

# %%
# TransformAction: scale continuous actions for MountainCarContinuous.

base5 = gym.make("MountainCarContinuous-v0")
# Clip action magnitude to 50%.
clipped_action_env = TransformAction(
    gym.make("MountainCarContinuous-v0"),
    func=lambda a: np.clip(a, -0.5, 0.5),
    action_space=base5.action_space,
)

base5.reset(seed=0)
clipped_action_env.reset(seed=0)
# Send 1.0, outside the clip range [-0.5, 0.5], and compare the trajectory.
obs_base, *_ = base5.step(np.array([1.0]))
obs_clipped, *_ = clipped_action_env.step(np.array([1.0]))
data2 = {
    "env_source": ["base", "clipped_action"],
    "next_obs": [str(obs_base), str(obs_clipped)],
}
display(pd.DataFrame(data2))
hdbg.dassert(
    not np.array_equal(obs_base, obs_clipped), "The action must be clipped"
)
# Check the exact value with the wrapper's own `action()` method.
clipped = clipped_action_env.action(np.array([1.0]))
print("clipped=", clipped)
hdbg.dassert(np.array_equal(clipped, np.array([0.5])), "Action must clip to 0.5")

base5.close()
clipped_action_env.close()

# %%
# TransformReward: scale and shift every reward.
base6 = gym.make("CartPole-v1")
scaled_reward_env = TransformReward(
    gym.make("CartPole-v1"), func=lambda r: 2.0 * r + 1.0
)

base6.reset(seed=0)
scaled_reward_env.reset(seed=0)
_, rew_base, *_ = base6.step(1)
_, rew_scaled, *_ = scaled_reward_env.step(1)
print(f"base reward: {rew_base}   scaled reward: {rew_scaled}")
hdbg.dassert_eq(
    rew_scaled, 2.0 * rew_base + 1.0, "The reward must be transformed"
)

base6.close()
scaled_reward_env.close()

# %% [markdown]
# ## Composition Example 1: `TimeLimit` — Truncating Episodes
#
# - **Mental model**: A `TimeLimit` wrapper truncates an episode after a fixed
#   number of steps, setting `truncated=True`  instead of `terminated=True`

# %%
from gymnasium.wrappers import TimeLimit


# Make a custom env with no natural termination.
class InfiniteCartPole(gym.Env):
    """
    A CartPole that never terminates (no time limit, no failure).
    """

    def __init__(self):
        super().__init__()
        self._env = gym.make("CartPole-v1")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, _terminated, truncated, info = self._env.step(action)
        # Never terminate naturally.
        return obs, reward, False, truncated, info

    def close(self):
        self._env.close()


# %%
# Without TimeLimit, the episode could run forever.
infinite_env = InfiniteCartPole()
# With TimeLimit, the episode is truncated after 5 steps.
limited_env = TimeLimit(infinite_env, max_episode_steps=5)

limited_env.reset(seed=0)
for i in range(6):
    obs, reward, terminated, truncated, info = limited_env.step(1)
    print(f"Step {i}: terminated={terminated} truncated={truncated}")
    if terminated or truncated:
        print(f"  Episode ended at step {i}")
        break

limited_env.close()

# %% [markdown]
# ## Composition Example 2: Multiple Wrappers Stacked
#
# - **Mental model**: wrappers compose by nesting.  Each layer transforms one
#   aspect (actions, rewards, observations).  The outermost wrapper sees the final
#   result

# %%
# Stack three wrappers: one for each channel (observation, action, reward).
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics, Autoreset

base7 = gym.make("CartPole-v1")

# Layer 1 (innermost): double the reward.
w1 = DoubleReward(base7, factor=2.0)
# Layer 2: flip the observation.
w2 = FlipObservation(w1)
# Layer 3 (outermost): swap the action.
w3 = SwapAction(w2)
print("w3=", w3)

# What is the actual action -> inner step action?
w3.reset(seed=0)
print("Outer wrapper type:", type(w3).__name__)
print("w3.env type:", type(w3.env).__name__)
print("w3.env.env type:", type(w3.env.env).__name__)
print("w3.unwrapped type:", type(w3.unwrapped).__name__)

# Flow: action=0
#   → SwapAction.action(0) → 1
#   → inner env step(1)
#   → FlipObservation.observation(obs) → -obs
#   → DoubleReward.reward(reward) → 2 * reward
obs, reward, *_ = w3.step(0)
print(f"  obs (negated): {obs}")
print(f"  reward (doubled): {reward}")

w3.close()

# %% [markdown]
# ## Composition Example 3: Custom Observation Wrapper
#
# - **Problem**: a discrete observation space (e.g., frozen lake grid) is hard to
#   visualize.  Wrap it to emit one-hot encoded observations

# %%
class OneHotObservation(ObservationWrapper):
    """
    Convert a discrete observation into a one-hot vector.
    """

    def __init__(self, env):
        super().__init__(env)
        n = int(env.observation_space.n)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def observation(self, observation):
        # Build a one-hot vector for the discrete observation index.
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[int(observation)] = 1.0
        return one_hot


# %%
# Build a FrozenLake environment and wrap it.
frozen = gym.make("FrozenLake-v1", is_slippery=False)
onehot_env = OneHotObservation(frozen)

obs, _ = onehot_env.reset(seed=0)
print("Original discrete obs:", frozen.reset(seed=0)[0])
print("One-hot obs:", obs)
print("One-hot shape:", obs.shape)

# Step and see the one-hot representation.
obs2, *_ = onehot_env.step(2)  # action 2 = move right (FrozenLake layout)
print("Obs after step right:", obs2)

onehot_env.close()

# %% [markdown]
# ## Composition Example 4: Custom Action Wrapper
#
# - **Problem**: discrete actions in {0, 1, 2, 3} need to be mapped to a smaller
#   set of allowed actions

# %%
class DiscreteActionMap(ActionWrapper):
    """
    Map discrete actions from the user to different actions for the env.

    E.g., map {0, 1} -> {0, 2} (only left and right, no up/down).
    """

    def __init__(self, env, mapping):
        super().__init__(env)
        self._mapping = mapping
        # The wrapper's action space is the size of the mapping (input actions).
        self.action_space = gym.spaces.Discrete(len(mapping))

    def action(self, action):
        return self._mapping[action]


# %%
# Use FrozenLake: map {0: 0, 1: 2} so action 1 means "right".
frozen2 = gym.make("FrozenLake-v1", is_slippery=False)
mapped_env = DiscreteActionMap(frozen2, mapping={0: 0, 1: 2})

mapped_env.reset(seed=0)
obs, reward, terminated, truncated, info = mapped_env.step(1)  # 1 -> right
print("action 1 mapped to right, obs:", obs)

# The wrapper's action space only allows {0, 1}.
print("Wrapper action space:", mapped_env.action_space)

mapped_env.close()

# %% [markdown]
# ## Composition Example 5: Custom Reward Wrapper
#
# - **Problem**: you want to shape the reward with a penalty term that depends on
#   the observation

# %%
class RewardFromObservation(Wrapper):
    """
    Penalize the agent based on the distance from the observation to the origin.

    Override `step()` since `RewardWrapper.reward()` cannot see the observation.
    """

    def __init__(self, env, penalty_weight=0.01):
        super().__init__(env)
        self._penalty_weight = penalty_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self._penalty_weight * float(np.linalg.norm(obs))
        return obs, reward, terminated, truncated, info


# %%
base8 = gym.make("CartPole-v1")
penalty_env = RewardFromObservation(gym.make("CartPole-v1"), penalty_weight=0.5)

base8.reset(seed=0)
penalty_env.reset(seed=0)
_, rew_a, *_ = base8.step(1)
_, rew_b, *_ = penalty_env.step(1)
print(f"base reward: {rew_a}   penalty reward: {rew_b}  diff={rew_a - rew_b}")
hdbg.dassert_lt(rew_b, rew_a, "The penalty must lower the reward")

base8.close()
penalty_env.close()

# %% [markdown]
# ## Composition Example 6: `RecordEpisodeStatistics`
#
# - **Mental model**: tracks cumulative reward, episode length, and wall-clock
#   time for completed episodes.  Adds an `"episode"` key to the `info` dict at
#   episode end, containing `"r"` (return), `"l"` (length), `"t"` (time)

# %%
base9 = gym.make("CartPole-v1")
stats_env = RecordEpisodeStatistics(base9, buffer_length=10)
stats_env.action_space.seed(0)

# Run 3 episodes.
for _ in range(3):
    obs, info = stats_env.reset(seed=42)
    terminated, truncated = False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = stats_env.step(
            stats_env.action_space.sample()
        )
    # At episode end, info contains the statistics.
    ep_info = info.get("episode")
    if ep_info:
        print(
            f"Episode return: {ep_info['r']:.1f}  "
            f"length: {ep_info['l']}  "
            f"time: {ep_info['t']:.3f}s"
        )

# Access the rolling buffers.
print(f"Return buffer: {list(stats_env.return_queue)}")
print(f"Length buffer: {list(stats_env.length_queue)}")

stats_env.close()

# %% [markdown]
# ## Composition Example 7: `Autoreset` — Never Manual Reset
#
# - **Mental model**: `Autoreset` automatically resets the environment on the step
#   AFTER an episode ends.  This simulates the behavior of vectorized
#   environments in a single environment

# %%
base10 = gym.make("CartPole-v1")
# Limit steps to 3 so episodes end quickly, then auto-reset.
limited = TimeLimit(base10, max_episode_steps=3)
auto_env = Autoreset(limited)

# Step many times - the env auto-resets without explicit reset() calls.
auto_env.reset(seed=0)
for i in range(12):
    obs, reward, terminated, truncated, info = auto_env.step(
        auto_env.action_space.sample()
    )
    print(f"  Step {i:2d}: terminated={terminated} truncated={truncated}")
    if terminated or truncated:
        # On the step that ends, terminated/truncated is True.
        # On the next step, the env has already been reset (autoreset).
        pass

auto_env.close()

# %% [markdown]
# ## API Patterns
#
# - **Inheritance pattern**: subclass `Wrapper`, `ObservationWrapper`,
#   `ActionWrapper`, or `RewardWrapper` and override the hook method
#   (`observation()`, `action()`, or `reward()`).  Full control, most common
#
# - **Lambda/function pattern**: use `TransformObservation(func=...)`,
#   `TransformAction(func=...)`, `TransformReward(func=...)` to apply a function
#   without writing a class.  Concise for simple transforms
#
# - **Composition/chaining pattern**: wrappers stack by nesting:
#   ```
#   env = WrapperC(WrapperB(WrapperA(base_env)))
#   ```
#   Each layer intercepts and can modify the data flow.  Order matters: the
#   outermost wrapper sees the action first and the observation/reward last
#
# - **Utility wrapper pattern**: builtin wrappers like `TimeLimit`,
#   `RecordEpisodeStatistics`, `Autoreset`, `OrderEnforcing` add infrastructure
#   behavior without changing the agent's view of the env
#
# - **Space modification pattern**: when a wrapper changes the type, shape, or
#   bounds of observations or actions, update `self.observation_space` or
#   `self.action_space` in `__init__()` so downstream code knows the new contract

# %% [markdown]
# ## Interactive Exploration
#
# - Experiment with the wrapper primitives

# %%
# Inspect the Wrapper class interface.
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
import helpers.hintrospection as hintros

for cls in [Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper]:
    print(f"\n=== {cls.__name__} ===")
    hintros.print_obj_info(cls)

# %%
# Experiment: what happens when you change the order of wrappers?
base_for_order = gym.make("CartPole-v1")

# Order A: flip obs first, then scale reward.
env_a = DoubleReward(FlipObservation(base_for_order), factor=10.0)
env_a.reset(seed=0)
obs_a, rew_a, *_ = env_a.step(1)
print(
    f"Order A (flip inside, reward outside): obs={obs_a[:2]}, reward={rew_a:.1f}"
)

# Order B: scale reward first, then flip obs.
base_for_order2 = gym.make("CartPole-v1")
env_b = FlipObservation(DoubleReward(base_for_order2, factor=10.0))
env_b.reset(seed=0)
obs_b, rew_b, *_ = env_b.step(1)
print(
    f"Order B (reward inside, flip outside): obs={obs_b[:2]}, reward={rew_b:.1f}"
)

# The observation is flipped in both cases, but the reward order doesn't matter
# for independent transforms. For linked transforms, order matters!
base_for_order.close()

# %%
# Inspect the wrapper chain with `get_wrapper_attr`.
env_chain = DoubleReward(
    FlipObservation(SwapAction(gym.make("CartPole-v1"))), factor=5.0
)
# `get_wrapper_attr` searches through the wrapper stack for an attribute.
print("get_wrapper_attr('_factor'):", env_chain.get_wrapper_attr("_factor"))

# %%
# Question: does `wrapper.render()` work when the base env has render_mode=None?
# (It warns and returns None: with no render mode there is nothing to draw.)
no_render_env = gym.make("CartPole-v1", render_mode=None)
wrapper_no_render = DoubleReward(no_render_env)
# Reset first: `OrderEnforcing` raises `ResetNeeded` if `render()` comes first.
wrapper_no_render.reset(seed=0)
frame = wrapper_no_render.render()
print("frame=", frame)
hdbg.dassert_is(frame, None, "Without a render mode there is no frame")

# %%
# Question: can you wrap a wrapped environment with the same wrapper type?
# (Yes, nesting is the fundamental composition mechanism.)
env_nested = DoubleReward(
    DoubleReward(gym.make("CartPole-v1"), factor=2.0), factor=3.0
)
env_nested.reset(seed=0)
_, rew_nested, *_ = env_nested.step(1)
print(f"Reward with double wrapping (2x inner, 3x outer): {rew_nested}")

# %%
# What wrapper attributes are accessible?
# `env.env`: the directly wrapped inner environment.
env_w_spec = TimeLimit(gym.make("CartPole-v1"), max_episode_steps=100)
print("env.env type:", type(env_w_spec.env).__name__)
print("env.unwrapped type:", type(env_w_spec.unwrapped).__name__)
# `env.spec` is the EnvSpec of the innermost environment.
print("env.spec:", env_w_spec.spec)
env_w_spec.close()

# %% [markdown]
# ## Summary: The Mental Model
#
# - A `Wrapper` is an environment that delegates every method to an inner
#   environment, allowing selective interception without modifying the original
#   environment code
# - Three specialized subclasses provide clean hooks: `ObservationWrapper`
#   (override `observation()`), `ActionWrapper` (override `action()`), and
#   `RewardWrapper` (override `reward()`)
# - Lambda wrappers (`TransformObservation`, `TransformAction`,
#   `TransformReward`) let you apply a function without writing a class
# - Wrappers compose by nesting: the outermost wrapper sees the action first and
#   the observation/reward last.  The order of wrapping determines the pipeline
# - Builtin utility wrappers (`TimeLimit`, `RecordEpisodeStatistics`,
#   `Autoreset`, `OrderEnforcing`) add infrastructure behavior that every RL
#   project needs
