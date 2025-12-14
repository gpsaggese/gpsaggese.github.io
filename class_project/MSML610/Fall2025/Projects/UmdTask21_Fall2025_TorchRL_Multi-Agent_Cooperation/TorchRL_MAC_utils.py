"""Single reusable module for TorchRL multi-agent cooperation notebooks (CPU-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agent_policy.agent_policy import SharedMLPPolicy
from src.envs.mpe_env import make_mpe_env
from src.wrappers.wrapper import MPEWrapper

__all__ = [
    "EnvConfig",
    "TrainConfig",
    "RolloutBatch",
    "Policy",
    "ValueFunction",
    "make_env",
    "build_shared_actor",
    "build_central_critic",
    "select_actions",
    "collect_episode",
    "compute_returns_advantages",
    "a2c_update",
    "train_ctde_a2c",
]


# ---------------------------------------------------------------------------
# Contract layer
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    env_name: str = "simple_spread"
    seed: int = 0
    max_steps: int = 25
    device: str = "cpu"


@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_episodes: int = 500
    log_every: int = 25
    device: str = "cpu"


@dataclass
class RolloutBatch:
    """Container for one episode of experience."""

    obs: torch.Tensor  # [T, n_agents, obs_dim]
    actions: torch.Tensor  # [T, n_agents]
    rewards: torch.Tensor  # [T, n_agents]
    dones: torch.Tensor  # [T]
    logprobs: torch.Tensor  # [T]
    values: torch.Tensor  # [T]
    entropies: torch.Tensor  # [T]


class Policy(Protocol):
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (actions [n_agents], mean_logprob scalar, mean_entropy scalar)."""


class ValueFunction(Protocol):
    def value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Return scalar value for concatenated observations."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def make_env(cfg: EnvConfig) -> MPEWrapper:
    """Instantiate wrapped MPE env (parallel) with seeding and max steps control."""
    torch.manual_seed(cfg.seed)
    wrapper = MPEWrapper(device=cfg.device, max_cycles=cfg.max_steps)
    wrapper.reset(seed=cfg.seed)
    return wrapper


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def build_shared_actor(obs_dim: int, n_actions: int, hidden_sizes: Iterable[int] = (128, 128)) -> nn.Module:
    """Build a shared actor that outputs logits for each agent observation."""
    return SharedMLPPolicy(obs_dim, n_actions, hidden_dims=list(hidden_sizes))


class CentralCritic(nn.Module):
    """Centralized critic over concatenated observations."""

    def __init__(self, joint_obs_dim: int, hidden_sizes: Iterable[int] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = joint_obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.net(joint_obs).squeeze(-1)

    def value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.forward(joint_obs)


def build_central_critic(joint_obs_dim: int, hidden_sizes: Iterable[int] = (128, 128)) -> nn.Module:
    """Build centralized critic mapping joint observation to scalar value."""
    return CentralCritic(joint_obs_dim, hidden_sizes)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


def select_actions(actor: nn.Module, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample discrete actions and return actions, mean logprob, mean entropy."""
    logits = actor(obs)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    logprob_mean = dist.log_prob(actions).mean()
    entropy_mean = dist.entropy().mean()
    return actions, logprob_mean, entropy_mean


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_episode(wrapper: MPEWrapper, actor: nn.Module, critic: nn.Module, env_cfg: EnvConfig) -> RolloutBatch:
    """Collect one episode of experience using current actor/critic."""
    obs_tensors: List[torch.Tensor] = []
    actions_tensors: List[torch.Tensor] = []
    rewards_tensors: List[torch.Tensor] = []
    logprobs_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    entropies_list: List[torch.Tensor] = []
    dones_list: List[torch.Tensor] = []

    obs = wrapper.reset(seed=env_cfg.seed)
    steps = 0

    while steps < env_cfg.max_steps:
        actions, logp_mean, ent_mean = select_actions(actor, obs)
        joint_obs = obs.reshape(-1)
        value = critic.value(joint_obs)

        next_obs, rewards, done_all = wrapper.step(actions)

        obs_tensors.append(obs)
        actions_tensors.append(actions)
        rewards_tensors.append(rewards)
        logprobs_list.append(logp_mean)
        values_list.append(value)
        entropies_list.append(ent_mean)
        dones_list.append(torch.tensor(done_all, dtype=torch.float32, device=obs.device))

        obs = next_obs
        steps += 1
        if done_all:
            break

    return RolloutBatch(
        obs=torch.stack(obs_tensors, dim=0),
        actions=torch.stack(actions_tensors, dim=0),
        rewards=torch.stack(rewards_tensors, dim=0),
        dones=torch.stack(dones_list, dim=0),
        logprobs=torch.stack(logprobs_list, dim=0),
        values=torch.stack(values_list, dim=0),
        entropies=torch.stack(entropies_list, dim=0),
    )


# ---------------------------------------------------------------------------
# Returns and advantages
# ---------------------------------------------------------------------------


def compute_returns_advantages(
    rewards_mean: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute simple discounted returns and advantages (no GAE)."""
    T = rewards_mean.shape[0]
    returns = torch.zeros_like(rewards_mean)
    future_return = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            future_return = 0.0
        future_return = rewards_mean[t] + gamma * future_return
        returns[t] = future_return
    advantages = returns - values
    return returns, advantages


# ---------------------------------------------------------------------------
# Loss and update
# ---------------------------------------------------------------------------


def a2c_update(
    actor: nn.Module,
    critic: nn.Module,
    batch: RolloutBatch,
    cfg: TrainConfig,
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One A2C-style update step; returns scalar metrics."""
    rewards_mean = batch.rewards.mean(dim=1)
    returns, advantages = compute_returns_advantages(rewards_mean, batch.values, batch.dones, cfg.gamma)

    policy_loss = -(batch.logprobs * advantages.detach()).mean()
    value_loss = F.mse_loss(batch.values, returns)
    entropy_term = -cfg.entropy_coef * batch.entropies.mean()
    total_loss = policy_loss + cfg.value_coef * value_loss + entropy_term

    optim_actor.zero_grad()
    optim_critic.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
    optim_actor.step()
    optim_critic.step()

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(batch.entropies.mean().detach().cpu()),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_ctde_a2c(env_cfg: EnvConfig, train_cfg: TrainConfig) -> Dict[str, List[float]]:
    """Train a CTDE A2C baseline; returns history dict for plotting."""
    device = torch.device(train_cfg.device)
    wrapper = make_env(env_cfg)
    obs0 = wrapper.reset(seed=env_cfg.seed)
    n_agents, obs_dim = obs0.shape
    n_actions = wrapper.n_actions
    joint_obs_dim = n_agents * obs_dim

    actor = build_shared_actor(obs_dim, n_actions).to(device)
    critic = build_central_critic(joint_obs_dim).to(device)

    optim_actor = torch.optim.Adam(actor.parameters(), lr=train_cfg.lr_actor)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=train_cfg.lr_critic)

    history: Dict[str, List[float]] = {
        "episode_return": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }

    for ep in range(train_cfg.n_episodes):
        batch = collect_episode(wrapper, actor, critic, env_cfg)
        rewards_mean = batch.rewards.mean(dim=1)
        returns, advantages = compute_returns_advantages(rewards_mean, batch.values, batch.dones, train_cfg.gamma)

        # reuse batch but with computed returns/adv
        metrics = a2c_update(actor, critic, batch, train_cfg, optim_actor, optim_critic)

        ep_return = float(rewards_mean.sum().detach().cpu())
        history["episode_return"].append(ep_return)
        history["policy_loss"].append(metrics["policy_loss"])
        history["value_loss"].append(metrics["value_loss"])
        history["entropy"].append(metrics["entropy"])

    return history


# ---------------------------------------------------------------------------
# Backward-compat helpers (kept for existing notebooks)
# ---------------------------------------------------------------------------


def build_wrapped_env(device: str = "cpu", **mpe_kwargs) -> MPEWrapper:
    """Alias to create a wrapped env (compat)."""
    return MPEWrapper(device=device, **mpe_kwargs)


def infer_action_obs_dims(env_factory: Callable[[], MPEWrapper]) -> Tuple[int, int]:
    env = env_factory()
    obs = env.reset()
    _, obs_dim = obs.shape
    raw_env = make_mpe_env()
    raw_env.reset()
    act_dim = raw_env.action_space(raw_env.agents[0]).n
    raw_env.close()
    return obs_dim, act_dim


def make_shared_policy(obs_dim: int, act_dim: int, hidden_dims=None, device: str = "cpu") -> SharedMLPPolicy:
    hidden_dims = hidden_dims or [128, 64]
    policy = SharedMLPPolicy(obs_dim, act_dim, hidden_dims=hidden_dims).to(device)
    return policy


def run_stateless_rollout(env: MPEWrapper, policy: SharedMLPPolicy, num_episodes: int = 1) -> Dict[int, torch.Tensor]:
    results: Dict[int, torch.Tensor] = {}
    device = next(policy.parameters()).device
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_rewards = torch.zeros(env.num_agents, device=device)
        while not done:
            actions = policy.act(obs)
            obs, rewards, done = env.step(actions)
            ep_rewards += rewards
        results[ep] = ep_rewards.detach().cpu()
    return results
