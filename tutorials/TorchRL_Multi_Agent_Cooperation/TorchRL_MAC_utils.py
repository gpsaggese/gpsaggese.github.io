"""
MAC (Multi-Agent Communication) Utilities for PettingZoo MPE
simple_reference with TorchRL.

Actor and Critic Neural Net Model Setup.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

_LOG = logging.getLogger(__name__)

# #############################################################################
# Configuration Setup
# #############################################################################


@dataclass
class MacConfig:
    """
    Configuration for Multi Agent training and evaluation.
    """

    # Environment.
    env_name: str = "simple_reference"
    num_envs: int = 4
    max_cycles: int = 25
    continuous_actions: bool = False
    # Network architecture.
    hidden_dim: int = 128
    actor_layers: int = 2
    critic_layers: int = 2
    # Training.
    num_iters: int = 1000
    rollout_len: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Evaluation.
    eval_episodes: int = 100
    success_threshold: float = -150.0  # Reward threshold for success.
    # System.
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10

    def to_dict(self) -> dict:
        """
        Convert config to dictionary.
        """
        return asdict(self)


def default_cfg() -> MacConfig:
    """
    Return default MacConfig instance.
    """
    return MacConfig()


# #############################################################################
# Environment Creation
# #############################################################################


def make_env(cfg: MacConfig):
    """
    Create PettingZoo MPE simple_reference environment wrapped with TorchRL.

    :param cfg: MacConfig instance with environment settings.
    :return: TorchRL-wrapped PettingZoo environment.
    :raises ImportError: If TorchRL PettingZoo wrapper not found.
    """
    # Set seed for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torchrl_wrapper = None
    try:
        from torchrl.envs.libs.pettingzoo import PettingZooEnv

        torchrl_wrapper = PettingZooEnv
    except ImportError:
        try:
            from torchrl.envs.libs.pettingzoo import PettingZooWrapper

            torchrl_wrapper = PettingZooWrapper
        except ImportError:
            raise ImportError(
                "Could not import TorchRL PettingZoo wrapper. "
                "Expected one of:\n"
                "  - torchrl.envs.libs.pettingzoo.PettingZooEnv\n"
                "  - torchrl.envs.libs.pettingzoo.PettingZooWrapper\n"
                "Please install torchrl with PettingZoo support."
            )
    env_version = "v3"
    try:
        from pettingzoo.mpe import simple_reference_v3  # noqa: F401
    except (ImportError, AttributeError):
        try:
            from pettingzoo.mpe import simple_reference_v2  # noqa: F401

            env_version = "v2"
        except (ImportError, AttributeError):
            raise ImportError(
                "Could not import simple_reference from pettingzoo.mpe. "
                "Please install pettingzoo: pip install pettingzoo[mpe]"
            )
    task_name = f"mpe/simple_reference_{env_version}"
    try:
        env = torchrl_wrapper(
            task=task_name,
            parallel=True,
            categorical_actions=not cfg.continuous_actions,
            seed=cfg.seed,
            done_on_any=False,
            max_cycles=cfg.max_cycles,
            continuous_actions=cfg.continuous_actions,
        )
    except (TypeError, RuntimeError):
        # Fallback: create environment manually without TorchRL wrapper.
        if env_version == "v3":
            from pettingzoo.mpe import simple_reference_v3

            env = simple_reference_v3.parallel_env(
                max_cycles=cfg.max_cycles,
                continuous_actions=cfg.continuous_actions,
            )
        else:
            from pettingzoo.mpe import simple_reference_v2

            env = simple_reference_v2.parallel_env(
                max_cycles=cfg.max_cycles,
                continuous_actions=cfg.continuous_actions,
            )
    return env


def get_agent_names(env) -> List[str]:
    """
    Get list of agent names from environment.

    :param env: TorchRL-wrapped PettingZoo environment.
    :return: List of agent names (e.g., ['speaker_0', 'listener_0']).
    """
    if hasattr(env, "_env"):
        pz_env = env._env
    elif hasattr(env, "env"):
        pz_env = env.env
    else:
        pz_env = env
    # Get agents list.
    if hasattr(pz_env, "possible_agents"):
        return pz_env.possible_agents
    elif hasattr(pz_env, "agents"):
        return pz_env.agents
    else:
        # Fallback for simple_reference.
        return ["speaker_0", "listener_0"]


# #############################################################################
# Neural Network Modules
# #############################################################################


def _orthogonal_init(layer, gain=1.0):
    """
    Apply orthogonal initialization to a linear layer.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


class MLPNetwork(nn.Module):
    """
    Enhanced Multi-layer perceptron network.

    Features: GELU activation, Layer Normalization, and Orthogonal
    Initialization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        layers = []
        # Input layer.
        fc1 = nn.Linear(input_dim, hidden_dim)
        _orthogonal_init(fc1)
        layers.append(fc1)
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        # Hidden layers (deeper and stabilized).
        for _ in range(num_layers - 1):
            fc = nn.Linear(hidden_dim, hidden_dim)
            _orthogonal_init(fc)
            layers.append(fc)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        # Output layer.
        fc_out = nn.Linear(hidden_dim, output_dim)
        _orthogonal_init(fc_out, gain=0.01)
        layers.append(fc_out)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)


class DiscreteActor(nn.Module):
    """
    Actor network for discrete action spaces (Categorical policy).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.mlp = MLPNetwork(obs_dim, action_dim, hidden_dim, num_layers)

    def forward(self, obs):
        """
        Forward pass computing action distribution.
        """
        logits = self.mlp(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, action):
        """
        Evaluate actions for PPO update.
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return log_prob, entropy

    def get_null_action(self, batch_size: int, device: str):
        """
        Return null (zero) action for the given batch size.
        """
        return torch.zeros(batch_size, dtype=torch.long, device=device)


class ContinuousActor(nn.Module):
    """
    Actor network for continuous action spaces (Gaussian policy).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_bounds: Tuple[float, float],
    ):
        super().__init__()
        self.mlp = MLPNetwork(obs_dim, 2 * action_dim, hidden_dim, num_layers)
        self.action_dim = action_dim
        self.action_low = action_bounds[0]
        self.action_high = action_bounds[1]
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

    def forward(self, obs):
        """
        Forward pass computing action distribution.
        """
        output = self.mlp(obs)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return Normal(mean, std)

    def get_action(self, obs, *, deterministic: bool = False):
        """
        Sample or compute deterministic action from the policy.
        """
        dist = self.forward(obs)
        if deterministic:
            action_raw = dist.mean
        else:
            action_raw = dist.rsample()
        action = torch.tanh(action_raw)
        log_prob = dist.log_prob(action_raw)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action_scaled = action * self.action_scale + self.action_bias
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_scaled, log_prob, entropy

    def evaluate_actions(self, obs, action):
        """
        Evaluate actions for PPO update (reverse the scaling).
        """
        # Unscale action to [-1, 1] range for tanh.
        action_norm = (action - self.action_bias) / (self.action_scale + 1e-8)
        action_norm = torch.clamp(action_norm, -0.999, 0.999)
        action_raw = torch.atanh(action_norm)
        dist = self.forward(obs)
        log_prob = dist.log_prob(action_raw)
        log_prob = log_prob - torch.log(1 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def get_null_action(self, batch_size: int, device: str):
        """
        Return null (zero) action for the given batch size.
        """
        return torch.zeros(batch_size, self.action_dim, device=device)


class CentralizedCritic(nn.Module):
    """
    Centralized critic network for multi-agent value estimation.
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.mlp = MLPNetwork(state_dim, 1, hidden_dim, num_layers)

    def forward(self, state):
        """
        Forward pass computing state value estimate.
        """
        return self.mlp(state)


def build_modules(cfg, env) -> Tuple[Dict[str, nn.Module], nn.Module, dict]:
    """
    Build actor networks and centralized critic using the enhanced MLP classes.
    """
    # Get agent names.
    if hasattr(env, "agents"):
        agent_names = env.agents
    else:
        # Fallback for some wrappers.
        agent_names = env.unwrapped.agents
    # Access underlying PettingZoo environment to get spaces.
    if hasattr(env, "_env"):
        pz_env = env._env
    elif hasattr(env, "env"):
        pz_env = env.env
    else:
        pz_env = env
    specs = {
        "agent_names": agent_names,
        "obs_dims": {},
        "action_dims": {},
        "action_types": {},
        "action_bounds": {},
    }
    total_obs_dim = 0
    for agent in agent_names:
        obs_space = pz_env.observation_space(agent)
        action_space = pz_env.action_space(agent)
        # Compute obs dim.
        if hasattr(obs_space, "shape") and len(obs_space.shape) > 0:
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = int(obs_space.n)
        specs["obs_dims"][agent] = obs_dim
        total_obs_dim += obs_dim
        # Compute action dim.
        if hasattr(action_space, "n"):
            action_dim = int(action_space.n)
            specs["action_types"][agent] = "discrete"
            specs["action_bounds"][agent] = None
        else:
            action_dim = int(np.prod(action_space.shape))
            specs["action_types"][agent] = "continuous"
            specs["action_bounds"][agent] = (
                float(action_space.low[0]),
                float(action_space.high[0]),
            )
        specs["action_dims"][agent] = action_dim
    actors = {}
    for agent in agent_names:
        obs_dim = specs["obs_dims"][agent]
        action_dim = specs["action_dims"][agent]
        action_type = specs["action_types"][agent]
        if action_type == "discrete":
            actor = DiscreteActor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.actor_layers,
            )
        else:
            action_bounds = specs["action_bounds"][agent]
            actor = ContinuousActor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.actor_layers,
                action_bounds=action_bounds,
            )
        actors[agent] = actor.to(cfg.device)
    critic = CentralizedCritic(
        state_dim=total_obs_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.critic_layers,
    ).to(cfg.device)
    return actors, critic, specs


# #############################################################################
# Rollout Collection
# #############################################################################


def collect_rollout(
    cfg: MacConfig,
    env,
    actors: Dict[str, nn.Module],
    critic: Optional[nn.Module] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Collect rollout from environment using current policies.

    :param cfg: MacConfig instance.
    :param env: TorchRL-wrapped environment.
    :param actors: Dict of actor networks per agent.
    :param critic: Centralized critic (optional).
    :param device: Device to store tensors on.
    :return: batch dictionary containing obs, actions, logp, entropy, reward,
        done, state, and values (if critic provided).
    """
    if device is None:
        device = cfg.device
    agent_names = get_agent_names(env)
    # Access underlying PettingZoo environment.
    if hasattr(env, "_env"):
        pz_env = env._env
    elif hasattr(env, "env"):
        pz_env = env.env
    else:
        pz_env = env
    # Initialize storage.
    batch = {
        "obs": {agent: [] for agent in agent_names},
        "actions": {agent: [] for agent in agent_names},
        "logp": {agent: [] for agent in agent_names},
        "entropy": {agent: [] for agent in agent_names},
        "reward": [],
        "done": [],
        "state": [],
    }
    if critic is not None:
        batch["values"] = []
    # Collect rollouts from multiple environments in parallel.
    for env_idx in range(cfg.num_envs):
        # Reset environment.
        reset_result = pz_env.reset(seed=cfg.seed + env_idx)
        # Handle TorchRL TensorDict or plain dict.
        if isinstance(reset_result, tuple):
            obs_dict = (
                reset_result[0] if len(reset_result) > 0 else reset_result
            )
        else:
            obs_dict = reset_result
        # Collect rollout_len steps.
        for step in range(cfg.rollout_len):
            # Convert observations to tensors.
            obs_tensors = {}
            for agent in agent_names:
                # Handle TensorDict from TorchRL.
                if hasattr(obs_dict, "get"):
                    obs = obs_dict.get(
                        agent, obs_dict.get(("agents", agent))
                    )
                else:
                    obs = (
                        obs_dict[agent]
                        if isinstance(obs_dict, dict)
                        else obs_dict
                    )
                if not isinstance(obs, torch.Tensor):
                    obs = torch.FloatTensor(obs).flatten()
                else:
                    obs = obs.flatten()
                obs_tensors[agent] = obs.unsqueeze(0).to(device)
            # Create global state (concatenate all observations).
            state = torch.cat(
                [obs_tensors[agent] for agent in agent_names], dim=-1
            )
            # Get actions and log probs from actors.
            actions_dict = {}
            logp_dict = {}
            entropy_dict = {}
            for agent in agent_names:
                actor = actors[agent]
                obs = obs_tensors[agent]
                with torch.no_grad():
                    if isinstance(actor, DiscreteActor):
                        dist = actor(obs)
                        action = dist.sample()
                        logp = dist.log_prob(action)
                        entropy = dist.entropy()
                        actions_dict[agent] = action.cpu().numpy()[0]
                    else:
                        # ContinuousActor.
                        action, logp, entropy = actor.get_action(obs)
                        actions_dict[agent] = action.cpu().numpy()[0]
                    logp_dict[agent] = logp
                    entropy_dict[agent] = entropy
            # Get value estimate.
            if critic is not None:
                with torch.no_grad():
                    value = critic(state)
            # Step environment.
            step_result = pz_env.step(actions_dict)
            # Handle TorchRL TensorDict or plain tuple.
            if len(step_result) == 5:
                (
                    next_obs_dict,
                    rewards_dict,
                    dones_dict,
                    truncs_dict,
                    infos_dict,
                ) = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 1:
                # TorchRL returns a single TensorDict.
                td = step_result[0]
                next_obs_dict = {
                    agent: td.get(agent, td.get(("agents", agent)))
                    for agent in agent_names
                }
                rewards_dict = {
                    agent: td.get(
                        ("next", "reward", agent),
                        td.get(("reward", agent), 0),
                    )
                    for agent in agent_names
                }
                dones_dict = {
                    agent: td.get(
                        ("next", "done", agent),
                        td.get(("done", agent), False),
                    )
                    for agent in agent_names
                }
                truncs_dict = {agent: False for agent in agent_names}
                infos_dict = {agent: {} for agent in agent_names}
            else:
                (
                    next_obs_dict,
                    rewards_dict,
                    dones_dict,
                    truncs_dict,
                    infos_dict,
                ) = (
                    step_result[0],
                    step_result[1],
                    step_result[2],
                    step_result[3],
                    step_result[4],
                )
            # Compute team reward (sum of individual rewards).
            team_reward = (
                sum(rewards_dict.values())
                if isinstance(rewards_dict, dict)
                else sum(rewards_dict)
            )
            # Check if episode done.
            done_vals = (
                dones_dict.values()
                if isinstance(dones_dict, dict)
                else dones_dict
            )
            trunc_vals = (
                truncs_dict.values()
                if isinstance(truncs_dict, dict)
                else truncs_dict
            )
            done = any(done_vals) or any(trunc_vals)
            # Store transition.
            for agent in agent_names:
                batch["obs"][agent].append(obs_tensors[agent].cpu())
                if isinstance(actors[agent], DiscreteActor):
                    batch["actions"][agent].append(
                        torch.LongTensor([actions_dict[agent]])
                    )
                else:
                    batch["actions"][agent].append(
                        torch.FloatTensor([actions_dict[agent]])
                    )
                batch["logp"][agent].append(logp_dict[agent].cpu())
                batch["entropy"][agent].append(entropy_dict[agent].cpu())
            batch["reward"].append(torch.FloatTensor([team_reward]))
            batch["done"].append(torch.FloatTensor([float(done)]))
            batch["state"].append(state.cpu())
            if critic is not None:
                batch["values"].append(value.cpu())
            # Update observation.
            obs_dict = next_obs_dict
            # Reset if done.
            if done:
                reset_result = pz_env.reset(
                    seed=cfg.seed + env_idx + step * 1000
                )
                if isinstance(reset_result, tuple):
                    obs_dict = (
                        reset_result[0]
                        if len(reset_result) > 0
                        else reset_result
                    )
                else:
                    obs_dict = reset_result
    # Stack tensors.
    for agent in agent_names:
        batch["obs"][agent] = torch.cat(batch["obs"][agent], dim=0)
        batch["actions"][agent] = torch.cat(batch["actions"][agent], dim=0)
        batch["logp"][agent] = torch.cat(batch["logp"][agent], dim=0)
        batch["entropy"][agent] = torch.cat(batch["entropy"][agent], dim=0)
    batch["reward"] = torch.cat(batch["reward"], dim=0)
    batch["done"] = torch.cat(batch["done"], dim=0)
    batch["state"] = torch.cat(batch["state"], dim=0)
    if critic is not None:
        batch["values"] = torch.cat(batch["values"], dim=0)
    return batch


# #############################################################################
# Loss Computation
# #############################################################################


def compute_loss(
    cfg,
    batch: dict,
    actors: Dict[str, nn.Module],
    critic: nn.Module,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PPO loss with GAE.
    """
    device = cfg.device
    agent_names = list(actors.keys())
    # 1. Prepare data.
    rewards = batch["reward"].to(device)  # [T, 1].
    dones = batch["done"].to(device)  # [T, 1].
    states = batch["state"].to(device)  # [T, state_dim].
    old_values = batch["values"].to(device).detach()  # [T, 1].
    T = rewards.shape[0]
    # Ensure rewards and dones are 1D.
    if rewards.dim() > 1:
        rewards = rewards.squeeze()
    if dones.dim() > 1:
        dones = dones.squeeze()
    # 2. Compute GAE (Generalized Advantage Estimation).
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    # Bootstrap value.
    with torch.no_grad():
        if dones[-1] < 0.5:
            next_value = critic(states[-1:]).squeeze().item()
        else:
            next_value = 0.0
    # Calculate GAE backwards.
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].item()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t].item()
            next_val = old_values[t + 1].item()
        delta = (
            rewards[t].item()
            + cfg.gamma * next_val * next_non_terminal
            - old_values[t].item()
        )
        # Using hardcoded lambda=0.95.
        lastgaelam = (
            delta + cfg.gamma * 0.95 * next_non_terminal * lastgaelam
        )
        advantages[t] = lastgaelam
    # Ensure old_values is 1D for proper addition.
    old_values_flat = (
        old_values.squeeze() if old_values.dim() > 1 else old_values
    )
    returns = advantages + old_values_flat
    # Normalize advantages.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 3. PPO Update Loop.
    # Config defaults if not present.
    ppo_epochs = getattr(cfg, "ppo_epochs", 4)
    clip_param = getattr(cfg, "clip_param", 0.2)
    mini_batch_size = T // 4  # Default to 4 minibatches.
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy_loss = 0
    # Create indices for shuffling.
    indices = np.arange(T)
    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = indices[start:end]
            # Get mini-batch data for shared critic.
            mb_states = states[mb_inds]
            mb_returns = returns[mb_inds]
            mb_advantages = advantages[mb_inds]
            # Ensure all tensors are 1D for this mini-batch.
            if mb_returns.dim() > 1:
                mb_returns = mb_returns.squeeze()
            if mb_advantages.dim() > 1:
                mb_advantages = mb_advantages.squeeze()
            # Critic update.
            new_values = critic(mb_states).squeeze()
            critic_loss = F.mse_loss(new_values, mb_returns)
            # Actor update (per agent).
            actor_loss_sum = 0
            entropy_sum = 0
            for agent in agent_names:
                # Get agent specific mini-batch data.
                mb_obs = batch["obs"][agent].to(device)[mb_inds]
                mb_actions = batch["actions"][agent].to(device)[mb_inds]
                mb_old_logp = (
                    batch["logp"][agent].to(device)[mb_inds].detach()
                )
                actor = actors[agent]
                # Evaluate current policy.
                new_logp, dist_entropy = actor.evaluate_actions(
                    mb_obs, mb_actions
                )
                # PPO ratio.
                ratio = torch.exp(new_logp - mb_old_logp).squeeze(-1)
                # Surrogate loss.
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                    * mb_advantages
                )
                # Maximize objective => minimize negative loss.
                agent_actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss_sum += agent_actor_loss
                entropy_sum += dist_entropy.mean()
            # Aggregate losses.
            loss = (
                actor_loss_sum
                + cfg.value_coef * critic_loss
                - cfg.entropy_coef * entropy_sum
            )
            # Track metrics.
            total_actor_loss += actor_loss_sum.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_sum.item()
    # Single pass calculation for gradient graph.
    final_critic_loss = F.mse_loss(critic(states), returns)
    final_actor_loss = 0
    final_entropy = 0
    for agent in agent_names:
        obs = batch["obs"][agent].to(device)
        actions = batch["actions"][agent].to(device)
        old_logp = batch["logp"][agent].to(device).detach()
        new_logp, entropy = actors[agent].evaluate_actions(obs, actions)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * advantages
        )
        final_actor_loss += -torch.min(surr1, surr2).mean()
        final_entropy += entropy.mean()
    final_loss = (
        final_actor_loss
        + cfg.value_coef * final_critic_loss
        - cfg.entropy_coef * final_entropy
    )
    info = {
        "loss": final_loss.item(),
        "actor_loss": final_actor_loss.item(),
        "critic_loss": final_critic_loss.item(),
        "entropy": final_entropy.item(),
        "mean_return": returns.mean().item(),
        "mean_value": old_values.mean().item(),
    }
    return final_loss, info


# #############################################################################
# Training
# #############################################################################


def train(cfg: MacConfig) -> Tuple[str, dict]:
    """
    Train multi-agent system with centralized A3C.

    :param cfg: MacConfig instance.
    :return: checkpoint_path (path to saved checkpoint) and stats (training
        statistics).
    """
    _LOG.info("Starting training with config:\n%s", cfg)
    # Create environment.
    env = make_env(cfg)
    # Build modules.
    actors, critic, specs = build_modules(cfg, env)
    # Optimizer.
    params = list(critic.parameters())
    for actor in actors.values():
        params.extend(actor.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    # Training loop.
    stats = {
        "losses": [],
        "returns": [],
        "entropies": [],
    }
    for iteration in range(cfg.num_iters):
        # Collect rollout.
        batch = collect_rollout(cfg, env, actors, critic, device=cfg.device)
        # Compute loss.
        loss, info = compute_loss(cfg, batch, actors, critic)
        # Optimize.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        # Log stats.
        stats["losses"].append(info["loss"])
        stats["returns"].append(info["mean_return"])
        stats["entropies"].append(info["entropy"])
        if (iteration + 1) % cfg.log_interval == 0:
            _LOG.info(
                "Iter %d/%d | Loss: %.3f | Return: %.3f | Entropy: %.3f",
                iteration + 1,
                cfg.num_iters,
                info["loss"],
                info["mean_return"],
                info["entropy"],
            )
    # Save checkpoint.
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        cfg.checkpoint_dir, f"mac_checkpoint_iter{cfg.num_iters}.pt"
    )
    checkpoint = {
        "actors_state": {
            agent: actor.state_dict() for agent, actor in actors.items()
        },
        "critic_state": critic.state_dict(),
        "cfg": cfg.to_dict(),
        "specs": specs,
        "iteration": cfg.num_iters,
    }
    torch.save(checkpoint, checkpoint_path)
    final_positive_score = -info["mean_return"]
    _LOG.info("Training complete! Checkpoint saved to: %s", checkpoint_path)
    _LOG.info("Final loss: %.3f", info["loss"])
    _LOG.info("Final success score: %.3f", final_positive_score)
    _LOG.info("Final entropy: %.3f", info["entropy"])
    return checkpoint_path, stats


# #############################################################################
# Checkpoint Loading
# #############################################################################


def load_checkpoint(path: str, device: Optional[str] = None) -> dict:
    """
    Load checkpoint from disk.
    """
    if device is None:
        device = "cpu"
    checkpoint = torch.load(path, map_location=device)
    # Reconstruct config.
    cfg = MacConfig(**checkpoint["cfg"])
    cfg.device = device
    # Reconstruct environment to get specs.
    env = make_env(cfg)
    # Rebuild modules.
    actors, critic, specs = build_modules(cfg, env)
    # Load state dicts.
    for agent, actor in actors.items():
        actor.load_state_dict(checkpoint["actors_state"][agent])
        actor.eval()
    critic.load_state_dict(checkpoint["critic_state"])
    critic.eval()
    return {
        "actors": actors,
        "critic": critic,
        "cfg": cfg,
        "specs": specs,
        "env": env,
        "iteration": checkpoint.get("iteration", 0),
    }


# #############################################################################
# Evaluation
# #############################################################################


def evaluate(
    cfg: MacConfig, ckpt_path: str, *, mode: str = "normal"
) -> dict:
    """
    Evaluate trained policy.

    :param cfg: MacConfig instance (used for eval_episodes, etc).
    :param ckpt_path: Path to checkpoint.
    :param mode: "normal" for normal eval, "no_comm" to silence speaker.
    :return: metrics dict with success_rate, comm_cost, comm_gain,
        comm_efficiency.
    """
    assert mode in ["normal", "no_comm"], f"Invalid mode: {mode}"
    _LOG.info("Evaluating in mode: %s", mode)
    # Load checkpoint.
    ckpt = load_checkpoint(ckpt_path, device=cfg.device)
    actors = ckpt["actors"]
    env = ckpt["env"]
    agent_names = ckpt["specs"]["agent_names"]
    speaker_name = None
    for agent in agent_names:
        if "speaker" in agent.lower():
            speaker_name = agent
            break
    if speaker_name is None:
        speaker_name = agent_names[0]
    # Access underlying PettingZoo environment.
    if hasattr(env, "_env"):
        pz_env = env._env
    elif hasattr(env, "env"):
        pz_env = env.env
    else:
        pz_env = env
    # Run evaluation episodes.
    successes = []
    comm_costs = []
    for ep in range(cfg.eval_episodes):
        reset_result = pz_env.reset(seed=cfg.seed + ep)
        if isinstance(reset_result, tuple):
            obs_dict = (
                reset_result[0] if len(reset_result) > 0 else reset_result
            )
        else:
            obs_dict = reset_result
        done = False
        episode_reward = 0
        episode_comm_cost = 0
        step_count = 0
        while not done and step_count < cfg.max_cycles:
            # Get actions.
            actions_dict = {}
            for agent in agent_names:
                # Handle TensorDict from TorchRL.
                if hasattr(obs_dict, "get"):
                    obs = obs_dict.get(
                        agent, obs_dict.get(("agents", agent))
                    )
                else:
                    obs = (
                        obs_dict[agent]
                        if isinstance(obs_dict, dict)
                        else obs_dict
                    )
                if not isinstance(obs, torch.Tensor):
                    obs = torch.FloatTensor(obs).flatten()
                else:
                    obs = obs.flatten()
                obs_tensor = obs.unsqueeze(0).to(cfg.device)
                actor = actors[agent]
                with torch.no_grad():
                    if isinstance(actor, DiscreteActor):
                        if mode == "no_comm" and agent == speaker_name:
                            # Force null action (action 0).
                            action = 0
                        else:
                            dist = actor(obs_tensor)
                            action = dist.sample().item()
                        actions_dict[agent] = action
                        # Track comm cost for discrete (nonzero action
                        # fraction).
                        if agent == speaker_name:
                            episode_comm_cost += float(action != 0)
                    else:
                        # ContinuousActor.
                        if mode == "no_comm" and agent == speaker_name:
                            # Force null action (zeros).
                            action = actor.get_null_action(
                                1, cfg.device
                            ).cpu().numpy()[0]
                        else:
                            action, _, _ = actor.get_action(
                                obs_tensor, deterministic=True
                            )
                            action = action.cpu().numpy()[0]
                        actions_dict[agent] = action
                        # Track comm cost for continuous (L2 norm).
                        if agent == speaker_name:
                            episode_comm_cost += float(
                                np.linalg.norm(action)
                            )
            # Environment step.
            step_result = pz_env.step(actions_dict)
            # Handle TorchRL TensorDict.
            if len(step_result) == 5:
                (
                    next_obs_dict,
                    rewards_dict,
                    dones_dict,
                    truncs_dict,
                    infos_dict,
                ) = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 1:
                # TorchRL returns a single TensorDict.
                td = step_result[0]
                next_obs_dict = {
                    agent: td.get(agent, td.get(("agents", agent)))
                    for agent in agent_names
                }
                rewards_dict = {
                    agent: td.get(
                        ("next", "reward", agent),
                        td.get(("reward", agent), 0),
                    )
                    for agent in agent_names
                }
                dones_dict = {
                    agent: td.get(
                        ("next", "done", agent),
                        td.get(("done", agent), False),
                    )
                    for agent in agent_names
                }
                truncs_dict = {agent: False for agent in agent_names}
                infos_dict = {agent: {} for agent in agent_names}
            else:
                (
                    next_obs_dict,
                    rewards_dict,
                    dones_dict,
                    truncs_dict,
                    infos_dict,
                ) = (
                    step_result[0],
                    step_result[1],
                    step_result[2],
                    step_result[3],
                    step_result[4],
                )
            episode_reward += (
                sum(rewards_dict.values())
                if isinstance(rewards_dict, dict)
                else sum(rewards_dict)
            )
            done_vals = (
                dones_dict.values()
                if isinstance(dones_dict, dict)
                else dones_dict
            )
            trunc_vals = (
                truncs_dict.values()
                if isinstance(truncs_dict, dict)
                else truncs_dict
            )
            done = any(done_vals) or any(trunc_vals)
            obs_dict = next_obs_dict
            step_count += 1
        # Determine success.
        success = False
        for agent_info in infos_dict.values():
            if isinstance(agent_info, dict):
                if "is_success" in agent_info:
                    success = agent_info["is_success"]
                    break
                elif "success" in agent_info:
                    success = agent_info["success"]
                    break
        # Fallback: infer from reward threshold.
        if not success:
            success = episode_reward >= cfg.success_threshold
        successes.append(float(success))
        # Average comm cost per step.
        avg_comm_cost = episode_comm_cost / max(step_count, 1)
        comm_costs.append(avg_comm_cost)
    # Compute metrics.
    success_rate = np.mean(successes)
    comm_cost = np.mean(comm_costs)
    avg_reward = (
        np.mean([episode_reward]) if "episode_reward" in locals() else 0
    )
    positive_score = -avg_reward
    metrics = {
        "success_rate": success_rate,
        "comm_cost": comm_cost,
        "positive_score": positive_score,
    }
    _LOG.info("  Success Rate: %.3f", success_rate)
    _LOG.info("  Comm Cost: %.4f", comm_cost)
    _LOG.info("  Positive Score: %.3f", positive_score)
    # If this is normal mode, we can't compute gain/efficiency yet.
    if mode == "normal":
        metrics["comm_gain"] = 0.0
        metrics["comm_efficiency"] = 0.0
    else:
        # For no_comm mode, set to 0 (would need normal results to compare).
        metrics["comm_gain"] = 0.0
        metrics["comm_efficiency"] = 0.0
    return metrics


def evaluate_with_comparison(cfg: MacConfig, ckpt_path: str) -> dict:
    """
    Evaluate in both normal and no_comm modes and compute full metrics.

    :param cfg: MacConfig instance.
    :param ckpt_path: Path to checkpoint.
    :return: metrics dict with all metrics including comm_gain and
        comm_efficiency.
    """
    # Evaluate normal mode.
    normal_metrics = evaluate(cfg, ckpt_path, mode="normal")
    # Evaluate no_comm mode.
    no_comm_metrics = evaluate(cfg, ckpt_path, mode="no_comm")
    # Compute gain and efficiency.
    success_normal = normal_metrics["success_rate"]
    success_no_comm = no_comm_metrics["success_rate"]
    comm_cost = normal_metrics["comm_cost"]
    comm_gain = success_normal - success_no_comm
    comm_efficiency = comm_gain / (comm_cost + 1e-8)
    metrics = {
        "success_rate": success_normal,
        "comm_cost": comm_cost,
        "comm_gain": comm_gain,
        "comm_efficiency": comm_efficiency,
        "success_rate_no_comm": success_no_comm,
    }
    _LOG.info("=== Final Metrics ===")
    _LOG.info("Success Rate (normal): %.3f", success_normal)
    _LOG.info("Success Rate (no_comm): %.3f", success_no_comm)
    _LOG.info("Comm Cost: %.4f", comm_cost)
    _LOG.info("Comm Gain: %.3f", comm_gain)
    _LOG.info("Comm Efficiency: %.3f", comm_efficiency)
    return metrics


def train_wrapper(cfg):
    """
    Train and plot training curves.
    """
    checkpoint_path, stats = train(cfg)
    # Plot training curves.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Loss.
    axes[0].plot(stats["losses"], "b-", linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    # Returns.
    axes[1].plot(stats["returns"], "g-", linewidth=2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean Return")
    axes[1].set_title("Episode Returns")
    axes[1].grid(True, alpha=0.3)
    # Entropy.
    axes[2].plot(stats["entropies"], "r-", linewidth=2)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Mean Entropy")
    axes[2].set_title("Policy Entropy")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mac_training_curves.png", dpi=100, bbox_inches="tight")
    plt.show()
    _LOG.info("Training curves saved to: mac_training_curves.png")


# #############################################################################
# Main (for testing)
# #############################################################################


if __name__ == "__main__":
    _LOG.info("Testing MAC utilities...")
    cfg = default_cfg()
    cfg.num_iters = 2
    cfg.rollout_len = 8
    cfg.num_envs = 2
    _LOG.info("1. Creating environment...")
    env = make_env(cfg)
    _LOG.info("   Agents: %s", get_agent_names(env))
    _LOG.info("2. Building modules...")
    actors, critic, specs = build_modules(cfg, env)
    _LOG.info("   Actor networks: %s", list(actors.keys()))
    _LOG.info("   Obs dims: %s", specs["obs_dims"])
    _LOG.info("   Action dims: %s", specs["action_dims"])
    _LOG.info("3. Collecting rollout...")
    batch = collect_rollout(cfg, env, actors, critic)
    _LOG.info("   Collected %d transitions", batch["reward"].shape[0])
    _LOG.info("4. Computing loss...")
    loss, info = compute_loss(cfg, batch, actors, critic)
    _LOG.info("   Loss: %.3f", loss.item())
    _LOG.info("5. Running short training...")
    ckpt_path, stats = train(cfg)
    _LOG.info("6. Loading checkpoint...")
    ckpt = load_checkpoint(ckpt_path)
    _LOG.info(
        "   Loaded checkpoint from iteration %d", ckpt["iteration"]
    )
    _LOG.info("7. Evaluating...")
    cfg.eval_episodes = 5
    metrics = evaluate_with_comparison(cfg, ckpt_path)
    _LOG.info("All tests passed!")
