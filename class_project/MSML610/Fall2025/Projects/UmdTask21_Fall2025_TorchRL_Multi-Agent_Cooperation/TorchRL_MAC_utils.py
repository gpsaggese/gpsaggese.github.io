"""Single reusable module for TorchRL multi-agent cooperation notebooks (MPS/CPU-safe)."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Protocol, Tuple

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
    "collect_n_steps",
    "compute_returns_advantages",
    "a3c_update",
    "train_ctde_a3c",
    "get_device",
    "resolve_device",
]
class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared states for A3C (works with spawn)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # move optimizer state to shared memory
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    # initialize if missing
                    if "step" not in state:
                        state["step"] = torch.zeros(1)
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p.data)
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # share
                    state["step"].share_memory_()
                    state["exp_avg"].share_memory_()
                    state["exp_avg_sq"].share_memory_()



# ---------------------------------------------------------------------------
# Contract layer
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    env_name: str = "simple_spread"
    seed: int = 0
    max_steps: int = 25
    device: str = "mps"  # "mps" = auto-fallback to CPU if unavailable, or "cpu" explicit


@dataclass
class TrainConfig:
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    entropy_coef: float = 0.005  # Reduced from 0.01 to allow convergence
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_episodes: int = 500  # Alias for max_episodes (for backward compatibility)
    log_every: int = 25
    device: str = "mps"  # "mps" = auto-fallback to CPU if unavailable, or "cpu" explicit
    debug_grads: bool = False  # If True, print gradient debugging info
    # A3C-specific parameters
    num_workers: int = 4  # Number of parallel A3C workers
    n_steps: int = 10  # Rollout length per update (for A3C)
    max_episodes: int = 300  # Global stop criterion for A3C
    sync_every: int = 1  # Episodes between local->global sync
    seed: int = 0  # Base seed for workers


@dataclass
class RolloutBatch:
    """Container for one episode/rollout of experience."""

    obs: torch.Tensor  # [T, n_agents, obs_dim]
    actions: torch.Tensor  # [T, n_agents]
    rewards: torch.Tensor  # [T, n_agents]
    dones: torch.Tensor  # [T] (boolean as floats)


class Policy(Protocol):
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (actions [n_agents], mean_logprob scalar, mean_entropy scalar)."""


class ValueFunction(Protocol):
    def value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Return scalar value for concatenated observations."""


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    """Resolve device string to torch.device with MPS fallback to CPU."""
    if requested == "cpu":
        return torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            # Fallback to CPU if MPS not available
            return torch.device("cpu")
    else:
        # Allow other explicit devices like "cuda:0"
        return torch.device(requested)


def get_device(device: str | None = None) -> str:
    """Legacy helper: returns device string (use resolve_device for torch.device)."""
    if device is None:
        device = "mps"
    resolved = resolve_device(device)
    return str(resolved).replace("mps:0", "mps")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def make_env(cfg: EnvConfig) -> MPEWrapper:
    """Instantiate wrapped MPE env (parallel) with seeding and max steps control."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    wrapper = MPEWrapper(device=str(device), max_cycles=cfg.max_steps)
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


def collect_episode(
    wrapper: MPEWrapper, actor: nn.Module, env_cfg: EnvConfig, max_steps: int | None = None
) -> RolloutBatch:
    """
    Collect one rollout of experience using current actor (no critic needed).
    Does NOT store logprobs/values/entropies - they will be recomputed during update.
    """
    obs_tensors: List[torch.Tensor] = []
    actions_tensors: List[torch.Tensor] = []
    rewards_tensors: List[torch.Tensor] = []
    dones_list: List[torch.Tensor] = []

    obs = wrapper.reset(seed=None)  # Don't fix seed - allow diverse episodes
    steps = 0
    max_steps = max_steps or env_cfg.max_steps

    while steps < max_steps:
        with torch.no_grad():
            actions, _, _ = select_actions(actor, obs)

        next_obs, rewards, done_all = wrapper.step(actions)

        obs_tensors.append(obs)
        actions_tensors.append(actions)
        rewards_tensors.append(rewards)
        # Use .item() to convert bool to Python bool for proper conditional logic
        dones_list.append(torch.tensor(1.0 if done_all else 0.0, dtype=torch.float32, device=obs.device))

        obs = next_obs
        steps += 1
        if done_all:
            break

    return RolloutBatch(
        obs=torch.stack(obs_tensors, dim=0),
        actions=torch.stack(actions_tensors, dim=0),
        rewards=torch.stack(rewards_tensors, dim=0),
        dones=torch.stack(dones_list, dim=0),
    )


def collect_n_steps(
    wrapper: MPEWrapper, actor: nn.Module, env_cfg: EnvConfig, n_steps: int, obs: torch.Tensor | None = None
) -> Tuple[RolloutBatch, torch.Tensor, bool]:
    """
    Collect n-step rollout for A3C.
    
    Args:
        wrapper: Environment wrapper
        actor: Policy network
        env_cfg: Environment configuration
        n_steps: Number of steps to collect
        obs: Current observation (if None, will reset env)
    
    Returns:
        batch: RolloutBatch with collected experience
        next_obs: Final observation after n steps
        done_all: Whether episode ended
    """
    obs_tensors: List[torch.Tensor] = []
    actions_tensors: List[torch.Tensor] = []
    rewards_tensors: List[torch.Tensor] = []
    dones_list: List[torch.Tensor] = []
    
    if obs is None:
        obs = wrapper.reset(seed=None)
    
    done_all = False
    for step in range(n_steps):
        with torch.no_grad():
            actions, _, _ = select_actions(actor, obs)
        
        next_obs, rewards, done_all = wrapper.step(actions)
        
        obs_tensors.append(obs)
        actions_tensors.append(actions)
        rewards_tensors.append(rewards)
        dones_list.append(torch.tensor(1.0 if done_all else 0.0, dtype=torch.float32, device=obs.device))
        
        obs = next_obs
        if done_all:
            break
    
    batch = RolloutBatch(
        obs=torch.stack(obs_tensors, dim=0),
        actions=torch.stack(actions_tensors, dim=0),
        rewards=torch.stack(rewards_tensors, dim=0),
        dones=torch.stack(dones_list, dim=0),
    )
    
    return batch, obs, done_all


# ---------------------------------------------------------------------------
# Returns and advantages
# ---------------------------------------------------------------------------


def compute_returns_advantages(
    rewards_team: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    bootstrap_value: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute discounted returns and advantages with optional n-step bootstrapping.
    Uses consistent team reward aggregation (sum over agents).

    Args:
        rewards_team: [T] tensor of team rewards (already summed over agents)
        values: [T] tensor of value estimates
        dones: [T] tensor of done flags (1.0 or 0.0)
        gamma: discount factor
        bootstrap_value: scalar tensor V(s_T) when rollout truncates without done
    """
    T = rewards_team.shape[0]
    returns = torch.zeros_like(rewards_team)

    if bootstrap_value is None:
        future_return = torch.zeros((), device=rewards_team.device)
    else:
        future_return = bootstrap_value  # scalar tensor

    for t in reversed(range(T)):
        # Reset future return if episode actually terminated at t
        if float(dones[t].item()) > 0.5:
            future_return = torch.zeros((), device=rewards_team.device)
        future_return = rewards_team[t] + gamma * future_return
        returns[t] = future_return
    advantages = returns - values
    return returns, advantages


# ---------------------------------------------------------------------------
# Loss and update
# ---------------------------------------------------------------------------


def a3c_update(
    actor: nn.Module,
    critic: nn.Module,
    batch: RolloutBatch,
    cfg: TrainConfig,
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
) -> Dict[str, float]:
    """
    One A3C-style update step; returns scalar metrics.
    Recomputes logits/logprobs/entropy/values from stored obs/actions.
    Uses consistent team reward aggregation (sum over agents).
    
    NOTE: .item() is ONLY used on detached tensors after backward pass (in return dict).
    Losses (policy_loss, value_loss, entropy_bonus) keep gradients during backward.
    """
    # Extract dimensions and get n_actions from actor output
    T, n_agents, obs_dim = batch.obs.shape
    obs_flat = batch.obs.reshape(T * n_agents, obs_dim)
    
    # Forward pass through actor to get logits
    logits_flat = actor(obs_flat)  # [T*n_agents, n_actions]
    n_actions = logits_flat.shape[-1]
    logits = logits_flat.reshape(T, n_agents, n_actions)  # [T, n_agents, n_actions]
    
    # Create distribution and compute log probs and entropy
    dist = Categorical(logits=logits)
    logprob_agents = dist.log_prob(batch.actions)  # [T, n_agents]
    logprob_mean = logprob_agents.mean(dim=1)  # [T]
    entropy_mean = dist.entropy().mean(dim=1)  # [T]
    
    # Compute values from joint observations
    joint_obs = batch.obs.reshape(T, -1)  # [T, n_agents*obs_dim]
    values = critic.value(joint_obs)  # [T]
    
    # Aggregate rewards as team reward (sum over agents)
    rewards_team = batch.rewards.sum(dim=1)  # [T]
    
    # Compute returns and advantages
    returns, advantages = compute_returns_advantages(rewards_team, values, batch.dones, cfg.gamma)
    
    # Normalize advantages for stable gradients
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute losses (keep gradients!)
    policy_loss = -(logprob_mean * advantages.detach()).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = -cfg.entropy_coef * entropy_mean.mean()
    total_loss = policy_loss + cfg.value_coef * value_loss + entropy_bonus
    
    # Debug: print gradient info if enabled (before backward)
    if cfg.debug_grads:
        print(f"\n=== Gradient Debug (Pre-Backward) ===")
        print(f"policy_loss.requires_grad: {policy_loss.requires_grad}")
        print(f"value_loss.requires_grad: {value_loss.requires_grad}")
        print(f"total_loss.requires_grad: {total_loss.requires_grad}")
    
    # Backpropagation
    optim_actor.zero_grad()
    optim_critic.zero_grad()
    total_loss.backward()
    
    # Debug: print mean abs gradient of actor params (after backward)
    if cfg.debug_grads:
        actor_grad_sum = 0.0
        actor_param_count = 0
        for param in actor.parameters():
            if param.grad is not None:
                actor_grad_sum += param.grad.abs().sum().item()
                actor_param_count += param.grad.numel()
        mean_abs_grad = actor_grad_sum / actor_param_count if actor_param_count > 0 else 0.0
        print(f"Mean abs grad of actor params: {mean_abs_grad:.6e}")
        print(f"Entropy mean: {entropy_mean.mean().item():.4f} (expected to decrease from ln(n_actions)={float('inf') if n_actions != 5 else 1.609:.4f})")
        print(f"======================================\n")
    
    # Clip gradients and update
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm)
    optim_actor.step()
    optim_critic.step()
    
    # Return metrics (safe to use .item() on detached tensors)
    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_mean.mean().detach().cpu()),
        "episode_return": float(rewards_team.sum().detach().cpu()),
    }


# ---------------------------------------------------------------------------
# A3C Training
# ---------------------------------------------------------------------------


def _a3c_worker(
    rank: int,
    global_actor: nn.Module,
    global_critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    lock: Any,
    counter: Any,
    queue: Any,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
):
    """
    A3C worker process.
    
    Each worker:
    - Maintains its own environment
    - Has local actor/critic copies
    - Collects n-step rollouts
    - Computes gradients locally
    - Copies gradients to global networks under lock
    - Syncs weights from global periodically
    """
    # Set unique seed for this worker
    worker_seed = train_cfg.seed + rank * 1000
    torch.manual_seed(worker_seed)
    
    # Force CPU for A3C workers (MPS doesn't work well with multiprocessing)
    device = torch.device("cpu")
    
    # Create local environment
    local_env_cfg = EnvConfig(
        env_name=env_cfg.env_name,
        seed=worker_seed,
        max_steps=env_cfg.max_steps,
        device="cpu"
    )
    wrapper = make_env(local_env_cfg)
    obs = wrapper.reset(seed=worker_seed)
    n_agents, obs_dim = obs.shape
    n_actions = wrapper.n_actions
    joint_obs_dim = n_agents * obs_dim
    
    # Create local actor and critic
    local_actor = build_shared_actor(obs_dim, n_actions).to(device)
    local_critic = build_central_critic(joint_obs_dim).to(device)
    
    # Sync initial weights from global
    with lock:
        local_actor.load_state_dict(global_actor.state_dict())
        local_critic.load_state_dict(global_critic.state_dict())
    
    episode_count = 0
    obs = None  # Will trigger reset on first collect_n_steps
    
    while True:
        # Check if we should stop
        with lock:
            if counter.value >= train_cfg.max_episodes:
                break
        
        # Collect n-step rollout
        batch, obs, done = collect_n_steps(wrapper, local_actor, local_env_cfg, train_cfg.n_steps, obs)
        
        # If episode ended, reset obs for next iteration
        if done:
            obs = None
            episode_count += 1
        
        # Compute gradients locally
        local_actor.zero_grad()
        local_critic.zero_grad()
        
        # Recompute everything from batch
        T, n_agents, obs_dim = batch.obs.shape
        obs_flat = batch.obs.reshape(T * n_agents, obs_dim)
        logits_flat = local_actor(obs_flat)
        n_actions = logits_flat.shape[-1]
        logits = logits_flat.reshape(T, n_agents, n_actions)
        
        dist = Categorical(logits=logits)
        logprob_agents = dist.log_prob(batch.actions)
        logprob_mean = logprob_agents.mean(dim=1)
        entropy_mean = dist.entropy().mean(dim=1)
        
        joint_obs = batch.obs.reshape(T, -1)
        values = local_critic.value(joint_obs)
        
        rewards_team = batch.rewards.sum(dim=1)
        # Bootstrap with value of next joint observation if rollout truncated by n_steps
        bootstrap = None
        if not done:
            with torch.no_grad():
                next_joint_obs = obs.reshape(1, -1)  # obs is next_obs here
                bootstrap = local_critic.value(next_joint_obs).squeeze(0)
        returns, advantages = compute_returns_advantages(
            rewards_team, values, batch.dones, train_cfg.gamma, bootstrap
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute loss
        policy_loss = -(logprob_mean * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_bonus = -train_cfg.entropy_coef * entropy_mean.mean()
        total_loss = policy_loss + train_cfg.value_coef * value_loss + entropy_bonus
        
        # Debug: print gradient info if enabled (before backward)
        if train_cfg.debug_grads and episode_count % train_cfg.log_every == 0 and episode_count > 0:
            print(f"\n[Worker {rank}] Gradient Debug (Episode {episode_count}):")
            print(f"  policy_loss.requires_grad: {policy_loss.requires_grad}")
            print(f"  entropy_mean.mean(): {entropy_mean.mean().item():.4f}")
        
        # Backward
        total_loss.backward()
        
        # Debug: print mean abs gradient after backward
        if train_cfg.debug_grads and episode_count % train_cfg.log_every == 0 and episode_count > 0:
            actor_grad_sum = 0.0
            actor_param_count = 0
            for param in local_actor.parameters():
                if param.grad is not None:
                    actor_grad_sum += param.grad.abs().sum().item()
                    actor_param_count += param.grad.numel()
            mean_abs_grad = actor_grad_sum / actor_param_count if actor_param_count > 0 else 0.0
            print(f"  Mean abs grad of actor params: {mean_abs_grad:.6e}\n")
        
        # Copy gradients to global networks and update under lock
        with lock:
            # Clear existing grads on global parameters
            optimizer.zero_grad(set_to_none=True)
            # Copy gradients from local to global (assign to .grad)
            for local_param, global_param in zip(local_actor.parameters(), global_actor.parameters()):
                if local_param.grad is not None:
                    global_param.grad = local_param.grad
            for local_param, global_param in zip(local_critic.parameters(), global_critic.parameters()):
                if local_param.grad is not None:
                    global_param.grad = local_param.grad
            # Clip and step optimizer
            torch.nn.utils.clip_grad_norm_(
                list(global_actor.parameters()) + list(global_critic.parameters()),
                train_cfg.max_grad_norm
            )
            optimizer.step()
            
            # If episode ended, increment counter and log
            if done:
                episode_return = rewards_team.sum().item()
                counter.value += 1
                queue.put({
                    "episode_return": episode_return,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy_mean.mean().item(),
                    "rank": rank,
                })
            
            # Sync local weights from global every sync_every episodes
            if episode_count % train_cfg.sync_every == 0:
                local_actor.load_state_dict(global_actor.state_dict())
                local_critic.load_state_dict(global_critic.state_dict())


def train_ctde_a3c(env_cfg: EnvConfig, train_cfg: TrainConfig) -> Dict[str, List[float]]:
    """
    Train a CTDE A3C model with asynchronous multi-process workers.
    
    Note: A3C with multiprocessing uses CPU for workers (MPS doesn't support shared memory).
    Global networks are kept on CPU for stability.
    
    Returns:
        history: Dict with episode_return, policy_loss, value_loss, entropy lists
    """
    # Set multiprocessing start method (must be done inside function, not at import)
    mp.set_start_method("spawn", force=True)
    
    # Force CPU for A3C multiprocessing (log once)
    if train_cfg.num_workers > 1:
        print(f"[A3C] Using CPU for {train_cfg.num_workers} workers (MPS doesn't support multiprocessing)")
        device = torch.device("cpu")
    else:
        device = resolve_device(train_cfg.device)
    
    torch.manual_seed(train_cfg.seed)
    
    # Initialize environment to get dimensions
    wrapper = make_env(env_cfg)
    obs0 = wrapper.reset(seed=train_cfg.seed)
    n_agents, obs_dim = obs0.shape
    n_actions = wrapper.n_actions
    joint_obs_dim = n_agents * obs_dim
    wrapper.env.close()
    
    # Create global networks on CPU (for shared memory)
    global_actor = build_shared_actor(obs_dim, n_actions).to(device)
    global_critic = build_central_critic(joint_obs_dim).to(device)
    
    # Enable shared memory for global networks
    global_actor.share_memory()
    global_critic.share_memory()
    
    # Create shared optimizer
    optimizer = SharedAdam(
        list(global_actor.parameters()) + list(global_critic.parameters()),
        lr=train_cfg.lr_actor
    )
    
    # Shared counters and queues
    lock = mp.Lock()
    counter = mp.Value('i', 0)  # Episode counter
    queue = mp.Queue()  # Metrics queue
    
    # Spawn workers
    processes = []
    for rank in range(train_cfg.num_workers):
        p = mp.Process(
            target=_a3c_worker,
            args=(rank, global_actor, global_critic, optimizer, lock, counter, queue, env_cfg, train_cfg)
        )
        p.start()
        processes.append(p)
    
    # Collect metrics from queue
    history: Dict[str, List[float]] = {
        "episode_return": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }
    
    last_logged = 0
    expected_entropy = float('inf')  # Expected entropy: ln(n_actions) for uniform distribution
    while True:
        # Check if training is done
        with lock:
            episodes_done = counter.value
        
        if episodes_done >= train_cfg.max_episodes:
            break
        
        # Collect metrics from queue
        while not queue.empty():
            try:
                metrics = queue.get_nowait()
                history["episode_return"].append(metrics["episode_return"])
                history["policy_loss"].append(metrics["policy_loss"])
                history["value_loss"].append(metrics["value_loss"])
                history["entropy"].append(metrics["entropy"])
            except:
                break
        
        # Log progress with rolling average and entropy validation
        if episodes_done >= last_logged + train_cfg.log_every and episodes_done > 0:
            if len(history["episode_return"]) > 0:
                recent_returns = history["episode_return"][-train_cfg.log_every:]
                avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0
                
                recent_entropy = history["entropy"][-train_cfg.log_every:] if history["entropy"] else []
                avg_entropy = sum(recent_entropy) / len(recent_entropy) if recent_entropy else 0
                
                # Entropy validation: should start near ln(n_actions) and decrease
                if expected_entropy == float('inf') and len(history["entropy"]) > 0:
                    expected_entropy = history["entropy"][0]  # First entropy value
                    print(f"[Entropy Check] Initial entropy: {expected_entropy:.4f} (expected ~ln(5)={float('inf') if n_actions != 5 else 1.609:.4f})")
                
                entropy_trend = "(steady)" if len(history["entropy"]) < 2 else \
                    ("(decreasing)" if history["entropy"][-1] < history["entropy"][max(0, len(history["entropy"])-train_cfg.log_every-1)] else "(increasing)")
                
                print(f"Episode {episodes_done:3d}/{train_cfg.max_episodes} | "
                      f"Avg Return: {avg_return:8.2f} | "
                      f"Entropy: {avg_entropy:.4f} {entropy_trend}")
            last_logged = episodes_done
    
    # Wait for all workers to finish
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    # Drain remaining metrics
    while not queue.empty():
        try:
            metrics = queue.get_nowait()
            history["episode_return"].append(metrics["episode_return"])
            history["policy_loss"].append(metrics["policy_loss"])
            history["value_loss"].append(metrics["value_loss"])
            history["entropy"].append(metrics["entropy"])
        except:
            break
    
    return history


# ---------------------------------------------------------------------------
# Backward-compat helpers (kept for existing notebooks)
# ---------------------------------------------------------------------------


def build_wrapped_env(device: str = "mps", **mpe_kwargs) -> MPEWrapper:
    """Alias to create a wrapped env (compat)."""
    resolved = resolve_device(device)
    return MPEWrapper(device=str(resolved), **mpe_kwargs)


def infer_action_obs_dims(env_factory: Callable[[], MPEWrapper]) -> Tuple[int, int]:
    env = env_factory()
    obs = env.reset()
    _, obs_dim = obs.shape
    raw_env = make_mpe_env()
    raw_env.reset()
    act_dim = raw_env.action_space(raw_env.agents[0]).n
    raw_env.close()
    return obs_dim, act_dim


def make_shared_policy(obs_dim: int, act_dim: int, hidden_dims=None, device: str = "mps") -> SharedMLPPolicy:
    hidden_dims = hidden_dims or [128, 64]
    resolved = resolve_device(device)
    policy = SharedMLPPolicy(obs_dim, act_dim, hidden_dims=hidden_dims).to(resolved)
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
