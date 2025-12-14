"""Wrapper to expose PettingZoo MPE env as fixed-order PyTorch tensors."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from src.envs.mpe_env import make_mpe_env

__all__ = ["MPEWrapper"]


class MPEWrapper:
    """Adapter around the PettingZoo MPE environment with stable agent ordering."""

    def __init__(self, device: str = "cpu", **mpe_kwargs):
        self.device = torch.device(device)
        self.env = make_mpe_env(**mpe_kwargs)
        self.agents = list(self.env.agents)
        first_obs, _ = self.env.reset(seed=None)
        self.num_agents = len(self.agents)
        self.obs_dim = len(first_obs[self.agents[0]])
        try:
            self.n_actions = int(self.env.action_space(self.agents[0]).n)
        except Exception:
            self.n_actions = None

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset env and return observations as tensor [num_agents, obs_dim]."""
        obs_dict, _ = self.env.reset(seed=seed)
        return self._obs_dict_to_tensor(obs_dict)

    def step(self, actions_tensor: torch.Tensor):
        """
        Step env with discrete actions [num_agents]; return (obs, rewards, done_all).
        
        done_all semantics: episode ends when ALL agents are done/truncated (not any).
        Uses fast tensor conversion with .item() for boolean checks.
        """
        if actions_tensor.ndim != 1 or actions_tensor.shape[0] != self.num_agents:
            raise ValueError(f"actions shape must be [{self.num_agents}], got {tuple(actions_tensor.shape)}")
        actions_tensor = actions_tensor.to("cpu").long()

        if self.n_actions is not None:
            # Use .item() for fast boolean conversion
            if not ((0 <= actions_tensor) & (actions_tensor < self.n_actions)).all().item():
                raise ValueError(f"actions must be in [0, {self.n_actions-1}]")

        actions = {agent: int(actions_tensor[i].item()) for i, agent in enumerate(self.agents)}
        obs_dict, rewards_dict, dones_dict, truncs_dict, _ = self.env.step(actions)

        obs_tensor = self._obs_dict_to_tensor(obs_dict)
        rewards_tensor = self._rewards_dict_to_tensor(rewards_dict)
        # done_all = True when ALL agents are done or truncated
        done_all = all(dones_dict[a] or truncs_dict[a] for a in self.agents)
        return obs_tensor, rewards_tensor, done_all

    def _obs_dict_to_tensor(self, obs_dict: Dict[str, Iterable[float]]) -> torch.Tensor:
        """Convert obs dict to tensor [num_agents, obs_dim] using numpy for speed."""
        obs_list = [np.asarray(obs_dict[a], dtype=np.float32) for a in self.agents]
        obs_array = np.stack(obs_list, axis=0)
        return torch.from_numpy(obs_array).to(self.device)

    def _rewards_dict_to_tensor(self, rewards_dict: Dict[str, float]) -> torch.Tensor:
        """Convert rewards dict to tensor [num_agents] using numpy for speed."""
        rew_array = np.array([rewards_dict[a] for a in self.agents], dtype=np.float32)
        return torch.from_numpy(rew_array).to(self.device)
