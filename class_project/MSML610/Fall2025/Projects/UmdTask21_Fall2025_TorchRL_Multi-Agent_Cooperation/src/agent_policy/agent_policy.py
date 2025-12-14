"""Shared multi-agent policy network for discrete MPE actions."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

__all__ = ["SharedMLPPolicy"]


class SharedMLPPolicy(nn.Module):
    """Parameter-shared MLP policy for all agents (discrete actions)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int] | None = None):
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [128, 64]
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits for each agent given observations [num_agents, obs_dim]."""
        logits = self.net(obs)
        return logits  # [num_agents, act_dim]

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample integer actions for each agent; returns tensor [num_agents]."""
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        return actions

