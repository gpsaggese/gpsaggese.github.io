import torch
import torch.nn as nn

class SharedMLPPolicy(nn.Module):
    """
    Shared policy: same network used for all agents.
    Input: observations [num_agents, obs_dim]
    Output: Categorical actions for each agent.
    """

    def __init__(self, obs_dim, act_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        # obs: [num_agents, obs_dim]
        logits = self.net(obs)
        return logits  # [num_agents, act_dim]

    def act(self, obs):
        """
        Returns integer actions for each agent as a tensor [num_agents].
        Currently: random-ish, because network is untrained.
        """
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        return actions

