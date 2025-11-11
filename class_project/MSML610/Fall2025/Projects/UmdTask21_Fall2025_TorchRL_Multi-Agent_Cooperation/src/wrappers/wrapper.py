 # src/train/mpe_wrapper.py
import torch
from src.envs.mpe_env import make_mpe_env  # should create a PettingZoo MPE env (parallel or aec with .step(actions_dict))

class MPEWrapper:
    """
    Adapter around the PettingZoo MPE environment.
    - Internally: dict[agent] obs/reward from PettingZoo.
    - Externally: PyTorch tensors with fixed agent ordering.
    Assumes discrete actions (set continuous_actions=False in make_mpe_env()).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.env = make_mpe_env()
        self.agents = list(self.env.agents)  # fixed order snapshot
        first_obs, _ = self.env.reset(seed=None)
        self.num_agents = len(self.agents)
        self.obs_dim = len(first_obs[self.agents[0]])
        # Discrete action count (PettingZoo action_space has .n)
        try:
            self.n_actions = int(self.env.action_space(self.agents[0]).n)
        except Exception:
            self.n_actions = None  

    def reset(self, seed: int | None = None):
        obs_dict, _ = self.env.reset(seed=seed)
        return self._obs_dict_to_tensor(obs_dict)

    def step(self, actions_tensor: torch.Tensor):
        """
        actions_tensor: Long/int tensor [num_agents] with discrete actions.
        Returns:
          obs_tensor: [num_agents, obs_dim] float32
          rewards_tensor: [num_agents] float32
          done_all: bool
        """
        if actions_tensor.ndim != 1 or actions_tensor.shape[0] != self.num_agents:
            raise ValueError(f"actions shape must be [{self.num_agents}], got {tuple(actions_tensor.shape)}")
        actions_tensor = actions_tensor.to("cpu").long()  # env expects Python ints

        # validating range for discrete actions
        if self.n_actions is not None:
            if not bool(((0 <= actions_tensor) & (actions_tensor < self.n_actions)).all()):
                raise ValueError(f"actions must be in [0, {self.n_actions-1}]")

        actions = {agent: int(actions_tensor[i].item()) for i, agent in enumerate(self.agents)}
        obs_dict, rewards_dict, dones_dict, truncs_dict, _ = self.env.step(actions)

        obs_tensor = self._obs_dict_to_tensor(obs_dict)
        rewards_tensor = self._rewards_dict_to_tensor(rewards_dict)
        done_all = all(dones_dict[a] or truncs_dict[a] for a in self.agents)
        return obs_tensor, rewards_tensor, done_all

    # helpers 
    def _obs_dict_to_tensor(self, obs_dict):
        obs_list = [obs_dict[a] for a in self.agents]
        return torch.tensor(obs_list, dtype=torch.float32, device=self.device)

    def _rewards_dict_to_tensor(self, rewards_dict):
        rew_list = [rewards_dict[a] for a in self.agents]
        return torch.tensor(rew_list, dtype=torch.float32, device=self.device)
