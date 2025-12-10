import torch
from trl import PPOConfig


def get_ppo_config() -> PPOConfig:
    """
    Return a PPOConfig that is:
    - CPU friendly
    - Small batch size to avoid OOM in WSL
    """
    cfg = PPOConfig()  # start from defaults so we don't fight version changes

    # Small, safe batch size – we will enforce this in main.py
    cfg.batch_size = 64
    cfg.mini_batch_size = 8

    # Conservative learning rate
    cfg.learning_rate = 1e-5

    # Make sure we stay on CPU
    if hasattr(cfg, "device"):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"


    # Turn off fp16 / bf16 if those flags exist in this TRL version
    for attr in ["bf16", "fp16", "use_fp16", "use_bf16"]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, False)

    # Different TRL versions use different names for the PPO epoch count
    if hasattr(cfg, "num_ppo_epochs"):
        cfg.num_ppo_epochs = 1
    if hasattr(cfg, "ppo_epochs"):
        cfg.ppo_epochs = 1

    return cfg
