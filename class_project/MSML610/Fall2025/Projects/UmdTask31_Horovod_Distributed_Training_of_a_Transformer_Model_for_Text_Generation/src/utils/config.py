"""
Configuration management utilities.

Handles loading and saving YAML configuration files.
"""

import os
from typing import Any, Dict, Optional
import yaml


class Config:
    """
    Configuration container with nested attribute access.
    
    Allows accessing config values via dot notation:
        config.model.d_model instead of config['model']['d_model']
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize from a dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any):
        """Support dictionary-style assignment."""
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_dict()})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with nested attribute access.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Config object to save.
        save_path: Path where to save the configuration.
    """
    dir_path = os.path.dirname(save_path)
    if dir_path and not os.path.exists(dir_path):  # Only create directory if path contains a directory
       os.makedirs(dir_path, exist_ok=True)
    
    config_dict = config.to_dict() if isinstance(config, Config) else config
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge override values into base configuration.
    
    Args:
        base_config: Base configuration.
        override_dict: Dictionary with values to override.
        
    Returns:
        New Config with merged values.
    """
    base_dict = base_config.to_dict()
    
    def recursive_merge(base: dict, override: dict) -> dict:
        """Recursively merge nested dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = recursive_merge(base_dict, override_dict)
    return Config(merged_dict)

