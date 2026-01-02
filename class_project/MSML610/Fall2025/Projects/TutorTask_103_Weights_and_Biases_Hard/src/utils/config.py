# src/utils/config.py

"""
Configuration Manager Utility
=============================
This module provides a centralized way to load and access configuration files.
It uses the Singleton pattern to ensure only one config instance exists.

Why Singleton?
- Prevents loading config files multiple times (wasteful)
- Ensures all parts of code use same config values
- Easy to mock in tests (replace singleton instance)
"""

import yaml  # Library to parse YAML files
import os  # For file system operations
from pathlib import Path  # Modern way to handle file paths (better than os.path)
from typing import Dict, Any  # Type hints for better code documentation


class ConfigManager:
    """
    Manages loading and accessing configuration files.
    
    This class follows the Singleton pattern - only one instance should exist.
    All parts of your code can import the same config_manager instance.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing config files (default: "config")
            
        Why __init__ doesn't load files immediately:
        - Lazy loading - only load when needed
        - Faster startup if config not needed
        - Can change config_dir before loading
        """
        self.config_dir = Path(config_dir)  # Convert to Path object
        # Why Path? Better than os.path - cross-platform, cleaner syntax
        # Example: Path("config") / "params.yaml" works on Windows/Mac/Linux
        
        self.params = None  # Will store params.yaml contents
        # Why None initially? Lazy loading - load when first accessed
        
        self.wandb_config = None  # Will store wandb.yaml contents
        # Why None initially? Same reason - load when needed
    
    def load_params(self, filename: str = "params.yaml") -> Dict[str, Any]:
        """
        Load parameters from YAML file.
        
        Args:
            filename: Name of the params file (default: "params.yaml")
            
        Returns:
            Dictionary containing all parameters from YAML file
            
        Why this method?
        - Separates file loading logic from access logic
        - Can reload config if file changes (useful for long-running scripts)
        - Returns dict so caller knows what they got
        """
        params_path = self.config_dir / filename  # Build full path
        # Example: Path("config") / "params.yaml" = "config/params.yaml"
        # Works on all operating systems
        
        if not params_path.exists():  # Check if file exists
            # Why check? Better error message than YAML parser error
            raise FileNotFoundError(f"Params file not found: {params_path}")
            # Why FileNotFoundError? Specific exception type
            # Other code can catch this specific error and handle it
        
        with open(params_path, 'r') as file:  # Open file in read mode
            # Why 'with' statement? Automatically closes file when done
            # Prevents file handle leaks if code crashes
            
            self.params = yaml.safe_load(file)  # Parse YAML to Python dict
            # Why safe_load? Only loads basic YAML (no code execution)
            # yaml.load() can execute Python code (security risk!)
            # safe_load() is safe - only loads data structures
        
        return self.params  # Return loaded params
        # Why return? Allows caller to use return value if needed
        # Also stores in self.params for later access
    
    def load_wandb_config(self, filename: str = "wandb.yaml") -> Dict[str, Any]:
        """
        Load W&B configuration from YAML file.
        
        Args:
            filename: Name of the wandb config file (default: "wandb.yaml")
            
        Returns:
            Dictionary containing W&B configuration
            
        Why separate method from load_params?
        - Separation of concerns - different configs, different methods
        - Can load one without the other (if only need params)
        - Clearer code - explicit about what you're loading
        """
        wandb_path = self.config_dir / filename  # Build path to wandb.yaml
        
        if not wandb_path.exists():  # Check file exists
            raise FileNotFoundError(f"W&B config file not found: {wandb_path}")
        
        with open(wandb_path, 'r') as file:  # Open file
            self.wandb_config = yaml.safe_load(file)  # Parse YAML
        
        return self.wandb_config  # Return config dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        This is the main method you'll use to access config values.
        Example: config.get('model.lstm.learning_rate')
        
        Args:
            key: Dot-separated key path (e.g., 'data_collection.ticker_symbol')
            default: Default value if key not found (default: None)
            
        Returns:
            Configuration value, or default if not found
            
        Why dot notation?
        - Cleaner than nested dict access: config['model']['lstm']['learning_rate']
        - More readable: config.get('model.lstm.learning_rate')
        - Handles missing keys gracefully (returns default)
        
        Example usage:
            ticker = config.get('data_collection.ticker_symbol', 'AAPL')
            learning_rate = config.get('model.lstm.learning_rate', 0.001)
        """
        if self.params is None:  # Check if params loaded
            # Why check? Lazy loading - auto-load if not already loaded
            self.load_params()  # Load params.yaml automatically
        
        keys = key.split('.')  # Split 'model.lstm.learning_rate' into ['model', 'lstm', 'learning_rate']
        # Why split? Need to navigate nested dictionary structure
        
        value = self.params  # Start at root of params dict
        # Why start at root? Will navigate down through nested keys
        
        for k in keys:  # Iterate through each key level
            # Example: for 'model.lstm.learning_rate'
            # First iteration: k='model', value = params['model']
            # Second iteration: k='lstm', value = params['model']['lstm']
            # Third iteration: k='learning_rate', value = params['model']['lstm']['learning_rate']
            
            if isinstance(value, dict) and k in value:  # Check if current level is dict and key exists
                # isinstance(value, dict) = is value a dictionary?
                # k in value = does this key exist in the dictionary?
                # Why both checks? Prevents errors if structure is wrong
                
                value = value[k]  # Navigate one level deeper
                # Example: value = value['model'] moves from root to 'model' section
            else:
                return default  # Key not found, return default value
                # Why return default? Graceful failure - don't crash, use default
                # Example: config.get('nonexistent.key', 'default_value') returns 'default_value'
        
        return value  # Return final value
        # Example: Returns 0.001 for 'model.lstm.learning_rate'


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
# Create one global instance that all code can import and use
config_manager = ConfigManager()
# Why create instance here? Singleton pattern - one instance for entire program
# Any file can do: from src.utils.config import config_manager
# All files get the same instance, same config values

# Why not create instance in each file?
# - Would load config files multiple times (wasteful)
# - Different instances might have different values if config changes
# - Singleton ensures consistency across entire codebase