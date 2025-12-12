# src/logging/logger.py

"""
W&B Logger Module
==================
This module provides a unified logging interface that combines:
1. Python's standard logging (console + file)
2. Weights & Biases experiment tracking

Why combine both?
- Python logging: Debug issues locally, persistent logs
- W&B logging: Track experiments, compare runs, share with team
- Best of both worlds!
"""

import os
import wandb  # Weights & Biases library for experiment tracking
import logging  # Python's built-in logging module
from pathlib import Path  # For file path handling
from typing import Optional, Dict, Any  # Type hints
import sys  # For stdout/stderr access

# Import ConfigManager - we'll use it to load config
from src.utils.config import ConfigManager
# Why import here? Need config to initialize W&B with project settings


class WandbLogger:
    """
    Unified logger for Python logging and W&B tracking.
    
    This class wraps both Python's logging and W&B's API to provide
    a single interface for all logging needs.
    
    Why a class instead of functions?
    - Maintains state (W&B run, logger instance)
    - Can initialize once, use many times
    - Cleaner than passing logger around everywhere
    """
    
    def __init__(self, config_path: str = "config"):
        """
        Initialize W&B logger.
        
        Args:
            config_path: Path to config directory (default: "config")
            
        Why __init__ doesn't start W&B run?
        - Lazy initialization - only start W&B when needed
        - Can create logger without starting W&B (useful for testing)
        - More flexible - start W&B later with custom settings
        """
        # Create ConfigManager instance to load configs
        self.config_manager = ConfigManager(config_path)
        # Why ConfigManager? Reuse our config loading logic
        
        # Load W&B config (but don't initialize W&B yet)
        self.wandb_config = self.config_manager.load_wandb_config()
        # Why load now? Need config values for initialization later
        
        # Load params (for W&B run config)
        self.params = self.config_manager.load_params()
        # Why load params? W&B tracks hyperparameters - we'll log these
        
        # Initialize W&B run as None (will be set when init_run() called)
        self.run = None
        # Why None? W&B run only created when explicitly started
        # Allows creating logger without starting W&B (useful for testing)
        
        # Set up Python logging (this happens immediately)
        self.logger = self._setup_logger()
        # Why underscore prefix (_setup_logger)? Indicates "private" method
        # Convention: methods starting with _ are internal, not part of public API
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up Python logging system.
        
        Creates a logger that writes to both:
        1. File (logs/training.log) - persistent record
        2. Console (stdout) - see logs in real-time
        
        Returns:
            Configured logging.Logger instance
            
        Why separate method?
        - Keeps __init__ clean
        - Can be called again to reconfigure if needed
        - Easier to test separately
        """
        # Create logger instance with module name
        logger = logging.getLogger(__name__)
        # Why __name__? Creates logger named after this module
        # Example: logger name = "src.logging.logger"
        # Helps identify where log messages come from
        
        # Set logging level to INFO
        logger.setLevel(logging.INFO)
        # Why INFO? Shows info, warnings, errors (not debug messages)
        # Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
        # INFO level = normal operation messages
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        # Why mkdir? Ensures directory exists before creating file
        # exist_ok=True = don't error if directory already exists
        
        # ====================================================================
        # FILE HANDLER - Write logs to file
        # ====================================================================
        file_handler = logging.FileHandler(log_dir / "training.log")
        # Why FileHandler? Writes logs to a file
        # Path: logs/training.log (relative to project root)
        
        file_handler.setLevel(logging.INFO)
        # Why set level? Can have different levels for file vs console
        # Example: File gets DEBUG, console gets INFO (less noisy)
        
        # ====================================================================
        # CONSOLE HANDLER - Print logs to terminal
        # ====================================================================
        console_handler = logging.StreamHandler(sys.stdout)
        # Why StreamHandler? Writes to stdout (terminal)
        # sys.stdout = standard output stream
        
        console_handler.setLevel(logging.INFO)
        # Same level as file handler
        
        # ====================================================================
        # FORMATTER - How log messages look
        # ====================================================================
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Format breakdown:
        # %(asctime)s = timestamp (e.g., "2025-01-15 10:30:45")
        # %(name)s = logger name (e.g., "src.logging.logger")
        # %(levelname)s = log level (e.g., "INFO", "ERROR")
        # %(message)s = actual log message
        
        # Apply formatter to both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # Why same formatter? Consistent log format everywhere
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # Why add both? Now logger writes to both file and console
        
        return logger  # Return configured logger
    
    def init_run(self, 
                 run_name: Optional[str] = None,
                 tags: Optional[list] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new W&B run.
        
        This starts experiment tracking in W&B. All subsequent logs
        will be associated with this run.
        
        Args:
            run_name: Custom name for this run (optional)
            tags: List of tags for organizing runs (optional)
            config: Additional config to log (optional)
            
        Why optional parameters?
        - Can use defaults from config files
        - Can override for specific experiments
        - Flexible for different use cases
        """
        # Get W&B config section from wandb.yaml
        wandb_config = self.wandb_config.get('wandb', {})
        # Why .get('wandb', {})? Safe access - returns {} if 'wandb' key missing
        # wandb.yaml has structure: {wandb: {...}}, so we get the inner dict
        
        params = self.params  # Get params from params.yaml
        
        # Merge configs - combine params.yaml and any additional config
        run_config = {
            **params,  # All params from params.yaml
            **(config or {})  # Any additional config passed in (if provided)
        }
        # Why merge? W&B tracks hyperparameters - we want to log everything
        # **params = "unpack" dictionary (spread operator)
        # Example: {**{'a': 1}, **{'b': 2}} = {'a': 1, 'b': 2}
        
        # Allow environment-variable overrides (useful in Docker / CI).
        # - If you belong to an organization, W&B may require using a TEAM entity.
        # - Setting WANDB_ENTITY/WANDB_PROJECT here avoids hardcoding org/team names in files.
        env_project = os.getenv("WANDB_PROJECT")
        env_entity = os.getenv("WANDB_ENTITY")
        project = env_project or wandb_config.get('project', 'stock_price_forecasting')
        entity = env_entity or wandb_config.get('entity')
        if isinstance(entity, str) and entity.strip().lower() in {"", "none", "null"}:
            entity = None

        # Initialize W&B run
        self.run = wandb.init(
            # Project name from wandb.yaml
            project=project,
            # Why .get() with default? Safe access - uses default if key missing
            
            # Entity (username/team) from wandb.yaml
            entity=entity,
            # Why None allowed? W&B uses your personal account if None
            
            # Run name - use custom name or default from params.yaml
            name=run_name or params.get('wandb', {}).get('experiment_name', 'time_series_forecast'),
            # Why or operator? Use run_name if provided, else use default from params
            # run_name or default = first truthy value
            
            # Tags for organizing runs
            tags=tags or params.get('wandb', {}).get('tags', []),
            # Why or operator? Use provided tags or defaults from params
            
            # Hyperparameters/config to track
            config=run_config,
            # Why log config? W&B tracks hyperparameters automatically
            # Can compare runs with different hyperparameters
            
            # Job type (training, evaluation, etc.)
            job_type=wandb_config.get('job_type', 'training'),
            
            # Group related experiments
            group=wandb_config.get('group', 'time_series_models'),
            # Why group? If running LSTM, ARIMA, Prophet - group them together
            # Makes comparison easier in W&B dashboard
            
            # Save code snapshot
            save_code=wandb_config.get('save_code', True)
            # Why save_code? Reproducibility - know exact code version
        )
        
        # Log that W&B run started
        self.logger.info(f"W&B run initialized: {self.run.name}")
        # Why log? Confirms W&B started successfully
        # self.run.name = W&B auto-generated or custom run name
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Metrics are numerical values you want to track over time.
        Examples: loss, accuracy, MAE, RMSE
        
        Args:
            metrics: Dictionary of metric names and values
                    Example: {'loss': 0.5, 'mae': 10.2, 'rmse': 15.3}
            step: Optional step number (epoch, iteration, etc.)
                  
        Why dictionary format?
        - Can log multiple metrics at once
        - Cleaner than separate function calls
        - Matches W&B API
        
        Example usage:
            logger.log_metrics({'loss': 0.5, 'val_loss': 0.6}, step=10)
        """
        if self.run:  # Check if W&B run is active
            # Why check? Can't log if W&B not initialized
            # Prevents errors if someone forgets to call init_run()
            
            self.run.log(metrics, step=step)
            # Why self.run.log()? W&B's API for logging metrics
            # step=step = optional step number (for plotting over time)
            
            self.logger.info(f"Logged metrics: {metrics}")
            # Why also log to Python logger? See metrics in console/file too
        else:
            self.logger.warning("W&B run not initialized. Metrics not logged to W&B.")
            # Why warning? Not an error - code still works, just not tracking
    
    def log_plot(self, plot, plot_name: str) -> None:
        """
        Log a plot to W&B.
        
        Plots are visualizations (matplotlib figures, plotly graphs, etc.)
        
        Args:
            plot: Matplotlib figure or plotly graph object
            plot_name: Name for the plot in W&B dashboard
            
        Why log plots?
        - Visualize results in W&B dashboard
        - Compare plots across different runs
        - Share visualizations with team
        
        Example usage:
            fig, ax = plt.subplots()
            ax.plot(x, y)
            logger.log_plot(fig, "training_curve")
        """
        if self.run:
            # Convert plot to W&B image format
            self.run.log({plot_name: wandb.Image(plot)})
            # Why wandb.Image()? W&B's format for images/plots
            # Wraps matplotlib/plotly figures for W&B
            
            self.logger.info(f"Logged plot: {plot_name}")
        else:
            self.logger.warning("W&B run not initialized. Plot not logged to W&B.")
    
    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "model") -> None:
        """
        Log an artifact (file) to W&B.
        
        Artifacts are files you want to save: models, datasets, plots, etc.
        W&B versions artifacts automatically.
        
        Args:
            file_path: Path to the file to upload
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            
        Why artifacts?
        - Version control for models/datasets
        - Download models later
        - Share artifacts with team
        - Track which model produced which results
        
        Example usage:
            logger.log_artifact("artifacts/model.h5", "trained_lstm", "model")
        """
        if self.run:
            # Create W&B artifact object
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            # Why Artifact? W&B's way of versioning files
            # type=artifact_type = categorize artifact (model, dataset, etc.)
            
            artifact.add_file(file_path)
            # Why add_file? Tells W&B which file to upload
            
            self.run.log_artifact(artifact)
            # Why log_artifact? Uploads file to W&B servers
            # W&B versions it automatically (v0, v1, v2, etc.)
            
            self.logger.info(f"Logged artifact: {artifact_name}")
        else:
            self.logger.warning("W&B run not initialized. Artifact not logged to W&B.")
    
    def log_table(self, table_name: str, data, columns: Optional[list] = None) -> None:
        """
        Log a table to W&B.

        Args:
            table_name: Name for the table
            data: Data to log (pandas DataFrame or list of lists)
            columns: Column names (if data is list of lists)
        """
        if self.run:
            # Check if data is pandas DataFrame
            if hasattr(data, "to_dict"):
                # W&B Table requires column names to be simple types (str/int).
                # yfinance often returns MultiIndex / tuple columns, so we stringify.
                df = data.copy()
                df = df.reset_index()
                df.columns = [str(c) for c in df.columns]
                table = wandb.Table(dataframe=df)
            else:
                table = wandb.Table(columns=columns or [], data=data)

            self.run.log({table_name: table})
            self.logger.info(f"Logged table: {table_name}")
        else:
            self.logger.warning("W&B run not initialized. Table not logged to W&B.")
    
    def finish(self) -> None:
        """
        Finish the W&B run.
        
        Call this when experiment is complete. W&B will finalize
        the run and mark it as finished.
        
        Why explicit finish?
        - W&B finalizes run (uploads remaining data)
        - Marks run as complete in dashboard
        - Good practice - clean shutdown
        """
        if self.run:
            wandb.finish()  # W&B API to finish run
            self.logger.info("W&B run finished")
        else:
            self.logger.warning("No W&B run to finish.")
    
    # ========================================================================
    # CONVENIENCE METHODS - Python logging shortcuts
    # ========================================================================
    # These methods delegate to Python logger for consistency
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        # Why wrapper? Consistent API - logger.info() instead of logger.logger.info()
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)