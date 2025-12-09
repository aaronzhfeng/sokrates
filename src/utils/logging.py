"""
Centralized logging and experiment tracking for SOKRATES.

Provides:
- Timestamped experiment directories
- File-based logging with timestamps
- Metrics tracking for later plotting
- JSON-serializable run configs
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(
    base_dir: str,
    experiment_name: str,
    timestamp: Optional[str] = None,
    create_latest_symlink: bool = True,
) -> Path:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir: Base output directory (e.g., "outputs")
        experiment_name: Name of the experiment (e.g., "sft", "oak_dpo")
        timestamp: Optional timestamp (auto-generated if None)
        create_latest_symlink: Create/update "latest" symlink
    
    Returns:
        Path to experiment directory (e.g., outputs/sft/20241209_143022)
    
    Example:
        outputs/
        └── sft/
            ├── 20241209_143022/    # Previous run (preserved)
            ├── 20241209_151530/    # Current run
            └── latest -> 20241209_151530/  # Symlink to most recent
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    exp_dir = Path(base_dir) / experiment_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create/update "latest" symlink for easy access
    if create_latest_symlink:
        latest_link = Path(base_dir) / experiment_name / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(timestamp)
        except OSError:
            pass  # Symlinks may not work on all systems
    
    return exp_dir


def setup_logging(
    experiment_dir: Path,
    log_filename: str = "run.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logging to both file and console with timestamps.
    
    Args:
        experiment_dir: Directory to save log file
        log_filename: Name of log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("sokrates")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # File handler
    log_path = experiment_dir / log_filename
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_path}")
    
    return logger


@dataclass
class MetricsTracker:
    """
    Track metrics over time for later plotting.
    
    Saves metrics incrementally to a JSONL file.
    """
    
    experiment_dir: Path
    filename: str = "metrics.jsonl"
    _metrics_file: Path = field(init=False)
    _step: int = field(default=0, init=False)
    
    def __post_init__(self):
        self._metrics_file = self.experiment_dir / self.filename
        # Write header comment
        with open(self._metrics_file, 'w') as f:
            f.write(f"# SOKRATES Metrics Log - {datetime.now().isoformat()}\n")
    
    def log(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        """
        Log metrics at a given step.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step number (auto-incremented if None)
            phase: Phase name (e.g., "train", "eval", "oak_iter")
        """
        if step is None:
            step = self._step
            self._step += 1
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "phase": phase,
            **metrics
        }
        
        with open(self._metrics_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_iteration(
        self,
        iteration: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log metrics for an OaK iteration."""
        self.log(metrics, step=iteration, phase="oak_iteration")
    
    def log_eval(
        self,
        metrics: dict[str, Any],
        dataset_name: str = "test",
    ) -> None:
        """Log evaluation metrics."""
        self.log(metrics, phase=f"eval_{dataset_name}")
    
    def get_all_metrics(self) -> list[dict]:
        """Load all logged metrics."""
        metrics = []
        with open(self._metrics_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                metrics.append(json.loads(line.strip()))
        return metrics


@dataclass
class ExperimentConfig:
    """
    Serializable experiment configuration.
    
    Captures all settings for reproducibility.
    """
    
    experiment_name: str
    timestamp: str
    model_name: str = ""
    dataset: str = ""
    
    # Training settings
    num_epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    
    # OaK settings
    oak_iterations: int = 0
    samples_per_problem: int = 0
    dpo_beta: float = 0.0
    
    # Hardware
    gpu: str = ""
    seed: int = 42
    
    # Additional settings
    extra: dict = field(default_factory=dict)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentLogger:
    """
    Complete experiment logging setup.
    
    Creates timestamped directory, sets up logging,
    tracks metrics, and saves configuration.
    
    Usage:
        exp = ExperimentLogger("sft", base_dir="outputs")
        exp.log_config({"model": "llama", "epochs": 3})
        exp.logger.info("Starting training...")
        exp.metrics.log({"loss": 0.5, "accuracy": 0.8})
        exp.finish()
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "outputs",
        timestamp: Optional[str] = None,
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of experiment
            base_dir: Base output directory
            timestamp: Optional timestamp (auto-generated if None)
        """
        self.experiment_name = experiment_name
        self.timestamp = timestamp or get_timestamp()
        
        # Create experiment directory
        self.exp_dir = get_experiment_dir(base_dir, experiment_name, self.timestamp)
        
        # Set up logging
        self.logger = setup_logging(self.exp_dir)
        
        # Set up metrics tracking
        self.metrics = MetricsTracker(self.exp_dir)
        
        # Config placeholder
        self.config: Optional[ExperimentConfig] = None
        
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        self.logger.info(f"Output dir: {self.exp_dir}")
    
    def log_config(self, config: dict) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = ExperimentConfig(
            experiment_name=self.experiment_name,
            timestamp=self.timestamp,
            **{k: v for k, v in config.items() if k in ExperimentConfig.__dataclass_fields__}
        )
        # Store extra settings
        for k, v in config.items():
            if k not in ExperimentConfig.__dataclass_fields__:
                self.config.extra[k] = v
        
        config_path = self.exp_dir / "config.json"
        self.config.save(config_path)
        self.logger.info(f"Config saved to: {config_path}")
    
    def log_artifact(self, name: str, data: Any, as_json: bool = True) -> Path:
        """
        Save an artifact (data file) to the experiment directory.
        
        Args:
            name: Filename (without extension for JSON)
            data: Data to save
            as_json: Whether to save as JSON
        
        Returns:
            Path to saved file
        """
        if as_json:
            if not name.endswith('.json'):
                name = f"{name}.json"
            path = self.exp_dir / name
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            path = self.exp_dir / name
            with open(path, 'w') as f:
                f.write(str(data))
        
        self.logger.info(f"Saved artifact: {path}")
        return path
    
    def finish(self, final_metrics: Optional[dict] = None) -> None:
        """
        Finish experiment and save summary.
        
        Args:
            final_metrics: Optional final metrics to log
        """
        if final_metrics:
            self.metrics.log(final_metrics, phase="final")
        
        # Save summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "exp_dir": str(self.exp_dir),
            "finished_at": datetime.now().isoformat(),
            "final_metrics": final_metrics,
        }
        
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info("=" * 50)
        self.logger.info("Experiment finished!")
        self.logger.info(f"Results saved to: {self.exp_dir}")
        self.logger.info("=" * 50)


def list_experiments(base_dir: str = "outputs") -> list[dict]:
    """
    List all experiments in the output directory.
    
    Returns:
        List of experiment info dicts
    """
    experiments = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return experiments
    
    for exp_type_dir in base_path.iterdir():
        if not exp_type_dir.is_dir():
            continue
        
        for run_dir in exp_type_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            summary_path = run_dir / "summary.json"
            config_path = run_dir / "config.json"
            
            info = {
                "experiment_name": exp_type_dir.name,
                "timestamp": run_dir.name,
                "path": str(run_dir),
            }
            
            if summary_path.exists():
                with open(summary_path) as f:
                    info["summary"] = json.load(f)
            
            if config_path.exists():
                with open(config_path) as f:
                    info["config"] = json.load(f)
            
            experiments.append(info)
    
    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return experiments

