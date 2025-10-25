"""
Enhanced logging utilities for comprehensive training monitoring
"""

import os
import sys
import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def setup_enhanced_logging(experiment_name: str, 
                         log_type: str = "training",
                         log_level: int = logging.INFO) -> Tuple[logging.Logger, Path]:
    """
    Setup enhanced logging with file and console handlers
    
    Args:
        experiment_name: Name of the experiment
        log_type: Type of logging (training, preprocessing, etc.)
        log_level: Logging level
    
    Returns:
        Tuple of (logger, log_directory_path)
    """
    # Create logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{experiment_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger(f"{experiment_name}_{log_type}")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = log_dir / f"{experiment_name}_{log_type}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log system info
    logger.info(f"Logging initialized for {experiment_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    return logger, log_dir


def setup_basic_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup basic logging for simple operations
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler only
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


class MetricsLogger:
    """
    Logger for training metrics with CSV and JSON output
    """
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # CSV file for metrics
        self.csv_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.csv_initialized = False
        
        # JSON file for experiment config
        self.config_file = self.log_dir / f"{experiment_name}_config.json"
        
        # Metrics history
        self.metrics_history = []
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        # Add epoch to metrics
        epoch_metrics = {'epoch': epoch, **metrics}
        self.metrics_history.append(epoch_metrics)
        
        # Initialize CSV if needed
        if not self.csv_initialized:
            self._initialize_csv(epoch_metrics.keys())
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
            writer.writerow(epoch_metrics)
    
    def _initialize_csv(self, fieldnames):
        """Initialize CSV file with headers"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self.csv_initialized = True
    
    def get_best_metrics(self, metric_name: str, maximize: bool = True) -> Dict[str, Any]:
        """Get best metrics based on a specific metric"""
        if not self.metrics_history:
            return {}
        
        if maximize:
            best_epoch = max(self.metrics_history, key=lambda x: x.get(metric_name, 0))
        else:
            best_epoch = min(self.metrics_history, key=lambda x: x.get(metric_name, float('inf')))
        
        return best_epoch


class TrainingLogger:
    """
    Comprehensive training logger combining file logging and metrics
    """
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        
        # Setup logging
        self.logger, self.log_dir = setup_enhanced_logging(experiment_name, "training")
        
        # Setup metrics logging
        self.metrics_logger = MetricsLogger(self.log_dir, experiment_name)
        self.metrics_logger.log_config(config)
        
        # Training state
        self.start_time = datetime.now()
        self.epoch_times = []
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
        self.epoch_start_time = datetime.now()
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch completion with metrics"""
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.epoch_times.append(epoch_time)
        
        # Add timing to metrics
        metrics_with_time = {
            **metrics,
            'epoch_time': epoch_time,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times)
        }
        
        # Log to file
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        # Log metrics
        self.metrics_logger.log_metrics(epoch, metrics_with_time)
    
    def log_training_complete(self, best_metrics: Dict[str, Any]):
        """Log training completion"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time / 3600:.2f} hours")
        self.logger.info(f"Average epoch time: {sum(self.epoch_times) / len(self.epoch_times):.2f}s")
        
        if best_metrics:
            self.logger.info("Best metrics:")
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.6f}")
                else:
                    self.logger.info(f"  {key}: {value}")


def log_system_info(logger: logging.Logger):
    """Log comprehensive system information"""
    import platform
    
    logger.info("System Information:")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {os.cpu_count()}")
    
    # Optional psutil for RAM info
    try:
        import psutil
        logger.info(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except ImportError:
        logger.info("  RAM: Unknown (psutil not installed)")
    
    # GPU info
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"  GPUs: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("  GPUs: None (CPU only)")
            
    except ImportError:
        logger.info("  PyTorch: Not installed")


# Compatibility function for existing code
def setup_logging(experiment_name: str) -> Tuple[logging.Logger, Path]:
    """
    Compatibility function for existing code
    """
    return setup_enhanced_logging(experiment_name, "training")


if __name__ == "__main__":
    # Test logging setup
    logger, log_dir = setup_enhanced_logging("test_experiment", "test")
    log_system_info(logger)
    
    # Test metrics logging
    metrics_logger = MetricsLogger(log_dir, "test_experiment")
    metrics_logger.log_config({"batch_size": 8, "learning_rate": 0.001})
    
    for epoch in range(3):
        metrics = {
            "train_loss": 0.5 - epoch * 0.1,
            "val_loss": 0.6 - epoch * 0.08,
            "val_accuracy": 0.7 + epoch * 0.1
        }
        metrics_logger.log_metrics(epoch + 1, metrics)
    
    print(f"✅ Logging test completed. Check: {log_dir}")