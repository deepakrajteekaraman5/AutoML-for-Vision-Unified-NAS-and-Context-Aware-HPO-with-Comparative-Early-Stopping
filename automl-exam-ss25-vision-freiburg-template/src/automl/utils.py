# src/automl/utils.py
"""
Core utilities for AutoML Pipeline
Provides logging, configuration, timing, and reproducibility utilities
UPDATED: Added quick test configurations and reduced trial settings
"""

import logging
import random
import numpy as np
import torch
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager
import pickle
from datetime import datetime

class AutoMLConfig:
    """Configuration management for AutoML pipeline - WITH QUICK TEST SUPPORT"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for emotions dataset - WITH REDUCED TRIALS"""
        return {
            # Basic settings
            'random_seed': 42,
            'log_level': 'INFO',
            'verbose_logging': True,  # NEW: Set to False for quiet mode
            'device': 'auto',  # auto, cpu, cuda
            
            # Directory settings
            'checkpoint_dir': './checkpoints',
            'results_dir': './results',
            'logs_dir': './logs',
            
            # Time budget settings
            'time_budget_hours': 24,
            'architecture_search_ratio': 0.70,  # 70% for arch search
            'final_training_ratio': 0.20,       # 20% for final training
            'buffer_ratio': 0.10,               # 10% buffer
            
            # Training settings
            'min_epochs_per_architecture': 5,
            'max_epochs_per_architecture': 50,
            'validation_split': 0.2,
            'test_split': 0.1,
            'early_stopping_patience': 10,
            'performance_metric': 'accuracy',
            
            # Resource settings
            'max_gpu_memory_gb': 16,
            'max_cpu_cores': 8,
            'batch_size_auto': True,
            
            # Dataset-specific (emotions default)
            'dataset_name': 'emotions',
            'image_size': 48,
            'num_classes': 7,
            'channels': 1,
            
            # NEW: HPO Configuration (REDUCED TRIALS)
            'hpo_base_trials': 8,              # Reduced from 50
            'hpo_max_trials_bayesian': 12,     # Reduced from 100
            'hpo_max_trials_random': 10,       # Reduced from 80
            'hpo_max_trials_successive': 16,   # Reduced from 200
            
            # NEW: Quick test mode settings
            'quick_test': False,
            'quick_test_time_budget': 1.0,     # 1 hour for quick test
            'quick_test_max_epochs': 10,       # 10 epochs per trial in quick mode
            'quick_test_trials': 5,            # 5 trials per architecture in quick mode
            'quick_test_max_architectures': 2, # Only 2 architectures in quick mode
            
            # NEW: Performance thresholds
            'early_success_threshold': 0.95,   # Stop HPO if score > 95%
            'min_trials_before_early_stop': 5, # Minimum trials before early stopping
            
            # NEW: Visualization settings
            'enable_plotting': True,
            'plot_training_curves': True,
            'plot_hpo_progress': True,
            'plot_architecture_comparison': True,
            'save_plots': True,
            'plot_format': 'png',              # png, pdf, svg
            'plot_dpi': 150,
        }
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        self.config.update(user_config)
        logging.getLogger('AutoML').info(f"Loaded config from {config_path}")
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logging.getLogger('AutoML').info(f"Saved config to {config_path}")
    
    def get(self, key: str, default: Any = None):
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value
    
    def update(self, **kwargs):
        """Update multiple config values"""
        self.config.update(kwargs)
    
    def enable_quick_test_mode(self):
        """NEW: Enable quick test mode with reduced settings"""
        self.config.update({
            'quick_test': True,
            'time_budget_hours': self.config['quick_test_time_budget'],
            'hpo_base_trials': self.config['quick_test_trials'],
            'max_epochs_per_architecture': self.config['quick_test_max_epochs'],
            'min_epochs_per_architecture': 3,
            'early_stopping_patience': 5,
        })
        logging.getLogger('AutoML').info("Quick test mode enabled - reduced time budget and trials")
    
    def get_quick_test_architectures(self) -> List[str]:
        """NEW: Get reduced architecture list for quick testing"""
        if self.config['quick_test']:
            # Return only 2 fast architectures for quick testing
            return ['resnet18', 'efficientnet_b0']
        else:
            # Return None to use full strategic selection
            return None
    
    def print_configuration_summary(self):
        """NEW: Print current configuration for verification"""
        print("\n" + "="*60)
        print("AUTOML CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Mode: {'QUICK TEST' if self.config['quick_test'] else 'FULL PIPELINE'}")
        print(f"Time Budget: {self.config['time_budget_hours']} hours")
        print(f"Dataset: {self.config['dataset_name']} ({self.config['num_classes']} classes)")
        print(f"HPO Base Trials: {self.config['hpo_base_trials']}")
        print(f"Max Epochs per Architecture: {self.config['max_epochs_per_architecture']}")
        print(f"Early Success Threshold: {self.config['early_success_threshold']}")
        print(f"Plotting Enabled: {self.config['enable_plotting']}")
        
        if self.config['quick_test']:
            print(f"\nQuick Test Settings:")
            print(f"  Max Architectures: {self.config['quick_test_max_architectures']}")
            print(f"  Trials per Arch: {self.config['quick_test_trials']}")
            print(f"  Max Epochs: {self.config['quick_test_max_epochs']}")
        
        print("="*60)

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging for AutoML pipeline - ENHANCED"""
    
    # Create main logger
    logger = logging.getLogger('AutoML')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter with timestamp and better formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with colors (if available)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Always show INFO+ on console
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    # Create child loggers for different components
    component_loggers = ['DataManager', 'ModelFactory', 'EarlyStopping', 
                        'HPOSelector', 'BudgetManager', 'AutoMLPipeline', 'Trainer']
    
    for component in component_loggers:
        child_logger = logging.getLogger(f'AutoML.{component}')
        child_logger.setLevel(getattr(logging, level.upper()))
    
    logger.info("=== AutoML Pipeline Logging Initialized ===")
    logger.info(f"Log level: {level}, File: {log_file or 'Console only'}")
    return logger

def set_seed(seed: int = 42):
    """Set random seeds for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # FIXED: Use ASCII-only logging to avoid Windows encoding issues
    logging.getLogger('AutoML').info(f"Random seed set to {seed} for reproducibility")

class Timer:
    """Context manager and standalone timer for operation timing - ENHANCED"""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger('AutoML')
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start()
        self.logger.info(f" Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        duration = self.elapsed
        self.logger.info(f" {self.name} completed in {self._format_duration(duration)}")
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        """Stop the timer"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"

class MetricTracker:
    """Track performance metrics across training runs - ENHANCED WITH PLOTTING SUPPORT"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        self.logger = logging.getLogger('AutoML.MetricTracker')
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        timestamp = time.time()
        
        # Update metric lists
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Store timestamped history
        entry = {'timestamp': timestamp, **kwargs}
        self.history.append(entry)
        
        # Log the update
        metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in kwargs.items()])
        self.logger.debug(f"Metrics updated - {metric_str}")
    
    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric in self.metrics and self.metrics[metric]:
            return self.metrics[metric][-1]
        return None
    
    def get_best(self, metric: str, higher_is_better: bool = True) -> Optional[float]:
        """Get best value for a metric"""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        
        values = self.metrics[metric]
        return max(values) if higher_is_better else min(values)
    
    def get_average(self, metric: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average of recent metric values"""
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        
        values = self.metrics[metric]
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_trend(self, metric: str, window: int = 5) -> str:
        """Get trend direction (improving/declining/stable)"""
        if metric not in self.metrics or len(self.metrics[metric]) < window:
            return "insufficient_data"
        
        recent_values = self.metrics[metric][-window:]
        
        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if abs(slope) < 1e-4:
            return "stable"
        elif slope > 0:
            return "improving" 
        else:
            return "declining"
    
    def export_for_plotting(self) -> Dict[str, Any]:
        """NEW: Export data in format suitable for plotting"""
        return {
            'metrics': self.metrics.copy(),
            'history': self.history.copy(),
            'summary': {
                'num_entries': len(self.history),
                'metrics_tracked': list(self.metrics.keys()),
                'duration': self.history[-1]['timestamp'] - self.history[0]['timestamp'] if self.history else 0
            }
        }

class ProgressTracker:
    """NEW: Track progress across the entire AutoML pipeline"""
    
    def __init__(self, total_architectures: int):
        self.total_architectures = total_architectures
        self.completed_architectures = 0
        self.current_architecture = None
        self.current_trial = 0
        self.total_trials = 0
        self.start_time = time.time()
        self.logger = logging.getLogger('AutoML.ProgressTracker')
    
    def start_architecture(self, architecture_name: str, n_trials: int):
        """Start processing a new architecture"""
        self.current_architecture = architecture_name
        self.current_trial = 0
        self.total_trials = n_trials
        
        progress_pct = (self.completed_architectures / self.total_architectures) * 100
        self.logger.info(f" Starting {architecture_name} ({self.completed_architectures + 1}/{self.total_architectures}) - {progress_pct:.1f}% complete")
    
    def update_trial(self, trial_idx: int, score: float):
        """Update trial progress"""
        self.current_trial = trial_idx
        trial_progress = (trial_idx / self.total_trials) * 100
        self.logger.info(f"  Trial {trial_idx}/{self.total_trials} ({trial_progress:.1f}%) - Score: {score:.4f}")
    
    def complete_architecture(self, best_score: float):
        """Mark architecture as completed"""
        self.completed_architectures += 1
        elapsed = time.time() - self.start_time
        
        overall_progress = (self.completed_architectures / self.total_architectures) * 100
        self.logger.info(f" {self.current_architecture} completed - Best: {best_score:.4f}")
        self.logger.info(f" Overall progress: {self.completed_architectures}/{self.total_architectures} ({overall_progress:.1f}%) - Elapsed: {elapsed/3600:.1f}h")
    
    def get_eta(self) -> str:
        """Get estimated time to completion"""
        if self.completed_architectures == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        time_per_arch = elapsed / self.completed_architectures
        remaining_archs = self.total_architectures - self.completed_architectures
        eta_seconds = remaining_archs * time_per_arch
        
        if eta_seconds < 3600:
            return f"{eta_seconds/60:.0f} minutes"
        else:
            return f"{eta_seconds/3600:.1f} hours"

def save_checkpoint(obj: Any, filepath: str, metadata: Optional[Dict] = None):
    """Save object to checkpoint with metadata"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint_data = {
        'object': obj,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    if filepath.endswith('.pt') or filepath.endswith('.pth'):
        torch.save(checkpoint_data, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    logging.getLogger('AutoML').info(f" Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str) -> Any:
    """Load object from checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    if filepath.endswith('.pt') or filepath.endswith('.pth'):
        checkpoint_data = torch.load(filepath, map_location='cpu')
    else:
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
    
    # Handle both new format (with metadata) and old format (raw object)
    if isinstance(checkpoint_data, dict) and 'object' in checkpoint_data:
        logging.getLogger('AutoML').info(f" Loaded checkpoint from {filepath} "
                                        f"(saved: {checkpoint_data.get('timestamp', 'unknown')})")
        return checkpoint_data['object']
    else:
        # Old format - just return the object
        logging.getLogger('AutoML').info(f" Loaded checkpoint from {filepath}")
        return checkpoint_data

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device - ENHANCED REPORTING"""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.getLogger('AutoML').info(f" Using GPU: {gpu_name} ({memory_gb:.1f}GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.getLogger('AutoML').info(" Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        logging.getLogger('AutoML').info(" Using CPU")
    
    return device

def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'size_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
    }

def format_time_remaining(seconds: float) -> str:
    """NEW: Format remaining time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        return f"{seconds/3600:.1f}h"

def print_performance_summary(results: Dict[str, Any]):
    """NEW: Print a quick performance summary"""
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*52)
    
    if results.get('best_model') and results['best_model'].get('architecture'):
        best = results['best_model']
        print(f" Best Architecture: {best['architecture']}")
        print(f" Test Accuracy: {best['test_accuracy']:.4f}")
        print(f" Total Time: {results['pipeline_config']['total_execution_time_hours']:.2f}h")
    else:
        print(" No models completed training")
    
    print("="*52)

# Test the utilities if run directly
if __name__ == "__main__":
    print("Testing AutoML utilities...")
    
    # Test configuration
    config = AutoMLConfig()
    print(f"✓ Config loaded - Time budget: {config.get('time_budget_hours')} hours")
    print(f"✓ HPO base trials: {config.get('hpo_base_trials')}")
    
    # Test quick mode
    config.enable_quick_test_mode()
    print(f"✓ Quick test enabled - Time budget: {config.get('time_budget_hours')} hours")
    config.print_configuration_summary()
    
    # Test logging
    logger = setup_logging(level='INFO')
    logger.info("✓ Logging system working")
    
    # Test reproducibility
    set_seed(42)
    random_val = np.random.random()
    print(f"✓ Reproducibility test - Random value: {random_val}")
    
    # Test timer
    with Timer("Test operation") as timer:
        time.sleep(0.1)
    print(f"✓ Timer working - Elapsed: {timer.elapsed:.2f}s")
    
    # Test device detection
    device = get_device()
    print(f"✓ Device detection - Using: {device}")
    
    # Test progress tracker
    progress = ProgressTracker(total_architectures=4)
    progress.start_architecture('resnet18', 10)
    progress.update_trial(1, 0.85)
    progress.complete_architecture(0.87)
    print(f"✓ Progress tracker - ETA: {progress.get_eta()}")
    
    print("All utilities working correctly!")
