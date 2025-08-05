# src/automl/hpo_selection.py
"""
Meta-HPO Selection Engine for AutoML Pipeline
Core innovation: Context-aware automatic selection of optimal HPO methods for each architecture
UPDATED: Reduced trial counts for faster execution
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import ParameterSampler
import random

from .utils import AutoMLConfig, Timer, MetricTracker

class HPOMethod(Enum):
    """Available HPO methods - Only Bayesian Optimization"""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

@dataclass
class HPOConfig:
    """Configuration for HPO method"""
    method: HPOMethod
    n_trials: int
    timeout_seconds: Optional[int] = None
    early_stopping_rounds: Optional[int] = None
    resource_budget: float = 1.0  # Fraction of total resource budget
    method_specific_params: Dict[str, Any] = None

@dataclass
class HPOResult:
    """Result from HPO optimization"""
    architecture: str
    method_used: HPOMethod
    best_params: Dict[str, Any]
    best_score: float
    n_trials_completed: int
    optimization_time: float
    convergence_history: List[float]
    method_efficiency: float  # Score per unit time
    early_stopped: bool = False  # NEW: Track if stopped early

class BaseHPOMethod(ABC):
    """Base class for HPO methods"""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.logger = logging.getLogger(f'AutoML.HPO.{self.config.method.value}')
        self.optimization_history = []
        self.early_stop_threshold = 0.95  # NEW: Stop if score > 95%
        self.min_trials_before_early_stop = 5  # NEW: Minimum trials before early stopping
    
    @abstractmethod
    def optimize(self, 
                objective_function: callable,
                search_space: Dict[str, Any],
                architecture_name: str) -> HPOResult:
        """Optimize hyperparameters using this method"""
        pass
    
    def _evaluate_trial(self, trial_params: Dict[str, Any], objective_function: callable) -> float:
        """Evaluate a single trial"""
        try:
            score = objective_function(trial_params)
            self.optimization_history.append(score)
            
            # NEW: Check for early success
            if (len(self.optimization_history) >= self.min_trials_before_early_stop and 
                score >= self.early_stop_threshold):
                self.logger.info(f"Excellent score {score:.4f} achieved! Considering early stop...")
                
            return score
        except Exception as e:
            self.logger.warning(f"Trial evaluation failed: {e}")
            return float('-inf')
    
    def _should_stop_early(self) -> bool:
        """NEW: Check if we should stop HPO early due to excellent performance"""
        if len(self.optimization_history) < self.min_trials_before_early_stop:
            return False
        
        best_score = max(self.optimization_history)
        return best_score >= self.early_stop_threshold

class BayesianOptimizationHPO(BaseHPOMethod):
    """Bayesian Optimization using Optuna TPE - REDUCED TRIALS"""
    
    def optimize(self, objective_function: callable, search_space: Dict[str, Any], architecture_name: str) -> HPOResult:
        self.logger.info(f"Starting Bayesian Optimization for {architecture_name}")
        
        start_time = time.time()
        early_stopped = False
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)  # REDUCED
        )
        
        def optuna_objective(trial):
            # Convert search space to Optuna suggestions
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    if param_config.get('log_scale', False):
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['range'][0], 
                            param_config['range'][1],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['range'][0], 
                            param_config['range'][1]
                        )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['range'][0],
                        param_config['range'][1]
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            score = self._evaluate_trial(params, objective_function)
            
            # NEW: Check for early stopping
            if self._should_stop_early():
                study.stop()
                
            return score
        
        # Run optimization
        try:
            study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                catch=(Exception,)  # Continue on individual trial failures
            )
        except Exception as e:
            self.logger.warning(f"Study optimization stopped: {e}")
        
        # Check if we stopped early due to excellent performance
        if self._should_stop_early():
            early_stopped = True
            self.logger.info(f"Stopped early - excellent performance achieved!")
        
        optimization_time = time.time() - start_time
        
        # Calculate efficiency
        if optimization_time > 0 and study.best_value is not None:
            efficiency = study.best_value / optimization_time
        else:
            efficiency = 0.0
        
        return HPOResult(
            architecture=architecture_name,
            method_used=HPOMethod.BAYESIAN_OPTIMIZATION,
            best_params=study.best_params if study.best_params else {},
            best_score=study.best_value if study.best_value is not None else float('-inf'),
            n_trials_completed=len(study.trials),
            optimization_time=optimization_time,
            convergence_history=self.optimization_history.copy(),
            method_efficiency=efficiency,
            early_stopped=early_stopped
        )


class MetaHPOSelector:
    """
    Meta-optimizer that selects the best HPO method for each architecture and context
    UPDATED: Adaptive trial calculation for faster execution
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.MetaHPOSelector')
        
        # NEW: Adaptive trial configuration
        self.base_trials = config.get('hpo_base_trials', 8)  # REDUCED from 50
        self.quick_mode = config.get('quick_test', False)
        
        # Historical performance tracking
        self.method_performance_history: Dict[str, List[float]] = {
            method.value: [] for method in HPOMethod
        }
        
        # Context-aware selection rules
        self.selection_rules = self._initialize_selection_rules()
        
        # HPO method factories - Only Bayesian Optimization
        self.hpo_methods = {
            HPOMethod.BAYESIAN_OPTIMIZATION: BayesianOptimizationHPO,
        }
        
        self.logger.info(f"MetaHPOSelector initialized - Base trials: {self.base_trials}, Quick mode: {self.quick_mode}")
    
    def _initialize_selection_rules(self) -> Dict[str, Any]:
        """Initialize selection rules based on context - Only Bayesian Optimization"""
        return {
            'architecture_preferences': {
                'resnet': HPOMethod.BAYESIAN_OPTIMIZATION,
                'efficientnet': HPOMethod.BAYESIAN_OPTIMIZATION,
                'convnext': HPOMethod.BAYESIAN_OPTIMIZATION,
                'mobilenet': HPOMethod.BAYESIAN_OPTIMIZATION,
                'densenet': HPOMethod.BAYESIAN_OPTIMIZATION,
            },
            'dataset_complexity_thresholds': {
                'low_complexity': {
                    'threshold': 3.0,
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION
                },
                'medium_complexity': {
                    'threshold': 6.0,
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION
                },
                'high_complexity': {
                    'threshold': 10.0,
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION
                }
            },
            'time_budget_thresholds': {
                'very_limited': {
                    'threshold_hours': 2.0,
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION,
                    'max_trials': 5  # REDUCED
                },
                'limited': {
                    'threshold_hours': 6.0,
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION,
                    'max_trials': 12  # REDUCED
                },
                'sufficient': {
                    'threshold_hours': float('inf'),
                    'preferred_method': HPOMethod.BAYESIAN_OPTIMIZATION,
                    'max_trials': 15  # REDUCED from 100
                }
            }
        }
    
    def _calculate_trial_budget(self, 
                               architecture_name: str,
                               architecture_characteristics: Dict[str, Any],
                               dataset_complexity: float,
                               available_time_hours: float) -> int:
        """NEW: Calculate adaptive trial budget based on context"""
        
        # Start with base trials
        trials = self.base_trials
        
        # Quick mode override
        if self.quick_mode:
            return min(5, trials)
        
        # Adjust for architecture complexity
        arch_complexity = architecture_characteristics.get('complexity_score', 3.0)
        if arch_complexity > 4.0:
            trials += 3  # More trials for complex architectures
        elif arch_complexity < 2.5:
            trials -= 2  # Fewer trials for simple architectures
        
        # Adjust for dataset complexity
        if dataset_complexity > 7.0:
            trials += 2  # More trials for complex datasets
        elif dataset_complexity < 3.0:
            trials -= 2  # Fewer trials for simple datasets
        
        # Adjust for available time
        if available_time_hours < 2.0:
            trials = max(5, trials // 2)  # Aggressive reduction for limited time
        elif available_time_hours < 6.0:
            trials = max(6, int(trials * 0.7))  # Moderate reduction
        
        # Apply bounds
        trials = max(5, min(trials, 20))  # Between 5 and 20 trials
        
        self.logger.info(f"Trial budget for {architecture_name}: {trials} trials")
        self.logger.info(f"  Arch complexity: {arch_complexity}, Dataset complexity: {dataset_complexity}")
        self.logger.info(f"  Available time: {available_time_hours:.1f}h")
        
        return trials
    
    def select_hpo_method(self, 
                         architecture_name: str,
                         architecture_characteristics: Dict[str, Any],
                         dataset_complexity: float,
                         available_time_hours: float,
                         search_space_size: int) -> HPOConfig:
        """
        Select optimal HPO method based on context - WITH ADAPTIVE TRIALS
        """
        
        self.logger.info(f"Selecting HPO method for {architecture_name}")
        self.logger.info(f"  Dataset complexity: {dataset_complexity:.1f}")
        self.logger.info(f"  Available time: {available_time_hours:.1f} hours")
        self.logger.info(f"  Search space size: {search_space_size}")
        
        # Collect selection factors
        factors = self._analyze_selection_factors(
            architecture_name, architecture_characteristics, 
            dataset_complexity, available_time_hours, search_space_size
        )
        
        # Apply selection logic
        selected_method = self._apply_selection_logic(factors)
        
        # NEW: Calculate adaptive trial budget
        n_trials = self._calculate_trial_budget(
            architecture_name, architecture_characteristics,
            dataset_complexity, available_time_hours
        )
        
        # Configure method parameters
        hpo_config = self._configure_method_parameters(
            selected_method, factors, available_time_hours, n_trials
        )
        
        self.logger.info(f"Selected {selected_method.value} for {architecture_name}")
        self.logger.info(f"  Trials: {hpo_config.n_trials}")
        self.logger.info(f"  Timeout: {hpo_config.timeout_seconds}s")
        
        return hpo_config
    
    def _analyze_selection_factors(self, 
                                  architecture_name: str,
                                  architecture_characteristics: Dict[str, Any],
                                  dataset_complexity: float,
                                  available_time_hours: float,
                                  search_space_size: int) -> Dict[str, Any]:
        """Analyze all factors that influence HPO method selection"""
        
        # Architecture family
        arch_family = architecture_characteristics.get('family', 'unknown')
        arch_complexity = architecture_characteristics.get('complexity_score', 5.0)
        arch_speed = architecture_characteristics.get('speed', 'medium')
        
        # Time constraints
        time_pressure = self._categorize_time_pressure(available_time_hours)
        
        # Search space analysis
        search_complexity = self._categorize_search_complexity(search_space_size)
        
        # Historical performance
        method_performance = self._get_historical_performance()
        
        return {
            'architecture_name': architecture_name,
            'architecture_family': arch_family,
            'architecture_complexity': arch_complexity,
            'architecture_speed': arch_speed,
            'dataset_complexity': dataset_complexity,
            'available_time_hours': available_time_hours,
            'time_pressure': time_pressure,
            'search_space_size': search_space_size,
            'search_complexity': search_complexity,
            'historical_performance': method_performance
        }
    
    def _apply_selection_logic(self, factors: Dict[str, Any]) -> HPOMethod:
        """Apply selection logic based on analyzed factors"""
        
        # Score each method based on context
        method_scores = {}
        
        for method in HPOMethod:
            score = self._score_method_for_context(method, factors)
            method_scores[method] = score
        
        # Select method with highest score
        best_method = max(method_scores.items(), key=lambda x: x[1])
        
        self.logger.debug(f"Method scores: {[(m.value, s) for m, s in method_scores.items()]}")
        
        return best_method[0]
    
    def _score_method_for_context(self, method: HPOMethod, factors: Dict[str, Any]) -> float:
        """Score a method for given context - Only Bayesian Optimization available"""
        # Since we only have Bayesian Optimization, always return a high score
        if method == HPOMethod.BAYESIAN_OPTIMIZATION:
            return 1.0
        else:
            return 0.0
    
    def _categorize_time_pressure(self, available_hours: float) -> str:
        """Categorize time pressure"""
        if available_hours < 2.0:
            return 'very_limited'
        elif available_hours < 6.0:
            return 'limited'
        else:
            return 'sufficient'
    
    def _categorize_search_complexity(self, search_space_size: int) -> str:
        """Categorize search space complexity"""
        if search_space_size < 100:
            return 'small'
        elif search_space_size < 1000:
            return 'medium'
        else:
            return 'large'
    
    def _get_historical_performance(self) -> Dict[str, float]:
        """Get normalized historical performance for each method"""
        performance = {}
        
        for method_name, history in self.method_performance_history.items():
            if history:
                # Use recent average performance
                recent_performance = np.mean(history[-10:])  # Last 10 results
                performance[method_name] = min(recent_performance, 1.0)
            else:
                performance[method_name] = 0.5  # Default neutral score
        
        return performance
    
    def _configure_method_parameters(self, 
                                   method: HPOMethod, 
                                   factors: Dict[str, Any],
                                   available_time_hours: float,
                                   n_trials: int) -> HPOConfig:
        """Configure parameters for Bayesian Optimization"""
        
        # Use calculated trial budget
        timeout_seconds = int(available_time_hours * 3600 * 0.6)  # Use 60% of available time
        
        # Bayesian optimization configuration
        final_trials = min(n_trials, 15)  # Cap at 15 trials
        
        return HPOConfig(
            method=method,
            n_trials=final_trials,
            timeout_seconds=timeout_seconds,
            early_stopping_rounds=5,
            resource_budget=1.0,
            method_specific_params={}
        )
    
    def optimize_hyperparameters(self,
                                architecture_name: str,
                                objective_function: callable,
                                search_space: Dict[str, Any],
                                architecture_characteristics: Dict[str, Any],
                                dataset_complexity: float,
                                available_time_hours: float) -> HPOResult:
        """
        Complete HPO pipeline: select method and optimize - WITH PROGRESS TRACKING
        """
        
        # Select HPO method
        search_space_size = self._estimate_search_space_size(search_space)
        hpo_config = self.select_hpo_method(
            architecture_name, architecture_characteristics,
            dataset_complexity, available_time_hours, search_space_size
        )
        
        # Create HPO method instance - Only Bayesian Optimization available
        hpo_method = self.hpo_methods[hpo_config.method](hpo_config)
        
        # Run optimization with progress tracking
        self.logger.info(f" Starting HPO for {architecture_name} with {hpo_config.n_trials} trials")
        start_time = time.time()
        
        result = hpo_method.optimize(objective_function, search_space, architecture_name)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f" HPO completed for {architecture_name} in {elapsed_time:.1f}s")
        self.logger.info(f"   Best score: {result.best_score:.4f} ({result.n_trials_completed} trials)")
        if result.early_stopped:
            self.logger.info(f"    Stopped early due to excellent performance!")
        
        # Update historical performance
        self.method_performance_history[result.method_used.value].append(result.method_efficiency)
        
        return result
    
    def _estimate_search_space_size(self, search_space: Dict[str, Any]) -> int:
        """Estimate the size of the search space"""
        total_size = 1
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'categorical':
                total_size *= len(param_config['choices'])
            elif param_config['type'] == 'int':
                low, high = param_config['range']
                total_size *= (high - low + 1)
            else:  # float
                total_size *= 50  # REDUCED: Approximate discretization (was 100)
        
        return min(total_size, 100000)  # REDUCED: Cap at reasonable size (was 1M)
    
    def get_method_performance_summary(self) -> Dict[str, Any]:
        """Get summary of HPO method performance"""
        summary = {}
        
        for method_name, history in self.method_performance_history.items():
            if history:
                summary[method_name] = {
                    'num_uses': len(history),
                    'average_efficiency': np.mean(history),
                    'recent_efficiency': np.mean(history[-5:]) if len(history) >= 5 else np.mean(history),
                    'improvement_trend': 'improving' if len(history) > 1 and history[-1] > history[0] else 'stable'
                }
            else:
                summary[method_name] = {
                    'num_uses': 0,
                    'average_efficiency': 0.0,
                    'recent_efficiency': 0.0,
                    'improvement_trend': 'no_data'
                }
        
        return summary
    
    def save_state(self, filepath: str):
        """Save HPO selector state"""
        state = {
            'method_performance_history': self.method_performance_history,
            'selection_rules': self.selection_rules,
            'performance_summary': self.get_method_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"HPO selector state saved to {filepath}")

# Test the HPO selection engine
if __name__ == "__main__":
    # Simple test
    config = AutoMLConfig()
    selector = MetaHPOSelector(config)
    
    # Test method selection
    arch_characteristics = {
        'family': 'efficientnet',
        'complexity_score': 4.0,
        'speed': 'medium'
    }
    
    hpo_config = selector.select_hpo_method(
        'efficientnet_b0',
        arch_characteristics,
        dataset_complexity=5.0,
        available_time_hours=4.0,
        search_space_size=500
    )
    
    print(f"Selected method: {hpo_config.method.value}")
    print(f"Trials: {hpo_config.n_trials}")
    print(f"Timeout: {hpo_config.timeout_seconds}s")
    
    # Test dummy optimization
    def dummy_objective(params):
        return np.random.random()
    
    search_space = {
        'learning_rate': {'type': 'float', 'range': (1e-5, 1e-1), 'log_scale': True},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        'dropout': {'type': 'float', 'range': (0.0, 0.5)}
    }
    
    result = selector.optimize_hyperparameters(
        'efficientnet_b0',
        dummy_objective,
        search_space,
        arch_characteristics,
        dataset_complexity=5.0,
        available_time_hours=0.1  # Short test
    )
    
    print(f"Optimization result: {result.best_score:.3f}")
    print(f"Best params: {result.best_params}")
    print(f"Method used: {result.method_used.value}")
    print(f"Early stopped: {result.early_stopped}")
