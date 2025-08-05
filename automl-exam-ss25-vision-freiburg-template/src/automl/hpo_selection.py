# src/automl/hpo_selection.py
"""
Meta-HPO Selection Engine for AutoML Pipeline
Core innovation: Context-aware automatic selection of optimal HPO methods for each architecture
UPDATED: Using ASHA from Optuna and BOHB from HpBandSter
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
import random
import tempfile
import os

# HpBandSter imports
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
from hpbandster.core.worker import Worker
import threading

from .utils import AutoMLConfig, Timer, MetricTracker

class HPOMethod(Enum):
    """Available HPO methods"""
    ASHA = "asha"
    BOHB = "bohb"

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
    early_stopped: bool = False  # Track if stopped early

class BaseHPOMethod(ABC):
    """Base class for HPO methods"""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.logger = logging.getLogger(f'AutoML.HPO.{self.config.method.value}')
        self.optimization_history = []
        self.early_stop_threshold = 0.95  # Stop if score > 95%
        self.min_trials_before_early_stop = 5  # Minimum trials before early stopping
    
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
            
            # Check for early success
            if (len(self.optimization_history) >= self.min_trials_before_early_stop and 
                score >= self.early_stop_threshold):
                self.logger.info(f"Excellent score {score:.4f} achieved! Considering early stop...")
                
            return score
        except Exception as e:
            self.logger.warning(f"Trial evaluation failed: {e}")
            return float('-inf')
    
    def _should_stop_early(self) -> bool:
        """Check if we should stop HPO early due to excellent performance"""
        if len(self.optimization_history) < self.min_trials_before_early_stop:
            return False
        
        best_score = max(self.optimization_history)
        return best_score >= self.early_stop_threshold

class ASHAOptimizerHPO(BaseHPOMethod):
    """ASHA (Asynchronous Successive Halving Algorithm) using Optuna"""
    
    def optimize(self, objective_function: callable, search_space: Dict[str, Any], architecture_name: str) -> HPOResult:
        self.logger.info(f"Starting ASHA optimization for {architecture_name}")
        
        start_time = time.time()
        early_stopped = False
        
        # Create Optuna study with ASHA pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.RandomSampler(seed=42),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,  # Minimum epochs/resource
                reduction_factor=3,  # Factor by which to reduce candidates
                min_early_stopping_rate=0  # Allow pruning from the start
            )
        )
        
        def optuna_objective(trial):
            # SMART TRIAL SCHEDULING: Check if we should start this trial
            elapsed_time = time.time() - start_time
            remaining_time = max(0, self.config.timeout_seconds - elapsed_time)
            trials_completed = len(study.trials)
            trials_remaining = max(0, self.config.n_trials - trials_completed)
            
            # Don't start new trial if insufficient time and conditions met
            if (remaining_time < 1800 and  # Less than 30 minutes remaining
                trials_remaining <= 2 and  # 2 or fewer trials remaining
                trials_completed >= 2):    # At least 2 trials completed
                
                self.logger.info(f"SMART SCHEDULING: Skipping trial - insufficient time")
                self.logger.info(f"  Remaining time: {remaining_time/60:.1f}m, trials left: {trials_remaining}")
                study.stop()  # Stop the study
                raise optuna.exceptions.TrialPruned()
            
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
            
            # ASHA multi-fidelity: MINIMAL epochs for 2-hour limit
            max_epochs = 25  # REDUCED from 50
            for epoch in [3, 8, 15, 25]:  # MINIMAL successive halving points
                # Simulate partial training (in practice, you'd train for 'epoch' epochs)
                score = self._evaluate_trial(params, objective_function)
                
                # Report intermediate value for pruning
                trial.report(score, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    self.logger.debug(f"Trial pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()
                
                # Check for early stopping due to excellent performance
                if self._should_stop_early():
                    study.stop()
                    break
            
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
            self.logger.info(f"ASHA stopped early - excellent performance achieved!")
        
        optimization_time = time.time() - start_time
        
        # Calculate efficiency
        if optimization_time > 0 and study.best_value is not None:
            efficiency = study.best_value / optimization_time
        else:
            efficiency = 0.0
        
        return HPOResult(
            architecture=architecture_name,
            method_used=HPOMethod.ASHA,
            best_params=study.best_params if study.best_params else {},
            best_score=study.best_value if study.best_value is not None else float('-inf'),
            n_trials_completed=len(study.trials),
            optimization_time=optimization_time,
            convergence_history=self.optimization_history.copy(),
            method_efficiency=efficiency,
            early_stopped=early_stopped
        )

class BOHBWorker(Worker):
    """BOHB Worker class for evaluating configurations"""
    
    def __init__(self, objective_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_function = objective_function
        self.optimization_history = []
        self.early_stop_threshold = 0.95
        self.min_trials_before_early_stop = 5
    
    def compute(self, config, budget, **kwargs):
        """Compute the loss for a given configuration and budget"""
        try:
            # Convert config to standard format
            params = dict(config)
            
            # Simulate training with the given budget (budget represents epochs/resources)
            score = self.objective_function(params)
            self.optimization_history.append(score)
            
            # BOHB minimizes, so we return negative score
            loss = 1.0 - score
            
            return {
                'loss': loss,
                'info': {
                    'score': score,
                    'budget_used': budget,
                    'params': params
                }
            }
        except Exception as e:
            # Return a high loss for failed evaluations
            return {
                'loss': 1.0,
                'info': {
                    'error': str(e),
                    'budget_used': budget
                }
            }

class BOHBOptimizerHPO(BaseHPOMethod):
    """BOHB (Bayesian Optimization and HyperBand) using HpBandSter"""
    
    def optimize(self, objective_function: callable, search_space: Dict[str, Any], architecture_name: str) -> HPOResult:
        self.logger.info(f"Starting BOHB optimization for {architecture_name}")
        
        start_time = time.time()
        early_stopped = False
        
        # Convert search space to ConfigSpace format
        config_space = self._create_config_space(search_space)
        
        # Start nameserver
        NS = hpns.NameServer(run_id='bohb_optimization', host='127.0.0.1', port=None)
        NS.start()
        
        try:
            # Create worker
            worker = BOHBWorker(
                objective_function=objective_function,
                nameserver='127.0.0.1',
                nameserver_port=NS.port,
                run_id='bohb_optimization'
            )
            worker.run(background=True)
            
            # Create BOHB optimizer - MINIMAL parameters for 2-hour limit
            bohb = BOHB(
                configspace=config_space,
                run_id='bohb_optimization',
                nameserver='127.0.0.1',
                nameserver_port=NS.port,
                min_budget=3,     # MINIMAL: Minimum budget (e.g., epochs)
                max_budget=25,    # MINIMAL: Maximum budget (e.g., epochs)
                eta=3,           # Halving factor
                random_fraction=0.5  # More random for speed
            )
            
            # Run optimization - MINIMAL iterations for 2-hour limit
            res = bohb.run(
                n_iterations=max(1, min(self.config.n_trials, 2)),  # MAX 2 iterations
                min_n_workers=1
            )
            
            # Shutdown
            bohb.shutdown(shutdown_workers=True)
            NS.shutdown()
            
            # Extract results
            all_runs = res.get_all_runs()
            
            if all_runs:
                # Find best configuration - FIXED: Use correct attribute access
                best_run = min(all_runs, key=lambda x: x.loss)
                
                # FIXED: Access config correctly from HpBandSter Run object
                if hasattr(best_run, 'config'):
                    best_params = dict(best_run.config)
                elif hasattr(best_run, 'config_id'):
                    # Try to get config from results
                    best_config_id = best_run.config_id
                    best_params = {}
                    for run in all_runs:
                        if hasattr(run, 'config_id') and run.config_id == best_config_id:
                            if hasattr(run, 'info') and 'params' in run.info:
                                best_params = run.info['params']
                                break
                else:
                    # Fallback: try to extract from info
                    if hasattr(best_run, 'info') and 'params' in best_run.info:
                        best_params = best_run.info['params']
                    else:
                        best_params = {}
                
                best_score = 1.0 - best_run.loss  # Convert back from loss to score
                
                # Build convergence history - FIXED: Handle different Run object structures
                convergence_history = []
                for run in sorted(all_runs, key=lambda x: getattr(x, 'time_stamps', {}).get('finished', 0)):
                    if hasattr(run, 'info') and isinstance(run.info, dict) and 'score' in run.info:
                        convergence_history.append(run.info['score'])
                    else:
                        convergence_history.append(1.0 - run.loss)
            else:
                best_params = {}
                best_score = float('-inf')
                convergence_history = []
            
        except Exception as e:
            self.logger.error(f"BOHB optimization failed: {e}")
            # Cleanup
            try:
                NS.shutdown()
            except:
                pass
            
            best_params = {}
            best_score = float('-inf')
            convergence_history = []
            all_runs = []
        
        optimization_time = time.time() - start_time
        
        # Calculate efficiency
        if optimization_time > 0 and best_score > float('-inf'):
            efficiency = best_score / optimization_time
        else:
            efficiency = 0.0
        
        return HPOResult(
            architecture=architecture_name,
            method_used=HPOMethod.BOHB,
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=len(all_runs) if all_runs else 0,
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            method_efficiency=efficiency,
            early_stopped=early_stopped
        )
    
    def _create_config_space(self, search_space: Dict[str, Any]) -> CS.ConfigurationSpace:
        """Convert search space to ConfigSpace format"""
        config_space = CS.ConfigurationSpace()
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                low, high = param_config['range']
                if param_config.get('log_scale', False):
                    hp = CS.UniformFloatHyperparameter(
                        param_name, 
                        lower=low, 
                        upper=high, 
                        log=True
                    )
                else:
                    hp = CS.UniformFloatHyperparameter(
                        param_name, 
                        lower=low, 
                        upper=high
                    )
                config_space.add_hyperparameter(hp)
                
            elif param_config['type'] == 'int':
                low, high = param_config['range']
                hp = CS.UniformIntegerHyperparameter(
                    param_name,
                    lower=low,
                    upper=high
                )
                config_space.add_hyperparameter(hp)
                
            elif param_config['type'] == 'categorical':
                hp = CS.CategoricalHyperparameter(
                    param_name,
                    choices=param_config['choices']
                )
                config_space.add_hyperparameter(hp)
        
        return config_space

class MetaHPOSelector:
    """
    Meta-optimizer that selects between ASHA and BOHB based on context
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.MetaHPOSelector')
        
        # Trial configuration - ULTRA AGGRESSIVE for 2-hour limit
        self.base_trials = config.get('hpo_base_trials', 4)  # REDUCED from 8
        self.quick_mode = config.get('quick_test', False)
        # Historical performance tracking
        self.method_performance_history: Dict[str, List[float]] = {
            method.value: [] for method in HPOMethod
        }
        
        # Context-aware selection rules
        self.selection_rules = self._initialize_selection_rules()
        
        # HPO method factories
        self.hpo_methods = {
            HPOMethod.ASHA: ASHAOptimizerHPO,
            HPOMethod.BOHB: BOHBOptimizerHPO,
        }
        
        self.logger.info(f"MetaHPOSelector initialized - Base trials: {self.base_trials}, Quick mode: {self.quick_mode}")
        self.logger.info(f"UNIFIED: Per-model time managed by BudgetManager")
    
    def _initialize_selection_rules(self) -> Dict[str, Any]:
        """Initialize selection rules based on context"""
        return {
            'architecture_preferences': {
                'resnet': HPOMethod.BOHB,      # ResNets benefit from Bayesian optimization
                'efficientnet': HPOMethod.BOHB, # Complex scaling relationships
                'convnext': HPOMethod.BOHB,     # New architecture, needs exploration
                'mobilenet': HPOMethod.ASHA,    # Simpler, can use aggressive pruning
                'densenet': HPOMethod.ASHA,     # Many configurations, good for pruning
            },
            'dataset_complexity_thresholds': {
                'low_complexity': {
                    'threshold': 3.0,
                    'preferred_method': HPOMethod.ASHA
                },
                'medium_complexity': {
                    'threshold': 6.0,
                    'preferred_method': HPOMethod.BOHB
                },
                'high_complexity': {
                    'threshold': 10.0,
                    'preferred_method': HPOMethod.BOHB
                }
            },
            'time_budget_thresholds': {
                'very_limited': {
                    'threshold_hours': 1.0,  # REDUCED from 2.0
                    'preferred_method': HPOMethod.ASHA,
                    'max_trials': 2  # REDUCED from 4
                },
                'limited': {
                    'threshold_hours': 1.5,  # REDUCED from 4.0
                    'preferred_method': HPOMethod.ASHA,
                    'max_trials': 3  # REDUCED from 6
                },
                'sufficient': {
                    'threshold_hours': float('inf'),
                    'preferred_method': HPOMethod.ASHA,  # Keep ASHA for speed
                    'max_trials': 4  # REDUCED from 8
                }
            }
        }
    
    def _calculate_trial_budget(self, 
                               architecture_name: str,
                               architecture_characteristics: Dict[str, Any],
                               dataset_complexity: float,
                               available_time_hours: float) -> int:
        """Calculate adaptive trial budget based on context - UNIFIED VERSION"""
        
        # UNIFIED: Use available time directly (managed by BudgetManager)
        effective_time = available_time_hours
        
        # Start with base trials (now 4)
        trials = self.base_trials
        
        # Quick mode override
        if self.quick_mode:
            return min(2, trials)
        
        # Time-based reduction
        if effective_time <= 1.0:
            trials = 2  # Minimal trials for very short time
        elif effective_time <= 1.5:
            trials = 3  # Few trials for limited time
        else:
            trials = 4  # Max trials even with sufficient time
        
        # Apply strict bounds
        trials = max(2, min(trials, 4))  # Between 2 and 4 trials MAX
        
        self.logger.info(f"Trial budget for {architecture_name}: {trials} trials")
        self.logger.info(f"  Available time: {effective_time:.1f}h (managed by BudgetManager)")
        
        return trials
    
    def select_hpo_method(self, 
                         architecture_name: str,
                         architecture_characteristics: Dict[str, Any],
                         dataset_complexity: float,
                         available_time_hours: float,
                         search_space_size: int) -> HPOConfig:
        """Select optimal HPO method based on context"""
        
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
        
        # Calculate adaptive trial budget
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
        """Score a method for given context - UPDATED: Architecture-based selection"""
        score = 0.0
        
        # Architecture preference (50% weight) - INCREASED from 30%
        arch_family = factors['architecture_family']
        if arch_family in self.selection_rules['architecture_preferences']:
            preferred_method = self.selection_rules['architecture_preferences'][arch_family]
            if method == preferred_method:
                score += 0.5
        
        # Architecture complexity matching (30% weight) - REPLACED dataset complexity
        arch_complexity = factors['architecture_complexity']
        if method == HPOMethod.BOHB and arch_complexity > 3.5:
            score += 0.3  # BOHB for complex architectures (ResNet34+, EfficientNet-B1+, etc.)
        elif method == HPOMethod.ASHA and arch_complexity <= 3.5:
            score += 0.3  # ASHA for simpler architectures (ResNet18, EfficientNet-B0, etc.)
        
        # Time budget consideration (10% weight) - REDUCED from 25%
        time_pressure = factors['time_pressure']
        if time_pressure in ['very_limited', 'limited'] and method == HPOMethod.ASHA:
            score += 0.1  # ASHA is faster due to pruning
        elif time_pressure == 'sufficient' and method == HPOMethod.BOHB:
            score += 0.1  # BOHB when we have time for sophistication
        
        # Historical performance (10% weight) - KEPT same
        historical_perf = factors['historical_performance'].get(method.value, 0.5)
        score += 0.1 * historical_perf
        
        return score
    
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
                recent_performance = np.mean(history[-5:])  # Last 5 results
                performance[method_name] = min(recent_performance, 1.0)
            else:
                performance[method_name] = 0.5  # Default neutral score
        
        return performance
    
    def _configure_method_parameters(self, 
                                   method: HPOMethod, 
                                   factors: Dict[str, Any],
                                   available_time_hours: float,
                                   n_trials: int) -> HPOConfig:
        """Configure parameters for selected method - UNIFIED VERSION"""
        
        # UNIFIED: Use available time directly (managed by BudgetManager)
        effective_time = available_time_hours
        timeout_seconds = int(effective_time * 3600 * 0.9)  # Use 90% of available time
        
        # Method-specific adjustments
        if method == HPOMethod.BOHB:
            # BOHB: Limit iterations for time constraint
            final_trials = max(1, min(n_trials // 2, 2))  # MAX 2 iterations
            
        elif method == HPOMethod.ASHA:
            # ASHA: Use calculated trials directly
            final_trials = n_trials
            if effective_time < 1.5:
                final_trials = min(final_trials, 3)  # Extra limit for short time
        
        else:
            final_trials = n_trials
        
        self.logger.info(f"Configured {method.value}: {final_trials} trials/iterations, {timeout_seconds}s timeout")
        
        return HPOConfig(
            method=method,
            n_trials=final_trials,
            timeout_seconds=timeout_seconds,
            early_stopping_rounds=1,
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
        """Complete HPO pipeline: select method and optimize"""
        
        # Select HPO method
        search_space_size = self._estimate_search_space_size(search_space)
        hpo_config = self.select_hpo_method(
            architecture_name, architecture_characteristics,
            dataset_complexity, available_time_hours, search_space_size
        )
        
        # Create HPO method instance
        hpo_method = self.hpo_methods[hpo_config.method](hpo_config)
        
        # Run optimization with progress tracking
        self.logger.info(f"Starting HPO for {architecture_name} with {hpo_config.n_trials} trials")
        start_time = time.time()
        
        result = hpo_method.optimize(objective_function, search_space, architecture_name)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"HPO completed for {architecture_name} in {elapsed_time:.1f}s")
        self.logger.info(f"  Best score: {result.best_score:.4f} ({result.n_trials_completed} trials)")
        if result.early_stopped:
            self.logger.info(f"  Stopped early due to excellent performance!")
        
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
                total_size *= 50  # Approximate discretization
        
        return min(total_size, 50000)  # Cap at reasonable size
    
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
        # Simulate a realistic objective function
        lr = params.get('learning_rate', 1e-3)
        batch_size = params.get('batch_size', 32)
        dropout = params.get('dropout', 0.2)
        
        # Simple heuristic: penalize extreme values
        score = 0.8
        if lr < 1e-4 or lr > 1e-2:
            score -= 0.1
        if batch_size not in [16, 32, 64]:
            score -= 0.05
        if dropout > 0.4:
            score -= 0.1
            
        # Add some noise
        score += np.random.normal(0, 0.05)
        return max(0.0, min(1.0, score))
    
    search_space = {
        'learning_rate': {'type': 'float', 'range': (1e-5, 1e-1), 'log_scale': True},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        'dropout': {'type': 'float', 'range': (0.0, 0.5)}
    }
    
    print("\nRunning optimization test...")
    result = selector.optimize_hyperparameters(
        'efficientnet_b0',
        dummy_objective,
        search_space,
        arch_characteristics,
        dataset_complexity=5.0,
        available_time_hours=0.1  # Short test
    )
    
    print(f"\nOptimization result:")
    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Best params: {result.best_params}")
    print(f"  Method used: {result.method_used.value}")
    print(f"  Trials completed: {result.n_trials_completed}")
    print(f"  Optimization time: {result.optimization_time:.2f}s")
    print(f"  Method efficiency: {result.method_efficiency:.3f}")
    print(f"  Early stopped: {result.early_stopped}")
    
    # Test both methods
    print("\n" + "="*50)
    print("Testing both methods:")
    
    for method_name in ['ASHA', 'BOHB']:
        print(f"\nTesting {method_name}...")
        
        # Force method selection
        if method_name == 'ASHA':
            test_config = HPOConfig(
                method=HPOMethod.ASHA,
                n_trials=5,
                timeout_seconds=30
            )
            test_method = ASHAOptimizerHPO(test_config)
        else:
            test_config = HPOConfig(
                method=HPOMethod.BOHB,
                n_trials=2,  # BOHB iterations
                timeout_seconds=30
            )
            test_method = BOHBOptimizerHPO(test_config)
        
        try:
            test_result = test_method.optimize(
                dummy_objective,
                search_space,
                f'test_{method_name.lower()}'
            )
            
            print(f"  {method_name} Result:")
            print(f"    Best score: {test_result.best_score:.3f}")
            print(f"    Best params: {test_result.best_params}")
            print(f"    Trials: {test_result.n_trials_completed}")
            print(f"    Time: {test_result.optimization_time:.2f}s")
            
        except Exception as e:
            print(f"  {method_name} failed: {e}")
    
    print("\nPerformance summary:")
    summary = selector.get_method_performance_summary()
    for method, stats in summary.items():
        print(f"  {method}: {stats['num_uses']} uses, "
              f"avg efficiency: {stats['average_efficiency']:.3f}")
    
    print("\nTest completed successfully!")
