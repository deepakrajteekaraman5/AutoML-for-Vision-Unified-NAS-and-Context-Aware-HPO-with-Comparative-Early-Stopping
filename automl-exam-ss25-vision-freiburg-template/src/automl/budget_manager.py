# src/automl/budget_manager.py
"""
Budget Manager for AutoML Pipeline
Core responsibility: Intelligent allocation and management of the 24-hour time budget
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

# Note: Update this import once utils.py is implemented
try:
    from .utils import AutoMLConfig, Timer, MetricTracker
except ImportError:
    # Fallback for testing
    class AutoMLConfig:
        def __init__(self):
            self.config = {}
        def get(self, key, default=None):
            return self.config.get(key, default)
        def set(self, key, value):
            self.config[key] = value
    
    class Timer:
        def __init__(self):
            self.start_time = time.time()
        def elapsed(self):
            return time.time() - self.start_time
    
    class MetricTracker:
        def __init__(self):
            self.metrics = {}
    
    def get_time_config():
        from dataclasses import dataclass
        @dataclass
        class TimeConfig:
            MAX_HOURS_PER_MODEL: float = 2.0
            MAX_HOURS_FINAL_TRAINING: float = 2.0
            BUFFER_HOURS: float = 2.0
        return TimeConfig()

class ExecutionPhase(Enum):
    """Different phases of AutoML execution"""
    INITIALIZATION = "initialization"
    ARCHITECTURE_SEARCH = "architecture_search"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    FINAL_TRAINING = "final_training"
    EVALUATION = "evaluation"
    COMPLETED = "completed"

@dataclass
class ResourceAllocation:
    """Resource allocation for an architecture"""
    architecture: str
    allocated_time_hours: float
    allocated_memory_gb: float
    priority_score: float
    phase: ExecutionPhase
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    time_used: float = 0.0
    memory_used: float = 0.0
    is_active: bool = True
    performance_score: float = 0.0
    efficiency_score: float = 0.0  # Performance per unit time

@dataclass
class BudgetSnapshot:
    """Snapshot of budget state at a point in time - UNIFIED VERSION"""
    timestamp: float
    elapsed_hours: float  # Only track elapsed time, no remaining/total
    active_architectures: int
    completed_architectures: int
    stopped_architectures: int
    phase: ExecutionPhase
    resource_allocations: Dict[str, ResourceAllocation]

class BudgetManager:
    """
    UNIFIED Time Management for AutoML Pipeline
    
    SINGLE SOURCE OF TRUTH for all time-related functionality:
    - Per-model time limits (user configurable)
    - Smart completion (finish last epoch/trial)
    - No final training time limits
    - Strict enforcement with grace periods
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.BudgetManager')
        
        # UNIFIED TIME CONFIGURATION - User sets per-model time only
        self.hours_per_model = config.get('hours_per_model', 2.0)  # ONLY user-configurable time constraint
        
        # NO PIPELINE TIME LIMITS - only per-model limits
        # NO FINAL TRAINING TIME LIMITS - final training runs without time constraints
        
        # Smart completion settings
        self.smart_completion_enabled = config.get('smart_completion', True)
        self.grace_period_minutes = config.get('grace_period_minutes', 10)  # Allow 10min to finish last epoch/trial
        
        # Resource constraints (non-time related)
        self.max_gpu_memory_gb = config.get('max_gpu_memory_gb', 16)
        self.max_parallel_architectures = config.get('max_parallel_architectures', 1)  # Sequential by default
        
        # State tracking
        self.start_time: Optional[float] = None
        self.current_phase = ExecutionPhase.INITIALIZATION
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.architecture_queue: List[str] = []
        self.completed_architectures: Set[str] = set()
        self.stopped_architectures: Set[str] = set()
        
        # Per-model time tracking
        self.model_start_times: Dict[str, float] = {}
        self.model_timeouts: Dict[str, bool] = {}
        
        # Budget tracking
        self.budget_snapshots: List[BudgetSnapshot] = []
        self.reallocation_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info(f"UNIFIED BudgetManager initialized:")
        self.logger.info(f"  Per-model time limit: {self.hours_per_model:.1f} hours (STRICT)")
        self.logger.info(f"  Smart completion: {self.smart_completion_enabled}")
        self.logger.info(f"  Grace period: {self.grace_period_minutes} minutes")
        self.logger.info(f"  Final training: NO TIME LIMITS")
        self.logger.info(f"  Pipeline: NO TOTAL TIME LIMITS")
    
    @staticmethod
    def get_model_count_for_complexity(complexity_score: float) -> int:
        """
        Determine number of models based on dataset complexity
        
        Args:
            complexity_score: Dataset complexity score (0-10)
            
        Returns:
            Number of models to evaluate
        """
        LOW_COMPLEXITY_MODELS = 3  # For complexity <= 3.0
        HIGH_COMPLEXITY_MODELS = 4  # For complexity > 3.0
        
        if complexity_score <= 3.0:
            return LOW_COMPLEXITY_MODELS
        else:
            return HIGH_COMPLEXITY_MODELS
    
    def calculate_total_budget(self, num_models: int) -> dict:
        """
        Calculate total budget breakdown
        
        Args:
            num_models: Number of models to evaluate
            
        Returns:
            Dictionary with budget breakdown
        """
        max_hours_per_model = self.hours_per_model
        architecture_search_hours = num_models * max_hours_per_model
        final_training_hours = max_hours_per_model  # 1 model for final training
        buffer_hours = 0.5  # Small buffer
        total_hours = architecture_search_hours + final_training_hours + buffer_hours
        
        return {
            'max_hours_per_model': max_hours_per_model,
            'architecture_search_hours': architecture_search_hours,
            'final_training_hours': final_training_hours,
            'buffer_hours': buffer_hours,
            'total_hours': total_hours
        }
    
    def start_execution(self, architectures: List[str]):
        """Start the AutoML execution with given architectures"""
        with self._lock:
            self.start_time = time.time()
            self.current_phase = ExecutionPhase.ARCHITECTURE_SEARCH
            self.architecture_queue = architectures.copy()
            
            # Handle edge case: no architectures
            if not architectures:
                self.logger.warning("No architectures provided - creating empty execution")
                self._take_budget_snapshot()
                return
            
            # Initial resource allocation
            self._perform_initial_allocation(architectures)
            
            # Take initial snapshot
            self._take_budget_snapshot()
            
            self.logger.info(f"Execution started with {len(architectures)} architectures")
            self.logger.info(f"Architecture queue: {architectures}")
    
    def _perform_initial_allocation(self, architectures: List[str]):
        """Perform initial allocation - UNIFIED TIME MANAGEMENT"""
        
        # Handle edge case: no architectures
        if not architectures:
            self.logger.warning("No architectures to allocate resources to")
            return
        
        # UNIFIED: Each architecture gets exactly hours_per_model time
        time_per_architecture = self.hours_per_model
        
        # Memory allocation (sequential processing, so full memory available)
        memory_per_architecture = self.max_gpu_memory_gb * 0.8  # 80% safety margin
        
        for arch in architectures:
            allocation = ResourceAllocation(
                architecture=arch,
                allocated_time_hours=time_per_architecture,
                allocated_memory_gb=memory_per_architecture,
                priority_score=1.0,  # Initial equal priority
                phase=ExecutionPhase.ARCHITECTURE_SEARCH,
                is_active=False  # Start as inactive, will be activated when training starts
            )
            self.resource_allocations[arch] = allocation
        
        self.logger.info(f"UNIFIED allocation: {time_per_architecture:.2f}h per architecture (STRICT)")
        self.logger.info(f"Memory allocation: {memory_per_architecture:.1f}GB per architecture")
    
    def get_architecture_allocation(self, architecture: str) -> Optional[ResourceAllocation]:
        """Get current resource allocation for an architecture"""
        with self._lock:
            return self.resource_allocations.get(architecture)
    
    def start_architecture_training(self, architecture: str) -> bool:
        """Mark architecture as starting training and allocate resources"""
        with self._lock:
            if architecture not in self.resource_allocations:
                self.logger.error(f"Architecture {architecture} not in allocation table")
                return False
            
            allocation = self.resource_allocations[architecture]
            
            # Check if we have available resources
            if not self._has_available_resources(allocation):
                self.logger.warning(f"Insufficient resources for {architecture}")
                return False
            
            # Start tracking time
            allocation.start_time = time.time()
            allocation.is_active = True
            
            self.logger.info(f"Started training {architecture} with {allocation.allocated_time_hours:.2f}h budget")
            return True
    
    def update_architecture_progress(self, 
                                   architecture: str,
                                   performance_score: float,
                                   time_used_hours: float,
                                   memory_used_gb: float):
        """Update progress tracking for an architecture"""
        with self._lock:
            if architecture not in self.resource_allocations:
                self.logger.warning(f"Cannot update progress for unknown architecture: {architecture}")
                return
            
            allocation = self.resource_allocations[architecture]
            allocation.time_used = time_used_hours
            allocation.memory_used = memory_used_gb
            allocation.performance_score = performance_score
            
            # Calculate efficiency score
            if time_used_hours > 0:
                allocation.efficiency_score = performance_score / time_used_hours
            
            # Check for budget overrun
            if time_used_hours > allocation.allocated_time_hours * 1.1:  # 10% tolerance
                self.logger.warning(f"{architecture} exceeded time budget: {time_used_hours:.2f}h / {allocation.allocated_time_hours:.2f}h")
            
            self.logger.debug(f"Updated {architecture}: perf={performance_score:.3f}, time={time_used_hours:.2f}h, eff={allocation.efficiency_score:.3f}")
    
    def architecture_stopped_early(self, 
                                  architecture: str, 
                                  reason: str,
                                  final_performance: float) -> float:
        """
        Handle early stopping of an architecture and return freed time
        
        Returns:
            float: Hours of time freed up for reallocation
        """
        with self._lock:
            if architecture not in self.resource_allocations:
                self.logger.warning(f"Cannot stop unknown architecture: {architecture}")
                return 0.0
            
            allocation = self.resource_allocations[architecture]
            allocation.is_active = False
            allocation.end_time = time.time()
            allocation.performance_score = final_performance
            
            # Calculate freed time
            freed_time = max(0, allocation.allocated_time_hours - allocation.time_used)
            
            # Add to stopped architectures
            self.stopped_architectures.add(architecture)
            
            # Record reallocation event
            reallocation_event = {
                'timestamp': time.time(),
                'type': 'architecture_stopped',
                'architecture': architecture,
                'reason': reason,
                'freed_time_hours': freed_time,
                'final_performance': final_performance,
                'efficiency': allocation.efficiency_score
            }
            self.reallocation_history.append(reallocation_event)
            
            self.logger.info(f"Architecture {architecture} stopped early: {reason}")
            self.logger.info(f"  Final performance: {final_performance:.3f}")
            self.logger.info(f"  Time used: {allocation.time_used:.2f}h / {allocation.allocated_time_hours:.2f}h")
            self.logger.info(f"  Freed time: {freed_time:.2f}h")
            
            # Trigger reallocation
            if freed_time > 0.1:  # Only reallocate significant amounts
                self._reallocate_freed_time(freed_time, reason)
            
            return freed_time
    
    def _reallocate_freed_time(self, freed_time_hours: float, reason: str):
        """Reallocate freed time to remaining active architectures"""
        
        # Get active architectures sorted by efficiency
        active_architectures = [
            (arch, alloc) for arch, alloc in self.resource_allocations.items()
            if alloc.is_active and arch not in self.stopped_architectures
        ]
        
        if not active_architectures:
            self.logger.info(f"No active architectures to reallocate {freed_time_hours:.2f}h to")
            return
        
        # Sort by efficiency (performance per unit time) in descending order
        active_architectures.sort(key=lambda x: x[1].efficiency_score, reverse=True)
        
        # Reallocation strategy: Give more time to efficient architectures
        total_efficiency = sum(alloc.efficiency_score for _, alloc in active_architectures)
        
        if total_efficiency <= 0:
            # Fallback to equal distribution
            time_per_arch = freed_time_hours / len(active_architectures)
            for arch, alloc in active_architectures:
                alloc.allocated_time_hours += time_per_arch
                self.logger.info(f"Reallocated {time_per_arch:.2f}h to {arch} (equal distribution)")
        else:
            # Efficiency-based distribution
            for arch, alloc in active_architectures:
                efficiency_ratio = alloc.efficiency_score / total_efficiency
                additional_time = freed_time_hours * efficiency_ratio
                alloc.allocated_time_hours += additional_time
                
                self.logger.info(f"Reallocated {additional_time:.2f}h to {arch} (efficiency: {alloc.efficiency_score:.3f})")
        
        # Record reallocation
        reallocation_event = {
            'timestamp': time.time(),
            'type': 'time_reallocation',
            'freed_time_hours': freed_time_hours,
            'reason': reason,
            'beneficiaries': [(arch, alloc.efficiency_score) for arch, alloc in active_architectures],
            'distribution_strategy': 'efficiency_based' if total_efficiency > 0 else 'equal'
        }
        self.reallocation_history.append(reallocation_event)
    
    def should_start_final_training(self) -> bool:
        """Check if we should transition to final training phase - UNIFIED VERSION"""
        with self._lock:
            if self.current_phase != ExecutionPhase.ARCHITECTURE_SEARCH:
                return False
            
            # UNIFIED: NO TIME-BASED TRANSITION - only when all architectures are done
            # Count architectures that have actually been processed (have results or were stopped)
            processed_count = len(self.completed_architectures) + len(self.stopped_architectures)
            total_architectures = len(self.resource_allocations)
            
            # Only consider "all done" if we've actually processed all architectures
            all_architectures_done = (processed_count >= total_architectures) and (total_architectures > 0)
            
            if all_architectures_done:
                self.logger.info(f"Transitioning to final training phase:")
                self.logger.info(f"  All architectures processed: {processed_count}/{total_architectures}")
                self.logger.info(f"  NO TIME LIMITS for final training")
            
            return all_architectures_done
    
    def start_final_training_phase(self) -> List[str]:
        """
        Start final training phase and return architectures to train - UNIFIED VERSION
        
        Returns:
            List of architectures selected for final training
        """
        with self._lock:
            self.current_phase = ExecutionPhase.FINAL_TRAINING
            
            # Select top architectures for final training
            final_candidates = self._select_final_training_candidates()
            
            # UNIFIED: NO TIME ALLOCATION for final training - runs without limits
            self.logger.info(f"Starting final training phase:")
            self.logger.info(f"  Selected architectures: {final_candidates}")
            self.logger.info(f"  NO TIME LIMITS for final training")
            
            return final_candidates
    
    def _select_final_training_candidates(self, max_candidates: int = 3) -> List[str]:
        """Select best architectures for final training"""
        
        # Handle empty architecture case
        if not self.resource_allocations:
            self.logger.warning("No architectures available for final training")
            return []
        
        # Get architectures that completed (not stopped early)
        completed_architectures = [
            (arch, alloc) for arch, alloc in self.resource_allocations.items()
            if not alloc.is_active and arch not in self.stopped_architectures
        ]
        
        # Also consider architectures that are still active but performing well
        active_architectures = [
            (arch, alloc) for arch, alloc in self.resource_allocations.items()
            if alloc.is_active and alloc.performance_score > 0
        ]
        
        # Combine and sort by performance
        all_candidates = completed_architectures + active_architectures
        all_candidates.sort(key=lambda x: x[1].performance_score, reverse=True)
        
        # Select top candidates
        selected = [arch for arch, _ in all_candidates[:max_candidates]]
        
        # Ensure we have at least one candidate
        if not selected and self.resource_allocations:
            # Fall back to best performing architecture
            best_arch = max(self.resource_allocations.items(), key=lambda x: x[1].performance_score)
            selected = [best_arch[0]]
        
        return selected
    
    # UNIFIED: These methods are no longer needed since final training has no time limits
    def _calculate_remaining_time(self) -> float:
        """DEPRECATED: Final training has no time limits in unified system"""
        return float('inf')  # Unlimited time for final training
    
    def _allocate_final_training_time(self, candidates: List[str], total_time: float):
        """DEPRECATED: Final training has no time limits in unified system"""
        # Just mark candidates as final training phase - no time allocation needed
        for arch in candidates:
            if arch in self.resource_allocations:
                self.resource_allocations[arch].phase = ExecutionPhase.FINAL_TRAINING
                self.resource_allocations[arch].is_active = True  # Reactivate for final training
                self.logger.info(f"Final training candidate: {arch} (NO TIME LIMITS)")
    
    def _has_available_resources(self, required_allocation: ResourceAllocation) -> bool:
        """Check if required resources are available"""
        
        # Check memory availability - compare against current active memory usage
        current_memory_usage = sum(
            alloc.allocated_memory_gb for alloc in self.resource_allocations.values()
            if alloc.is_active
        )
        
        if current_memory_usage + required_allocation.allocated_memory_gb > self.max_gpu_memory_gb:
            self.logger.debug(f"Memory constraint: {current_memory_usage:.1f} + {required_allocation.allocated_memory_gb:.1f} > {self.max_gpu_memory_gb}")
            return False
        
        # Check parallel architecture limit
        active_count = sum(1 for alloc in self.resource_allocations.values() if alloc.is_active)
        if active_count >= self.max_parallel_architectures:
            self.logger.debug(f"Parallel limit constraint: {active_count} >= {self.max_parallel_architectures}")
            return False
        
        # ENHANCED LOGGING: Log resource check details
        self.logger.debug(f"Resource check for {required_allocation.architecture}:")
        self.logger.debug(f"  Current memory usage: {current_memory_usage:.1f}GB")
        self.logger.debug(f"  Required memory: {required_allocation.allocated_memory_gb:.1f}GB")
        self.logger.debug(f"  Max memory: {self.max_gpu_memory_gb:.1f}GB")
        self.logger.debug(f"  Active architectures: {active_count}")
        self.logger.debug(f"  Max parallel: {self.max_parallel_architectures}")
        
        return True
    
    def get_elapsed_time_hours(self) -> float:
        """Get elapsed time since execution start"""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0
    
    def _take_budget_snapshot(self):
        """Take a snapshot of current budget state - UNIFIED VERSION"""
        elapsed_time = self.get_elapsed_time_hours()
        
        active_count = sum(1 for alloc in self.resource_allocations.values() if alloc.is_active)
        completed_count = len(self.completed_architectures)
        stopped_count = len(self.stopped_architectures)
        
        snapshot = BudgetSnapshot(
            timestamp=time.time(),
            elapsed_hours=elapsed_time,
            active_architectures=active_count,
            completed_architectures=completed_count,
            stopped_architectures=stopped_count,
            phase=self.current_phase,
            resource_allocations=self.resource_allocations.copy()
        )
        
        self.budget_snapshots.append(snapshot)
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary - UNIFIED VERSION"""
        with self._lock:
            elapsed = self.get_elapsed_time_hours()
            
            # Architecture statistics
            active_architectures = [arch for arch, alloc in self.resource_allocations.items() if alloc.is_active]
            completed_architectures = list(self.completed_architectures)
            stopped_architectures = list(self.stopped_architectures)
            
            # Resource usage statistics
            total_memory_used = sum(alloc.memory_used for alloc in self.resource_allocations.values())
            avg_efficiency = np.mean([alloc.efficiency_score for alloc in self.resource_allocations.values()]) if self.resource_allocations else 0.0
            
            # Performance statistics
            best_performance = max((alloc.performance_score for alloc in self.resource_allocations.values()), default=0.0)
            performance_scores = [(arch, alloc.performance_score) for arch, alloc in self.resource_allocations.items()]
            performance_scores.sort(key=lambda x: x[1], reverse=True)
            
            summary = {
                'execution_status': {
                    'phase': self.current_phase.value,
                    'elapsed_hours': elapsed,
                    'budget_utilization': 0.0,  # UNIFIED: No total budget concept
                    'is_budget_exhausted': False  # UNIFIED: Never exhausted
                },
                'architecture_statistics': {
                    'total_architectures': len(self.resource_allocations),
                    'active': len(active_architectures),
                    'completed': len(completed_architectures),
                    'stopped_early': len(stopped_architectures),
                    'active_list': active_architectures,
                    'completed_list': completed_architectures,
                    'stopped_list': stopped_architectures
                },
                'resource_usage': {
                    'total_memory_used_gb': total_memory_used,
                    'max_memory_gb': self.max_gpu_memory_gb,
                    'memory_utilization': total_memory_used / self.max_gpu_memory_gb if self.max_gpu_memory_gb > 0 else 0,
                    'parallel_limit': self.max_parallel_architectures
                },
                'performance_metrics': {
                    'best_performance': best_performance,
                    'average_efficiency': avg_efficiency,
                    'performance_rankings': performance_scores[:5],  # Top 5
                    'total_reallocations': len(self.reallocation_history)
                },
                'phase_breakdown': {
                    'per_model_hours': self.hours_per_model,  # UNIFIED: Only per-model time matters
                    'smart_completion': self.smart_completion_enabled,
                    'grace_period_minutes': self.grace_period_minutes,
                    'final_training_unlimited': True
                }
            }
            
            return summary
    
    def print_budget_status(self):
        """Print human-readable budget status - UNIFIED VERSION"""
        summary = self.get_budget_summary()
        
        print(f"\n=== AutoML Budget Status ===")
        print(f"Phase: {summary['execution_status']['phase'].upper()}")
        print(f"Time: {summary['execution_status']['elapsed_hours']:.1f}h elapsed (NO TIME LIMITS)")
        
        print(f"\nArchitectures:")
        print(f"  Active: {summary['architecture_statistics']['active']} {summary['architecture_statistics']['active_list']}")
        print(f"  Completed: {summary['architecture_statistics']['completed']} {summary['architecture_statistics']['completed_list']}")
        print(f"  Stopped: {summary['architecture_statistics']['stopped_early']} {summary['architecture_statistics']['stopped_list']}")
        
        print(f"\nPerformance:")
        print(f"  Best score: {summary['performance_metrics']['best_performance']:.3f}")
        print(f"  Avg efficiency: {summary['performance_metrics']['average_efficiency']:.3f}")
        print(f"  Reallocations: {summary['performance_metrics']['total_reallocations']}")
        
        if summary['performance_metrics']['performance_rankings']:
            print(f"  Top performers:")
            for arch, score in summary['performance_metrics']['performance_rankings']:
                print(f"    {arch}: {score:.3f}")
    
    def save_state(self, filepath: str):
        """Save budget manager state - UNIFIED VERSION"""
        state = {
            'config': {
                'hours_per_model': self.hours_per_model,
                'smart_completion_enabled': self.smart_completion_enabled,
                'grace_period_minutes': self.grace_period_minutes,
                'max_gpu_memory_gb': self.max_gpu_memory_gb,
                'max_parallel_architectures': self.max_parallel_architectures
            },
            'execution_state': {
                'start_time': self.start_time,
                'current_phase': self.current_phase.value,
                'elapsed_hours': self.get_elapsed_time_hours()
            },
            'budget_summary': self.get_budget_summary(),
            'resource_allocations': {
                arch: {
                    'allocated_time_hours': alloc.allocated_time_hours,
                    'allocated_memory_gb': alloc.allocated_memory_gb,
                    'time_used': alloc.time_used,
                    'memory_used': alloc.memory_used,
                    'performance_score': alloc.performance_score,
                    'efficiency_score': alloc.efficiency_score,
                    'is_active': alloc.is_active,
                    'phase': alloc.phase.value
                }
                for arch, alloc in self.resource_allocations.items()
            },
            'reallocation_history': self.reallocation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"UNIFIED Budget manager state saved to {filepath}")

# Test the budget manager
if __name__ == "__main__":
    # Simple test
    config = AutoMLConfig()
    budget_manager = BudgetManager(config)
    
    # Test with sample architectures
    architectures = ['resnet18', 'efficientnet_b0', 'densenet121', 'mobilenetv3_small_100']
    
    print("Testing Budget Manager...")
    budget_manager.start_execution(architectures)
    budget_manager.print_budget_status()
    
    # Simulate some training progress
    budget_manager.start_architecture_training('resnet18')
    budget_manager.update_architecture_progress('resnet18', 0.85, 1.5, 2.0)
    
    budget_manager.start_architecture_training('efficientnet_b0')
    budget_manager.update_architecture_progress('efficientnet_b0', 0.87, 2.0, 3.0)
    
    # Simulate early stopping
    freed_time = budget_manager.architecture_stopped_early('densenet121', 'poor_performance', 0.75)
    print(f"\nFreed time from early stopping: {freed_time:.2f} hours")
    
    budget_manager.print_budget_status()
    
    print("\nBudget Manager test completed!")
