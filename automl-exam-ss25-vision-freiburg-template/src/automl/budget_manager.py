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
    """Snapshot of budget state at a point in time"""
    timestamp: float
    total_time_elapsed: float
    total_time_remaining: float
    active_architectures: int
    completed_architectures: int
    stopped_architectures: int
    phase: ExecutionPhase
    resource_allocations: Dict[str, ResourceAllocation]
    reallocation_events: List[Dict[str, Any]] = field(default_factory=list)

class BudgetManager:
    """
    Intelligent budget manager for AutoML pipeline
    
    Core Innovation: Dynamic resource reallocation based on architecture performance
    and early stopping decisions
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.BudgetManager')
        
        # Time budget configuration - UPDATED: Better resource allocation
        self.total_time_hours = config.get('time_budget_hours', 24)
        self.architecture_search_ratio = config.get('architecture_search_ratio', 0.60)  # Reduced from 70%
        self.final_training_ratio = config.get('final_training_ratio', 0.30)           # Increased from 20%
        self.buffer_ratio = config.get('buffer_ratio', 0.10)                           # Keep 10%
        
        # Calculate phase budgets
        self.architecture_search_hours = self.total_time_hours * self.architecture_search_ratio
        self.final_training_hours = self.total_time_hours * self.final_training_ratio
        self.buffer_hours = self.total_time_hours * self.buffer_ratio
        
        # Resource constraints
        self.max_gpu_memory_gb = config.get('max_gpu_memory_gb', 16)
        self.max_parallel_architectures = config.get('max_parallel_architectures', 4)
        
        # State tracking
        self.start_time: Optional[float] = None
        self.current_phase = ExecutionPhase.INITIALIZATION
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.architecture_queue: List[str] = []
        self.completed_architectures: Set[str] = set()
        self.stopped_architectures: Set[str] = set()
        
        # Budget tracking
        self.budget_snapshots: List[BudgetSnapshot] = []
        self.reallocation_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info(f"BudgetManager initialized:")
        self.logger.info(f"  Total budget: {self.total_time_hours:.1f} hours")
        self.logger.info(f"  Architecture search: {self.architecture_search_hours:.1f} hours ({self.architecture_search_ratio*100:.0f}%)")
        self.logger.info(f"  Final training: {self.final_training_hours:.1f} hours ({self.final_training_ratio*100:.0f}%)")
        self.logger.info(f"  Buffer: {self.buffer_hours:.1f} hours ({self.buffer_ratio*100:.0f}%)")
    
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
        """Perform initial equal allocation of resources"""
        
        # Handle edge case: no architectures
        if not architectures:
            self.logger.warning("No architectures to allocate resources to")
            return
        
        # Equal time allocation during search phase
        time_per_architecture = self.architecture_search_hours / len(architectures)
        
        # Equal memory allocation (with safety margin)
        # Fix: Don't exceed parallel limit when calculating memory per architecture
        effective_parallel_limit = min(len(architectures), self.max_parallel_architectures)
        memory_per_architecture = (self.max_gpu_memory_gb * 0.8) / effective_parallel_limit
        
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
        
        self.logger.info(f"Initial allocation: {time_per_architecture:.2f}h and {memory_per_architecture:.1f}GB per architecture")
    
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
        """Check if we should transition to final training phase - FIXED VERSION"""
        with self._lock:
            if self.current_phase != ExecutionPhase.ARCHITECTURE_SEARCH:
                return False
            
            # Check time condition
            elapsed_hours = self.get_elapsed_time_hours()
            search_phase_complete = elapsed_hours >= self.architecture_search_hours
            
            # FIXED: Check if all architectures are done properly
            # Count architectures that have actually been processed (have results or were stopped)
            processed_count = len(self.completed_architectures) + len(self.stopped_architectures)
            total_architectures = len(self.resource_allocations)
            
            # Only consider "all done" if we've actually processed all architectures
            all_architectures_done = (processed_count >= total_architectures) and (total_architectures > 0)
            
            should_transition = search_phase_complete or all_architectures_done
            
            if should_transition:
                self.logger.info(f"Transitioning to final training phase:")
                self.logger.info(f"  Time condition: {elapsed_hours:.2f}h >= {self.architecture_search_hours:.2f}h ({search_phase_complete})")
                self.logger.info(f"  All done condition: {processed_count}/{total_architectures} processed ({all_architectures_done})")
            
            return should_transition
    
    def start_final_training_phase(self) -> List[str]:
        """
        Start final training phase and return architectures to train
        
        Returns:
            List of architectures selected for final training
        """
        with self._lock:
            self.current_phase = ExecutionPhase.FINAL_TRAINING
            
            # Select top architectures for final training
            final_candidates = self._select_final_training_candidates()
            
            # Allocate remaining time to final training
            remaining_time = self._calculate_remaining_time()
            self._allocate_final_training_time(final_candidates, remaining_time)
            
            self.logger.info(f"Starting final training phase:")
            self.logger.info(f"  Selected architectures: {final_candidates}")
            self.logger.info(f"  Remaining time: {remaining_time:.2f}h")
            
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
    
    def _calculate_remaining_time(self) -> float:
        """Calculate remaining time for final training"""
        elapsed_hours = self.get_elapsed_time_hours()
        remaining_total = max(0, self.total_time_hours - elapsed_hours)
        
        # Reserve buffer time
        usable_remaining = max(0, remaining_total - self.buffer_hours)
        
        return usable_remaining
    
    def _allocate_final_training_time(self, candidates: List[str], total_time: float):
        """Allocate time for final training among selected candidates"""
        if not candidates:
            self.logger.warning("No candidates for final training time allocation")
            return
        
        # Strategy: Give more time to better performing architectures
        candidate_performances = []
        for arch in candidates:
            if arch in self.resource_allocations:
                performance = self.resource_allocations[arch].performance_score
                candidate_performances.append((arch, performance))
        
        total_performance = sum(perf for _, perf in candidate_performances)
        
        if total_performance <= 0:
            # Equal distribution fallback
            time_per_candidate = total_time / len(candidates)
            for arch in candidates:
                if arch in self.resource_allocations:
                    self.resource_allocations[arch].allocated_time_hours = time_per_candidate
                    self.resource_allocations[arch].phase = ExecutionPhase.FINAL_TRAINING
        else:
            # Performance-weighted distribution
            for arch, performance in candidate_performances:
                if arch in self.resource_allocations:
                    weight = performance / total_performance
                    allocated_time = total_time * weight
                    self.resource_allocations[arch].allocated_time_hours = allocated_time
                    self.resource_allocations[arch].phase = ExecutionPhase.FINAL_TRAINING
                    self.resource_allocations[arch].is_active = True  # Reactivate for final training
                    
                    self.logger.info(f"Final training allocation for {arch}: {allocated_time:.2f}h (weight: {weight:.2f})")
    
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
        
        return True
    
    def get_elapsed_time_hours(self) -> float:
        """Get elapsed time since execution start"""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0
    
    def get_remaining_time_hours(self) -> float:
        """Get remaining time in total budget"""
        elapsed = self.get_elapsed_time_hours()
        return max(0, self.total_time_hours - elapsed)
    
    def get_phase_remaining_time_hours(self) -> float:
        """Get remaining time in current phase"""
        elapsed = self.get_elapsed_time_hours()
        
        if self.current_phase == ExecutionPhase.ARCHITECTURE_SEARCH:
            return max(0, self.architecture_search_hours - elapsed)
        elif self.current_phase == ExecutionPhase.FINAL_TRAINING:
            final_training_start = self.architecture_search_hours
            final_training_elapsed = max(0, elapsed - final_training_start)
            return max(0, self.final_training_hours - final_training_elapsed)
        else:
            return self.get_remaining_time_hours()
    
    def is_budget_exhausted(self) -> bool:
        """Check if time budget is exhausted"""
        return self.get_remaining_time_hours() <= 0.1  # 6 minutes tolerance
    
    def _take_budget_snapshot(self):
        """Take a snapshot of current budget state"""
        elapsed_time = self.get_elapsed_time_hours()
        remaining_time = self.get_remaining_time_hours()
        
        active_count = sum(1 for alloc in self.resource_allocations.values() if alloc.is_active)
        completed_count = len(self.completed_architectures)
        stopped_count = len(self.stopped_architectures)
        
        snapshot = BudgetSnapshot(
            timestamp=time.time(),
            total_time_elapsed=elapsed_time,
            total_time_remaining=remaining_time,
            active_architectures=active_count,
            completed_architectures=completed_count,
            stopped_architectures=stopped_count,
            phase=self.current_phase,
            resource_allocations=self.resource_allocations.copy()
        )
        
        self.budget_snapshots.append(snapshot)
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary"""
        with self._lock:
            elapsed = self.get_elapsed_time_hours()
            remaining = self.get_remaining_time_hours()
            phase_remaining = self.get_phase_remaining_time_hours()
            
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
                    'remaining_hours': remaining,
                    'phase_remaining_hours': phase_remaining,
                    'budget_utilization': elapsed / self.total_time_hours if self.total_time_hours > 0 else 0,
                    'is_budget_exhausted': self.is_budget_exhausted()
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
                    'architecture_search_hours': self.architecture_search_hours,
                    'final_training_hours': self.final_training_hours,
                    'buffer_hours': self.buffer_hours,
                    'total_budget_hours': self.total_time_hours
                }
            }
            
            return summary
    
    def print_budget_status(self):
        """Print human-readable budget status"""
        summary = self.get_budget_summary()
        
        print(f"\n=== AutoML Budget Status ===")
        print(f"Phase: {summary['execution_status']['phase'].upper()}")
        print(f"Time: {summary['execution_status']['elapsed_hours']:.1f}h elapsed, "
              f"{summary['execution_status']['remaining_hours']:.1f}h remaining "
              f"({summary['execution_status']['budget_utilization']*100:.1f}% used)")
        
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
        """Save budget manager state"""
        state = {
            'config': {
                'total_time_hours': self.total_time_hours,
                'architecture_search_ratio': self.architecture_search_ratio,
                'final_training_ratio': self.final_training_ratio,
                'buffer_ratio': self.buffer_ratio
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
        
        self.logger.info(f"Budget manager state saved to {filepath}")

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
