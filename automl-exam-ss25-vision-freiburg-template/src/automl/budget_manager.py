# src/automl/budget_manager.py
"""
Budget Manager for AutoML Pipeline - ADVANCED INTEGRATION VERSION
Core responsibility: Intelligent allocation and management with full component coordination
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
    ADVANCED: Intelligent budget manager with full component coordination
    
    Core Innovation: Dynamic resource reallocation with intelligent coordination
    between early stopping, HPO selection, and training systems
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.BudgetManager')
        
        # Time budget configuration
        self.total_time_hours = config.get('time_budget_hours', 24)
        self.architecture_search_ratio = config.get('architecture_search_ratio', 0.70)  # 70%
        self.final_training_ratio = config.get('final_training_ratio', 0.20)           # 20%
        self.buffer_ratio = config.get('buffer_ratio', 0.10)                           # 10%
        
        # Calculate phase budgets
        self.architecture_search_hours = self.total_time_hours * self.architecture_search_ratio
        self.final_training_hours = self.total_time_hours * self.final_training_ratio
        self.buffer_hours = self.total_time_hours * self.buffer_ratio
        
        # Resource constraints
        self.max_gpu_memory_gb = config.get('max_gpu_memory_gb', 16)
        self.max_parallel_architectures = config.get('max_parallel_architectures', 4)
        
        # State tracking with thread safety
        self._lock = threading.RLock()
        self.start_time: Optional[float] = None
        self.current_phase = ExecutionPhase.INITIALIZATION
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.architecture_queue: List[str] = []
        self.completed_architectures: Set[str] = set()
        self.stopped_architectures: Set[str] = set()
        
        # Budget tracking
        self.budget_snapshots: List[BudgetSnapshot] = []
        self.reallocation_history: List[Dict[str, Any]] = []
        
        # NEW: Integration components
        self.early_stopping_engine = None  # Will be set by pipeline
        self.hpo_selector = None           # Will be set by pipeline
        
        # NEW: Anti-thrashing protection
        self.reallocation_cooldown = 1800  # 30 minutes
        self.last_reallocation_time = {}
        self.max_reallocations_per_hour = 3
        self.recent_decisions = deque(maxlen=100)  # Track recent decisions
        
        # NEW: Coordination state
        self.pending_early_stops = {}  # Architectures pending early stop confirmation
        self.resource_change_notifications = []  # Queue for notifying other systems
        
        self.logger.info(f"BudgetManager initialized with advanced integration:")
        self.logger.info(f"  Total budget: {self.total_time_hours:.1f} hours")
        self.logger.info(f"  Architecture search: {self.architecture_search_hours:.1f} hours ({self.architecture_search_ratio*100:.0f}%)")
        self.logger.info(f"  Final training: {self.final_training_hours:.1f} hours ({self.final_training_ratio*100:.0f}%)")
        self.logger.info(f"  Buffer: {self.buffer_hours:.1f} hours ({self.buffer_ratio*100:.0f}%)")
    
    def set_integration_components(self, early_stopping_engine=None, hpo_selector=None):
        """NEW: Set references to other pipeline components for coordination"""
        self.early_stopping_engine = early_stopping_engine
        self.hpo_selector = hpo_selector
        self.logger.info("Integration components set for coordination")
    
    def start_execution(self, architectures: List[str]):
        """Start execution with full integration - THREAD SAFE"""
        print("DEBUG: BudgetManager.start_execution called")  # ADD THIS
        with self._lock:
            try:
                print("DEBUG: Got lock in start_execution")  # ADD THIS
                self.start_time = time.time()
                self.current_phase = ExecutionPhase.ARCHITECTURE_SEARCH
                self.architecture_queue = architectures.copy()
                
                if not architectures:
                    self.logger.warning("No architectures provided - creating empty execution")
                    self._take_budget_snapshot()
                    return
                
                print("DEBUG: About to call _perform_initial_allocation")  # ADD THIS
                # Initial resource allocation
                self._perform_initial_allocation(architectures)

                print("DEBUG: About to call _take_budget_snapshot")  # ADD THIS
                
                # Take initial snapshot
                
                self._take_budget_snapshot()

                print("DEBUG: BudgetManager.start_execution completing")  # ADD THIS
                
                self.logger.info(f"Execution started with {len(architectures)} architectures")
                
            except Exception as e:
                self.logger.error(f"Failed to start execution: {e}")
                print(f"DEBUG: Exception in start_execution: {e}")  # ADD THIS
                raise
    
    def _perform_initial_allocation(self, architectures: List[str]):
        print("DEBUG: _perform_initial_allocation called")  # ADD THIS
        """Perform initial allocation with error handling"""
        try:
            if not architectures:
                self.logger.warning("No architectures to allocate resources to")
                return
            
            print("DEBUG: Creating resource allocations")  # ADD THIS
            
            # Equal time allocation during search phase
            time_per_architecture = self.architecture_search_hours / len(architectures)
            
            # Equal memory allocation (with safety margin)
            effective_parallel_limit = min(len(architectures), self.max_parallel_architectures)
            memory_per_architecture = (self.max_gpu_memory_gb * 0.8) / effective_parallel_limit
            
            for arch in architectures:
                allocation = ResourceAllocation(
                    architecture=arch,
                    allocated_time_hours=time_per_architecture,
                    allocated_memory_gb=memory_per_architecture,
                    priority_score=1.0,
                    phase=ExecutionPhase.ARCHITECTURE_SEARCH,
                    is_active=False
                )
                self.resource_allocations[arch] = allocation

            print("DEBUG: _perform_initial_allocation completing")  # ADD THIS
            self.logger.info(f"Initial allocation: {time_per_architecture:.2f}h and {memory_per_architecture:.1f}GB per architecture")
            
        except Exception as e:
            print(f"DEBUG: Exception in _perform_initial_allocation: {e}")  # ADD THIS
            self.logger.error(f"Initial allocation failed: {e}")
            raise
    
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
        """Update progress with early stopping coordination - THREAD SAFE"""
        try:
            with self._lock:
                if architecture not in self.resource_allocations:
                    self.logger.warning(f"Cannot update progress for unknown architecture: {architecture}")
                    return
                
                allocation = self.resource_allocations[architecture]
                if allocation is None:
                    self.logger.error(f"Allocation for {architecture} is None")
                    return
                
                # Update allocation state
                allocation.time_used = time_used_hours
                allocation.memory_used = memory_used_gb
                allocation.performance_score = performance_score
                
                # Calculate efficiency score
                if time_used_hours > 0:
                    allocation.efficiency_score = performance_score / time_used_hours
                
                # NEW: Check for budget overrun
                if time_used_hours > allocation.allocated_time_hours * 1.1:  # 10% tolerance
                    self.logger.warning(f"{architecture} exceeded time budget: {time_used_hours:.2f}h / {allocation.allocated_time_hours:.2f}h")
                    
                    # NEW: Notify early stopping about resource pressure
                    if self.early_stopping_engine:
                        self._notify_early_stopping_resource_pressure(architecture, time_used_hours / allocation.allocated_time_hours)
                
                self.logger.debug(f"Updated {architecture}: perf={performance_score:.3f}, time={time_used_hours:.2f}h, eff={allocation.efficiency_score:.3f}")
                
        except Exception as e:
            self.logger.error(f"Failed to update progress for {architecture}: {e}")
    
    def request_early_stopping_decision(self, architecture: str, current_performance: float) -> bool:
        """NEW: Coordinated early stopping request from HPO"""
        try:
            with self._lock:
                if architecture not in self.resource_allocations:
                    return False
                
                # Check if early stopping is appropriate from budget perspective
                allocation = self.resource_allocations[architecture]
                
                # Don't stop if we haven't used enough time to make a decision
                min_time_for_decision = allocation.allocated_time_hours * 0.3  # At least 30% of allocated time
                if allocation.time_used < min_time_for_decision:
                    self.logger.debug(f"Not enough time used for {architecture} early stopping decision")
                    return False
                
                # Check if other architectures need the resources more
                resource_pressure = self._calculate_resource_pressure()
                
                # If high resource pressure and this architecture is underperforming, allow early stop
                if resource_pressure > 0.8 and current_performance < self._get_performance_threshold():
                    self.logger.info(f"Approving early stopping for {architecture} due to resource pressure")
                    return True
                
                # Otherwise defer to early stopping engine
                if self.early_stopping_engine:
                    should_stop, reason, confidence = self.early_stopping_engine.should_stop_architecture(
                        architecture, 'val_accuracy'
                    )
                    
                    if should_stop:
                        # Record pending early stop
                        self.pending_early_stops[architecture] = {
                            'reason': reason,
                            'confidence': confidence,
                            'timestamp': time.time()
                        }
                        
                        self.logger.info(f"Early stopping approved for {architecture}: {reason.value}")
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Early stopping decision failed for {architecture}: {e}")
            return False
    
    def complete_architecture_hpo(self, 
                                 architecture: str, 
                                 final_score: float, 
                                 time_used_hours: float) -> bool:
        """
        Mark architecture as completed (not stopped early) - THREAD SAFE
        
        Args:
            architecture: Architecture name
            final_score: Final performance score
            time_used_hours: Total time used in hours
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._lock:
                if architecture not in self.resource_allocations:
                    self.logger.error(f"Architecture {architecture} not found in allocations")
                    return False
                
                allocation = self.resource_allocations[architecture]
                if allocation is None:
                    self.logger.error(f"Allocation for {architecture} is None")
                    return False
                
                allocation.is_active = False
                allocation.end_time = time.time()
                allocation.performance_score = final_score
                allocation.time_used = time_used_hours
                
                # Add to completed architectures
                self.completed_architectures.add(architecture)
                
                self.logger.info(f"Architecture {architecture} completed HPO:")
                self.logger.info(f"  Final score: {final_score:.4f}")
                self.logger.info(f"  Time used: {time_used_hours:.2f}h / {allocation.allocated_time_hours:.2f}h")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to complete architecture {architecture}: {e}")
            return False

    def architecture_stopped_early(self, 
                                  architecture: str, 
                                  reason: str,
                                  final_performance: float) -> float:
        """
        Handle early stopping with full integration - THREAD SAFE AND ROBUST
        """
        try:
            with self._lock:
                # DEFENSIVE PROGRAMMING
                if architecture not in self.resource_allocations:
                    self.logger.error(f"Architecture {architecture} not found in allocations")
                    return 0.0
                
                allocation = self.resource_allocations.get(architecture)
                if allocation is None:
                    self.logger.error(f"Allocation for {architecture} is None")
                    return 0.0
                
                # Update allocation state
                allocation.is_active = False
                allocation.end_time = time.time()
                allocation.performance_score = final_performance
                
                # Calculate freed time
                freed_time = max(0, allocation.allocated_time_hours - allocation.time_used)
                
                # Add to stopped architectures
                self.stopped_architectures.add(architecture)
                
                # Clear pending early stop if exists
                self.pending_early_stops.pop(architecture, None)
                
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
                
                # Trigger reallocation with coordination
                if freed_time > 0.1:  # Only reallocate significant amounts
                    self._coordinated_reallocation(freed_time, reason, architecture)
                
                return freed_time
                
        except Exception as e:
            self.logger.error(f"Failed to handle early stopping for {architecture}: {e}")
            return 0.0
    
    def _coordinated_reallocation(self, freed_time_hours: float, reason: str, stopped_architecture: str):
        """NEW: Coordinated reallocation with other pipeline components"""
        
        # Check if reallocation should proceed (anti-thrashing)
        if not self._can_reallocate_now():
            self.logger.info(f"Skipping reallocation of {freed_time_hours:.2f}h - cooldown active")
            return
        
        # Get active architectures with error handling
        active_architectures = []
        try:
            for arch, alloc in self.resource_allocations.items():
                if arch == stopped_architecture:
                    continue
                    
                if alloc and alloc.is_active and arch not in self.stopped_architectures:
                    active_architectures.append((arch, alloc))
        except Exception as e:
            self.logger.error(f"Error getting active architectures: {e}")
            return
        
        if not active_architectures:
            self.logger.info(f"No active architectures to reallocate {freed_time_hours:.2f}h to")
            return
        
        # Sort by efficiency (performance per unit time) in descending order
        try:
            active_architectures.sort(key=lambda x: x[1].efficiency_score or 0, reverse=True)
        except Exception as e:
            self.logger.warning(f"Error sorting architectures: {e}")
            # Continue with original order
        
        # Reallocation strategy: Give more time to efficient architectures
        total_efficiency = sum(max(alloc.efficiency_score or 0, 0.1) for _, alloc in active_architectures)
        
        if total_efficiency <= 0:
            # Fallback to equal distribution
            time_per_arch = freed_time_hours / len(active_architectures)
            for arch, alloc in active_architectures:
                try:
                    alloc.allocated_time_hours += time_per_arch
                    self.logger.info(f"Reallocated {time_per_arch:.2f}h to {arch} (equal distribution)")
                except Exception as e:
                    self.logger.error(f"Error reallocating to {arch}: {e}")
        else:
            # Efficiency-based distribution
            for arch, alloc in active_architectures:
                try:
                    efficiency_ratio = max(alloc.efficiency_score or 0, 0.1) / total_efficiency
                    additional_time = freed_time_hours * efficiency_ratio
                    alloc.allocated_time_hours += additional_time
                    
                    self.logger.info(f"Reallocated {additional_time:.2f}h to {arch} (efficiency: {alloc.efficiency_score or 0:.3f})")
                    
                    # NEW: Notify early stopping about resource increase
                    if self.early_stopping_engine:
                        self._notify_early_stopping_resource_increase(arch, additional_time)
                        
                except Exception as e:
                    self.logger.error(f"Error reallocating to {arch}: {e}")
        
        # Record reallocation
        reallocation_event = {
            'timestamp': time.time(),
            'type': 'coordinated_reallocation',
            'freed_time_hours': freed_time_hours,
            'reason': reason,
            'stopped_architecture': stopped_architecture,
            'beneficiaries': [(arch, alloc.efficiency_score or 0) for arch, alloc in active_architectures],
            'distribution_strategy': 'efficiency_based' if total_efficiency > 0 else 'equal'
        }
        self.reallocation_history.append(reallocation_event)
        
        # Update last reallocation time
        self.last_reallocation_time['global'] = time.time()
    
    def _can_reallocate_now(self) -> bool:
        """Check if reallocation is allowed (anti-thrashing protection)"""
        now = time.time()
        
        # Global cooldown
        last_global_realloc = self.last_reallocation_time.get('global', 0)
        if now - last_global_realloc < self.reallocation_cooldown:
            return False
        
        # Rate limiting
        recent_reallocations = [
            r for r in self.reallocation_history 
            if now - r['timestamp'] < 3600  # Last hour
        ]
        
        if len(recent_reallocations) >= self.max_reallocations_per_hour:
            return False
        
        return True
    
    def _calculate_resource_pressure(self) -> float:
        """Calculate current resource pressure (0.0 = low, 1.0 = high)"""
        try:
            elapsed_ratio = self.get_elapsed_time_hours() / max(self.total_time_hours, 1)
            
            # Count active vs total architectures
            active_count = sum(1 for alloc in self.resource_allocations.values() if alloc.is_active)
            total_count = len(self.resource_allocations)
            
            if total_count == 0:
                return 0.0
            
            progress_ratio = (total_count - active_count) / total_count
            
            # High pressure if time is running out but many architectures still active
            pressure = elapsed_ratio * (1 - progress_ratio) * 2  # Scale up pressure
            
            return min(pressure, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating resource pressure: {e}")
            return 0.5  # Default medium pressure
    
    def _get_performance_threshold(self) -> float:
        """Get performance threshold for early stopping decisions"""
        try:
            # Get performance statistics from active architectures
            performances = []
            for alloc in self.resource_allocations.values():
                if alloc.performance_score > 0:
                    performances.append(alloc.performance_score)
            
            if not performances:
                return 0.5  # Default threshold
            
            # Threshold is median performance minus 10%
            median_perf = np.median(performances)
            threshold = median_perf - 0.10
            
            return max(threshold, 0.3)  # Minimum threshold of 30%
            
        except Exception as e:
            self.logger.error(f"Error calculating performance threshold: {e}")
            return 0.5
    
    def _notify_early_stopping_resource_pressure(self, architecture: str, pressure_ratio: float):
        """NEW: Notify early stopping engine about resource pressure"""
        try:
            if self.early_stopping_engine and hasattr(self.early_stopping_engine, 'update_resource_pressure'):
                self.early_stopping_engine.update_resource_pressure(architecture, pressure_ratio)
        except Exception as e:
            self.logger.debug(f"Could not notify early stopping about resource pressure: {e}")
    
    def _notify_early_stopping_resource_increase(self, architecture: str, additional_hours: float):
        """NEW: Notify early stopping engine about resource increase"""
        try:
            if self.early_stopping_engine and hasattr(self.early_stopping_engine, 'update_resource_allocation'):
                self.early_stopping_engine.update_resource_allocation(architecture, additional_hours)
        except Exception as e:
            self.logger.debug(f"Could not notify early stopping about resource increase: {e}")
    
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
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """NEW: Get status of coordination between components"""
        with self._lock:
            return {
                'integration_components': {
                    'early_stopping_engine': self.early_stopping_engine is not None,
                    'hpo_selector': self.hpo_selector is not None
                },
                'pending_early_stops': len(self.pending_early_stops),
                'recent_reallocations': len([
                    r for r in self.reallocation_history 
                    if time.time() - r['timestamp'] < 3600
                ]),
                'resource_pressure': self._calculate_resource_pressure(),
                'coordination_health': 'GOOD' if len(self.pending_early_stops) < 3 else 'STRESSED'
            }
    
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
                },
                'coordination_status': self.get_coordination_status()
            }
            
            return summary
    
    def print_budget_status(self):
        print("DEBUG: print_budget_status called")  # ADD THIS
        """Print human-readable budget status"""
        summary = self.get_budget_summary()

        print("DEBUG: get_budget_summary completed")  # ADD THIS
        
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
        
        # NEW: Coordination status
        coord_status = summary['coordination_status']
        print(f"\nCoordination:")
        print(f"  Health: {coord_status['coordination_health']}")
        print(f"  Resource pressure: {coord_status['resource_pressure']:.2f}")
        print(f"  Pending decisions: {coord_status['pending_early_stops']}")
    
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
            'reallocation_history': self.reallocation_history,
            'coordination_status': self.get_coordination_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Budget manager state saved to {filepath}")

# Test the advanced budget manager
if __name__ == "__main__":
    # Simple test
    config = AutoMLConfig()
    budget_manager = BudgetManager(config)
    
    # Test with sample architectures
    architectures = ['resnet18', 'efficientnet_b0', 'densenet121', 'mobilenetv3_small_100']
    
    print("Testing Advanced Budget Manager...")
    budget_manager.start_execution(architectures)
    budget_manager.print_budget_status()
    
    # Simulate some training progress
    budget_manager.start_architecture_training('resnet18')
    budget_manager.update_architecture_progress('resnet18', 0.85, 1.5, 2.0)
    
    budget_manager.start_architecture_training('efficientnet_b0')
    budget_manager.update_architecture_progress('efficientnet_b0', 0.87, 2.0, 3.0)
    
    # Simulate early stopping with coordination
    freed_time = budget_manager.architecture_stopped_early('densenet121', 'poor_performance', 0.75)
    print(f"\nFreed time from coordinated early stopping: {freed_time:.2f} hours")
    
    # Test coordination status
    coord_status = budget_manager.get_coordination_status()
    print(f"Coordination status: {coord_status}")
    
    budget_manager.print_budget_status()
    
    print("\nAdvanced Budget Manager test completed!")
