# src/automl/early_stopping.py
"""
Comparative Early Stopping Engine for AutoML Pipeline - FIXED VERSION
Core innovation: Stop architectures based on cross-architecture performance comparison
FIXES: Conservative thresholds, multiple evidence requirement, anti-thrashing protection
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from collections import defaultdict, deque
import json
import time
import threading

from .utils import AutoMLConfig, MetricTracker, Timer

class StoppingReason(Enum):
    """Enumeration of possible stopping reasons"""
    STATISTICAL_SIGNIFICANCE = "statistically_significantly_worse"
    PERFORMANCE_GAP = "large_performance_gap"
    LEARNING_PLATEAU = "learning_plateau"
    RESOURCE_EFFICIENCY = "poor_resource_efficiency"
    CONVERGENCE_PREDICTION = "predicted_poor_convergence"
    MULTIPLE_EVIDENCE = "multiple_evidence_sources"  # NEW
    USER_REQUESTED = "user_requested"
    TIME_BUDGET = "time_budget_exceeded"

@dataclass
class ArchitecturePerformance:
    """Container for architecture performance data"""
    architecture: str
    epochs: List[int]
    metrics: Dict[str, List[float]]  # metric_name -> [values over epochs]
    timestamps: List[float]
    training_times: List[float]  # Time per epoch
    is_active: bool = True
    stopped_epoch: Optional[int] = None
    stopping_reason: Optional[StoppingReason] = None
    confidence_score: float = 0.0
    # NEW: Decision tracking
    decision_history: List[Dict] = None
    false_positive_indicators: List[str] = None

    def __post_init__(self):
        if self.decision_history is None:
            self.decision_history = []
        if self.false_positive_indicators is None:
            self.false_positive_indicators = []

@dataclass
class ComparisonResult:
    """Result of comparing two architectures"""
    architecture_a: str
    architecture_b: str
    metric: str
    a_better: bool
    confidence: float
    p_value: float
    effect_size: float
    recommendation: str

class LearningCurvePredictor:
    """Predicts future performance from current learning curves"""
    
    def __init__(self, min_points: int = 8):  # INCREASED from 5 to 8
        self.min_points = min_points
        self.logger = logging.getLogger('AutoML.LearningCurvePredictor')
    
    def predict_final_performance(self, 
                                 epochs: List[int], 
                                 values: List[float],
                                 target_epoch: int = 50) -> Tuple[float, float]:
        """
        Predict performance at target_epoch based on current trend
        
        Returns:
            (predicted_value, confidence_interval_width)
        """
        if len(epochs) < self.min_points:
            return values[-1] if values else 0.0, float('inf')
        
        try:
            # Convert to numpy arrays
            X = np.array(epochs).reshape(-1, 1)
            y = np.array(values)
            
            # Fit multiple models and ensemble
            predictions = []
            
            # Linear regression
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            linear_pred = linear_model.predict([[target_epoch]])[0]
            predictions.append(linear_pred)
            
            # Logarithmic model (common in ML training)
            if min(epochs) > 0:
                log_X = np.log(X)
                log_model = LinearRegression()
                log_model.fit(log_X, y)
                log_pred = log_model.predict(np.log([[target_epoch]]))[0]
                predictions.append(log_pred)
            
            # Power law model
            if min(epochs) > 0 and min(values) > 0:
                try:
                    log_epochs = np.log(epochs)
                    log_values = np.log(values)
                    power_model = LinearRegression()
                    power_model.fit(log_epochs.reshape(-1, 1), log_values)
                    log_pred = power_model.predict(np.log([[target_epoch]]))[0]
                    power_pred = np.exp(log_pred)
                    predictions.append(power_pred)
                except:
                    pass  # Skip if power law fails
            
            # Ensemble prediction
            if predictions:
                mean_prediction = np.mean(predictions)
                confidence_width = np.std(predictions) if len(predictions) > 1 else 0.1
            else:
                mean_prediction = values[-1]
                confidence_width = 0.1
            
            return mean_prediction, confidence_width
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}, using last value")
            return values[-1] if values else 0.0, 0.1
    
    def detect_plateau(self, values: List[float], window: int = 5, threshold: float = 0.005) -> bool:  # INCREASED threshold from 0.001 to 0.005
        """Detect if learning has plateaued - MADE MORE CONSERVATIVE"""
        if len(values) < window * 2:  # Require more data
            return False
        
        recent_values = values[-window:]
        
        # Check if improvement is below threshold
        improvement = max(recent_values) - min(recent_values)
        
        # NEW: Also check trend - plateau should have near-zero slope
        if len(recent_values) >= 3:
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # Both conditions must be met
            return improvement < threshold and abs(slope) < threshold
        
        return improvement < threshold
    
    def get_learning_trend(self, epochs: List[int], values: List[float]) -> str:
        """Get overall learning trend: improving, declining, stable"""
        if len(values) < 5:  # INCREASED from 3 to 5
            return "insufficient_data"
        
        # Linear regression on recent points
        recent_window = min(len(values), 10)
        recent_epochs = epochs[-recent_window:]
        recent_values = values[-recent_window:]
        
        if len(recent_epochs) >= 2:
            slope = np.polyfit(recent_epochs, recent_values, 1)[0]
            
            # INCREASED threshold for stability detection
            if abs(slope) < 5e-4:  # Was 1e-4
                return "stable"
            elif slope > 0:
                return "improving"
            else:
                return "declining"
        
        return "stable"

class StatisticalComparator:
    """Performs statistical comparisons between architectures"""
    
    def __init__(self, confidence_level: float = 0.9):  # INCREASED from 0.8 to 0.9
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.logger = logging.getLogger('AutoML.StatisticalComparator')
    
    def compare_architectures(self, 
                            arch_a_values: List[float],
                            arch_b_values: List[float],
                            arch_a_name: str = "A",
                            arch_b_name: str = "B") -> ComparisonResult:
        """
        Compare two architectures statistically - MADE MORE CONSERVATIVE
        
        Returns ComparisonResult with detailed comparison
        """
        
        # Ensure we have enough data - INCREASED minimum samples
        min_samples = min(len(arch_a_values), len(arch_b_values))
        if min_samples < 5:  # INCREASED from 3 to 5
            return ComparisonResult(
                architecture_a=arch_a_name,
                architecture_b=arch_b_name,
                metric="unknown",
                a_better=False,
                confidence=0.0,
                p_value=1.0,
                effect_size=0.0,
                recommendation="insufficient_data"
            )
        
        # Use common length for comparison
        a_vals = np.array(arch_a_values[-min_samples:])
        b_vals = np.array(arch_b_values[-min_samples:])
        
        # Perform paired t-test (since we're comparing at same time points)
        try:
            t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(a_vals) + np.var(b_vals)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(a_vals) - np.mean(b_vals)) / pooled_std
            else:
                effect_size = 0.0
            
            # Determine winner
            a_better = np.mean(a_vals) > np.mean(b_vals)
            confidence = 1.0 - p_value
            
            # Generate recommendation - MADE MORE CONSERVATIVE
            if p_value < self.alpha and abs(effect_size) > 0.5:  # INCREASED from 0.2 to 0.5
                if a_better:
                    recommendation = f"{arch_a_name}_significantly_better"
                else:
                    recommendation = f"{arch_b_name}_significantly_better"
            elif abs(effect_size) > 0.8:  # INCREASED from 0.5 to 0.8
                if a_better:
                    recommendation = f"{arch_a_name}_likely_better"
                else:
                    recommendation = f"{arch_b_name}_likely_better"
            else:
                recommendation = "no_significant_difference"
            
            return ComparisonResult(
                architecture_a=arch_a_name,
                architecture_b=arch_b_name,
                metric="comparison",
                a_better=a_better,
                confidence=confidence,
                p_value=p_value,
                effect_size=effect_size,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.warning(f"Statistical comparison failed: {e}")
            return ComparisonResult(
                architecture_a=arch_a_name,
                architecture_b=arch_b_name,
                metric="unknown",
                a_better=False,
                confidence=0.0,
                p_value=1.0,
                effect_size=0.0,
                recommendation="comparison_failed"
            )
    
    def rank_architectures(self, performances: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        """Rank architectures by their current performance"""
        rankings = []
        
        for arch_name, values in performances.items():
            if values:
                # Use recent average performance - INCREASED window
                recent_window = min(len(values), 8)  # INCREASED from 5 to 8
                recent_performance = np.mean(values[-recent_window:])
                rankings.append((arch_name, recent_performance))
        
        # Sort by performance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

class ComparativeEarlyStopping:
    """
    Main early stopping engine that makes comparative stopping decisions - FIXED VERSION
    
    Core Innovation: Conservative stopping with multiple evidence requirements and anti-thrashing protection
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.ComparativeEarlyStopping')
        
        # Configuration parameters - MADE MORE CONSERVATIVE
        self.confidence_threshold = config.get('early_stopping_confidence', 0.9)  # INCREASED from 0.8
        self.min_epochs_before_stopping = config.get('min_epochs_before_stopping', 15)  # INCREASED from 10
        self.max_epochs_per_architecture = config.get('max_epochs_per_architecture', 50)
        self.performance_gap_threshold = config.get('performance_gap_threshold', 0.08)  # INCREASED from 0.05
        self.patience = config.get('early_stopping_patience', 15)  # INCREASED from 10
        
        # NEW: Anti-thrashing protection
        self.decision_cooldown = 1800  # 30 minutes between decisions for same architecture
        self.last_decision_time = {}
        self.decision_history = []
        self.false_positive_tracking = {}
        
        # Decision weights for multi-criteria stopping - REBALANCED
        self.decision_weights = {
            'statistical_significance': 0.40,  # INCREASED - most reliable
            'performance_gap': 0.30,           # INCREASED - clear indicator
            'learning_trend': 0.20,            # DECREASED - less reliable
            'resource_efficiency': 0.10        # DECREASED - least important
        }
        
        # NEW: Multiple evidence requirements
        self.min_evidence_sources = 3  # Require at least 3 evidence sources
        self.min_decision_confidence = 0.85  # Overall decision confidence threshold
        
        # Components
        self.predictor = LearningCurvePredictor()
        self.comparator = StatisticalComparator(self.confidence_threshold)
        
        # State tracking with thread safety
        self._lock = threading.RLock()
        self.architecture_performances: Dict[str, ArchitecturePerformance] = {}
        self.stopping_history: List[Dict] = []
        self.comparison_cache: Dict[str, ComparisonResult] = {}
        
        self.logger.info(f"ComparativeEarlyStopping initialized (FIXED VERSION)")
        self.logger.info(f"Conservative settings: confidence={self.confidence_threshold}, gap_threshold={self.performance_gap_threshold}")
        self.logger.info(f"Anti-thrashing: cooldown={self.decision_cooldown}s, min_evidence={self.min_evidence_sources}")
    
    def register_architecture(self, architecture_name: str):
        """Register a new architecture for tracking - THREAD SAFE"""
        with self._lock:
            if architecture_name not in self.architecture_performances:
                self.architecture_performances[architecture_name] = ArchitecturePerformance(
                    architecture=architecture_name,
                    epochs=[],
                    metrics=defaultdict(list),
                    timestamps=[],
                    training_times=[]
                )
                self.logger.info(f"Registered architecture: {architecture_name}")
    
    def update_performance(self, 
                         architecture_name: str,
                         epoch: int,
                         metrics: Dict[str, float],
                         training_time: float):
        """Update performance data for an architecture - THREAD SAFE"""
        
        with self._lock:
            if architecture_name not in self.architecture_performances:
                self.register_architecture(architecture_name)
            
            perf = self.architecture_performances[architecture_name]
            
            # Only update if architecture is still active
            if not perf.is_active:
                return
            
            # Add new data
            perf.epochs.append(epoch)
            perf.timestamps.append(time.time())
            perf.training_times.append(training_time)
            
            for metric_name, value in metrics.items():
                perf.metrics[metric_name].append(value)
            
            self.logger.debug(f"Updated {architecture_name} epoch {epoch}: {metrics}")
    
    def should_stop_architecture(self, 
                                architecture_name: str,
                                primary_metric: str = 'val_accuracy') -> Tuple[bool, StoppingReason, float]:
        """
        Main decision function: Should we stop this architecture? - FIXED VERSION
        
        Returns:
            (should_stop, stopping_reason, confidence)
        """
        
        with self._lock:
            if architecture_name not in self.architecture_performances:
                return False, StoppingReason.USER_REQUESTED, 0.0
            
            perf = self.architecture_performances[architecture_name]
            
            # Don't stop if not enough data - INCREASED minimum
            if len(perf.epochs) < self.min_epochs_before_stopping:
                return False, StoppingReason.USER_REQUESTED, 0.0
            
            # Don't stop if already stopped
            if not perf.is_active:
                return True, perf.stopping_reason, perf.confidence_score
            
            # NEW: Anti-thrashing check
            now = time.time()
            last_decision = self.last_decision_time.get(architecture_name, 0)
            if now - last_decision < self.decision_cooldown:
                self.logger.debug(f"Skipping decision for {architecture_name} - cooldown active")
                return False, StoppingReason.USER_REQUESTED, 0.0
            
            # Get current metrics
            if primary_metric not in perf.metrics or not perf.metrics[primary_metric]:
                return False, StoppingReason.USER_REQUESTED, 0.0
            
            current_values = perf.metrics[primary_metric]
            
            # Multi-criteria decision making - ENHANCED
            evidence_sources = []
            decision_scores = {}
            
            # Criterion 1: Statistical significance compared to other architectures
            stat_score, stat_reason, stat_evidence = self._evaluate_statistical_significance(
                architecture_name, primary_metric
            )
            decision_scores['statistical_significance'] = stat_score
            if stat_evidence:
                evidence_sources.append(('statistical_significance', stat_score, stat_reason))
            
            # Criterion 2: Performance gap analysis
            gap_score, gap_reason, gap_evidence = self._evaluate_performance_gap(
                architecture_name, primary_metric
            )
            decision_scores['performance_gap'] = gap_score
            if gap_evidence:
                evidence_sources.append(('performance_gap', gap_score, gap_reason))
            
            # Criterion 3: Learning trend analysis
            trend_score, trend_reason, trend_evidence = self._evaluate_learning_trend(
                architecture_name, primary_metric
            )
            decision_scores['learning_trend'] = trend_score
            if trend_evidence:
                evidence_sources.append(('learning_trend', trend_score, trend_reason))
            
            # Criterion 4: Resource efficiency
            efficiency_score, efficiency_reason, efficiency_evidence = self._evaluate_resource_efficiency(
                architecture_name, primary_metric
            )
            decision_scores['resource_efficiency'] = efficiency_score
            if efficiency_evidence:
                evidence_sources.append(('resource_efficiency', efficiency_score, efficiency_reason))
            
            # NEW: Multiple evidence requirement
            if len(evidence_sources) < self.min_evidence_sources:
                self.logger.debug(f"Insufficient evidence for {architecture_name}: {len(evidence_sources)}/{self.min_evidence_sources}")
                return False, StoppingReason.USER_REQUESTED, 0.0
            
            # Combine scores using weights
            final_score = sum(
                self.decision_weights[criterion] * score
                for criterion, score in decision_scores.items()
            )
            
            # Decision threshold - INCREASED
            stop_threshold = 0.8  # INCREASED from 0.7
            
            # NEW: Require high overall confidence
            overall_confidence = np.mean([score for _, score, _ in evidence_sources])
            
            should_stop = (final_score > stop_threshold and 
                          overall_confidence > self.min_decision_confidence)
            
            # Record decision attempt
            self.last_decision_time[architecture_name] = now
            
            # Determine primary stopping reason
            if should_stop:
                primary_reason_score = max(evidence_sources, key=lambda x: x[1])
                
                if primary_reason_score[0] == 'statistical_significance':
                    reason = StoppingReason.STATISTICAL_SIGNIFICANCE
                elif primary_reason_score[0] == 'performance_gap':
                    reason = StoppingReason.PERFORMANCE_GAP
                elif primary_reason_score[0] == 'learning_trend':
                    reason = StoppingReason.LEARNING_PLATEAU
                else:
                    reason = StoppingReason.RESOURCE_EFFICIENCY
                
                # NEW: Record decision for tracking
                decision_record = {
                    'architecture': architecture_name,
                    'timestamp': now,
                    'evidence_sources': evidence_sources,
                    'final_score': final_score,
                    'overall_confidence': overall_confidence,
                    'decision': 'STOP',
                    'reason': reason.value
                }
                perf.decision_history.append(decision_record)
            else:
                reason = StoppingReason.USER_REQUESTED
            
            # Log decision details - SIMPLIFIED
            if should_stop:
                self.logger.info(f" {architecture_name}: STOPPED ({reason.value}, confidence: {overall_confidence:.2f})")
            else:
                # Only log occasionally to reduce spam
                if len(evidence_sources) >= 2:  # Only when getting close to stopping
                    self.logger.info(f" {architecture_name}: Continuing (score: {final_score:.2f}/{stop_threshold}, evidence: {len(evidence_sources)}/{self.min_evidence_sources})")
            
            return should_stop, reason, overall_confidence if should_stop else 0.0
    
    def _evaluate_statistical_significance(self, architecture_name: str, metric: str) -> Tuple[float, str, bool]:
        """Evaluate based on statistical comparison with other architectures - ENHANCED"""
        
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        
        if len(current_values) < 8:  # INCREASED from 5 to 8
            return 0.0, "insufficient_data", False
        
        # Compare with all other active architectures
        significantly_worse_count = 0
        total_comparisons = 0
        high_confidence_comparisons = 0
        
        for other_arch, other_perf in self.architecture_performances.items():
            if other_arch == architecture_name or not other_perf.is_active:
                continue
            
            if metric in other_perf.metrics and len(other_perf.metrics[metric]) >= 8:  # INCREASED threshold
                comparison = self.comparator.compare_architectures(
                    current_values, other_perf.metrics[metric],
                    architecture_name, other_arch
                )
                
                total_comparisons += 1
                
                # Check if this architecture is significantly worse - MADE MORE CONSERVATIVE
                if (not comparison.a_better and 
                    comparison.confidence > self.confidence_threshold and
                    abs(comparison.effect_size) > 0.5):  # INCREASED from 0.2 to 0.5
                    significantly_worse_count += 1
                    
                    if comparison.confidence > 0.95:  # Very high confidence
                        high_confidence_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0, "no_comparisons_possible", False
        
        # Score based on proportion of significantly worse comparisons
        worse_ratio = significantly_worse_count / total_comparisons
        high_confidence_ratio = high_confidence_comparisons / total_comparisons
        
        # NEW: Require both high worse ratio AND high confidence comparisons
        has_evidence = worse_ratio > 0.6 and high_confidence_ratio > 0.3  # At least 60% worse with 30% high confidence
        
        # High score if significantly worse than most other architectures
        score = worse_ratio * (1 + high_confidence_ratio)  # Boost score for high confidence
        reason = f"worse_than_{significantly_worse_count}_of_{total_comparisons}_hc_{high_confidence_comparisons}"
        
        return score, reason, has_evidence
    
    def _evaluate_performance_gap(self, architecture_name: str, metric: str) -> Tuple[float, str, bool]:
        """Evaluate based on absolute performance gap with leaders - ENHANCED"""
        
        # Get recent performance of this architecture
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        
        if len(current_values) < 5:  # INCREASED from 3
            return 0.0, "insufficient_data", False
        
        recent_performance = np.mean(current_values[-5:])  # INCREASED window
        
        # Find best performing architectures (top 2, not just top 1)
        performance_list = []
        for other_arch, other_perf in self.architecture_performances.items():
            if not other_perf.is_active or other_arch == architecture_name:
                continue
            
            if metric in other_perf.metrics and len(other_perf.metrics[metric]) >= 5:
                other_recent = np.mean(other_perf.metrics[metric][-5:])
                performance_list.append((other_arch, other_recent))
        
        if not performance_list:
            return 0.0, "no_other_architectures", False
        
        # Sort by performance
        performance_list.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate gap from top performers
        top_2_avg = np.mean([perf for _, perf in performance_list[:2]])  # Average of top 2
        performance_gap = top_2_avg - recent_performance
        
        # NEW: Require sustained gap, not just current gap
        if len(current_values) >= 10:
            # Check if gap has been consistent over last few measurements
            gaps = []
            for i in range(-5, 0):  # Last 5 measurements
                if i + len(current_values) >= 5:  # Ensure we have enough data
                    historical_performance = current_values[i]
                    # Recalculate historical top performance (simplified)
                    historical_gap = top_2_avg - historical_performance  # Approximation
                    gaps.append(historical_gap)
            
            if gaps:
                gap_consistency = np.std(gaps) < 0.02  # Low variance in gaps
                avg_gap = np.mean(gaps)
                
                # Use consistent gap if available
                if gap_consistency and len(gaps) >= 3:
                    performance_gap = avg_gap
        
        # Score based on gap size (normalized) - INCREASED threshold
        score = min(performance_gap / self.performance_gap_threshold, 1.0)
        score = max(score, 0.0)  # Don't go negative
        
        # Evidence requires significant and sustained gap
        has_evidence = performance_gap > self.performance_gap_threshold and score > 0.8
        
        best_arch = performance_list[0][0] if performance_list else "unknown"
        reason = f"gap_{performance_gap:.3f}_behind_top2_avg_{top_2_avg:.3f}"
        
        return score, reason, has_evidence
    
    def _evaluate_learning_trend(self, architecture_name: str, metric: str) -> Tuple[float, str, bool]:
        """Evaluate based on learning curve trends - ENHANCED"""
        
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        epochs = perf.epochs
        
        if len(current_values) < 8:  # INCREASED from 5
            return 0.0, "insufficient_data", False
        
        # Check for plateau
        plateau_detected = self.predictor.detect_plateau(current_values)
        
        # Get learning trend
        trend = self.predictor.get_learning_trend(epochs, current_values)
        
        # Predict future performance
        predicted_final, confidence_width = self.predictor.predict_final_performance(
            epochs, current_values, target_epoch=self.max_epochs_per_architecture
        )
        
        # NEW: More sophisticated trend analysis
        score = 0.0
        evidence_factors = []
        reason_parts = [trend]
        
        if plateau_detected:
            score += 0.3
            evidence_factors.append("plateau")
            reason_parts.append("plateau")
        
        if trend == "declining":
            score += 0.4
            evidence_factors.append("declining")
        elif trend == "stable" and len(current_values) > 15:  # Only if we have enough data
            score += 0.2
            evidence_factors.append("stable_long")
        
        # Add prediction uncertainty
        if confidence_width > 0.15:  # INCREASED threshold
            score += 0.2
            evidence_factors.append("uncertain")
            reason_parts.append("uncertain")
        
        # NEW: Check if predicted final performance is poor
        if len(current_values) >= 10:
            current_best = max(current_values)
            predicted_improvement = predicted_final - current_best
            
            if predicted_improvement < 0.02:  # Less than 2% predicted improvement
                score += 0.3
                evidence_factors.append("poor_prediction")
                reason_parts.append(f"pred_improve_{predicted_improvement:.3f}")
        
        # Evidence requires multiple indicators
        has_evidence = len(evidence_factors) >= 2 and score > 0.6
        
        reason = "_".join(reason_parts)
        
        return min(score, 1.0), reason, has_evidence
    
    def _evaluate_resource_efficiency(self, architecture_name: str, metric: str) -> Tuple[float, str, bool]:
        """Evaluate based on resource efficiency (performance per unit time) - ENHANCED"""
        
        perf = self.architecture_performances[architecture_name]
        
        if len(perf.training_times) < 5 or len(perf.metrics[metric]) < 5:  # INCREASED from 3
            return 0.0, "insufficient_data", False
        
        # Calculate performance per unit time
        recent_performance = np.mean(perf.metrics[metric][-5:])  # INCREASED window
        average_time_per_epoch = np.mean(perf.training_times[-5:])
        
        if average_time_per_epoch <= 0:
            return 0.0, "invalid_timing", False
        
        efficiency = recent_performance / average_time_per_epoch
        
        # Compare with other architectures' efficiency
        efficiency_comparisons = []
        for other_arch, other_perf in self.architecture_performances.items():
            if (other_arch != architecture_name and other_perf.is_active and
                len(other_perf.training_times) >= 5 and len(other_perf.metrics[metric]) >= 5):
                
                other_performance = np.mean(other_perf.metrics[metric][-5:])
                other_time = np.mean(other_perf.training_times[-5:])
                
                if other_time > 0:
                    other_efficiency = other_performance / other_time
                    efficiency_comparisons.append((other_arch, other_efficiency))
        
        if not efficiency_comparisons:
            return 0.0, "no_efficiency_comparisons", False
        
        # Find efficiency ranking
        efficiency_comparisons.append((architecture_name, efficiency))
        efficiency_comparisons.sort(key=lambda x: x[1], reverse=True)
        
        # Find rank
        rank = next(i for i, (arch, _) in enumerate(efficiency_comparisons) if arch == architecture_name)
        total_architectures = len(efficiency_comparisons)
        
        # Score based on efficiency rank - poor efficiency gets higher stop score
        efficiency_percentile = rank / total_architectures
        score = efficiency_percentile  # Higher rank (worse efficiency) = higher score
        
        # NEW: Require consistently poor efficiency
        has_evidence = (efficiency_percentile > 0.7 and  # Bottom 30%
                       total_architectures >= 3)  # Need at least 3 architectures to compare
        
        best_efficiency = efficiency_comparisons[0][1]
        reason = f"efficiency_{efficiency:.4f}_rank_{rank+1}_of_{total_architectures}_best_{best_efficiency:.4f}"
        
        return score, reason, has_evidence
    
    def stop_architecture(self, architecture_name: str, reason: StoppingReason, confidence: float):
        """Mark an architecture as stopped - THREAD SAFE"""
        
        with self._lock:
            if architecture_name in self.architecture_performances:
                perf = self.architecture_performances[architecture_name]
                perf.is_active = False
                perf.stopped_epoch = perf.epochs[-1] if perf.epochs else 0
                perf.stopping_reason = reason
                perf.confidence_score = confidence
                
                # Record stopping decision
                stopping_record = {
                    'timestamp': time.time(),
                    'architecture': architecture_name,
                    'epoch': perf.stopped_epoch,
                    'reason': reason.value,
                    'confidence': confidence,
                    'final_metrics': {k: v[-1] if v else 0.0 for k, v in perf.metrics.items()},
                    'decision_history': perf.decision_history.copy()  # NEW: Include decision history
                }
                self.stopping_history.append(stopping_record)
                
                # NEW: Start tracking for potential false positive
                self.false_positive_tracking[architecture_name] = {
                    'stopped_at': time.time(),
                    'final_performance': perf.metrics.get('val_accuracy', [0.0])[-1] if perf.metrics.get('val_accuracy') else 0.0,
                    'confidence': confidence,
                    'reason': reason.value
                }
                
                self.logger.info(f"STOPPED {architecture_name} at epoch {perf.stopped_epoch}")
                self.logger.info(f"  Reason: {reason.value}")
                self.logger.info(f"  Confidence: {confidence:.3f}")
                self.logger.info(f"  Decision history: {len(perf.decision_history)} evaluations")
    
    def get_active_architectures(self) -> List[str]:
        """Get list of architectures that are still active - THREAD SAFE"""
        with self._lock:
            return [
                arch for arch, perf in self.architecture_performances.items()
                if perf.is_active
            ]
    
    def get_stopped_architectures(self) -> List[str]:
        """Get list of architectures that have been stopped - THREAD SAFE"""
        with self._lock:
            return [
                arch for arch, perf in self.architecture_performances.items()
                if not perf.is_active
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary - THREAD SAFE"""
        
        with self._lock:
            summary = {
                'active_architectures': self.get_active_architectures(),
                'stopped_architectures': self.get_stopped_architectures(),
                'stopping_history': self.stopping_history,
                'current_rankings': {},
                'total_comparisons': len(self.comparison_cache),
                # NEW: Enhanced tracking
                'decision_stats': {
                    'total_decisions': len(self.decision_history),
                    'cooldown_blocks': sum(1 for arch in self.last_decision_time.keys() 
                                         if time.time() - self.last_decision_time[arch] < self.decision_cooldown),
                    'false_positive_tracking': len(self.false_positive_tracking)
                },
                'configuration': {
                    'performance_gap_threshold': self.performance_gap_threshold,
                    'min_epochs_before_stopping': self.min_epochs_before_stopping,
                    'min_evidence_sources': self.min_evidence_sources,
                    'min_decision_confidence': self.min_decision_confidence,
                    'decision_cooldown': self.decision_cooldown
                }
            }
            
            # Add current rankings for each metric
            all_metrics = set()
            for perf in self.architecture_performances.values():
                all_metrics.update(perf.metrics.keys())
            
            for metric in all_metrics:
                metric_performances = {}
                for arch, perf in self.architecture_performances.items():
                    if metric in perf.metrics and perf.metrics[metric]:
                        recent_performance = np.mean(perf.metrics[metric][-5:])  # INCREASED window
                        metric_performances[arch] = recent_performance
                
                rankings = self.comparator.rank_architectures(
                    {arch: [perf] for arch, perf in metric_performances.items()}
                )
                summary['current_rankings'][metric] = rankings
            
            return summary
    
    def validate_stopping_decisions(self) -> Dict[str, Any]:
        """NEW: Validate past stopping decisions to detect false positives"""
        
        with self._lock:
            validation_results = {
                'total_stopped': len(self.false_positive_tracking),
                'potential_false_positives': [],
                'validation_summary': {}
            }
            
            current_time = time.time()
            
            for arch_name, tracking_info in self.false_positive_tracking.items():
                time_since_stop = current_time - tracking_info['stopped_at']
                
                # Only validate if enough time has passed (at least 1 hour)
                if time_since_stop > 3600:
                    # Check if other architectures significantly outperformed stopped architecture
                    stopped_performance = tracking_info['final_performance']
                    
                    current_best = 0.0
                    for other_arch, other_perf in self.architecture_performances.items():
                        if other_arch != arch_name and other_perf.metrics.get('val_accuracy'):
                            other_best = max(other_perf.metrics['val_accuracy'])
                            current_best = max(current_best, other_best)
                    
                    # If stopped architecture's final performance was within 5% of current best,
                    # it might have been a false positive
                    performance_gap = current_best - stopped_performance
                    
                    if performance_gap < 0.05:  # Within 5%
                        validation_results['potential_false_positives'].append({
                            'architecture': arch_name,
                            'stopped_performance': stopped_performance,
                            'current_best': current_best,
                            'gap': performance_gap,
                            'confidence_used': tracking_info['confidence'],
                            'reason': tracking_info['reason'],
                            'time_since_stop_hours': time_since_stop / 3600
                        })
            
            # Summary statistics
            total_fps = len(validation_results['potential_false_positives'])
            total_stopped = validation_results['total_stopped']
            
            validation_results['validation_summary'] = {
                'false_positive_rate': total_fps / total_stopped if total_stopped > 0 else 0.0,
                'total_validated': total_stopped,
                'potential_false_positives_count': total_fps,
                'validation_status': 'GOOD' if total_fps / max(total_stopped, 1) < 0.2 else 'CONCERNING'
            }
            
            return validation_results
    
    def update_resource_pressure(self, architecture: str, pressure_ratio: float):
        """NEW: Update resource pressure information from budget manager"""
        with self._lock:
            if architecture in self.architecture_performances:
                perf = self.architecture_performances[architecture]
                # Add resource pressure to decision context
                if not hasattr(perf, 'resource_pressure_history'):
                    perf.resource_pressure_history = []
                perf.resource_pressure_history.append({
                    'timestamp': time.time(),
                    'pressure_ratio': pressure_ratio
                })
                self.logger.debug(f"Updated resource pressure for {architecture}: {pressure_ratio:.2f}")
    
    def update_resource_allocation(self, architecture: str, additional_hours: float):
        """NEW: Update resource allocation information from budget manager"""
        with self._lock:
            if architecture in self.architecture_performances:
                perf = self.architecture_performances[architecture]
                # Add resource allocation to decision context
                if not hasattr(perf, 'resource_allocation_history'):
                    perf.resource_allocation_history = []
                perf.resource_allocation_history.append({
                    'timestamp': time.time(),
                    'additional_hours': additional_hours
                })
                self.logger.debug(f"Updated resource allocation for {architecture}: +{additional_hours:.2f}h")

    def save_state(self, filepath: str):
        """Save early stopping state for analysis - THREAD SAFE"""
        
        with self._lock:
            state = {
                'config': {
                    'confidence_threshold': self.confidence_threshold,
                    'min_epochs_before_stopping': self.min_epochs_before_stopping,
                    'performance_gap_threshold': self.performance_gap_threshold,
                    'min_evidence_sources': self.min_evidence_sources,
                    'min_decision_confidence': self.min_decision_confidence,
                    'decision_cooldown': self.decision_cooldown,
                    'decision_weights': self.decision_weights
                },
                'performance_summary': self.get_performance_summary(),
                'validation_results': self.validate_stopping_decisions(),  # NEW
                'architecture_details': {}
            }
            
            # Add detailed architecture performance data
            for arch, perf in self.architecture_performances.items():
                state['architecture_details'][arch] = {
                    'epochs': perf.epochs,
                    'metrics': dict(perf.metrics),
                    'training_times': perf.training_times,
                    'is_active': perf.is_active,
                    'stopped_epoch': perf.stopped_epoch,
                    'stopping_reason': perf.stopping_reason.value if perf.stopping_reason else None,
                    'confidence_score': perf.confidence_score,
                    'decision_history': perf.decision_history,  # NEW
                    'false_positive_indicators': perf.false_positive_indicators,  # NEW
                    # NEW: Coordination data
                    'resource_pressure_history': getattr(perf, 'resource_pressure_history', []),
                    'resource_allocation_history': getattr(perf, 'resource_allocation_history', [])
                }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Early stopping state saved to {filepath}")

# Test the fixed early stopping engine
if __name__ == "__main__":
    # Simple test with dummy data
    config = AutoMLConfig()
    early_stopper = ComparativeEarlyStopping(config)
    
    # Simulate training data with more conservative patterns
    architectures = ['resnet18', 'efficientnet_b0', 'densenet121']
    
    for arch in architectures:
        early_stopper.register_architecture(arch)
    
    # Simulate 30 epochs of training (increased from 20)
    np.random.seed(42)
    for epoch in range(1, 31):
        for arch in architectures:
            # Simulate different learning patterns with more realistic curves
            if arch == 'resnet18':
                # Good architecture - steady improvement
                accuracy = 0.65 + 0.25 * (1 - np.exp(-epoch/12)) + np.random.normal(0, 0.015)
            elif arch == 'efficientnet_b0':
                # Best architecture - fast initial improvement then plateau
                accuracy = 0.75 + 0.2 * (1 - np.exp(-epoch/8)) + np.random.normal(0, 0.01)
            else:  # densenet121
                # Poor architecture - slow improvement with plateau
                accuracy = 0.55 + 0.12 * (1 - np.exp(-epoch/20)) + np.random.normal(0, 0.025)
            
            early_stopper.update_performance(
                arch, epoch, 
                {'val_accuracy': accuracy, 'val_loss': 1.2 - accuracy},
                training_time=np.random.uniform(45, 90)
            )
            
            # Check stopping decision (only after minimum epochs)
            if epoch >= early_stopper.min_epochs_before_stopping:
                should_stop, reason, confidence = early_stopper.should_stop_architecture(arch)
                if should_stop:
                    early_stopper.stop_architecture(arch, reason, confidence)
    
    # Print summary
    summary = early_stopper.get_performance_summary()
    validation = early_stopper.validate_stopping_decisions()
    
    print("=== Fixed Early Stopping Test Results ===")
    print(f"Active architectures: {summary['active_architectures']}")
    print(f"Stopped architectures: {summary['stopped_architectures']}")
    print(f"Stopping decisions: {len(summary['stopping_history'])}")
    print(f"False positive rate: {validation['validation_summary']['false_positive_rate']:.2%}")
    print(f"Validation status: {validation['validation_summary']['validation_status']}")
    
    for decision in summary['stopping_history']:
        print(f"  {decision['architecture']} stopped at epoch {decision['epoch']}: {decision['reason']} (confidence: {decision['confidence']:.3f})")
    
    print("\n Fixed Early Stopping Engine test completed!")
