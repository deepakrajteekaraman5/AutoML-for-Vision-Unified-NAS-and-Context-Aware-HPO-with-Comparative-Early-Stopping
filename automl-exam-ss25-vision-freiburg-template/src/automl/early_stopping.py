# src/automl/early_stopping.py
"""
Comparative Early Stopping Engine for AutoML Pipeline
Core innovation: Stop architectures based on cross-architecture performance comparison
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

from .utils import AutoMLConfig, MetricTracker, Timer

class StoppingReason(Enum):
    """Enumeration of possible stopping reasons"""
    STATISTICAL_SIGNIFICANCE = "statistically_significantly_worse"
    PERFORMANCE_GAP = "large_performance_gap"
    LEARNING_PLATEAU = "learning_plateau"
    RESOURCE_EFFICIENCY = "poor_resource_efficiency"
    CONVERGENCE_PREDICTION = "predicted_poor_convergence"
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
    
    def __init__(self, min_points: int = 5):
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
    
    def detect_plateau(self, values: List[float], window: int = 5, threshold: float = 0.001) -> bool:
        """Detect if learning has plateaued"""
        if len(values) < window:
            return False
        
        recent_values = values[-window:]
        
        # Check if improvement is below threshold
        improvement = max(recent_values) - min(recent_values)
        return improvement < threshold
    
    def get_learning_trend(self, epochs: List[int], values: List[float]) -> str:
        """Get overall learning trend: improving, declining, stable"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Linear regression on recent points
        recent_window = min(len(values), 10)
        recent_epochs = epochs[-recent_window:]
        recent_values = values[-recent_window:]
        
        if len(recent_epochs) >= 2:
            slope = np.polyfit(recent_epochs, recent_values, 1)[0]
            
            if abs(slope) < 1e-4:
                return "stable"
            elif slope > 0:
                return "improving"
            else:
                return "declining"
        
        return "stable"

class StatisticalComparator:
    """Performs statistical comparisons between architectures"""
    
    def __init__(self, confidence_level: float = 0.8):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.logger = logging.getLogger('AutoML.StatisticalComparator')
    
    def compare_architectures(self, 
                            arch_a_values: List[float],
                            arch_b_values: List[float],
                            arch_a_name: str = "A",
                            arch_b_name: str = "B") -> ComparisonResult:
        """
        Compare two architectures statistically
        
        Returns ComparisonResult with detailed comparison
        """
        
        # Ensure we have enough data
        min_samples = min(len(arch_a_values), len(arch_b_values))
        if min_samples < 3:
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
            
            # Generate recommendation
            if p_value < self.alpha and abs(effect_size) > 0.2:
                if a_better:
                    recommendation = f"{arch_a_name}_significantly_better"
                else:
                    recommendation = f"{arch_b_name}_significantly_better"
            elif abs(effect_size) > 0.5:
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
                # Use recent average performance
                recent_window = min(len(values), 5)
                recent_performance = np.mean(values[-recent_window:])
                rankings.append((arch_name, recent_performance))
        
        # Sort by performance (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

class ComparativeEarlyStopping:
    """
    Main early stopping engine that makes comparative stopping decisions
    
    Core Innovation: Stops architectures based on comparison with other architectures,
    not just individual performance plateauing.
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.ComparativeEarlyStopping')
        
        # Configuration parameters
        self.confidence_threshold = config.get('early_stopping_confidence', 0.8)
        self.min_epochs_before_stopping = config.get('min_epochs_before_stopping', 10)
        self.max_epochs_per_architecture = config.get('max_epochs_per_architecture', 50)
        self.performance_gap_threshold = config.get('performance_gap_threshold', 0.05)
        self.patience = config.get('early_stopping_patience', 10)
        
        # Decision weights for multi-criteria stopping
        self.decision_weights = {
            'statistical_significance': 0.35,
            'performance_gap': 0.25,
            'learning_trend': 0.25,
            'resource_efficiency': 0.15
        }
        
        # Components
        self.predictor = LearningCurvePredictor()
        self.comparator = StatisticalComparator(self.confidence_threshold)
        
        # State tracking
        self.architecture_performances: Dict[str, ArchitecturePerformance] = {}
        self.stopping_history: List[Dict] = []
        self.comparison_cache: Dict[str, ComparisonResult] = {}
        
        self.logger.info(f"ComparativeEarlyStopping initialized with confidence={self.confidence_threshold}")
        self.logger.info(f"Decision weights: {self.decision_weights}")
    
    def register_architecture(self, architecture_name: str):
        """Register a new architecture for tracking"""
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
        """Update performance data for an architecture"""
        
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
        Main decision function: Should we stop this architecture?
        
        Returns:
            (should_stop, stopping_reason, confidence)
        """
        
        if architecture_name not in self.architecture_performances:
            return False, StoppingReason.USER_REQUESTED, 0.0
        
        perf = self.architecture_performances[architecture_name]
        
        # Don't stop if not enough data
        if len(perf.epochs) < self.min_epochs_before_stopping:
            return False, StoppingReason.USER_REQUESTED, 0.0
        
        # Don't stop if already stopped
        if not perf.is_active:
            return True, perf.stopping_reason, perf.confidence_score
        
        # Get current metrics
        if primary_metric not in perf.metrics or not perf.metrics[primary_metric]:
            return False, StoppingReason.USER_REQUESTED, 0.0
        
        current_values = perf.metrics[primary_metric]
        
        # Multi-criteria decision making
        decision_scores = {}
        
        # Criterion 1: Statistical significance compared to other architectures
        stat_score, stat_reason = self._evaluate_statistical_significance(
            architecture_name, primary_metric
        )
        decision_scores['statistical_significance'] = stat_score
        
        # Criterion 2: Performance gap analysis
        gap_score, gap_reason = self._evaluate_performance_gap(
            architecture_name, primary_metric
        )
        decision_scores['performance_gap'] = gap_score
        
        # Criterion 3: Learning trend analysis
        trend_score, trend_reason = self._evaluate_learning_trend(
            architecture_name, primary_metric
        )
        decision_scores['learning_trend'] = trend_score
        
        # Criterion 4: Resource efficiency
        efficiency_score, efficiency_reason = self._evaluate_resource_efficiency(
            architecture_name, primary_metric
        )
        decision_scores['resource_efficiency'] = efficiency_score
        
        # Combine scores using weights
        final_score = sum(
            self.decision_weights[criterion] * score
            for criterion, score in decision_scores.items()
        )
        
        # Decision threshold
        stop_threshold = 0.7  # If combined score > 0.7, stop the architecture
        
        should_stop = final_score > stop_threshold
        
        # Determine primary stopping reason
        if should_stop:
            primary_reason_score = max(decision_scores.items(), key=lambda x: x[1])
            
            if primary_reason_score[0] == 'statistical_significance':
                reason = StoppingReason.STATISTICAL_SIGNIFICANCE
            elif primary_reason_score[0] == 'performance_gap':
                reason = StoppingReason.PERFORMANCE_GAP
            elif primary_reason_score[0] == 'learning_trend':
                reason = StoppingReason.LEARNING_PLATEAU
            else:
                reason = StoppingReason.RESOURCE_EFFICIENCY
        else:
            reason = StoppingReason.USER_REQUESTED
        
        # Log decision details
        self.logger.info(f"Early stopping evaluation for {architecture_name}:")
        self.logger.info(f"  Decision scores: {decision_scores}")
        self.logger.info(f"  Final score: {final_score:.3f}")
        self.logger.info(f"  Decision: {'STOP' if should_stop else 'CONTINUE'}")
        if should_stop:
            self.logger.info(f"  Primary reason: {reason}")
        
        return should_stop, reason, final_score
    
    def _evaluate_statistical_significance(self, architecture_name: str, metric: str) -> Tuple[float, str]:
        """Evaluate based on statistical comparison with other architectures"""
        
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        
        if len(current_values) < 5:
            return 0.0, "insufficient_data"
        
        # Compare with all other active architectures
        significantly_worse_count = 0
        total_comparisons = 0
        
        for other_arch, other_perf in self.architecture_performances.items():
            if other_arch == architecture_name or not other_perf.is_active:
                continue
            
            if metric in other_perf.metrics and len(other_perf.metrics[metric]) >= 5:
                comparison = self.comparator.compare_architectures(
                    current_values, other_perf.metrics[metric],
                    architecture_name, other_arch
                )
                
                total_comparisons += 1
                
                # Check if this architecture is significantly worse
                if (not comparison.a_better and 
                    comparison.confidence > self.confidence_threshold and
                    abs(comparison.effect_size) > 0.2):
                    significantly_worse_count += 1
        
        if total_comparisons == 0:
            return 0.0, "no_comparisons_possible"
        
        # Score based on proportion of significantly worse comparisons
        worse_ratio = significantly_worse_count / total_comparisons
        
        # High score if significantly worse than most other architectures
        score = worse_ratio
        reason = f"worse_than_{significantly_worse_count}_of_{total_comparisons}"
        
        return score, reason
    
    def _evaluate_performance_gap(self, architecture_name: str, metric: str) -> Tuple[float, str]:
        """Evaluate based on absolute performance gap with leaders"""
        
        # Get recent performance of this architecture
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        
        if len(current_values) < 3:
            return 0.0, "insufficient_data"
        
        recent_performance = np.mean(current_values[-3:])
        
        # Find best performing architecture
        best_performance = recent_performance
        best_arch = architecture_name
        
        for other_arch, other_perf in self.architecture_performances.items():
            if not other_perf.is_active or other_arch == architecture_name:
                continue
            
            if metric in other_perf.metrics and len(other_perf.metrics[metric]) >= 3:
                other_recent = np.mean(other_perf.metrics[metric][-3:])
                if other_recent > best_performance:
                    best_performance = other_recent
                    best_arch = other_arch
        
        # Calculate performance gap
        performance_gap = best_performance - recent_performance
        
        # Score based on gap size (normalized)
        score = min(performance_gap / self.performance_gap_threshold, 1.0)
        score = max(score, 0.0)  # Don't go negative
        
        reason = f"gap_{performance_gap:.3f}_behind_{best_arch}"
        
        return score, reason
    
    def _evaluate_learning_trend(self, architecture_name: str, metric: str) -> Tuple[float, str]:
        """Evaluate based on learning curve trends"""
        
        perf = self.architecture_performances[architecture_name]
        current_values = perf.metrics[metric]
        epochs = perf.epochs
        
        if len(current_values) < 5:
            return 0.0, "insufficient_data"
        
        # Check for plateau
        plateau_detected = self.predictor.detect_plateau(current_values)
        
        # Get learning trend
        trend = self.predictor.get_learning_trend(epochs, current_values)
        
        # Predict future performance
        predicted_final, confidence_width = self.predictor.predict_final_performance(
            epochs, current_values, target_epoch=self.max_epochs_per_architecture
        )
        
        # Score based on trend and plateau
        score = 0.0
        reason = trend
        
        if plateau_detected:
            score += 0.4
            reason += "_plateau"
        
        if trend == "declining":
            score += 0.5
        elif trend == "stable":
            score += 0.3
        
        # Add prediction uncertainty
        if confidence_width > 0.1:  # High uncertainty
            score += 0.2
            reason += "_uncertain"
        
        return min(score, 1.0), reason
    
    def _evaluate_resource_efficiency(self, architecture_name: str, metric: str) -> Tuple[float, str]:
        """Evaluate based on resource efficiency (performance per unit time)"""
        
        perf = self.architecture_performances[architecture_name]
        
        if len(perf.training_times) < 3 or len(perf.metrics[metric]) < 3:
            return 0.0, "insufficient_data"
        
        # Calculate performance per unit time
        recent_performance = np.mean(perf.metrics[metric][-3:])
        average_time_per_epoch = np.mean(perf.training_times[-3:])
        
        if average_time_per_epoch <= 0:
            return 0.0, "invalid_timing"
        
        efficiency = recent_performance / average_time_per_epoch
        
        # Compare with other architectures' efficiency
        best_efficiency = efficiency
        for other_arch, other_perf in self.architecture_performances.items():
            if (other_arch != architecture_name and other_perf.is_active and
                len(other_perf.training_times) >= 3 and len(other_perf.metrics[metric]) >= 3):
                
                other_performance = np.mean(other_perf.metrics[metric][-3:])
                other_time = np.mean(other_perf.training_times[-3:])
                
                if other_time > 0:
                    other_efficiency = other_performance / other_time
                    best_efficiency = max(best_efficiency, other_efficiency)
        
        # Score based on efficiency gap
        efficiency_ratio = efficiency / best_efficiency if best_efficiency > 0 else 1.0
        score = 1.0 - efficiency_ratio  # Higher score for lower efficiency
        
        reason = f"efficiency_{efficiency:.4f}_vs_best_{best_efficiency:.4f}"
        
        return max(score, 0.0), reason
    
    def stop_architecture(self, architecture_name: str, reason: StoppingReason, confidence: float):
        """Mark an architecture as stopped"""
        
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
                'final_metrics': {k: v[-1] if v else 0.0 for k, v in perf.metrics.items()}
            }
            self.stopping_history.append(stopping_record)
            
            self.logger.info(f"STOPPED {architecture_name} at epoch {perf.stopped_epoch}")
            self.logger.info(f"  Reason: {reason.value}")
            self.logger.info(f"  Confidence: {confidence:.3f}")
    
    def get_active_architectures(self) -> List[str]:
        """Get list of architectures that are still active"""
        return [
            arch for arch, perf in self.architecture_performances.items()
            if perf.is_active
        ]
    
    def get_stopped_architectures(self) -> List[str]:
        """Get list of architectures that have been stopped"""
        return [
            arch for arch, perf in self.architecture_performances.items()
            if not perf.is_active
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        summary = {
            'active_architectures': self.get_active_architectures(),
            'stopped_architectures': self.get_stopped_architectures(),
            'stopping_history': self.stopping_history,
            'current_rankings': {},
            'total_comparisons': len(self.comparison_cache)
        }
        
        # Add current rankings for each metric
        all_metrics = set()
        for perf in self.architecture_performances.values():
            all_metrics.update(perf.metrics.keys())
        
        for metric in all_metrics:
            metric_performances = {}
            for arch, perf in self.architecture_performances.items():
                if metric in perf.metrics and perf.metrics[metric]:
                    recent_performance = np.mean(perf.metrics[metric][-3:])
                    metric_performances[arch] = recent_performance
            
            rankings = self.comparator.rank_architectures(
                {arch: [perf] for arch, perf in metric_performances.items()}
            )
            summary['current_rankings'][metric] = rankings
        
        return summary
    
    def save_state(self, filepath: str):
        """Save early stopping state for analysis"""
        state = {
            'config': {
                'confidence_threshold': self.confidence_threshold,
                'min_epochs_before_stopping': self.min_epochs_before_stopping,
                'decision_weights': self.decision_weights
            },
            'performance_summary': self.get_performance_summary(),
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
                'confidence_score': perf.confidence_score
            }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Early stopping state saved to {filepath}")

# Test the early stopping engine
if __name__ == "__main__":
    # Simple test with dummy data
    config = AutoMLConfig()
    early_stopper = ComparativeEarlyStopping(config)
    
    # Simulate training data
    architectures = ['resnet18', 'efficientnet_b0', 'densenet121']
    
    for arch in architectures:
        early_stopper.register_architecture(arch)
    
    # Simulate 20 epochs of training
    np.random.seed(42)
    for epoch in range(1, 21):
        for arch in architectures:
            # Simulate different learning patterns
            if arch == 'resnet18':
                accuracy = 0.6 + 0.2 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.02)
            elif arch == 'efficientnet_b0':
                accuracy = 0.7 + 0.25 * (1 - np.exp(-epoch/8)) + np.random.normal(0, 0.015)
            else:  # densenet121
                accuracy = 0.5 + 0.15 * (1 - np.exp(-epoch/15)) + np.random.normal(0, 0.03)
            
            early_stopper.update_performance(
                arch, epoch, 
                {'val_accuracy': accuracy, 'val_loss': 1.0 - accuracy},
                training_time=np.random.uniform(30, 60)
            )
            
            # Check stopping decision
            should_stop, reason, confidence = early_stopper.should_stop_architecture(arch)
            if should_stop:
                early_stopper.stop_architecture(arch, reason, confidence)
    
    # Print summary
    summary = early_stopper.get_performance_summary()
    print("=== Early Stopping Test Results ===")
    print(f"Active architectures: {summary['active_architectures']}")
    print(f"Stopped architectures: {summary['stopped_architectures']}")
    print(f"Stopping decisions: {len(summary['stopping_history'])}")
    
    for decision in summary['stopping_history']:
        print(f"  {decision['architecture']} stopped at epoch {decision['epoch']}: {decision['reason']}")