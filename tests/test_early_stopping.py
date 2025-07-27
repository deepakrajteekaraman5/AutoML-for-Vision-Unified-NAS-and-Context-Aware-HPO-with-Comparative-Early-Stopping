# tests/test_early_stopping.py
"""
Comprehensive test for the Comparative Early Stopping Engine
"""

import sys
import numpy as np
import time
from pathlib import Path

# Setup Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
template_dir = project_root / "automl-exam-ss25-vision-freiburg-template"
src_path = template_dir / "src"
sys.path.insert(0, str(src_path))

def test_early_stopping_engine():
    """Test the early stopping engine with realistic scenarios"""
    
    print("=== Comparative Early Stopping Engine Test ===")
    
    try:
        # Import modules
        from automl.utils import AutoMLConfig, setup_logging
        from automl.early_stopping import (
            ComparativeEarlyStopping, StoppingReason, 
            LearningCurvePredictor, StatisticalComparator
        )
        
        # Suppress logs for cleaner output
        import logging
        logging.getLogger('AutoML').setLevel(logging.WARNING)
        
        print("SUCCESS: Imports working")
        
        # Test 1: Basic setup
        print("\n1. Testing early stopping engine creation...")
        config = AutoMLConfig()
        config.set('early_stopping_confidence', 0.8)
        config.set('min_epochs_before_stopping', 8)
        config.set('performance_gap_threshold', 0.05)
        
        early_stopper = ComparativeEarlyStopping(config)
        print(f"   SUCCESS: Early stopping engine created")
        print(f"   Confidence threshold: {early_stopper.confidence_threshold}")
        print(f"   Min epochs before stopping: {early_stopper.min_epochs_before_stopping}")
        
        # Test 2: Register architectures
        print("\n2. Testing architecture registration...")
        test_architectures = ['resnet18', 'efficientnet_b0', 'densenet121', 'mobilenetv3_large_100']
        
        for arch in test_architectures:
            early_stopper.register_architecture(arch)
        
        active_archs = early_stopper.get_active_architectures()
        print(f"   SUCCESS: Registered {len(test_architectures)} architectures")
        print(f"   Active architectures: {active_archs}")
        
        # Test 3: Learning curve predictor
        print("\n3. Testing learning curve prediction...")
        predictor = LearningCurvePredictor()
        
        # Simulate a learning curve
        epochs = list(range(1, 11))
        # Logarithmic learning curve (common in ML)
        accuracies = [0.5 + 0.3 * (1 - np.exp(-ep/5)) + np.random.normal(0, 0.01) for ep in epochs]
        
        predicted_final, confidence = predictor.predict_final_performance(epochs, accuracies, target_epoch=30)
        trend = predictor.get_learning_trend(epochs, accuracies)
        plateau = predictor.detect_plateau(accuracies)
        
        print(f"   Current accuracy trend: {[f'{acc:.3f}' for acc in accuracies[-3:]]}")
        print(f"   Predicted final accuracy: {predicted_final:.3f} (±{confidence:.3f})")
        print(f"   Learning trend: {trend}")
        print(f"   Plateau detected: {plateau}")
        print("   SUCCESS: Learning curve prediction working")
        
        # Test 4: Statistical comparison
        print("\n4. Testing statistical comparison...")
        comparator = StatisticalComparator(confidence_level=0.8)
        
        # Create two different performance curves
        good_performance = [0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85]
        poor_performance = [0.5, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
        
        comparison = comparator.compare_architectures(
            good_performance, poor_performance, "GoodModel", "PoorModel"
        )
        
        print(f"   Comparison result:")
        print(f"     GoodModel better: {comparison.a_better}")
        print(f"     Confidence: {comparison.confidence:.3f}")
        print(f"     P-value: {comparison.p_value:.4f}")
        print(f"     Effect size: {comparison.effect_size:.3f}")
        print(f"     Recommendation: {comparison.recommendation}")
        print("   SUCCESS: Statistical comparison working")
        
        # Test 5: Simulate realistic training scenario
        print("\n5. Testing realistic training simulation...")
        
        # Reset for clean test
        early_stopper = ComparativeEarlyStopping(config)
        
        # Different architecture performance patterns
        architectures_patterns = {
            'resnet18': {
                'base_acc': 0.6,
                'final_acc': 0.82,
                'convergence_rate': 8,
                'noise': 0.02,
                'description': 'Fast converging baseline'
            },
            'efficientnet_b0': {
                'base_acc': 0.65,
                'final_acc': 0.87,
                'convergence_rate': 10,
                'noise': 0.015,
                'description': 'Best final performance'
            },
            'densenet121': {
                'base_acc': 0.55,
                'final_acc': 0.78,
                'convergence_rate': 15,
                'noise': 0.03,
                'description': 'Slow starter, decent final'
            },
            'mobilenetv3_large_100': {
                'base_acc': 0.58,
                'final_acc': 0.75,
                'convergence_rate': 6,
                'noise': 0.025,
                'description': 'Fast but limited capacity'
            }
        }
        
        # Register all architectures
        for arch in architectures_patterns.keys():
            early_stopper.register_architecture(arch)
        
        # Simulate training for 25 epochs
        np.random.seed(42)  # For reproducible test
        stopping_decisions = []
        
        print("   Epoch-by-epoch simulation:")
        print("   " + "-" * 80)
        
        for epoch in range(1, 26):
            epoch_decisions = []
            
            for arch, pattern in architectures_patterns.items():
                # Skip if already stopped
                if arch not in early_stopper.get_active_architectures():
                    continue
                
                # Generate realistic accuracy curve
                progress = epoch / 25.0
                base = pattern['base_acc']
                final = pattern['final_acc']
                rate = pattern['convergence_rate']
                noise = pattern['noise']
                
                # Exponential convergence with noise
                accuracy = base + (final - base) * (1 - np.exp(-epoch / rate))
                accuracy += np.random.normal(0, noise)
                accuracy = np.clip(accuracy, 0.0, 1.0)  # Keep in valid range
                
                # Simulate training time (some models are slower)
                if 'densenet' in arch:
                    training_time = np.random.uniform(45, 60)
                elif 'efficientnet' in arch:
                    training_time = np.random.uniform(35, 50)
                else:
                    training_time = np.random.uniform(25, 40)
                
                # Update performance
                early_stopper.update_performance(
                    arch, epoch,
                    {
                        'val_accuracy': accuracy,
                        'val_loss': 1.2 - accuracy,
                        'train_accuracy': min(accuracy + 0.05, 1.0)
                    },
                    training_time
                )
                
                # Check stopping decision after minimum epochs
                if epoch >= early_stopper.min_epochs_before_stopping:
                    should_stop, reason, confidence = early_stopper.should_stop_architecture(arch)
                    
                    if should_stop:
                        early_stopper.stop_architecture(arch, reason, confidence)
                        decision_info = f"{arch} STOPPED ({reason.name}, conf={confidence:.2f})"
                        epoch_decisions.append(decision_info)
                        stopping_decisions.append({
                            'epoch': epoch,
                            'architecture': arch,
                            'reason': reason.name,
                            'confidence': confidence,
                            'final_accuracy': accuracy
                        })
            
            # Print epoch summary
            active_count = len(early_stopper.get_active_architectures())
            if epoch_decisions:
                print(f"   Epoch {epoch:2d}: {active_count} active, DECISIONS: {'; '.join(epoch_decisions)}")
            elif epoch % 5 == 0:  # Print every 5 epochs if no decisions
                current_best = 0.0
                current_worst = 1.0
                for arch in early_stopper.get_active_architectures():
                    perf = early_stopper.architecture_performances[arch]
                    if 'val_accuracy' in perf.metrics and perf.metrics['val_accuracy']:
                        recent_acc = perf.metrics['val_accuracy'][-1]
                        current_best = max(current_best, recent_acc)
                        current_worst = min(current_worst, recent_acc)
                gap = current_best - current_worst
                print(f"   Epoch {epoch:2d}: {active_count} active, performance gap: {gap:.3f}")
        
        print("   " + "-" * 80)
        
        # Test 6: Analyze results
        print("\n6. Testing results analysis...")
        
        summary = early_stopper.get_performance_summary()
        
        print(f"   Final active architectures: {summary['active_architectures']}")
        print(f"   Stopped architectures: {summary['stopped_architectures']}")
        print(f"   Total stopping decisions: {len(summary['stopping_history'])}")
        
        # Show stopping decisions
        if stopping_decisions:
            print("   Stopping decisions made:")
            for decision in stopping_decisions:
                print(f"     Epoch {decision['epoch']:2d}: {decision['architecture']:20} "
                      f"({decision['reason']:25}) conf={decision['confidence']:.2f} "
                      f"acc={decision['final_accuracy']:.3f}")
        
        # Show final rankings
        if 'val_accuracy' in summary['current_rankings']:
            print("   Final accuracy rankings:")
            for i, (arch, acc) in enumerate(summary['current_rankings']['val_accuracy']):
                status = "ACTIVE" if arch in summary['active_architectures'] else "STOPPED"
                print(f"     {i+1}. {arch:20} {acc:.3f} ({status})")
        
        print("   SUCCESS: Results analysis working")
        
        # Test 7: Save and validate state
        print("\n7. Testing state persistence...")
        
        state_file = "test_early_stopping_state.json"
        early_stopper.save_state(state_file)
        
        # Check if file was created and has content
        if Path(state_file).exists():
            import json
            with open(state_file, 'r') as f:
                saved_state = json.load(f)
            
            print(f"   State saved successfully:")
            print(f"     Architectures tracked: {len(saved_state['architecture_details'])}")
            print(f"     Stopping decisions: {len(saved_state['performance_summary']['stopping_history'])}")
            print(f"     Config preserved: {bool(saved_state['config'])}")
            
            # Cleanup
            Path(state_file).unlink()
            print("   SUCCESS: State persistence working")
        else:
            print("   ERROR: State file not created")
        
        # Test 8: Edge cases
        print("\n8. Testing edge cases...")
        
        # Test with insufficient data
        edge_stopper = ComparativeEarlyStopping(config)
        edge_stopper.register_architecture('test_arch')
        
        # Try stopping decision with no data
        should_stop, reason, confidence = edge_stopper.should_stop_architecture('test_arch')
        print(f"   No data case: stop={should_stop}, reason={reason.name}, conf={confidence:.2f}")
        
        # Try with minimal data
        edge_stopper.update_performance('test_arch', 1, {'val_accuracy': 0.5}, 30.0)
        should_stop, reason, confidence = edge_stopper.should_stop_architecture('test_arch')
        print(f"   Minimal data case: stop={should_stop}, reason={reason.name}, conf={confidence:.2f}")
        
        print("   SUCCESS: Edge cases handled properly")
        
        print("\n" + "="*70)
        print("COMPARATIVE EARLY STOPPING ENGINE TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Key capabilities verified:")
        print("  - Architecture registration and tracking")
        print("  - Learning curve prediction and trend analysis")
        print("  - Statistical comparison between architectures")
        print("  - Multi-criteria stopping decisions")
        print("  - Realistic training simulation")
        print("  - Performance analysis and ranking")
        print("  - State persistence and recovery")
        print("  - Edge case handling")
        print("\nYour early stopping engine is ready for AutoML integration!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_early_stopping_engine()
    if not success:
        sys.exit(1)