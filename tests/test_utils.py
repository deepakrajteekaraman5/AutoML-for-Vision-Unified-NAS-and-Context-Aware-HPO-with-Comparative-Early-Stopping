# tests/simple_test_utils.py
"""
Simple test for utils.py - no fancy imports
"""

import sys
import time
import numpy as np
from pathlib import Path

# Setup Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
template_dir = project_root / "automl-exam-ss25-vision-freiburg-template"
src_path = template_dir / "src"

print("=== AutoML Utils Test ===")
print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Template directory: {template_dir}")
print(f"Source path: {src_path}")
print(f"Source exists: {src_path.exists()}")

if not src_path.exists():
    print("❌ Source directory not found!")
    exit(1)

# Add to Python path
sys.path.insert(0, str(src_path))
print(f"Added to Python path: {src_path}")

# Test imports
print("\n1. Testing imports...")
try:
    from automl.utils import (
        AutoMLConfig, setup_logging, set_seed, Timer, 
        MetricTracker, get_device
    )
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 1: Configuration
print("\n2. Testing Configuration...")
try:
    config = AutoMLConfig()
    print(f"   - Time budget: {config.get('time_budget_hours')} hours")
    print(f"   - Dataset: {config.get('dataset_name')}")
    print(f"   - Image size: {config.get('image_size')}x{config.get('image_size')}")
    print(f"   - Classes: {config.get('num_classes')}")
    print("✅ Configuration test passed!")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")

# Test 2: Logging
print("\n3. Testing Logging...")
try:
    logger = setup_logging(level='INFO')
    logger.info("Test log message from simple test")
    print("✅ Logging test passed!")
except Exception as e:
    print(f"❌ Logging test failed: {e}")

# Test 3: Reproducibility
print("\n4. Testing Reproducibility...")
try:
    set_seed(42)
    val1 = np.random.random()
    set_seed(42)
    val2 = np.random.random()
    print(f"   - First value: {val1}")
    print(f"   - Second value: {val2}")
    print(f"   - Values match: {'✅' if val1 == val2 else '❌'}")
    if val1 == val2:
        print("✅ Reproducibility test passed!")
    else:
        print("❌ Reproducibility test failed!")
except Exception as e:
    print(f"❌ Reproducibility test failed: {e}")

# Test 4: Timer
print("\n5. Testing Timer...")
try:
    with Timer("Simple test operation") as timer:
        time.sleep(0.1)
    print(f"   - Elapsed time: {timer.elapsed:.3f} seconds")
    if 0.08 <= timer.elapsed <= 0.15:
        print("✅ Timer test passed!")
    else:
        print("❌ Timer test failed!")
except Exception as e:
    print(f"❌ Timer test failed: {e}")

# Test 5: MetricTracker
print("\n6. Testing MetricTracker...")
try:
    tracker = MetricTracker()
    tracker.update(accuracy=0.85, loss=0.3)
    tracker.update(accuracy=0.87, loss=0.25)
    
    latest_acc = tracker.get_latest('accuracy')
    best_acc = tracker.get_best('accuracy')
    
    print(f"   - Latest accuracy: {latest_acc}")
    print(f"   - Best accuracy: {best_acc}")
    
    if latest_acc == 0.87 and best_acc == 0.87:
        print("✅ MetricTracker test passed!")
    else:
        print("❌ MetricTracker test failed!")
except Exception as e:
    print(f"❌ MetricTracker test failed: {e}")

# Test 6: Device Detection
print("\n7. Testing Device Detection...")
try:
    device = get_device()
    print(f"   - Detected device: {device}")
    if device.type in ['cpu', 'cuda', 'mps']:
        print("✅ Device detection test passed!")
    else:
        print("❌ Device detection test failed!")
except Exception as e:
    print(f"❌ Device detection test failed: {e}")

print("\n" + "=" * 50)
print("🎉 Simple Utils Test Complete!")
print("If you see mostly ✅ symbols above, utils.py is working correctly!")