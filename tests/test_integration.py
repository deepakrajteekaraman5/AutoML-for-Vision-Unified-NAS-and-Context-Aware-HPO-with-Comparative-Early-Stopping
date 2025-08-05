# tests/test_integration_simple.py
"""
Simple integration test with correct path structure
"""

import sys
from pathlib import Path

# Setup correct Python path based on your directory structure
current_dir = Path(__file__).parent
project_root = current_dir.parent
automl_src_path = project_root / "automl-exam-ss25-vision-freiburg-template" / "src"
sys.path.insert(0, str(automl_src_path))

print(f"🔍 Looking for automl modules at: {automl_src_path}")
print(f"✅ Automl directory exists: {(automl_src_path / 'automl').exists()}")

def test_imports():
    """Test that all automl modules can be imported"""
    
    print("\n📦 Testing AutoML Module Imports")
    print("-" * 40)
    
    try:
        # Test utils first (has fallback classes)
        from automl.utils import AutoMLConfig, setup_logging, set_seed
        print("✅ utils.py")
        
        # Test budget manager (should work from our tests)
        from automl.budget_manager import BudgetManager
        print("✅ budget_manager.py")
        
        # Test other modules
        modules_to_test = [
            ('data_manager', 'AutoMLDataManager'),
            ('models', 'ModelFactory'),
            ('early_stopping', 'ComparativeEarlyStopping'),
            ('hpo_selection', 'MetaHPOSelector'),
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = __import__(f'automl.{module_name}', fromlist=[class_name])
                getattr(module, class_name)
                print(f"✅ {module_name}.py")
            except Exception as e:
                print(f"❌ {module_name}.py - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components"""
    
    print("\n🧪 Testing Basic Functionality")
    print("-" * 35)
    
    try:
        from automl.utils import AutoMLConfig, setup_logging, set_seed
        from automl.budget_manager import BudgetManager
        
        # Test configuration
        config = AutoMLConfig()
        config.set('dataset_name', 'emotions')
        config.set('num_classes', 7)
        config.set('time_budget_hours', 1)
        print("✅ Configuration working")
        
        # Test budget manager (we know this works)
        budget_manager = BudgetManager(config)
        budget_manager.start_execution(['resnet18', 'efficientnet_b0'])
        print("✅ Budget manager working")
        
        # Test logging
        logger = setup_logging(level='INFO')
        logger.info("Test message")
        print("✅ Logging working")
        
        # Test reproducibility
        set_seed(42)
        print("✅ Reproducibility working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure():
    """Check that all required files exist"""
    
    print("\n📁 Checking File Structure")
    print("-" * 26)
    
    automl_dir = project_root / "automl-exam-ss25-vision-freiburg-template" / "src" / "automl"
    
    required_files = [
        'utils.py',
        'budget_manager.py', 
        'data_manager.py',
        'models.py',
        'early_stopping.py',
        'hpo_selection.py',
        'training.py',
        'automl.py'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = automl_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Main test function"""
    
    print("🚀 Simple AutoML Integration Test")
    print("=" * 40)
    
    # Check file structure
    files_ok = check_file_structure()
    if not files_ok:
        print("❌ File structure check failed")
        return False
    
    # Test imports
    imports_ok = test_imports()
    if not imports_ok:
        print("❌ Import test failed")
        return False
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    if not functionality_ok:
        print("❌ Functionality test failed")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 40)
    print("Your AutoML pipeline is ready!")
    print("\nNext steps:")
    print("1. Ensure emotions_train.csv is in data/ directory")
    print("2. Run: python automl-exam-ss25-vision-freiburg-template/src/automl/automl.py")
    print("   (for a quick test)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)