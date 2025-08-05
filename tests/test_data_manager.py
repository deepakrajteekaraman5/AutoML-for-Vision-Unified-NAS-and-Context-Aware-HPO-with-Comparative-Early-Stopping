# tests/windows_test_data_manager.py
"""
Windows-friendly test for AutoML Data Manager
"""

import sys
from pathlib import Path
import torch

# Setup Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
template_dir = project_root / "automl-exam-ss25-vision-freiburg-template"
src_path = template_dir / "src"
sys.path.insert(0, str(src_path))

def test_data_manager():
    """Test function wrapped in if __name__ == '__main__' for Windows compatibility"""
    
    print("=== AutoML Data Manager Test (Windows) ===")
    print(f"Testing data loading for emotions dataset...")

    try:
        # Import required modules
        from automl.utils import AutoMLConfig, setup_logging
        from automl.data_manager import AutoMLDataManager
        
        print("SUCCESS: Imports successful!")
        
        # Setup logging (suppress INFO for cleaner output)
        import logging
        logging.getLogger('AutoML').setLevel(logging.WARNING)
        
        # Test 1: Configuration and DataManager creation
        print("\n1. Testing DataManager creation...")
        config = AutoMLConfig()
        config.set('dataset_name', 'emotions')
        
        data_manager = AutoMLDataManager(config)
        print(f"   SUCCESS: DataManager created for {data_manager.dataset_name}")
        
        # Test 2: Dataset setup
        print("\n2. Testing dataset setup...")
        data_root = str(project_root / "data")
        
        # Temporarily suppress logs for clean output
        logging.getLogger('AutoML').setLevel(logging.ERROR)
        setup_info = data_manager.setup_datasets(root=data_root)
        logging.getLogger('AutoML').setLevel(logging.WARNING)
        
        print(f"   Train samples: {setup_info['num_train']}")
        print(f"   Validation samples: {setup_info['num_val']}")
        print(f"   Test samples: {setup_info['num_test']}")
        print(f"   Setup time: {setup_info['setup_time']:.2f}s")
        
        characteristics = setup_info['characteristics']
        print(f"   Image size: {characteristics['image_width']}x{characteristics['image_height']}")
        print(f"   Channels: {characteristics['channels']}")
        print(f"   Classes: {characteristics['num_classes']}")
        print(f"   Complexity: {characteristics['complexity_score']:.2f}/10")
        print("   SUCCESS: Dataset setup completed!")
        
        # Test 3: DataLoader creation (Windows-friendly)
        print("\n3. Testing DataLoader creation...")
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(
            batch_size=16,
            image_size=48,
            augmentation_strategy='light',
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        print("   SUCCESS: DataLoaders created!")
        
        # Test 4: Load and inspect batches
        print("\n4. Testing batch loading...")
        
        # Test train loader
        print("   Testing train loader...")
        train_batch = next(iter(train_loader))
        train_images, train_labels = train_batch
        
        print(f"   Train batch shape: {train_images.shape}")
        print(f"   Train batch labels: {train_labels.shape}")
        print(f"   Image dtype: {train_images.dtype}")
        print(f"   Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"   Label range: [{train_labels.min()}, {train_labels.max()}]")
        print(f"   Unique labels: {sorted(torch.unique(train_labels).tolist())}")
        
        # Verify dimensions
        batch_size, channels, height, width = train_images.shape
        if channels == 1 and height == 48 and width == 48:
            print("   SUCCESS: Image dimensions correct (1x48x48)!")
        else:
            print(f"   WARNING: Unexpected dimensions ({channels}x{height}x{width})")
        
        # Test validation loader
        print("   Testing validation loader...")
        val_batch = next(iter(val_loader))
        val_images, val_labels = val_batch
        print(f"   Validation batch shape: {val_images.shape}")
        
        # Test test loader
        print("   Testing test loader...")
        test_batch = next(iter(test_loader))
        test_images, test_labels = test_batch
        print(f"   Test batch shape: {test_images.shape}")
        
        print("   SUCCESS: All loaders working!")
        
        # Test 5: Different configurations
        print("\n5. Testing different batch sizes...")
        
        # Test smaller batch
        small_loader, _, _ = data_manager.get_dataloaders(
            batch_size=8,
            augmentation_strategy='none',
            num_workers=0
        )
        small_batch = next(iter(small_loader))
        print(f"   Small batch (8): {small_batch[0].shape}")
        
        # Test larger batch  
        large_loader, _, _ = data_manager.get_dataloaders(
            batch_size=32,
            augmentation_strategy='medium', 
            num_workers=0
        )
        large_batch = next(iter(large_loader))
        print(f"   Large batch (32): {large_batch[0].shape}")
        
        print("   SUCCESS: Different configurations working!")
        
        # Test 6: Augmentation comparison
        print("\n6. Testing augmentation strategies...")
        
        # No augmentation
        none_loader, _, _ = data_manager.get_dataloaders(
            batch_size=4, augmentation_strategy='none', num_workers=0
        )
        none_batch = next(iter(none_loader))
        none_images = none_batch[0]
        
        # Medium augmentation
        medium_loader, _, _ = data_manager.get_dataloaders(
            batch_size=4, augmentation_strategy='medium', num_workers=0
        )
        medium_batch = next(iter(medium_loader))
        medium_images = medium_batch[0]
        
        print(f"   No augmentation range: [{none_images.min():.3f}, {none_images.max():.3f}]")
        print(f"   Medium augmentation range: [{medium_images.min():.3f}, {medium_images.max():.3f}]")
        print("   SUCCESS: Augmentation strategies working!")
        
        print("\n" + "="*60)
        print("DATA MANAGER TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Dataset: {data_manager.dataset_name.upper()}")
        print(f"Images: {characteristics['image_width']}x{characteristics['image_height']} ({characteristics['channels']} channel)")
        print(f"Classes: {characteristics['num_classes']}")
        print(f"Training samples: {setup_info['num_train']:,}")
        print(f"Validation samples: {setup_info['num_val']:,}")
        print(f"Test samples: {setup_info['num_test']:,}")
        print(f"Complexity score: {characteristics['complexity_score']:.1f}/10")
        print(f"Recommended batch size: {characteristics['recommended_batch_size']}")
        print("\nYour data pipeline is ready for model training!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Windows multiprocessing protection
if __name__ == '__main__':
    success = test_data_manager()
    if success:
        print("\nAll tests passed! Ready to build models.")
    else:
        print("\nSome tests failed.")
        sys.exit(1)