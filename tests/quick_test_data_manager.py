# tests/quick_test_data_manager.py
"""
Quick test to verify the Albumentations fix
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

def main():
    print("=== Quick Data Manager Test ===")
    
    try:
        # Import modules
        from automl.utils import AutoMLConfig, setup_logging
        from automl.data_manager import AutoMLDataManager
        
        # Suppress logs for cleaner output
        import logging
        logging.getLogger('AutoML').setLevel(logging.ERROR)
        
        print("SUCCESS: Imports working")
        
        # Create data manager
        config = AutoMLConfig()
        config.set('dataset_name', 'emotions')
        data_manager = AutoMLDataManager(config)
        
        print("SUCCESS: DataManager created")
        
        # Setup datasets
        data_root = str(project_root / "data")
        setup_info = data_manager.setup_datasets(root=data_root)
        
        print(f"SUCCESS: Dataset setup - {setup_info['num_train']} train samples")
        
        # Create dataloaders with minimal configuration
        print("Testing DataLoader creation...")
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(
            batch_size=4,  # Very small batch for quick test
            image_size=48,
            augmentation_strategy='none',  # Start with no augmentation
            num_workers=0
        )
        
        print(f"SUCCESS: DataLoaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test loading a single batch
        print("Testing batch loading...")
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"SUCCESS: Batch loaded")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels: {labels.tolist()}")
        
        # Test with light augmentation
        print("Testing with light augmentation...")
        aug_train_loader, _, _ = data_manager.get_dataloaders(
            batch_size=4,
            augmentation_strategy='light',
            num_workers=0
        )
        
        aug_batch = next(iter(aug_train_loader))
        aug_images, aug_labels = aug_batch
        
        print(f"SUCCESS: Augmented batch loaded")
        print(f"  Augmented batch shape: {aug_images.shape}")
        print(f"  Augmented image range: [{aug_images.min():.3f}, {aug_images.max():.3f}]")
        
        print("\n" + "="*50)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("The Albumentations fix is working!")
        print("Your data pipeline is ready!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)