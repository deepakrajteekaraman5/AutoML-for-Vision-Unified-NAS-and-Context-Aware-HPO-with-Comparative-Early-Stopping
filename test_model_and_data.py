#!/usr/bin/env python3
"""
Test script to identify label corruption issue
Load saved model and test on dataset to find where corruption occurs
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir
src_path = project_root / "automl-exam-ss25-vision-freiburg-template" / "src"
sys.path.insert(0, str(src_path))

from automl.datasets import SkinCancerDataset
from automl.data_manager import AutoMLDataManager
from automl.utils import AutoMLConfig
from automl.models import ModelFactory

def test_dataset_directly():
    """Test the dataset directly without any processing"""
    print("="*60)
    print("TESTING DATASET DIRECTLY")
    print("="*60)
    
    try:
        # Load dataset directly
        dataset = SkinCancerDataset(
            root="automl-exam-ss25-vision-freiburg-template/data", 
            split='test',  # Use test split
            transform=None
        )
        print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
        # Test first few samples
        print(f"\nTesting first 10 samples:")
        for i in range(min(10, len(dataset))):
            try:
                image, label = dataset[i]
                print(f"  Sample {i}: label = {label} (type: {type(label)})")
                
                # Check for corruption
                if isinstance(label, (int, np.integer)):
                    if label < 0 or label >= 7:
                        print(f"    ❌ Invalid label: {label}")
                    elif label == -9223372036854775808:
                        print(f"    ❌ CORRUPTED label detected: {label}")
                elif isinstance(label, torch.Tensor):
                    label_val = label.item()
                    if label_val < 0 or label_val >= 7:
                        print(f"    ❌ Invalid tensor label: {label_val}")
                    elif label_val == -9223372036854775808:
                        print(f"    ❌ CORRUPTED tensor label detected: {label_val}")
                        
            except Exception as e:
                print(f"  ❌ Error loading sample {i}: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset directly: {e}")
        return False

def test_dataloader():
    """Test the dataloader to see if corruption happens there"""
    print("\n" + "="*60)
    print("TESTING DATALOADER")
    print("="*60)
    
    try:
        # Setup data manager
        config = AutoMLConfig()
        config.set('dataset_name', 'skin_cancer')
        config.set('num_classes', 7)
        config.set('channels', 3)
        config.set('image_size', 450)
        
        data_manager = AutoMLDataManager(config)
        setup_info = data_manager.setup_datasets(root="automl-exam-ss25-vision-freiburg-template/data")
        print(f"✅ Data manager setup successful")
        
        # Create dataloaders with minimal processing
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(
            batch_size=4, 
            num_workers=0,  # No multiprocessing
            augmentation_strategy='none'  # No augmentation
        )
        print(f"✅ DataLoaders created successfully")
        
        # Test first few batches
        print(f"\nTesting first 3 batches from test loader:")
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 3:
                break
                
            print(f"\n  Batch {batch_idx}:")
            print(f"    Images shape: {images.shape}, dtype: {images.dtype}")
            print(f"    Labels shape: {labels.shape}, dtype: {labels.dtype}")
            print(f"    Label values: {labels.tolist()}")
            print(f"    Label range: {labels.min().item()} to {labels.max().item()}")
            
            # Check for corruption
            corrupted_mask = labels == torch.iinfo(torch.int64).min
            if corrupted_mask.any():
                print(f"    ❌ CORRUPTED labels found: {labels[corrupted_mask].tolist()}")
                print(f"    ❌ Corruption detected in batch {batch_idx}")
                return False
            
            invalid_labels = (labels < 0) | (labels >= 7)
            if invalid_labels.any():
                print(f"    ❌ Invalid labels found: {labels[invalid_labels].tolist()}")
            else:
                print(f"    ✅ All labels are valid (0-6)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_saved_model():
    """Test the saved model"""
    print("\n" + "="*60)
    print("TESTING SAVED MODEL")
    print("="*60)
    
    try:
        # Load saved model
        checkpoint_path = "automl-exam-ss25-vision-freiburg-template/checkpoints/final_resnet18.pt"
        if not Path(checkpoint_path).exists():
            checkpoint_path = "automl-exam-ss25-vision-freiburg-template/checkpoints/final_efficientnet_b0.pt"
        
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        
        if 'architecture_name' in checkpoint:
            print(f"Architecture: {checkpoint['architecture_name']}")
        if 'test_results' in checkpoint:
            print(f"Test results: {checkpoint['test_results']}")
            
        # Try to recreate the model
        config = AutoMLConfig()
        config.set('dataset_name', 'skin_cancer')
        config.set('num_classes', 7)
        config.set('channels', 3)
        config.set('image_size', 450)
        
        model_factory = ModelFactory(config)
        
        if 'architecture_name' in checkpoint:
            arch_name = checkpoint['architecture_name']
            model = model_factory.create_model(arch_name, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model recreated and loaded: {arch_name}")
            
            # Test model on a few samples
            model.eval()
            
            # Create a simple test loader
            dataset = SkinCancerDataset(
                root="automl-exam-ss25-vision-freiburg-template/data", 
                split='test',
                transform=None
            )
            
            from torch.utils.data import DataLoader
            test_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            
            print(f"\nTesting model inference:")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx >= 2:
                        break
                    
                    print(f"  Batch {batch_idx} before processing:")
                    print(f"    Labels: {labels.tolist()}")
                    print(f"    Label types: {[type(l.item()) for l in labels]}")
                    
                    # Check for corruption before any processing
                    if torch.any(labels == torch.iinfo(torch.int64).min):
                        print(f"    ❌ CORRUPTION detected BEFORE processing!")
                        return False
                    
                    # Convert to float and back to see if that causes issues
                    images = images.float()
                    labels = labels.long()
                    
                    print(f"  After type conversion:")
                    print(f"    Labels: {labels.tolist()}")
                    
                    if torch.any(labels == torch.iinfo(torch.int64).min):
                        print(f"    ❌ CORRUPTION detected AFTER type conversion!")
                        return False
                    
                    # Try model inference (this is where the error was occurring)
                    try:
                        outputs = model(images)
                        print(f"    ✅ Model inference successful, output shape: {outputs.shape}")
                    except Exception as e:
                        print(f"    ❌ Model inference failed: {e}")
                        return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Saved model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("INVESTIGATING LABEL CORRUPTION ISSUE")
    print("="*60)
    
    # Test 1: Dataset directly
    dataset_ok = test_dataset_directly()
    
    # Test 2: DataLoader
    dataloader_ok = test_dataloader()
    
    # Test 3: Saved model
    model_ok = test_saved_model()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset direct test: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    print(f"DataLoader test: {'✅ PASS' if dataloader_ok else '❌ FAIL'}")
    print(f"Saved model test: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if not dataset_ok:
        print("\n🔍 ISSUE: Problem is in the dataset itself")
    elif not dataloader_ok:
        print("\n🔍 ISSUE: Problem is in the DataLoader/data processing")
    elif not model_ok:
        print("\n🔍 ISSUE: Problem is in model loading/inference")
    else:
        print("\n✅ All tests passed - corruption might be intermittent or context-specific")

if __name__ == "__main__":
    main()
