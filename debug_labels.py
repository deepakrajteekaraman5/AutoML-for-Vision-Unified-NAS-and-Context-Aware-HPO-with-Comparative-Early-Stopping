#!/usr/bin/env python3
"""
Debug script to check label issues in the skin cancer dataset
"""

import sys
import torch
import pandas as pd
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

def debug_dataset_labels():
    """Debug the skin cancer dataset labels"""
    print("🔍 Debugging skin cancer dataset labels...")
    
    # Check CSV directly
    csv_path = Path("automl-exam-ss25-vision-freiburg-template/data/skin_cancer/train.csv")
    if csv_path.exists():
        print(f"\n📊 Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        if 'label' in df.columns:
            unique_labels = sorted(df['label'].unique())
            print(f"Unique labels in CSV: {unique_labels}")
            print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
            print(f"Number of unique labels: {len(unique_labels)}")
            
            # Check for any invalid labels
            if min(unique_labels) < 0:
                print("❌ ERROR: Found negative labels!")
            if max(unique_labels) >= 7:
                print("❌ ERROR: Found labels >= 7 (should be 0-6 for 7 classes)!")
            
            # Show label distribution
            label_counts = df['label'].value_counts().sort_index()
            print(f"\nLabel distribution:")
            for label, count in label_counts.items():
                print(f"  Class {label}: {count} samples")
        else:
            print("❌ ERROR: 'label' column not found in CSV!")
    else:
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        return False
    
    # Test dataset loading
    print(f"\n🔧 Testing dataset loading...")
    try:
        dataset = SkinCancerDataset(
            root="automl-exam-ss25-vision-freiburg-template/data", 
            split='train', 
            transform=None
        )
        print(f"Dataset loaded successfully: {len(dataset)} samples")
        print(f"Dataset num_classes: {dataset.num_classes}")
        
        # Check first few samples
        print(f"\nChecking first 10 samples:")
        for i in range(min(10, len(dataset))):
            try:
                image, label = dataset[i]
                print(f"  Sample {i}: label = {label} (type: {type(label)})")
                
                # Check if label is valid
                if isinstance(label, (int, np.integer)):
                    if label < 0 or label >= 7:
                        print(f"    ❌ Invalid label: {label} (should be 0-6)")
                elif isinstance(label, torch.Tensor):
                    label_val = label.item()
                    if label_val < 0 or label_val >= 7:
                        print(f"    ❌ Invalid tensor label: {label_val} (should be 0-6)")
                else:
                    print(f"    ❌ Unexpected label type: {type(label)}")
                    
            except Exception as e:
                print(f"  ❌ Error loading sample {i}: {e}")
                
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {e}")
        return False
    
    # Test data manager
    print(f"\n🔧 Testing data manager...")
    try:
        config = AutoMLConfig()
        config.set('dataset_name', 'skin_cancer')
        config.set('num_classes', 7)
        config.set('channels', 3)
        config.set('image_size', 450)
        
        data_manager = AutoMLDataManager(config)
        setup_info = data_manager.setup_datasets(root="automl-exam-ss25-vision-freiburg-template/data")
        print(f"Data manager setup successful")
        
        # Test dataloader
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(batch_size=4, num_workers=0)
        print(f"DataLoaders created successfully")
        
        # Check first batch
        print(f"\nChecking first batch:")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"  Batch {batch_idx}:")
            print(f"    Images shape: {images.shape}, dtype: {images.dtype}")
            print(f"    Labels shape: {labels.shape}, dtype: {labels.dtype}")
            print(f"    Label values: {labels.tolist()}")
            print(f"    Label range: {labels.min().item()} to {labels.max().item()}")
            
            # Check for invalid labels
            invalid_labels = (labels < 0) | (labels >= 7)
            if invalid_labels.any():
                print(f"    ❌ Found invalid labels: {labels[invalid_labels].tolist()}")
            else:
                print(f"    ✅ All labels are valid (0-6)")
            
            break  # Only check first batch
            
    except Exception as e:
        print(f"❌ ERROR: Data manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def suggest_fixes():
    """Suggest potential fixes for the label issue"""
    print(f"\n🔧 POTENTIAL FIXES:")
    print(f"1. Check if labels need to be converted to 0-indexed")
    print(f"2. Ensure labels are converted to long/int64 dtype")
    print(f"3. Add label validation in dataset loading")
    print(f"4. Check for any data preprocessing that might corrupt labels")

if __name__ == "__main__":
    print("="*60)
    print("SKIN CANCER DATASET LABEL DEBUGGING")
    print("="*60)
    
    success = debug_dataset_labels()
    
    if not success:
        suggest_fixes()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
