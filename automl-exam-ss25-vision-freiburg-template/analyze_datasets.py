#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes all three datasets (emotions, fashion, flowers) and prints their properties
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_dataset(dataset_name, data_root="data"):
    """Analyze a single dataset and return its properties"""
    
    print(f"\n{'='*50}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    dataset_path = Path(data_root) / dataset_name
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return None
    
    # Check CSV files
    train_csv = dataset_path / "train.csv"
    test_csv = dataset_path / "test.csv"
    
    if not train_csv.exists():
        print(f"train.csv not found: {train_csv}")
        return None
    
    if not test_csv.exists():
        print(f"test.csv not found: {test_csv}")
        return None
    
    try:
        # Read CSV files
        print(f"Reading CSV files...")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        print(f"   train.csv: {len(train_df)} rows")
        print(f"   test.csv: {len(test_df)} rows")
        
        # Analyze labels
        print(f"\nLABEL ANALYSIS:")
        train_labels = sorted(train_df['label'].unique())
        test_labels = sorted(test_df['label'].unique())
        
        print(f"   Training labels: {train_labels}")
        print(f"   Test labels: {test_labels}")
        print(f"   Number of classes: {len(train_labels)}")
        print(f"   Label range: {min(train_labels)} to {max(train_labels)}")
        
        # Check if train and test have same labels
        if set(train_labels) != set(test_labels):
            print(f"   WARNING: Train and test have different labels!")
            print(f"   Train only: {set(train_labels) - set(test_labels)}")
            print(f"   Test only: {set(test_labels) - set(train_labels)}")
        
        # Label distribution in training set
        print(f"\nTRAINING LABEL DISTRIBUTION:")
        label_counts = train_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            percentage = (count / len(train_df)) * 100
            print(f"   Class {label}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Check image files
        print(f"\nIMAGE ANALYSIS:")
        sample_image_file = train_df.iloc[0]['image_file_name']
        images_train_dir = dataset_path / "images_train"
        sample_image_path = images_train_dir / sample_image_file
        
        print(f"   Sample image: {sample_image_file}")
        print(f"   Images directory: {images_train_dir}")
        
        if not images_train_dir.exists():
            print(f"   Images directory not found: {images_train_dir}")
            return None
        
        if not sample_image_path.exists():
            print(f"   Sample image not found: {sample_image_path}")
            return None
        
        # Analyze sample image
        with Image.open(sample_image_path) as img:
            width, height = img.size
            mode = img.mode
            
            # Convert to array to check channels
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                channels = 1
                color_type = "Grayscale"
            elif len(img_array.shape) == 3:
                channels = img_array.shape[2]
                color_type = f"Color ({channels} channels)"
            else:
                channels = "Unknown"
                color_type = "Unknown"
        
        print(f"   Image dimensions: {width} x {height}")
        print(f"   Image mode: {mode}")
        print(f"   Channels: {channels}")
        print(f"   Color type: {color_type}")
        print(f"   Total pixels: {width * height:,}")
        
        # Memory estimation
        memory_per_image_mb = (width * height * channels * 4) / (1024 * 1024)  # float32
        print(f"   Memory per image: {memory_per_image_mb:.3f} MB")
        
        # Check a few more images to verify consistency
        print(f"\n🔍 CONSISTENCY CHECK:")
        sample_files = train_df['image_file_name'].head(5).tolist()
        consistent = True
        
        for i, img_file in enumerate(sample_files):
            img_path = images_train_dir / img_file
            if img_path.exists():
                with Image.open(img_path) as img:
                    if img.size != (width, height):
                        print(f"   ⚠️  Image {img_file} has different size: {img.size}")
                        consistent = False
            else:
                print(f"   Image not found: {img_file}")
                consistent = False
        
        if consistent:
            print(f"   All sampled images have consistent dimensions")
        
        # Summary for hardcoding
        print(f"\nSUMMARY FOR HARDCODING:")
        print(f"   dataset_name: '{dataset_name}'")
        print(f"   num_classes: {len(train_labels)}")
        print(f"   image_width: {width}")
        print(f"   image_height: {height}")
        print(f"   channels: {channels}")
        print(f"   num_samples_train: {len(train_df)}")
        print(f"   num_samples_test: {len(test_df)}")
        
        return {
            'dataset_name': dataset_name,
            'num_classes': len(train_labels),
            'image_width': width,
            'image_height': height,
            'channels': channels,
            'num_samples_train': len(train_df),
            'num_samples_test': len(test_df),
            'label_range': (min(train_labels), max(train_labels)),
            'memory_per_image_mb': memory_per_image_mb
        }
        
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to analyze all datasets"""
    
    print("🔍 DATASET ANALYSIS SCRIPT")
    print("Analyzing emotions, fashion, and flowers datasets...")
    
    datasets = ['emotions', 'fashion', 'flowers']
    results = {}
    
    for dataset in datasets:
        result = analyze_dataset(dataset)
        if result:
            results[dataset] = result
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - CONFIGURATION VALUES")
    print(f"{'='*60}")
    
    if results:
        print("\n# Configuration for run_automl.py:")
        print("dataset_configs = {")
        
        for dataset_name, props in results.items():
            print(f"    '{dataset_name}': {{")
            print(f"        'dataset_name': '{dataset_name}',")
            print(f"        'num_classes': {props['num_classes']},")
            print(f"        'channels': {props['channels']},")
            print(f"        'image_size': {props['image_width']},  # or {props['image_height']} if different")
            print(f"        'csv_file': '{dataset_name}/train.csv'")
            print(f"    }},")
        
        print("}")
        
        print(f"\n# Configuration for datasets.py:")
        for dataset_name, props in results.items():
            class_name = f"{dataset_name.capitalize()}Dataset"
            print(f"\nclass {class_name}(BaseVisionDataset):")
            print(f'    _dataset_name = "{dataset_name}"')
            print(f"    width = {props['image_width']}")
            print(f"    height = {props['image_height']}")
            print(f"    channels = {props['channels']}")
            print(f"    num_classes = {props['num_classes']}")
    
    else:
        print("No datasets could be analyzed successfully")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
