# src/automl/data_manager.py
"""
Data Management System for AutoML Pipeline
Wraps official dataset classes with AutoML-specific features
FIXED: Albumentations compatibility issues
"""

import os
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from .utils import AutoMLConfig, Timer, ensure_dir
from .datasets import EmotionsDataset, FashionDataset, FlowersDataset, SkinCancerDataset

class AlbumentationsWrapper:
    """Wrapper to make Albumentations compatible with torchvision-style datasets"""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply Albumentations transform with named argument
        result = self.transform(image=image)
        return result['image']

class DatasetCharacteristics:
    """Analyze and store dataset characteristics for optimization"""
    
    def __init__(self, dataset_name: str, train_dataset: Dataset):
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(f'AutoML.DatasetCharacteristics')
        self.characteristics = self._analyze_dataset(train_dataset)
    
    def _analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset to extract characteristics"""
        self.logger.info(f"Analyzing {self.dataset_name} dataset characteristics...")
        
        with Timer(f"{self.dataset_name} dataset analysis") as timer:
            # Basic info from dataset attributes
            basic_info = {
                'name': self.dataset_name,
                'num_samples': len(dataset),
                'num_classes': dataset.num_classes,
                'image_width': dataset.width,
                'image_height': dataset.height,
                'channels': dataset.channels,
                'image_size_pixels': dataset.width * dataset.height,
                'is_grayscale': dataset.channels == 1,
                'is_color': dataset.channels == 3,
            }
            
            # Sample a subset for detailed analysis (for efficiency)
            sample_size = min(1000, len(dataset))
            sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            # Analyze class distribution with progress bar
            sample_labels = []
            print(f"Analyzing class distribution...")
            for idx in tqdm(sample_indices[:100], desc="Sampling labels", leave=False):
                _, label = dataset[idx]
                sample_labels.append(label)
            
            class_counts = Counter(sample_labels)
            class_distribution = {
                'class_counts': dict(class_counts),
                'num_classes_actual': len(class_counts),
                'is_balanced': max(class_counts.values()) / min(class_counts.values()) < 2.0,
                'most_common_class': class_counts.most_common(1)[0][0],
                'least_common_class': class_counts.most_common()[-1][0],
            }
            
            # Complexity assessment
            complexity_score = self._calculate_complexity_score(basic_info)
            
            characteristics = {
                **basic_info,
                'class_distribution': class_distribution,
                'complexity_score': complexity_score,
                'memory_per_sample_mb': (basic_info['image_size_pixels'] * basic_info['channels'] * 4) / (1024 * 1024),  # float32
                'recommended_batch_size': self._recommend_batch_size(basic_info),
                'analysis_time': timer.elapsed
            }
        
        self.logger.info(f"Dataset analysis complete: {sample_size} samples analyzed")
        self.logger.info(f"Complexity score: {complexity_score:.2f}/10")
        
        return characteristics
    
    def _calculate_complexity_score(self, info: Dict) -> float:
        """Calculate complexity score (0-10) based on dataset characteristics"""
        score = 0.0
        
        # Image size complexity (0-3 points)
        pixels = info['image_size_pixels']
        if pixels > 200000:  # 512x512 and above
            score += 3.0
        elif pixels > 50000:  # ~224x224
            score += 2.0
        elif pixels > 10000:  # ~100x100
            score += 1.0
        # else: 0 points for small images
        
        # Number of classes complexity (0-3 points)
        classes = info['num_classes']
        if classes > 100:
            score += 3.0
        elif classes > 50:
            score += 2.5
        elif classes > 20:
            score += 2.0
        elif classes > 10:
            score += 1.0
        # else: 0 points for few classes
        
        # Color vs grayscale (0-2 points)
        if info['channels'] == 3:
            score += 2.0  # Color is more complex
        elif info['channels'] == 1:
            score += 1.0  # Grayscale is simpler
        
        # Dataset size complexity (0-2 points)
        samples = info['num_samples']
        if samples < 5000:
            score += 2.0  # Small datasets are harder (overfitting risk)
        elif samples < 20000:
            score += 1.0
        # Large datasets are actually easier to train on
        
        return min(score, 10.0)  # Cap at 10
    
    def _recommend_batch_size(self, info: Dict) -> int:
        """Recommend batch size based on image characteristics"""
        # Start with a base batch size
        if info['image_size_pixels'] > 200000:  # Large images
            base_batch = 8
        elif info['image_size_pixels'] > 50000:  # Medium images  
            base_batch = 16
        else:  # Small images
            base_batch = 64
        
        # Adjust for channels
        if info['channels'] == 3:
            base_batch = base_batch // 2
        
        return base_batch
    
    def get_characteristic(self, key: str) -> Any:
        """Get specific characteristic"""
        return self.characteristics.get(key)
    
    def print_summary(self):
        """Print a human-readable summary"""
        c = self.characteristics
        print(f"\n=== {c['name'].upper()} Dataset Summary ===")
        print(f"Samples: {c['num_samples']:,}")
        print(f"Classes: {c['num_classes']}")
        print(f"Image Size: {c['image_width']}x{c['image_height']}x{c['channels']}")
        print(f"Type: {'Grayscale' if c['is_grayscale'] else 'Color'}")
        print(f"Complexity: {c['complexity_score']:.1f}/10")
        print(f"Memory/sample: {c['memory_per_sample_mb']:.3f} MB")
        print(f"Recommended batch size: {c['recommended_batch_size']}")
        
        # Class distribution
        dist = c['class_distribution']
        print(f"Classes: {dist['num_classes_actual']} ({'Balanced' if dist['is_balanced'] else 'Imbalanced'})")

class AugmentationFactory:
    """Create augmentation strategies based on dataset characteristics"""
    
    def __init__(self, dataset_characteristics: DatasetCharacteristics):
        self.characteristics = dataset_characteristics
        self.logger = logging.getLogger('AutoML.AugmentationFactory')
    
    def create_transforms(self, 
                         strategy: str = 'auto', 
                         image_size: Optional[int] = None,
                         is_training: bool = True) -> A.Compose:
        """Create augmentation transforms based on strategy and dataset"""
        
        # Use dataset's native size if not specified
        if image_size is None:
            image_size = self.characteristics.get_characteristic('image_width')
        
        # Auto-select strategy based on dataset characteristics
        if strategy == 'auto':
            strategy = self._auto_select_strategy()
            self.logger.info(f"Auto-selected augmentation strategy: {strategy}")
        
        # Base transforms (always applied)
        base_transforms = [
            A.Resize(image_size, image_size, interpolation=1),  # INTER_LINEAR
        ]
        
        # Training-specific augmentations
        if is_training:
            augmentations = self._get_augmentations_by_strategy(strategy)
        else:
            augmentations = []  # No augmentation for validation/test
        
        # Normalization (always last, before ToTensor)
        normalization = self._get_normalization()
        
        # Combine all transforms
        all_transforms = base_transforms + augmentations + normalization + [ToTensorV2()]
        
        albumentations_transform = A.Compose(all_transforms)
        return AlbumentationsWrapper(albumentations_transform)
    
    def _auto_select_strategy(self) -> str:
        """Auto-select augmentation strategy based on dataset characteristics"""
        complexity = self.characteristics.get_characteristic('complexity_score')
        num_samples = self.characteristics.get_characteristic('num_samples')
        
        # Small datasets need more augmentation
        if num_samples < 10000:
            return 'heavy'
        elif complexity > 7.0:
            return 'medium'  # Complex datasets benefit from some augmentation
        elif complexity < 3.0:
            return 'light'   # Simple datasets need less augmentation
        else:
            return 'medium'
    
    def _get_augmentations_by_strategy(self, strategy: str) -> List:
        """Get augmentations based on strategy - FIXED for Albumentations compatibility"""
        is_grayscale = self.characteristics.get_characteristic('is_grayscale')
        
        if strategy == 'light':
            augs = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.3),
            ]
        elif strategy == 'medium':
            augs = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ]
            
            # Add color augmentations for color images
            if not is_grayscale:
                augs.extend([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                ])
                
        elif strategy == 'heavy':
            # FIXED: Get image dimensions safely
            image_height = self.characteristics.get_characteristic('image_height')
            image_width = self.characteristics.get_characteristic('image_width')
            
            augs = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                # FIXED: Use variance_limit instead of var_limit
                A.GaussNoise(variance_limit=(10.0, 50.0), p=0.2),
                # FIXED: Use size parameter instead of height/width
                A.RandomResizedCrop(size=(image_height, image_width), 
                                   scale=(0.8, 1.0), p=0.3),
            ]
            
            # Add color augmentations for color images  
            if not is_grayscale:
                augs.extend([
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                ])
        else:
            augs = []  # No augmentation
        
        return augs
    
    def _get_normalization(self) -> List:
        """Get appropriate normalization for the dataset"""
        is_grayscale = self.characteristics.get_characteristic('is_grayscale')
        
        if is_grayscale:
            # ImageNet grayscale normalization
            return [A.Normalize(mean=[0.485], std=[0.229])]
        else:
            # ImageNet RGB normalization
            return [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

class AutoMLDataManager:
    """Main data management class for AutoML pipeline"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.dataset_name = config.get('dataset_name', 'emotions')
        self.logger = logging.getLogger('AutoML.DataManager')
        
        # Dataset class mapping
        self.dataset_classes = {
            'emotions': EmotionsDataset,
            'fashion': FashionDataset,
            'flowers': FlowersDataset,
            'skin_cancer': SkinCancerDataset
        }
        
        # Initialize dataset
        self.dataset_class = self.dataset_classes[self.dataset_name]
        self.characteristics = None
        self.augmentation_factory = None
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.logger.info(f"Initialized DataManager for {self.dataset_name} dataset")
    
    def setup_datasets(self, root: str = "data", val_split: float = 0.2) -> Dict[str, Any]:
        """Setup train, validation, and test datasets"""
        self.logger.info(f"Setting up {self.dataset_name} datasets...")
        
        with Timer(f"{self.dataset_name} dataset setup") as timer:
            # Load the training dataset
            train_full = self.dataset_class(root=root, split='train', transform=None)
            test_dataset = self.dataset_class(root=root, split='test', transform=None)
            
            # Analyze characteristics
            self.characteristics = DatasetCharacteristics(self.dataset_name, train_full)
            self.characteristics.print_summary()
            
            # Create augmentation factory
            self.augmentation_factory = AugmentationFactory(self.characteristics)
            
            # Split training data into train/validation and store indices
            self.train_indices, self.val_indices = self._create_stratified_split(train_full, val_split)
            
            self.logger.info(f"Data splits - Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(test_dataset)}")
        
        setup_info = {
            'dataset_name': self.dataset_name,
            'num_train': len(self.train_indices),
            'num_val': len(self.val_indices), 
            'num_test': len(test_dataset),
            'characteristics': self.characteristics.characteristics,
            'setup_time': timer.elapsed
        }
        
        return setup_info
    
    def _create_stratified_split(self, dataset: Dataset, val_split: float) -> Tuple[List[int], List[int]]:
        """Create stratified train/validation split"""
        # Extract labels for stratification with progress bar
        labels = []
        print(f"Creating stratified train/validation split...")
        for i in tqdm(range(len(dataset)), desc="Extracting labels", leave=False):
            _, label = dataset[i]
            labels.append(label)
        
        # Use stratified split to maintain class distribution
        print(f"Performing stratified split ({int((1-val_split)*100)}% train, {int(val_split*100)}% val)...")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=self.config.get('random_seed', 42))
        train_indices, val_indices = next(splitter.split(range(len(dataset)), labels))
        
        self.logger.info(f"Created stratified split: {len(train_indices)} train, {len(val_indices)} validation")
        
        return train_indices.tolist(), val_indices.tolist()
    
    def get_dataloaders(self, 
                       batch_size: Optional[int] = None,
                       image_size: Optional[int] = None,
                       augmentation_strategy: str = 'auto',
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train, validation, and test"""
        
        if self.characteristics is None:
            raise ValueError("Must call setup_datasets() first")
        
        # Use recommended batch size if not provided
        if batch_size is None:
            batch_size = self.characteristics.get_characteristic('recommended_batch_size')
            self.logger.info(f"Using recommended batch size: {batch_size}")
        
        # Use dataset's native image size if not provided
        if image_size is None:
            image_size = self.characteristics.get_characteristic('image_width')
        
        self.logger.info(f"Creating dataloaders - batch_size: {batch_size}, image_size: {image_size}")
        
        # Create transforms
        train_transform = self.augmentation_factory.create_transforms(
            strategy=augmentation_strategy, 
            image_size=image_size, 
            is_training=True
        )
        val_test_transform = self.augmentation_factory.create_transforms(
            strategy='none', 
            image_size=image_size, 
            is_training=False
        )
        
        # Load datasets with transforms
        root = "data"  # TODO: Make this configurable
        
        # Full datasets
        train_full = self.dataset_class(root=root, split='train', transform=train_transform)
        test_full = self.dataset_class(root=root, split='test', transform=val_test_transform)
        
        # Use stored train/val splits
        if not hasattr(self, 'train_indices') or not hasattr(self, 'val_indices'):
            raise ValueError("Must call setup_datasets() first to create train/val splits")
        
        # Create subset datasets
        train_dataset = Subset(train_full, self.train_indices)
        val_dataset = Subset(
            self.dataset_class(root=root, split='train', transform=val_test_transform), 
            self.val_indices
        )

        safe_num_workers = 0 if num_workers == 0 or sys.platform.startswith('win') else min(num_workers, 2)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=safe_num_workers,
            pin_memory=torch.cuda.is_available() and safe_num_workers > 0,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=safe_num_workers,
            pin_memory=torch.cuda.is_available() and safe_num_workers > 0
        )
        
        test_loader = DataLoader(
            test_full,
            batch_size=batch_size,
            shuffle=False,
            num_workers=safe_num_workers,
            pin_memory=torch.cuda.is_available() and safe_num_workers > 0
        )
        
        self.logger.info(f"DataLoaders created successfully")
        return train_loader, val_loader, test_loader
    
    def visualize_samples(self, num_samples: int = 8, save_path: Optional[str] = None):
        """Visualize random samples from the dataset"""
        if self.characteristics is None:
            raise ValueError("Must call setup_datasets() first")
        
        # Load dataset without transforms for visualization
        dataset = self.dataset_class(root="data", split='train', transform=None)
        
        # Sample random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Handle grayscale vs color
            if len(image.shape) == 2:  # Grayscale
                axes[i].imshow(image, cmap='gray')
            else:  # Color
                axes[i].imshow(image)
            
            axes[i].set_title(f'Class: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

# Test the data manager
if __name__ == "__main__":
    # Test configuration
    config = AutoMLConfig()
    config.set('dataset_name', 'emotions')
    
    # Create data manager
    data_manager = AutoMLDataManager(config)
    
    # Setup datasets
    setup_info = data_manager.setup_datasets()
    print(f"Setup complete: {setup_info['num_train']} train samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_manager.get_dataloaders(batch_size=32)
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        if batch_idx == 0:
            break
    
    print("Data manager test complete!")