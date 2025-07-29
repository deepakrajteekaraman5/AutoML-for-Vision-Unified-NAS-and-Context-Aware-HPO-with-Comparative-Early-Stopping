# src/automl/models.py
"""
Model Factory for AutoML Pipeline
Creates and manages different CNN architectures
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import timm
from pathlib import Path
import json

from .utils import AutoMLConfig, calculate_model_size, Timer

class ModelFactory:
    """Factory class for creating different CNN architectures"""
    
    # Define available model families as per project scope
    AVAILABLE_FAMILIES = {
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
        'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base'],
        'regnet': ['regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008'],
        'mobilenet': ['mobilenetv3_small_100', 'mobilenetv3_small_100'],
        'densenet': ['densenet121', 'densenet161', 'densenet169']
    }
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.ModelFactory')
        
        # Dataset-specific info
        self.num_classes = config.get('num_classes', 7)
        self.input_channels = config.get('channels', 1)
        self.image_size = config.get('image_size', 48)
        
        # Model compatibility cache
        self._compatibility_cache = {}
        
# src/automl/models.py
"""
Model Factory for AutoML Pipeline
Creates and manages different CNN architectures
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import timm
from pathlib import Path
import json

from .utils import AutoMLConfig, calculate_model_size, Timer

class ModelFactory:
    """Factory class for creating different CNN architectures"""
    
    # Strategic selection of 7 models for AutoML pipeline
    STRATEGIC_MODELS = {
        'resnet': ['resnet18', 'resnet34'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1'], 
        'convnext': ['convnext_tiny'],
        'mobilenet': ['mobilenetv3_small_100'],
        'densenet': ['densenet121']
    }
    
    # Flatten for easy access
    ALL_MODELS = [
        'resnet18', 'resnet34',
        'efficientnet_b0', 'efficientnet_b1',
        'convnext_tiny',
        'mobilenetv3_small_100', 
        'densenet121'
    ]
    
    # Model characteristics for optimization
    MODEL_CHARACTERISTICS = {
        'resnet18': {
            'family': 'resnet',
            'size': 'small',
            'speed': 'fast',
            'memory': 'low',
            'complexity_score': 2.0,
            'typical_lr': 1e-3,
            'good_for': ['small_images', 'debugging', 'baseline']
        },
        'resnet34': {
            'family': 'resnet', 
            'size': 'medium',
            'speed': 'medium',
            'memory': 'medium',
            'complexity_score': 3.0,
            'typical_lr': 1e-3,
            'good_for': ['medium_images', 'better_accuracy']
        },
        'efficientnet_b0': {
            'family': 'efficientnet',
            'size': 'small',
            'speed': 'medium',
            'memory': 'low',
            'complexity_score': 3.5,
            'typical_lr': 1e-3,
            'good_for': ['efficiency', 'small_datasets', 'all_sizes']
        },
        'efficientnet_b1': {
            'family': 'efficientnet',
            'size': 'medium', 
            'speed': 'medium',
            'memory': 'medium',
            'complexity_score': 4.0,
            'typical_lr': 8e-4,
            'good_for': ['better_accuracy', 'medium_datasets']
        },
        'convnext_tiny': {
            'family': 'convnext',
            'size': 'small',
            'speed': 'medium',
            'memory': 'medium',
            'complexity_score': 4.5,
            'typical_lr': 4e-3,
            'good_for': ['modern_architecture', 'complex_patterns']
        },
        'mobilenetv3_small_100': {
            'family': 'mobilenet',
            'size': 'small',
            'speed': 'very_fast',
            'memory': 'very_low',
            'complexity_score': 2.5,
            'typical_lr': 1e-3,
            'good_for': ['speed', 'small_images', 'limited_resources']
        },
        'densenet121': {
            'family': 'densenet',
            'size': 'medium',
            'speed': 'slow',
            'memory': 'high',
            'complexity_score': 4.0,
            'typical_lr': 1e-3,
            'good_for': ['feature_reuse', 'accuracy', 'complex_tasks']
        }
    }
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.ModelFactory')
        
        # Dataset-specific info
        self.num_classes = config.get('num_classes', 7)
        self.input_channels = config.get('channels', 1)
        self.image_size = config.get('image_size', 48)
        self.dataset_name = config.get('dataset_name', 'emotions')
        
        # Model compatibility cache
        self._model_cache = {}
        
        self.logger.info(f"ModelFactory initialized for {self.dataset_name}: "
                        f"{self.num_classes} classes, {self.input_channels} channels, "
                        f"{self.image_size}x{self.image_size} images")
        
        # Log available models
        self.logger.info(f"Strategic models available: {self.ALL_MODELS}")
    
    def get_available_models(self, family: Optional[str] = None) -> List[str]:
        """Get list of available models, optionally filtered by family"""
        if family:
            if family not in self.STRATEGIC_MODELS:
                raise ValueError(f"Unknown model family: {family}. Available: {list(self.STRATEGIC_MODELS.keys())}")
            return self.STRATEGIC_MODELS[family]
        else:
            return self.ALL_MODELS.copy()
    
    def get_model_characteristics(self, model_name: str) -> Dict[str, Any]:
        """Get characteristics of a specific model"""
        if model_name not in self.MODEL_CHARACTERISTICS:
            raise ValueError(f"Unknown model: {model_name}. Available: {self.ALL_MODELS}")
        return self.MODEL_CHARACTERISTICS[model_name].copy()
    
    def create_model(self, 
                    model_name: str, 
                    pretrained: bool = True,
                    dropout_rate: float = 0.0,
                    **kwargs) -> nn.Module:
        """Create a model with specified configuration"""
        
        if model_name not in self.ALL_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {self.ALL_MODELS}")
        
        self.logger.info(f"Creating {model_name} (pretrained={pretrained}, dropout={dropout_rate})")
        
        with Timer(f"{model_name} creation") as timer:
            try:
                # Create model using timm
                model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=self.num_classes,
                    in_chans=self.input_channels,
                    drop_rate=dropout_rate,
                    **kwargs
                )
                
                # Apply model-specific modifications if needed
                model = self._apply_model_modifications(model, model_name)
                
                # Calculate model info
                model_info = calculate_model_size(model)
                
                self.logger.info(f"{model_name} created successfully:")
                self.logger.info(f"  Parameters: {model_info['total_parameters']:,}")
                self.logger.info(f"  Trainable: {model_info['trainable_parameters']:,}")
                self.logger.info(f"  Model size: {model_info['size_mb']:.2f} MB")
                
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to create {model_name}: {e}")
                raise
    
    def _apply_model_modifications(self, model: nn.Module, model_name: str) -> nn.Module:
        """Apply any model-specific modifications"""
        
        # Handle grayscale input for models expecting RGB
        if self.input_channels == 1 and hasattr(model, 'conv1'):
            # For ResNet-style models, modify first conv layer
            if 'resnet' in model_name or 'densenet' in model_name:
                self._modify_first_conv_for_grayscale(model)
        
        return model
    
    def _modify_first_conv_for_grayscale(self, model: nn.Module):
        """Modify first conv layer to handle grayscale input"""
        if hasattr(model, 'conv1'):
            conv1 = model.conv1
            if conv1.in_channels == 3:  # Originally expecting RGB
                # Create new conv layer for grayscale
                new_conv1 = nn.Conv2d(
                    1, conv1.out_channels,
                    kernel_size=conv1.kernel_size,
                    stride=conv1.stride,
                    padding=conv1.padding,
                    bias=conv1.bias is not None
                )
                
                # Initialize with average of RGB weights if pretrained
                if conv1.weight.data.shape[1] == 3:
                    new_conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
                    if conv1.bias is not None:
                        new_conv1.bias.data = conv1.bias.data.clone()
                
                model.conv1 = new_conv1
                self.logger.info(f"Modified first conv layer for grayscale input")
    
    def get_recommended_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get recommended hyperparameters for a model"""
        if model_name not in self.MODEL_CHARACTERISTICS:
            raise ValueError(f"Unknown model: {model_name}")
        
        characteristics = self.MODEL_CHARACTERISTICS[model_name]
        
        # Base recommendations
        recommendations = {
            'learning_rate': characteristics['typical_lr'],
            'weight_decay': 1e-4,
            'batch_size': self._recommend_batch_size(model_name),
            'optimizer': 'adamw',
            'lr_scheduler': 'cosine',
            'warmup_epochs': 5
        }
        
        # Model-specific adjustments
        if 'convnext' in model_name:
            recommendations.update({
                'weight_decay': 5e-2,  # ConvNeXt uses higher weight decay
                'lr_scheduler': 'cosine',
                'layer_decay': 0.9  # Layer-wise learning rate decay
            })
        elif 'efficientnet' in model_name:
            recommendations.update({
                'optimizer': 'rmsprop',  # EfficientNet often uses RMSprop
                'weight_decay': 1e-5
            })
        elif 'mobilenet' in model_name:
            recommendations.update({
                'weight_decay': 4e-5,
                'batch_size': min(recommendations['batch_size'] * 2, 128)  # Can use larger batches
            })
        
        return recommendations
    
    def _recommend_batch_size(self, model_name: str) -> int:
        """Recommend batch size based on model and dataset characteristics"""
        characteristics = self.MODEL_CHARACTERISTICS[model_name]
        
        # Base batch size depends on image size and model memory usage
        if self.image_size <= 32:
            base_batch = 128
        elif self.image_size <= 64:
            base_batch = 64
        elif self.image_size <= 224:
            base_batch = 32
        else:
            base_batch = 16
        
        # Adjust for model memory usage
        if characteristics['memory'] == 'very_low':
            return min(base_batch * 2, 256)
        elif characteristics['memory'] == 'low':
            return base_batch
        elif characteristics['memory'] == 'medium':
            return max(base_batch // 2, 8)
        else:  # high memory
            return max(base_batch // 4, 4)
    
    def get_model_complexity_score(self, model_name: str) -> float:
        """Get complexity score for a model (used by early stopping)"""
        if model_name not in self.MODEL_CHARACTERISTICS:
            return 3.0  # Default medium complexity
        return self.MODEL_CHARACTERISTICS[model_name]['complexity_score']
    
    def get_models_by_speed(self, speed_preference: str = 'fast') -> List[str]:
        """Get models filtered by training speed"""
        speed_map = {
            'very_fast': ['mobilenetv3_small_100'],
            'fast': ['resnet18', 'mobilenetv3_small_100'],
            'medium': ['resnet34', 'efficientnet_b0', 'efficientnet_b1', 'convnext_tiny'],
            'slow': ['densenet121']
        }
        
        if speed_preference not in speed_map:
            return self.ALL_MODELS.copy()
        
        return speed_map[speed_preference]
    
    def get_models_by_dataset_characteristics(self, 
                                            complexity: str = 'medium',
                                            size: str = 'medium') -> List[str]:
        """Get recommended models based on dataset characteristics"""
        
        # Simple heuristic based on dataset complexity and size
        if complexity == 'low' and size == 'small':
            return ['resnet18', 'mobilenetv3_small_100', 'efficientnet_b0']
        elif complexity == 'high' and size == 'large':
            return ['efficientnet_b1', 'convnext_tiny', 'densenet121', 'resnet34']
        else:  # medium complexity/size
            return ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'densenet121']
    
    def print_model_summary(self):
        """Print summary of all available models"""
        print("\n=== AutoML Model Factory Summary ===")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input: {self.input_channels}x{self.image_size}x{self.image_size}")
        print(f"Classes: {self.num_classes}")
        print(f"\nStrategic Models ({len(self.ALL_MODELS)} total):")
        
        for family, models in self.STRATEGIC_MODELS.items():
            print(f"\n{family.upper()}:")
            for model in models:
                chars = self.MODEL_CHARACTERISTICS[model]
                print(f"  {model:25} | Speed: {chars['speed']:10} | "
                      f"Memory: {chars['memory']:8} | Complexity: {chars['complexity_score']}")

# Test the model factory
if __name__ == "__main__":
    # Test configuration
    config = AutoMLConfig()
    config.set('dataset_name', 'emotions')
    config.set('num_classes', 7)
    config.set('channels', 1)
    config.set('image_size', 48)
    
    # Create model factory
    factory = ModelFactory(config)
    factory.print_model_summary()
    
    # Test creating a model
    print(f"\n=== Testing Model Creation ===")
    model = factory.create_model('resnet18', pretrained=True, dropout_rate=0.1)
    print(f"Model created: {type(model).__name__}")
    
    # Test recommendations
    print(f"\n=== Hyperparameter Recommendations ===")
    for model_name in ['resnet18', 'efficientnet_b0', 'convnext_tiny']:
        recommendations = factory.get_recommended_hyperparameters(model_name)
        print(f"{model_name}: LR={recommendations['learning_rate']}, "
              f"Batch={recommendations['batch_size']}, "
              f"Optimizer={recommendations['optimizer']}")
    
    print("✅ Model factory test complete!")