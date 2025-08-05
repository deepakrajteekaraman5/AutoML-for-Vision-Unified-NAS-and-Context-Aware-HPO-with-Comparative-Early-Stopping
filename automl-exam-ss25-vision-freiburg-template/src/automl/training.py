# src/automl/training.py
"""
Training Engine for AutoML Pipeline
Handles model training, evaluation, and HPO objective functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .utils import AutoMLConfig, Timer, MetricTracker, get_device, calculate_model_size

class Trainer:
    """
    Core training engine for AutoML pipeline
    
    Handles:
    - Model training with different optimizers/schedulers
    - Model evaluation and metric tracking
    - HPO objective function
    - Early stopping integration
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.Trainer')
        self.device = get_device(prefer_gpu=True)
        
        # Training configuration
        self.max_epochs = config.get('max_epochs_per_architecture', 50)
        self.min_epochs = config.get('min_epochs_per_architecture', 5)
        self.patience = config.get('early_stopping_patience', 10)
        self.primary_metric = config.get('performance_metric', 'accuracy')
        
        # State tracking
        self.metric_tracker = MetricTracker()
        self.training_history = {}
        
        self.logger.info(f"Trainer initialized - Device: {self.device}")
        self.logger.info(f"Max epochs: {self.max_epochs}, Min epochs: {self.min_epochs}")
    
    def create_objective_function(self, 
                                architecture_name: str,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                model_factory: Any,
                                early_stopping_engine: Any = None,
                                budget_manager: Any = None,
                                time_enforcer: Any = None,
                                model_timer_id: str = None) -> Callable:
        """
        Create HPO objective function for a specific architecture
        
        Returns:
            Callable that takes hyperparameters and returns validation accuracy
        """
        
        def objective_function(hyperparams: Dict[str, Any]) -> float:
            """
            Objective function for HPO
            
            Args:
                hyperparams: Dictionary of hyperparameters to evaluate
                
            Returns:
                Validation accuracy (metric to maximize)
            """
            try:
                self.logger.info(f"Evaluating hyperparameters for {architecture_name}: {hyperparams}")
                
                # Clear GPU memory before creating new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create model with hyperparameters
                model = model_factory.create_model(
                    architecture_name,
                    pretrained=hyperparams.get('pretrained', True),
                    dropout_rate=hyperparams.get('dropout_rate', 0.0)
                )
                model = model.to(self.device)
                
                # Create optimizer
                optimizer = self._create_optimizer(model, hyperparams)
                
                # Create scheduler
                scheduler = self._create_scheduler(optimizer, hyperparams)
                
                # Create loss function with class weighting for imbalanced datasets
                criterion = self._create_loss_function(train_loader)
                
                # Training configuration
                epochs_to_run = min(
                    hyperparams.get('max_epochs', self.max_epochs),
                    self.max_epochs
                )
                
                # Track training progress
                training_start_time = time.time()
                best_val_accuracy = 0.0
                epoch_times = []
                
                # Training loop with progress bar and STRICT TIME ENFORCEMENT
                epoch_pbar = tqdm(range(1, epochs_to_run + 1), 
                                desc=f"Training {architecture_name}", 
                                unit="epoch",
                                leave=False,  # Don't leave progress bar after completion
                                position=0,   # Use position 0 to avoid conflicts
                                dynamic_ncols=True)  # Adjust to terminal width
                
                for epoch in epoch_pbar:
                    epoch_start_time = time.time()
                    
                    # SMART TIME ENFORCEMENT: Check timeout with smart completion
                    if time_enforcer and model_timer_id:
                        epochs_remaining = epochs_to_run - epoch + 1
                        is_timeout, elapsed, remaining = time_enforcer.check_timeout_with_smart_completion(
                            model_timer_id, epochs_remaining=epochs_remaining
                        )
                        
                        if is_timeout:
                            self.logger.warning(f"TIMEOUT: {architecture_name} exceeded time limit at epoch {epoch}")
                            self.logger.warning(f"Elapsed: {elapsed/3600:.2f}h, stopping training")
                            break
                        
                        # Update progress bar with time info
                        remaining_minutes = remaining / 60
                        if elapsed >= time_enforcer.time_config.get_timeout_seconds_per_model():
                            epoch_pbar.set_description(f"Training {architecture_name} (BUFFER: {remaining_minutes:.1f}m left)")
                        else:
                            epoch_pbar.set_description(f"Training {architecture_name} (Time left: {remaining_minutes:.1f}m)")
                    
                    # Training phase
                    train_metrics = self._train_epoch(
                        model, train_loader, optimizer, criterion, epoch
                    )
                    
                    # STRICT TIME ENFORCEMENT: Check timeout after training phase
                    if time_enforcer and model_timer_id:
                        is_timeout, elapsed, remaining = time_enforcer.check_timeout(model_timer_id)
                        if is_timeout:
                            self.logger.warning(f"TIMEOUT: {architecture_name} exceeded time limit during training phase")
                            break
                    
                    # Validation phase
                    val_metrics = self._validate_epoch(
                        model, val_loader, criterion, epoch
                    )
                    
                    # STRICT TIME ENFORCEMENT: Check timeout after validation phase
                    if time_enforcer and model_timer_id:
                        is_timeout, elapsed, remaining = time_enforcer.check_timeout(model_timer_id)
                        if is_timeout:
                            self.logger.warning(f"TIMEOUT: {architecture_name} exceeded time limit during validation phase")
                            break
                    
                    # Update progress bar with current metrics
                    epoch_pbar.set_postfix({
                        'Train Acc': f"{train_metrics['accuracy']:.3f}",
                        'Val Acc': f"{val_metrics['accuracy']:.3f}",
                        'Val Loss': f"{val_metrics['loss']:.3f}",
                        'Best': f"{best_val_accuracy:.3f}"
                    })
                    
                    # Learning rate scheduling
                    if scheduler:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_metrics['loss'])
                        else:
                            scheduler.step()
                    
                    epoch_time = time.time() - epoch_start_time
                    epoch_times.append(epoch_time)
                    
                    # Combine metrics
                    combined_metrics = {
                        'epoch': epoch,
                        'train_accuracy': train_metrics['accuracy'],
                        'train_loss': train_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch_time': epoch_time
                    }
                    
                    # Update tracking
                    self.metric_tracker.update(**combined_metrics)
                    
                    # Track best validation accuracy
                    if val_metrics['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['accuracy']
                    
                    # Update early stopping engine if provided
                    if early_stopping_engine:
                        early_stopping_engine.update_performance(
                            architecture_name, epoch, 
                            {'val_accuracy': val_metrics['accuracy'], 'val_loss': val_metrics['loss']},
                            training_time=epoch_time
                        )
                        
                        # Check if should stop early (only after minimum epochs)
                        if epoch >= self.min_epochs:
                            should_stop, reason, confidence = early_stopping_engine.should_stop_architecture(
                                architecture_name, 'val_accuracy'
                            )
                            
                            if should_stop:
                                self.logger.info(f"Early stopping triggered for {architecture_name} at epoch {epoch}")
                                self.logger.info(f"Reason: {reason}, Confidence: {confidence:.3f}")
                                break
                    
                    # Update budget manager if provided
                    if budget_manager:
                        total_training_time = (time.time() - training_start_time) / 3600.0  # Hours
                        budget_manager.update_architecture_progress(
                            architecture_name,
                            performance_score=val_metrics['accuracy'],
                            time_used_hours=total_training_time,
                            memory_used_gb=self._estimate_memory_usage(model)
                        )
                    
                    # Log progress with time enforcement info
                    if epoch % 5 == 0 or epoch == epochs_to_run:
                        time_info = ""
                        if time_enforcer and model_timer_id:
                            _, elapsed, remaining = time_enforcer.check_timeout(model_timer_id)
                            time_info = f", Time: {elapsed/3600:.2f}h/{remaining/3600:.2f}h left"
                        
                        self.logger.info(
                            f"Epoch {epoch}/{epochs_to_run} - "
                            f"Train Acc: {train_metrics['accuracy']:.4f}, "
                            f"Val Acc: {val_metrics['accuracy']:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Epoch Time: {epoch_time:.1f}s{time_info}"
                        )
                
                # Calculate efficiency metrics
                total_training_time = time.time() - training_start_time
                avg_epoch_time = np.mean(epoch_times)
                
                self.logger.info(f"Training completed for {architecture_name}")
                self.logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
                self.logger.info(f"Total training time: {total_training_time:.1f}s")
                self.logger.info(f"Average epoch time: {avg_epoch_time:.1f}s")
                
                return best_val_accuracy
                
            except Exception as e:
                self.logger.error(f"Training failed for {architecture_name}: {e}")
                # Clean up GPU memory on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return 0.0  # Return poor score for failed training
            
            finally:
                # Always clean up memory after training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Clean up model reference
                if 'model' in locals():
                    del model
        
        return objective_function
    
    def _train_epoch(self, 
                    model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    epoch: int) -> Dict[str, float]:
        """Train model for one epoch"""
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar for training batches
        train_pbar = tqdm(dataloader, 
                         desc=f"Train Epoch {epoch}", 
                         leave=False,
                         unit="batch")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Ensure correct data types
            images = images.float()
            labels = labels.long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_acc = correct_predictions / total_samples
            current_loss = running_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'Acc': f"{current_acc:.3f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def _validate_epoch(self,
                       model: nn.Module,
                       dataloader: DataLoader,
                       criterion: nn.Module,
                       epoch: int) -> Dict[str, float]:
        """Validate model for one epoch"""
        
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar for validation batches
        val_pbar = tqdm(dataloader, 
                       desc=f"Val Epoch {epoch}", 
                       leave=False,
                       unit="batch")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Ensure correct data types
                images = images.float()
                labels = labels.long()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar with current metrics
                current_acc = correct_predictions / total_samples
                current_loss = running_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'Loss': f"{current_loss:.4f}",
                    'Acc': f"{current_acc:.3f}"
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def _create_optimizer(self, model: nn.Module, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer based on hyperparameters"""
        
        optimizer_type = hyperparams.get('optimizer', 'adamw').lower()
        learning_rate = hyperparams.get('learning_rate', 1e-3)
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = hyperparams.get('momentum', 0.9)
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"Unknown optimizer {optimizer_type}, using AdamW")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def _create_scheduler(self, 
                         optimizer: torch.optim.Optimizer, 
                         hyperparams: Dict[str, Any]) -> Optional[Any]:
        """Create learning rate scheduler based on hyperparameters"""
        
        scheduler_type = hyperparams.get('lr_scheduler', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            T_max = hyperparams.get('max_epochs', self.max_epochs)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_type == 'step':
            step_size = hyperparams.get('step_size', 10)
            gamma = hyperparams.get('gamma', 0.1)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'none':
            scheduler = None
        else:
            self.logger.warning(f"Unknown scheduler {scheduler_type}, using cosine")
            T_max = hyperparams.get('max_epochs', self.max_epochs)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
        
        return scheduler
    
    def _create_loss_function(self, train_loader: DataLoader) -> nn.Module:
        """Create loss function with class weighting for imbalanced datasets using sklearn's robust method"""
        try:
            # Get the total number of classes from the dataset
            dataset = train_loader.dataset
            
            # Handle different dataset types (Subset, etc.)
            if hasattr(dataset, 'dataset'):
                # This is a Subset, get the underlying dataset
                underlying_dataset = dataset.dataset
            else:
                underlying_dataset = dataset
            
            # Get total number of classes from the dataset
            if hasattr(underlying_dataset, 'num_classes'):
                total_num_classes = underlying_dataset.num_classes
            else:
                self.logger.warning("Could not determine total number of classes, using standard loss")
                return nn.CrossEntropyLoss()
            
            # Collect labels from training data for class weight calculation
            all_labels = []
            
            # Sample a larger subset for more robust class weight calculation
            sample_batches = min(20, len(train_loader))  # Increased sample size for better estimation
            for batch_idx, (_, labels) in enumerate(train_loader):
                if batch_idx >= sample_batches:
                    break
                all_labels.extend(labels.cpu().numpy().tolist())
            
            if len(set(all_labels)) > 1:
                # Get unique classes present in the sample
                unique_classes_in_sample = np.unique(all_labels)
                
                # Compute class weights using sklearn's robust 'balanced' method
                # Only for classes actually present in the sample
                sample_class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes_in_sample,
                    y=all_labels
                )
                
                # Create weight array for ALL classes (0 to total_num_classes-1)
                class_weights_array = np.ones(total_num_classes, dtype=np.float32)
                
                # Assign computed weights to classes present in sample
                for i, class_idx in enumerate(unique_classes_in_sample):
                    class_weights_array[class_idx] = sample_class_weights[i]
                
                # For missing classes, use the mean weight of present classes
                mean_weight = np.mean(sample_class_weights)
                for class_idx in range(total_num_classes):
                    if class_idx not in unique_classes_in_sample:
                        class_weights_array[class_idx] = mean_weight
                
                # Convert to tensor
                class_weights_tensor = torch.FloatTensor(class_weights_array).to(self.device)
                
                # Log class weights for debugging
                class_counts = {cls: all_labels.count(cls) for cls in set(all_labels)}
                self.logger.info(f"Total classes in dataset: {total_num_classes}")
                self.logger.info(f"Classes found in sample: {len(set(all_labels))}")
                self.logger.info(f"Sample size: {len(all_labels)} labels")
                self.logger.info(f"Class distribution (sample): {dict(sorted(class_counts.items()))}")
                self.logger.info(f"Class weights tensor shape: {class_weights_tensor.shape}")
                self.logger.info(f"Sample class weights range: [{class_weights_tensor.min():.4f}, {class_weights_tensor.max():.4f}]")
                self.logger.info(f"Mean weight for missing classes: {mean_weight:.4f}")
                
                return nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                self.logger.info("Single class detected or no class distribution found, using standard loss")
                return nn.CrossEntropyLoss()
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate class weights: {e}, using standard loss")
            self.logger.warning(f"Error details: {str(e)}")
            return nn.CrossEntropyLoss()
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                memory_bytes = torch.cuda.memory_allocated()
                memory_gb = memory_bytes / (1024 ** 3)
                return memory_gb
            else:
                # Rough estimate for CPU
                model_info = calculate_model_size(model)
                return model_info['size_mb'] / 1024  # Convert MB to GB
        except:
            return 1.0  # Default estimate
    
    def _get_model_num_classes(self, model: nn.Module) -> int:
        """Get number of output classes from model"""
        try:
            # Common final layer names in different architectures
            final_layer_names = ['fc', 'classifier', 'head', 'linear', 'output']
            
            for name in final_layer_names:
                if hasattr(model, name):
                    layer = getattr(model, name)
                    if hasattr(layer, 'out_features'):
                        return layer.out_features
                    elif hasattr(layer, 'weight') and len(layer.weight.shape) >= 2:
                        return layer.weight.shape[0]
            
            # Fallback: try to get from a forward pass with dummy input
            model.eval()
            with torch.no_grad():
                # Create dummy input based on expected input size
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)  # Standard size
                try:
                    output = model(dummy_input)
                    if hasattr(output, 'shape') and len(output.shape) >= 2:
                        return output.shape[1]  # Number of classes
                except:
                    pass
            
            # Final fallback: assume 7 classes for skin cancer dataset
            self.logger.warning("Could not determine number of classes from model, assuming 7")
            return 7
            
        except Exception as e:
            self.logger.warning(f"Error getting model num_classes: {e}, assuming 7")
            return 7
    
    def evaluate_model(self, 
                      model: nn.Module,
                      test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        
        self.logger.info("Evaluating model on test set...")
        
        # SKIN CANCER DATASET FIX: Skip test evaluation for skin cancer (competition dataset)
        dataset_name = self.config.get('dataset_name', '')
        if dataset_name == 'skin_cancer':
            self.logger.info("Skin cancer dataset detected - skipping test evaluation")
            self.logger.info("Test labels are withheld for grading purposes")
            
            # Return placeholder results for skin cancer
            results = {
                'test_loss': 0.0,
                'test_accuracy': 0.0,
                'total_samples': 0,
                'correct_predictions': 0,
                'per_class_accuracy': {},
                'evaluation_skipped': True,
                'reason': 'Skin cancer dataset - test labels withheld for grading'
            }
            
            return results
        
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Per-class accuracy tracking
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # ENHANCED FIX: Ensure correct data types and validate labels
                images = images.float()  # Ensure float32
                labels = labels.long()   # Ensure long (int64)
                
                # CRITICAL FIX: Validate label range to prevent CUDA assertion error
                # Get number of classes from model
                num_classes = self._get_model_num_classes(model)
                
                # Check for invalid labels before forward pass
                if labels.min() < 0 or labels.max() >= num_classes:
                    self.logger.error(f"Invalid labels detected: min={labels.min().item()}, max={labels.max().item()}")
                    self.logger.error(f"Expected range: [0, {num_classes-1}]")
                    self.logger.error(f"Label values: {labels.cpu().numpy()}")
                    # Skip this batch to avoid CUDA error
                    continue
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Per-class statistics
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    
                    class_total[label] += 1
                    if pred == label:
                        class_correct[label] += 1
        
        # Calculate overall metrics
        test_loss = running_loss / len(test_loader)
        test_accuracy = correct_predictions / total_samples
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_idx in class_total:
            if class_total[class_idx] > 0:
                per_class_accuracy[class_idx] = class_correct[class_idx] / class_total[class_idx]
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'per_class_accuracy': per_class_accuracy
        }
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        self.logger.info(f"  Samples: {total_samples}")
        
        # Log per-class accuracy
        for class_idx, acc in per_class_accuracy.items():
            self.logger.info(f"  Class {class_idx} Accuracy: {acc:.4f}")
        
        return results
    
    def train_final_model(self,
                         architecture_name: str,
                         best_hyperparams: Dict[str, Any],
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader: DataLoader,
                         model_factory: Any,
                         save_path: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train final model with best hyperparameters
        
        Returns:
            (trained_model, training_results)
        """
        
        self.logger.info(f"Training final model: {architecture_name}")
        self.logger.info(f"Best hyperparameters: {best_hyperparams}")
        
        # Create model
        model = model_factory.create_model(
            architecture_name,
            pretrained=best_hyperparams.get('pretrained', True),
            dropout_rate=best_hyperparams.get('dropout_rate', 0.0)
        )
        model = model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer(model, best_hyperparams)
        scheduler = self._create_scheduler(optimizer, best_hyperparams)
        criterion = nn.CrossEntropyLoss()
        
        # Extended training for final model
        final_epochs = min(
            best_hyperparams.get('final_epochs', self.max_epochs * 2),
            self.max_epochs * 2
        )
        
        best_val_accuracy = 0.0
        best_model_state = None
        training_history = []
        
        self.logger.info(f"Training for {final_epochs} epochs...")
        
        # Training loop
        for epoch in range(1, final_epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader, criterion, epoch)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Track best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
            
            # Record history
            epoch_record = {
                'epoch': epoch,
                'train_accuracy': train_metrics['accuracy'],
                'train_loss': train_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            training_history.append(epoch_record)
            
            # Log progress
            if epoch % 10 == 0 or epoch == final_epochs:
                self.logger.info(
                    f"Epoch {epoch}/{final_epochs} - "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Best Val Acc: {best_val_accuracy:.4f}"
                )
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        test_results = self.evaluate_model(model, test_loader)
        
        # Save model if path provided
        if save_path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture_name': architecture_name,
                'hyperparameters': best_hyperparams,
                'training_history': training_history,
                'test_results': test_results
            }, save_path)
            self.logger.info(f"Final model saved to {save_path}")
        
        results = {
            'architecture_name': architecture_name,
            'best_hyperparameters': best_hyperparams,
            'best_val_accuracy': best_val_accuracy,
            'test_results': test_results,
            'training_history': training_history
        }
        
        return model, results

# Test the training engine
if __name__ == "__main__":
    # Simple test
    config = AutoMLConfig()
    trainer = Trainer(config)
    
    print("Training engine initialized successfully!")
    print(f"Device: {trainer.device}")
    print(f"Max epochs: {trainer.max_epochs}")
    print("Ready for integration with AutoML pipeline!")
