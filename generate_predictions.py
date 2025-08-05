#!/usr/bin/env python3
"""
Generate predictions for skin cancer test set
For Phase II submission to obtain test score
"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir
src_path = project_root / "automl-exam-ss25-vision-freiburg-template" / "src"
sys.path.insert(0, str(src_path))

from automl.datasets import SkinCancerDataset
from automl.models import ModelFactory
from automl.utils import AutoMLConfig
from automl.data_manager import AlbumentationsWrapper
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_test_transform(image_size: int = 450):
    """Create test-time transforms (no augmentation)"""
    transform = A.Compose([
        A.Resize(image_size, image_size, interpolation=1),  # INTER_LINEAR
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ToTensorV2()
    ])
    return AlbumentationsWrapper(transform)

def load_trained_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model info
    architecture_name = checkpoint['architecture_name']
    hyperparams = checkpoint.get('hyperparameters', {})
    
    print(f"Architecture: {architecture_name}")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create model factory
    config = AutoMLConfig()
    config.set('dataset_name', 'skin_cancer')
    config.set('num_classes', 7)
    config.set('channels', 3)
    config.set('image_size', 450)
    
    model_factory = ModelFactory(config)
    
    # Create model
    model = model_factory.create_model(
        architecture_name,
        pretrained=False,  # Don't load pretrained weights, we'll load our trained weights
        dropout_rate=hyperparams.get('dropout_rate', 0.0)
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, architecture_name

def generate_predictions(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Generate predictions on test set"""
    print("Generating predictions on test set...")
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            images = images.to(device)
            images = images.float()
            
            # Forward pass
            outputs = model(images)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)

def create_submission_file(predictions: np.array, 
                          test_csv_path: str, 
                          output_path: str,
                          probabilities: np.array = None):
    """Create submission file with predictions"""
    print(f"Creating submission file: {output_path}")
    
    # Load test CSV to get image file names
    test_df = pd.read_csv(test_csv_path)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'image_file_name': test_df['image_file_name'],
        'label': predictions
    })
    
    # Save submission file
    submission_df.to_csv(output_path, index=False)
    
    # Save predictions as .npy file
    npy_output_path = output_path.replace('.csv', '.npy')
    np.save(npy_output_path, predictions)
    print(f"✅ Predictions saved as numpy array to: {npy_output_path}")
    
    print(f"✅ Submission file created with {len(predictions)} predictions")
    print(f"Prediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(predictions)*100:.1f}%)")
    
    # Optionally save probabilities for analysis
    if probabilities is not None:
        prob_output_path = output_path.replace('.csv', '_probabilities.csv')
        prob_df = pd.DataFrame(probabilities, columns=[f'prob_class_{i}' for i in range(probabilities.shape[1])])
        prob_df['image_file_name'] = test_df['image_file_name']
        prob_df['predicted_label'] = predictions
        prob_df.to_csv(prob_output_path, index=False)
        print(f"✅ Probabilities saved to: {prob_output_path}")

def main():
    """Main prediction generation function"""
    parser = argparse.ArgumentParser(description='Generate predictions for skin cancer test set')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='automl-exam-ss25-vision-freiburg-template/data',
                       help='Root directory containing dataset')
    parser.add_argument('--output', type=str, default='skin_cancer_predictions.csv',
                       help='Output CSV file for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction')
    parser.add_argument('--image_size', type=int, default=450,
                       help='Image size for preprocessing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SKIN CANCER TEST PREDICTION GENERATOR")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print("="*60)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ ERROR: Checkpoint file not found: {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = Path("automl-exam-ss25-vision-freiburg-template/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pt"):
                print(f"  {ckpt}")
        return 1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load trained model
        model, architecture_name = load_trained_model(args.checkpoint, device)
        
        # Create test dataset and dataloader
        print("Loading test dataset...")
        test_transform = create_test_transform(args.image_size)
        
        test_dataset = SkinCancerDataset(
            root=args.data_root,
            split='test',
            transform=test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Important: don't shuffle for consistent ordering
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
        # Generate predictions
        predictions, probabilities = generate_predictions(model, test_loader, device)
        
        # Create submission file
        test_csv_path = Path(args.data_root) / "skin_cancer" / "test.csv"
        create_submission_file(predictions, test_csv_path, args.output, probabilities)
        
        print("\n" + "="*60)
        print("PREDICTION GENERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Submission file: {args.output}")
        print(f"🎯 Total predictions: {len(predictions)}")
        print(f"🏗️ Model architecture: {architecture_name}")
        print("\nNext steps:")
        print("1. Review the prediction distribution above")
        print("2. Submit the CSV file via GitHub for grading")
        print("3. Check probabilities file for confidence analysis")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: Prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
