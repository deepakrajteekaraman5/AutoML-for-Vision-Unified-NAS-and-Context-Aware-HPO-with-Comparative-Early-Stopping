# src/automl/run_automl.py
"""
Main execution script for AutoML Pipeline
Entry point for running the complete AutoML system
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# FIXED: Unicode encoding setup for Windows
if sys.platform.startswith('win'):
    # Set console encoding to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    # Set environment variable for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels to get to project root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import AutoML components
from automl.automl import AutoMLPipeline
from automl.utils import AutoMLConfig, setup_logging, set_seed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run AutoML Pipeline for Image Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on emotions dataset with default settings
  python -m src.automl.run_automl --dataset emotions
  
  # Run with custom time budget and specific architectures
  python -m src.automl.run_automl --dataset emotions --time_budget 12 --architectures resnet18 efficientnet_b0
  
  # Run with debug logging
  python -m src.automl.run_automl --dataset emotions --log_level DEBUG
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['emotions', 'fashion', 'flowers'],
        default='emotions',
        help='Dataset to use for training (default: emotions)'
    )
    
    # Data directory
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Root directory containing dataset (default: data)'
    )
    
    # Time budget
    parser.add_argument(
        '--time_budget',
        type=float,
        default=24.0,
        help='Time budget in hours (default: 24.0)'
    )
    
    # Architecture selection
    parser.add_argument(
        '--architectures',
        nargs='+',
        choices=['resnet18', 'resnet34', 'efficientnet_b0', 'efficientnet_b1', 
                'convnext_tiny', 'mobilenetv3_small_100', 'densenet121'],
        help='Specific architectures to evaluate (default: auto-select based on dataset)'
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Logging
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # Output directories
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory to save results (default: ./results)'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints (default: ./checkpoints)'
    )
    
    # Device selection
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for training (default: auto)'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run in quick test mode (reduced time budget and epochs)'
    )
    
    return parser.parse_args()

def setup_dataset_config(dataset_name: str) -> dict:
    """Setup dataset-specific configuration"""
    
    dataset_configs = {
        'emotions': {
            'dataset_name': 'emotions',
            'num_classes': 7,
            'channels': 1,  # Grayscale
            'image_size': 48,
            'csv_file': 'emotions/train.csv'
        },
        'fashion': {
            'dataset_name': 'fashion',
            'num_classes': 10,
            'channels': 1,  # Grayscale
            'image_size': 28,
            'csv_file': 'fashion/train.csv'
        },
        'flowers': {
            'dataset_name': 'flowers',
            'num_classes': 5,
            'channels': 3,  # Color
            'image_size': 224,
            'csv_file': 'flowers/train.csv'
        }
    }
    
    return dataset_configs[dataset_name]

def validate_environment(args):
    """Validate that the environment is properly set up"""
    
    print("Validating environment...")

    # Check data directory
    data_path = Path(args.data_root).resolve()
    print(f"[DEBUG] Checking data path: {data_path} (Exists? {data_path.exists()})")

    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        print("Please ensure the data directory exists and contains your dataset.")
        return False

    # Check dataset CSV file
    dataset_config = setup_dataset_config(args.dataset)
    csv_file = data_path / dataset_config['csv_file']
    print(f"[DEBUG] Checking CSV file: {csv_file} (Exists? {csv_file.exists()})")

    if not csv_file.exists():
        print(f"ERROR: Dataset CSV file not found: {csv_file}")
        print(f"Please ensure {dataset_config['csv_file']} exists in the data directory.")
        return False

    # Check PyTorch installation
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, will use CPU")
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False

    # Check other dependencies
    required_packages = ['timm', 'optuna', 'albumentations', 'pandas', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} - OK")
        except ImportError:
            print(f"ERROR: {package} not installed")
            return False

    print("Environment validation complete")
    return True


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Print banner (removed emojis for Windows compatibility)
    print("="*80)
    print("AutoML Pipeline for Image Classification")
    print("   Intelligent Architecture Search with Adaptive HPO Selection")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Time Budget: {args.time_budget} hours")
    print(f"Random Seed: {args.seed}")
    print(f"Log Level: {args.log_level}")
    print("="*80)
    
    # Validate environment
    if not validate_environment(args):
        print("ERROR: Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    
    # Setup logging
    log_file = Path(args.results_dir) / 'logs' / 'automl.log'
    logger = setup_logging(level=args.log_level, log_file=str(log_file))
    
    # Set random seed
    set_seed(args.seed)
    
    try:
        # Create configuration
        config = AutoMLConfig()
        
        # Dataset-specific configuration
        dataset_config = setup_dataset_config(args.dataset)
        config.update(**dataset_config)
        
        # Apply command line arguments
        config.set('time_budget_hours', args.time_budget)
        config.set('results_dir', args.results_dir)
        config.set('checkpoint_dir', args.checkpoint_dir)
        config.set('random_seed', args.seed)
        config.set('device', args.device)
        
        # Quick test mode adjustments
        if args.quick_test:
            logger.info("Running in quick test mode")
            config.set('time_budget_hours', min(args.time_budget, 2.0))
            config.set('max_epochs_per_architecture', 10)
            config.set('final_training_epochs', 20)
        
        # Create and run pipeline
        logger.info("Initializing AutoML Pipeline...")
        pipeline = AutoMLPipeline(config)
        
        logger.info("Starting AutoML execution...")
        results = pipeline.run(
            dataset_root=args.data_root,
            architectures=args.architectures,
            save_results=True
        )
        
        # Print success message
        print("\n" + "="*80)
        print("AUTOML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if results['best_model']['architecture']:
            print(f"Best Model: {results['best_model']['architecture']}")
            print(f"Test Accuracy: {results['best_model']['test_accuracy']:.4f}")
            print(f"Total Time: {results['pipeline_config']['total_execution_time_hours']:.2f} hours")
        else:
            print("No models completed training")
        
        print(f"Results saved to: {args.results_dir}")
        print(f"Models saved to: {args.checkpoint_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\nExecution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)