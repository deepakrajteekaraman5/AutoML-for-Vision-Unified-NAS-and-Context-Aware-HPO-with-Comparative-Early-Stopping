# src/automl/run_automl.py
"""
Main execution script for AutoML Pipeline
Entry point for running the complete AutoML system
UPDATED: Enhanced quick test mode with reduced trials and better progress tracking
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import time

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
from automl.utils import AutoMLConfig, setup_logging, set_seed, format_time_remaining, print_performance_summary

def parse_arguments():
    """Parse command line arguments - ENHANCED WITH QUICK TEST OPTIONS"""
    parser = argparse.ArgumentParser(
        description='Run AutoML Pipeline for Image Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (recommended for first run)
  python -m src.automl.run_automl --dataset emotions --quick_test
  
  # Run on emotions dataset with default settings
  python -m src.automl.run_automl --dataset emotions
  
  # Run with custom time budget and specific architectures
  python -m src.automl.run_automl --dataset emotions --time_budget 12 --architectures resnet18 efficientnet_b0
  
  # Run with debug logging
  python -m src.automl.run_automl --dataset emotions --log_level DEBUG
  
  # Ultra-quick test (5 trials per architecture, 2 architectures)
  python -m src.automl.run_automl --dataset emotions --quick_test --ultra_quick
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['emotions', 'fashion', 'flowers', 'skin_cancer'],
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
    
    # NEW: Enhanced quick test options
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run in quick test mode (1 hour, 8 trials per arch, 2 architectures)'
    )
    
    parser.add_argument(
        '--ultra_quick',
        action='store_true',
        help='Ultra-quick test mode (30 min, 5 trials per arch, 2 architectures). Requires --quick_test'
    )
    
    # NEW: HPO trial configuration
    parser.add_argument(
        '--hpo_trials',
        type=int,
        help='Number of HPO trials per architecture (overrides defaults)'
    )
    
    parser.add_argument(
        '--max_epochs',
        type=int,
        help='Maximum epochs per architecture (overrides defaults)'
    )
    
    # NEW: Visualization options
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plotting and visualization'
    )
    
    parser.add_argument(
        '--plot_format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Format for saved plots (default: png)'
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
            'num_classes': 102,  # FIXED: Was 5, now 102
            'channels': 3,  # Color
            'image_size': 512,  # FIXED: Was 224, now 512
            'csv_file': 'flowers/train.csv'
        },
        'skin_cancer': {
            'dataset_name': 'skin_cancer',
            'num_classes': 7,
            'channels': 3,  # Color
            'image_size': 450,
            'csv_file': 'skin_cancer/train.csv'
        }
    }
    
    return dataset_configs[dataset_name]

def validate_environment(args):
    """Validate that the environment is properly set up"""
    
    print("🔍 Validating environment...")

    # Check data directory
    data_path = Path(args.data_root).resolve()
    print(f" Checking data path: {data_path}")

    if not data_path.exists():
        print(f" ERROR: Data directory not found: {data_path}")
        print("Please ensure the data directory exists and contains your dataset.")
        return False

    # Check dataset CSV file
    dataset_config = setup_dataset_config(args.dataset)
    csv_file = data_path / dataset_config['csv_file']
    print(f" Checking CSV file: {csv_file}")

    if not csv_file.exists():
        print(f" ERROR: Dataset CSV file not found: {csv_file}")
        print(f"Please ensure {dataset_config['csv_file']} exists in the data directory.")
        return False

    # Check PyTorch installation
    try:
        import torch
        print(f" PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f" CUDA available: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f" GPU Memory: {memory_gb:.1f}GB")
        else:
            print(" CUDA not available, will use CPU")
    except ImportError:
        print(" ERROR: PyTorch not installed")
        return False

    # Check other dependencies
    required_packages = ['timm', 'optuna', 'albumentations', 'pandas', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package}")
        except ImportError:
            print(f" ERROR: {package} not installed")
            return False

    print(" Environment validation complete")
    return True

def apply_quick_test_configuration(config: AutoMLConfig, args):
    """Apply quick test configurations - FIXED VERSION"""
    
    if args.quick_test:
        print(" Enabling quick test mode...")
        config.enable_quick_test_mode()
        
        if args.ultra_quick:
            print(" Enabling ultra-quick mode...")
            # FIXED: Use individual set() calls instead of update()
            config.set('time_budget_hours', 0.5)      # 30 minutes
            config.set('hpo_base_trials', 5)          # 5 trials per arch
            config.set('max_epochs_per_architecture', 8)  # 8 epochs max
            config.set('min_epochs_per_architecture', 3)  # 3 epochs min
    
    # Apply command line overrides
    if args.hpo_trials:
        config.set('hpo_base_trials', args.hpo_trials)
        print(f"🔧 HPO trials set to: {args.hpo_trials}")
    
    if args.max_epochs:
        config.set('max_epochs_per_architecture', args.max_epochs)
        print(f"🔧 Max epochs set to: {args.max_epochs}")
    
    # Visualization settings
    if args.no_plots:
        config.set('enable_plotting', False)
        print(" Plotting disabled")
    else:
        config.set('plot_format', args.plot_format)
        print(f" Plot format: {args.plot_format}")

def estimate_runtime(config: AutoMLConfig, num_architectures: int) -> str:
    """NEW: Estimate total runtime based on configuration"""
    
    trials_per_arch = config.get('hpo_base_trials')
    epochs_per_trial = config.get('max_epochs_per_architecture') // 2  # Rough estimate
    
    # Rough timing estimates (very approximate)
    if config.get('dataset_name') == 'emotions':
        seconds_per_epoch = 10  # Small images, fast training
    elif config.get('dataset_name') == 'fashion':
        seconds_per_epoch = 8   # Very small images
    elif config.get('dataset_name') == 'skin_cancer':
        seconds_per_epoch = 25  # Medium-large images
    else:  # flowers
        seconds_per_epoch = 30  # Larger images
    
    # Calculate estimated time
    total_trials = num_architectures * trials_per_arch
    total_epochs = total_trials * epochs_per_trial
    estimated_seconds = total_epochs * seconds_per_epoch
    
    # Add overhead (30%)
    estimated_seconds *= 1.3
    
    return format_time_remaining(estimated_seconds)

def print_execution_plan(config: AutoMLConfig, architectures: list):
    """NEW: Print detailed execution plan"""
    
    print("\n" + "="*60 )
    print("EXECUTION PLAN")
    print("="*62)
    
    mode = "QUICK TEST" if config.get('quick_test') else "FULL PIPELINE"
    if config.get('quick_test') and config.get('time_budget_hours') <= 0.5:
        mode = "ULTRA-QUICK TEST"
    
    print(f" Mode: {mode}")
    print(f" Dataset: {config.get('dataset_name')} ({config.get('num_classes')} classes)")
    print(f" Time Budget: {config.get('time_budget_hours')} hours")
    print(f" Architectures: {len(architectures)} ({', '.join(architectures)})")
    print(f" HPO Trials per Architecture: {config.get('hpo_base_trials')}")
    print(f" Max Epochs per Trial: {config.get('max_epochs_per_architecture')}")
    
    # Estimate runtime
    estimated_runtime = estimate_runtime(config, len(architectures))
    print(f" Estimated Runtime: {estimated_runtime}")
    
    # Calculate total trials
    total_trials = len(architectures) * config.get('hpo_base_trials')
    print(f" Total HPO Trials: {total_trials}")
    
    print("="*62)
    
    # Ask for confirmation if long runtime
    if not config.get('quick_test') and len(architectures) > 3:
        response = input("\n This is a full pipeline run that may take several hours. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print(" Execution cancelled. Use --quick_test for faster testing.")
            sys.exit(0)

def main():
    """Main execution function - ENHANCED WITH BETTER UX"""
    
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate ultra_quick requires quick_test
    if args.ultra_quick and not args.quick_test:
        print(" ERROR: --ultra_quick requires --quick_test")
        sys.exit(1)
    
    # Print banner
    print("="*78)
    print("  ╔═══════════════════════════════════════════════════════════════════════════╗")
    print("  ║                    AutoML Pipeline for Image Classification               ║")
    print("  ║              Intelligent Architecture Search with Adaptive HPO           ║")
    print("  ╚═══════════════════════════════════════════════════════════════════════════╝")
    print("="*80)
    
    # Show basic info
    mode_indicator = " QUICK TEST" if args.quick_test else " FULL PIPELINE"
    if args.ultra_quick:
        mode_indicator = " ULTRA-QUICK"
    
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {mode_indicator}")
    print(f"Time Budget: {args.time_budget} hours")
    print(f"Random Seed: {args.seed}")
    print(f"Log Level: {args.log_level}")
    print("="*80)
    
    # Validate environment
    if not validate_environment(args):
        print(" Environment validation failed. Please fix the issues above.")
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
        
        # Apply quick test configurations
        apply_quick_test_configuration(config, args)
        
        # Determine architectures to use
        if args.architectures:
            selected_architectures = args.architectures
        elif config.get('quick_test'):
            selected_architectures = config.get_quick_test_architectures()
        else:
            selected_architectures = None  # Let pipeline auto-select
        
        # Print execution plan and get confirmation
        if selected_architectures:
            print_execution_plan(config, selected_architectures)
        else:
            print("\n Architecture selection will be determined automatically based on dataset characteristics")
        
        # Print configuration summary
        config.print_configuration_summary()
        
        # Create and run pipeline
        logger.info(" Initializing AutoML Pipeline...")
        pipeline = AutoMLPipeline(config)
        
        logger.info(" Starting AutoML execution...")
        print(f"\n Starting AutoML pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = pipeline.run(
            dataset_root=args.data_root,
            architectures=selected_architectures,
            save_results=True
        )
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Print success message
        print("\n" + "="*78 )
        print("AUTOML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print performance summary
        print_performance_summary(results)
        
        # Additional success details
        print(f"\n Results Summary:")
        if results['best_model']['architecture']:
            print(f"    Best Architecture: {results['best_model']['architecture']}")
            print(f"    Test Accuracy: {results['best_model']['test_accuracy']:.4f}")
            print(f"    Pipeline Time: {results['pipeline_config']['total_execution_time_hours']:.2f}h")
            print(f"    Architectures Evaluated: {len(results.get('architecture_search_results', {}))}")
        else:
            print("    No models completed training")
            print("    Try running with --quick_test for faster iteration")
        
        print(f"\n Results saved to: {args.results_dir}")
        print(f" Models saved to: {args.checkpoint_dir}")
        print(f" Total Runtime: {format_time_remaining(total_time)}")
        
        # Suggest next steps
        if args.quick_test:
            print(f"\n Next Steps:")
            print(f"   • Review results in {args.results_dir}")
            print(f"   • Run full pipeline: python -m src.automl.run_automl --dataset {args.dataset}")
            print(f"   • Try different datasets: --dataset flowers or --dataset fashion")
        
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        logger.info(" Execution interrupted by user")
        print(f"\n Execution interrupted by user after {format_time_remaining(elapsed)}")
        print(" You can resume with the same parameters to continue from checkpoints")
        return 1
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f" Pipeline failed: {e}")
        print(f"\n Pipeline failed after {format_time_remaining(elapsed)}: {e}")
        
        # Print debugging help
        print(f"\n Debugging Help:")
        print(f"   • Check logs: {log_file}")
        print(f"   • Try quick test: --quick_test")
        print(f"   • Reduce trials: --hpo_trials 5")
        print(f"   • Enable debug logging: --log_level DEBUG")
        
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
