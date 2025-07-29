# src/automl/automl.py
"""
Main AutoML Pipeline Orchestrator
Intelligent AutoML Pipeline for Image Classification with Adaptive Architecture Search
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np

from .utils import AutoMLConfig, Timer, setup_logging, set_seed, ensure_dir
from .data_manager import AutoMLDataManager
from .models import ModelFactory
from .training import Trainer
from .early_stopping import ComparativeEarlyStopping
from .hpo_selection import MetaHPOSelector
from .budget_manager import BudgetManager

class AutoMLPipeline:
    """
    Main AutoML Pipeline Orchestrator
    
    Core Innovation: Intelligent AutoML pipeline that combines Neural Architecture Search (NAS) 
    with intelligent Hyperparameter Optimization (HPO) method selection, featuring comparative 
    early stopping based on cross-architecture performance analysis.
    
    Key Features:
    - Comparative Early Stopping: Stop architectures based on performance comparison
    - Meta-HPO Selection: Context-aware automatic selection of optimal HPO methods
    - Dynamic Resource Allocation: Intelligent budget reallocation from poor to promising architectures
    - Integrated NAS + HPO: Unified pipeline combining architecture search with hyperparameter optimization
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = logging.getLogger('AutoML.Pipeline')
        
        # Initialize all components
        self.data_manager = AutoMLDataManager(config)
        self.model_factory = ModelFactory(config)
        self.trainer = Trainer(config)
        self.budget_manager = BudgetManager(config)
        self.early_stopping = ComparativeEarlyStopping(config)
        self.hpo_selector = MetaHPOSelector(config)
        
        # Pipeline state
        self.dataset_info = None
        self.architecture_results = {}
        self.final_model = None
        self.final_results = None
        
        # Results tracking
        self.execution_log = []
        self.decision_log = []
        
        self.logger.info("=== AutoML Pipeline Initialized ===")
        self.logger.info(f"Configuration: {config.get('dataset_name')} dataset")
        self.logger.info(f"Time budget: {config.get('time_budget_hours')} hours")
        self.logger.info(f"Target: {config.get('num_classes')} classes")
    
    def run(self, 
            dataset_root: str = "data",
            architectures: Optional[List[str]] = None,
            save_results: bool = True) -> Dict[str, Any]:
        """
        Main execution pipeline
        
        Args:
            dataset_root: Root directory containing dataset
            architectures: List of architectures to evaluate (None = use all strategic models)
            save_results: Whether to save results and models
            
        Returns:
            Dictionary containing comprehensive results
        """
        
        self.logger.info("Starting AutoML Pipeline Execution")
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: Dataset Analysis and Preparation
            self.logger.info("=== Phase 1: Dataset Analysis and Preparation ===")
            self.dataset_info = self._phase1_dataset_preparation(dataset_root)
            
            # Phase 2: Architecture Selection and Budget Allocation
            self.logger.info("=== Phase 2: Architecture Selection and Budget Allocation ===")
            selected_architectures = self._phase2_architecture_selection(architectures)
            
            # Phase 3: Intelligent Architecture Search with HPO
            self.logger.info("=== Phase 3: Intelligent Architecture Search ===")
            architecture_results = self._phase3_architecture_search(selected_architectures)
            
            # Phase 4: Final Training
            self.logger.info("=== Phase 4: Final Training ===")
            final_results = self._phase4_final_training()
            
            # Phase 5: Results Analysis and Reporting
            self.logger.info("=== Phase 5: Results Analysis ===")
            comprehensive_results = self._phase5_results_analysis(final_results)
            
            # Save results
            if save_results:
                self._save_results(comprehensive_results)
            
            total_time = time.time() - pipeline_start_time
            self.logger.info(f"AutoML Pipeline completed in {total_time/3600:.2f} hours")
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _phase1_dataset_preparation(self, dataset_root: str) -> Dict[str, Any]:
        """Phase 1: Analyze dataset and prepare data loaders"""
        
        with Timer("Dataset preparation") as timer:
            # Setup datasets and get characteristics
            dataset_info = self.data_manager.setup_datasets(
                root=dataset_root,
                val_split=self.config.get('validation_split', 0.2)
            )
            
            # Log dataset characteristics
            characteristics = dataset_info['characteristics']
            self.logger.info(f"Dataset Analysis Complete:")
            self.logger.info(f"  Name: {characteristics['name']}")
            self.logger.info(f"  Samples: {characteristics['num_samples']:,}")
            self.logger.info(f"  Classes: {characteristics['num_classes']}")
            self.logger.info(f"  Image size: {characteristics['image_width']}x{characteristics['image_height']}x{characteristics['channels']}")
            self.logger.info(f"  Complexity: {characteristics['complexity_score']:.1f}/10")
            self.logger.info(f"  Memory per sample: {characteristics['memory_per_sample_mb']:.3f} MB")
            
            self.execution_log.append({
                'phase': 'dataset_preparation',
                'duration_seconds': timer.elapsed,
                'dataset_info': dataset_info
            })
        
        return dataset_info
    
    def _phase2_architecture_selection(self, architectures: Optional[List[str]]) -> List[str]:
        """Phase 2: Select architectures and allocate budget"""
        
        with Timer("Architecture selection") as timer:
        # 1) Choose your candidates
        if architectures is None:
            # Select architectures based on dataset characteristics
            complexity = self.dataset_info['characteristics']['complexity_score']
            num_samples = self.dataset_info['characteristics']['num_samples']
            
            if num_samples < 10000:  # Small dataset
                selected_architectures = ['resnet18', 'efficientnet_b0', 'mobilenetv3_small_100']
            elif complexity > 6.0:   # Complex dataset
                selected_architectures = ['resnet34', 'efficientnet_b1', 'convnext_tiny', 'densenet121']
            else:                    # Medium complexity
                selected_architectures = ['resnet18', 'efficientnet_b0', 'convnext_tiny', 'densenet121']
        else:
            selected_architectures = architectures
        
        self.logger.info(f"Selected architectures (pre–ZeroCost): {selected_architectures}")
        
        
        # 2) Zero‑Cost NAS filtering
        try:
            from .zero_cost_nas import score_model
            zero_cost_method = self.config.get('zero_cost_method', 'synflow')
            top_k = self.config.get('zero_cost_top_k', len(selected_architectures))
            scores = []
            for arch in selected_architectures:
                # build model factory function
                model_fn = lambda name=arch: self.model_factory.create_model(name)
                # determine dummy input shape from your dataset
                c = self.dataset_info['characteristics']['channels']
                h = self.dataset_info['characteristics']['image_height']
                w = self.dataset_info['characteristics']['image_width']
                score = score_model(model_fn,
                                    method=zero_cost_method,
                                    input_shape=(1, c, h, w))
                scores.append((arch, score))
                self.logger.debug(f"Zero‑Cost score [{zero_cost_method}] for {arch}: {score:.4f}")
            # keep only top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            selected_architectures = [arch for arch, _ in scores[:top_k]]
            self.logger.info(f"Architectures after Zero‑Cost NAS filter (top {top_k}): {selected_architectures}")
        except Exception as e:
            self.logger.warning(f"Zero‑Cost NAS filtering failed, continuing with full set: {e}")
        
        
        # 3) Register for early stopping & budget allocation
        self.logger.info(f"Final architectures for HPO & training: {selected_architectures}")
        for arch in selected_architectures:
            self.early_stopping.register_architecture(arch)
        
        # Start budget allocation
        self.budget_manager.start_execution(selected_architectures)
        # Print initial budget status
        self.budget_manager.print_budget_status()
        
        self.execution_log.append({
            'phase': 'architecture_selection',
            'duration_seconds': timer.elapsed,
            'selected_architectures': selected_architectures
        })
    
        return selected_architectures
    
    def _phase3_architecture_search(self, architectures: List[str]) -> Dict[str, Any]:
        """Phase 3: Run architecture search with intelligent HPO - FIXED VERSION"""
        
        with Timer("Architecture search") as timer:
            # Create data loaders (will be reused for all architectures)
            self.logger.info("Creating data loaders...")
            train_loader, val_loader, test_loader = self.data_manager.get_dataloaders(
                batch_size=None,  # Will use recommended batch size
                image_size=None,  # Will use dataset's native size
                augmentation_strategy='auto',
                num_workers=2
            )
            
            # Store for later use
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Validation batches: {len(val_loader)}")
            self.logger.info(f"  Test batches: {len(test_loader)}")
            
            # Define hyperparameter search space
            search_space = self._get_hyperparameter_search_space()
            
            # Architecture search results
            architecture_results = {}
            
            # FIXED: Process each architecture sequentially
            for arch_name in architectures:
                self.logger.info(f"Processing architecture: {arch_name}")
                
                # Check if we should transition to final training before processing more architectures
                if self.budget_manager.should_start_final_training():
                    self.logger.info("Time budget reached, transitioning to final training")
                    break
                
                # Check if architecture can start training
                if not self.budget_manager.start_architecture_training(arch_name):
                    self.logger.warning(f"Cannot start training {arch_name} - resource constraints")
                    continue
                
                self.logger.info(f"Starting HPO for {arch_name}")
                
                # Get architecture characteristics for HPO selection
                arch_characteristics = self.model_factory.get_model_characteristics(arch_name)
                dataset_complexity = self.dataset_info['characteristics']['complexity_score']
                available_time = self.budget_manager.get_phase_remaining_time_hours()
                
                # Create objective function for this architecture
                objective_function = self.trainer.create_objective_function(
                    arch_name, train_loader, val_loader, self.model_factory,
                    early_stopping_engine=self.early_stopping,
                    budget_manager=self.budget_manager
                )
                
                # Run HPO for this architecture
                hpo_result = self.hpo_selector.optimize_hyperparameters(
                    arch_name,
                    objective_function,
                    search_space,
                    arch_characteristics,
                    dataset_complexity,
                    available_time
                )
                
                # Store results
                architecture_results[arch_name] = hpo_result
                self.budget_manager.completed_architectures.add(arch_name)

                self.logger.info(f"HPO completed for {arch_name}:")
                self.logger.info(f"  Best score: {hpo_result.best_score:.4f}")
                self.logger.info(f"  Best params: {hpo_result.best_params}")
                self.logger.info(f"  Method used: {hpo_result.method_used.value}")
                self.logger.info(f"  Trials completed: {hpo_result.n_trials_completed}")
                
                # Check for early stopping
                should_stop, reason, confidence = self.early_stopping.should_stop_architecture(arch_name)
                if should_stop:
                    self.early_stopping.stop_architecture(arch_name, reason, confidence)
                    freed_time = self.budget_manager.architecture_stopped_early(
                        arch_name, reason.value, hpo_result.best_score
                    )
                    
                    self.decision_log.append({
                        'type': 'early_stopping',
                        'architecture': arch_name,
                        'reason': reason.value,
                        'confidence': confidence,
                        'freed_time_hours': freed_time
                    })
                    
                    self.logger.info(f"Stopped {arch_name} early: {reason.value}")
                
                # Print current budget status
                self.budget_manager.print_budget_status()
            
            self.execution_log.append({
                'phase': 'architecture_search',
                'duration_seconds': timer.elapsed,
                'architecture_results': {k: {
                    'best_score': v.best_score,
                    'method_used': v.method_used.value,
                    'n_trials': v.n_trials_completed
                } for k, v in architecture_results.items()}
            })
        
        self.architecture_results = architecture_results
        return architecture_results
    
    def _phase4_final_training(self) -> Dict[str, Any]:
        """Phase 4: Final training of selected architectures - FIXED VERSION"""
        
        with Timer("Final training") as timer:
            # Select candidates for final training
            final_candidates = self.budget_manager.start_final_training_phase()
            
            if not final_candidates:
                self.logger.warning("No candidates selected for final training")
                return {}
            
            self.logger.info(f"Final training candidates: {final_candidates}")
            
            # FIXED: Ensure checkpoint directory exists
            checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
            ensure_dir(checkpoint_dir)
            
            final_results = {}
            best_overall_score = 0.0
            best_model_info = None
            
            for arch_name in final_candidates:
                if arch_name not in self.architecture_results:
                    self.logger.warning(f"No HPO results for {arch_name}, skipping final training")
                    continue
                
                self.logger.info(f"Final training for {arch_name}")
                
                # Get best hyperparameters from HPO
                hpo_result = self.architecture_results[arch_name]
                best_hyperparams = hpo_result.best_params.copy()
                
                # Extended training for final model
                best_hyperparams['final_epochs'] = self.config.get('final_training_epochs', 100)
                
                # FIXED: Create checkpoint path with proper directory
                checkpoint_path = checkpoint_dir / f"final_{arch_name}.pt"
                
                # Train final model
                final_model, training_results = self.trainer.train_final_model(
                    arch_name,
                    best_hyperparams,
                    self.train_loader,
                    self.val_loader,
                    self.test_loader,
                    self.model_factory,
                    save_path=str(checkpoint_path)  # Convert Path to string
                )
                
                final_results[arch_name] = training_results
                
                # Track best overall model
                test_accuracy = training_results['test_results']['test_accuracy']
                if test_accuracy > best_overall_score:
                    best_overall_score = test_accuracy
                    best_model_info = {
                        'architecture': arch_name,
                        'model': final_model,
                        'results': training_results
                    }
                
                self.logger.info(f"{arch_name} final training completed:")
                self.logger.info(f"  Test accuracy: {test_accuracy:.4f}")
                self.logger.info(f"  Best val accuracy: {training_results['best_val_accuracy']:.4f}")
            
            # Store best model - FIXED: Proper structure
            if best_model_info:
                self.final_model = best_model_info['model']
                self.final_results = {
                    'best_model': {
                        'architecture': best_model_info['architecture'],
                        'model': best_model_info['model'],
                        'results': best_model_info['results']
                    }
                }
                
                self.logger.info(f"Best overall model: {best_model_info['architecture']}")
                self.logger.info(f"Best test accuracy: {best_overall_score:.4f}")
            else:
                # FIXED: Ensure structure exists even if no models trained
                self.final_results = {
                    'best_model': {
                        'architecture': None,
                        'model': None,
                        'results': None
                    }
                }
            
            self.execution_log.append({
                'phase': 'final_training',
                'duration_seconds': timer.elapsed,
                'final_candidates': final_candidates,
                'best_architecture': best_model_info['architecture'] if best_model_info else None,
                'best_test_accuracy': best_overall_score
            })
        
        return final_results
    
    def _phase5_results_analysis(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Comprehensive results analysis and reporting"""
        
        with Timer("Results analysis") as timer:
            # Compile comprehensive results
            comprehensive_results = {
                'pipeline_config': {
                    'dataset_name': self.config.get('dataset_name'),
                    'time_budget_hours': self.config.get('time_budget_hours'),
                    'architectures_evaluated': list(self.architecture_results.keys()),
                    'total_execution_time_hours': sum(log['duration_seconds'] for log in self.execution_log) / 3600.0
                },
                'dataset_analysis': self.dataset_info,
                'architecture_search_results': {
                    arch: {
                        'hpo_method_used': result.method_used.value,
                        'best_score': result.best_score,
                        'best_hyperparameters': result.best_params,
                        'trials_completed': result.n_trials_completed,
                        'optimization_time': result.optimization_time,
                        'efficiency': result.method_efficiency
                    }
                    for arch, result in self.architecture_results.items()
                },
                'final_training_results': final_results,
                'best_model': {
                    'architecture': self.final_results.get('best_model', {}).get('architecture'),
                    'test_accuracy': (
                        self.final_results.get('best_model', {}).get('results', {}).get('test_results', {}).get('test_accuracy', 0.0)
                        if self.final_results and self.final_results.get('best_model', {}).get('results')
                        else 0.0
                    ),
                    'hyperparameters': (
                        self.final_results.get('best_model', {}).get('results', {}).get('best_hyperparameters', {})
                        if self.final_results and self.final_results.get('best_model', {}).get('results')
                        else {}
                    ),
                } if self.final_results else {'architecture': None, 'test_accuracy': 0.0, 'hyperparameters': {}},
                'budget_summary': self.budget_manager.get_budget_summary(),
                'early_stopping_summary': self.early_stopping.get_performance_summary(),
                'hpo_method_performance': self.hpo_selector.get_method_performance_summary(),
                'execution_log': self.execution_log,
                'decision_log': self.decision_log
            }
            
            # Generate insights
            insights = self._generate_insights(comprehensive_results)
            comprehensive_results['insights'] = insights
            
            # Print summary
            self._print_final_summary(comprehensive_results)
            
            self.execution_log.append({
                'phase': 'results_analysis',
                'duration_seconds': timer.elapsed
            })
        
        return comprehensive_results
    
    def _get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space for HPO"""
        
        return {
            # Core training parameters
            'learning_rate': {'type': 'float', 'range': (1e-5, 1e-1), 'log_scale': True},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'adamw']},
            'weight_decay': {'type': 'float', 'range': (1e-6, 1e-2), 'log_scale': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
            
            # Architecture-specific parameters
            'dropout_rate': {'type': 'float', 'range': (0.0, 0.7)},
            'pretrained': {'type': 'categorical', 'choices': [True, False]},
            
            # Training schedule parameters
            'lr_scheduler': {'type': 'categorical', 'choices': ['cosine', 'step', 'plateau']},
            'max_epochs': {'type': 'int', 'range': (20, 50)},
        }
    
    def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from results"""
        
        insights = {
            'dataset_insights': [],
            'architecture_insights': [],
            'hpo_insights': [],
            'recommendations': []
        }
        
        # Dataset insights
        if self.dataset_info:
            complexity = self.dataset_info['characteristics']['complexity_score']
            if complexity < 3.0:
                insights['dataset_insights'].append("Low complexity dataset - simpler models may be sufficient")
            elif complexity > 7.0:
                insights['dataset_insights'].append("High complexity dataset - sophisticated models recommended")
        
        # Architecture insights
        if self.architecture_results:
            best_arch = max(self.architecture_results.items(), key=lambda x: x[1].best_score)
            insights['architecture_insights'].append(f"Best performing architecture: {best_arch[0]} ({best_arch[1].best_score:.4f})")
            
            # HPO method analysis
            hpo_methods_used = [result.method_used.value for result in self.architecture_results.values()]
            most_common_hpo = max(set(hpo_methods_used), key=hpo_methods_used.count)
            insights['hpo_insights'].append(f"Most frequently selected HPO method: {most_common_hpo}")
        
        # Recommendations
        insights['recommendations'].append("Consider ensemble methods for further improvement")
        insights['recommendations'].append("Experiment with advanced augmentation strategies")
        
        return insights
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive final summary - FIXED VERSION"""
        
        print("\n" + "="*80)
        print("AUTOML PIPELINE EXECUTION COMPLETED")
        print("="*80)
        
        # Basic info
        config = results['pipeline_config']
        print(f"Dataset: {config['dataset_name']}")
        print(f"Total execution time: {config['total_execution_time_hours']:.2f} hours")
        print(f"Architectures evaluated: {len(config['architectures_evaluated'])}")
        
        # FIXED: Safe access to best model results
        best_model = results.get('best_model', {})
        if best_model and best_model.get('architecture'):
            print(f"\nBEST MODEL RESULTS:")
            print(f"   Architecture: {best_model['architecture']}")
            print(f"   Test Accuracy: {best_model.get('test_accuracy', 0):.4f}")
            print(f"   Hyperparameters: {best_model.get('hyperparameters', {})}")
        else:
            print(f"\nNO MODELS COMPLETED TRAINING")
            print(f"   Reason: Pipeline stopped before any architecture finished HPO")
        
        # Architecture comparison - only if we have results
        arch_results = results.get('architecture_search_results', {})
        if arch_results:
            print(f"\nARCHITECTURE COMPARISON:")
            for arch, result in arch_results.items():
                print(f"   {arch:20} | Score: {result['best_score']:.4f} | "
                      f"HPO: {result['hpo_method_used']:15} | Trials: {result['trials_completed']:3}")
        else:
            print(f"\nARCHITECTURE COMPARISON: No architectures completed")
        
        # Budget utilization
        budget = results.get('budget_summary', {})
        if budget:
            exec_status = budget.get('execution_status', {})
            phase_breakdown = budget.get('phase_breakdown', {})
            print(f"\nBUDGET UTILIZATION:")
            print(f"   Time used: {exec_status.get('elapsed_hours', 0):.1f}h / "
                  f"{phase_breakdown.get('total_budget_hours', 24):.1f}h "
                  f"({exec_status.get('budget_utilization', 0)*100:.1f}%)")
            
            perf_metrics = budget.get('performance_metrics', {})
            print(f"   Reallocations: {perf_metrics.get('total_reallocations', 0)}")
        
        # Early stopping summary
        es_summary = results.get('early_stopping_summary', {})
        if es_summary:
            print(f"\nEARLY STOPPING SUMMARY:")
            print(f"   Active architectures: {len(es_summary.get('active_architectures', []))}")
            print(f"   Stopped architectures: {len(es_summary.get('stopped_architectures', []))}")
        
        # Insights
        insights = results.get('insights', {})
        if insights and insights.get('recommendations'):
            print(f"\nKEY INSIGHTS:")
            for insight in insights['recommendations']:
                print(f"   • {insight}")
        
        print("="*80)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results"""
        
        # Create results directory
        results_dir = Path(self.config.get('results_dir', './results'))
        ensure_dir(results_dir)
        
        # Save main results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"automl_results_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save final model separately if available
        if self.final_model is not None:
            model_file = results_dir / f"best_model_{timestamp}.pt"
            torch.save({
                'model_state_dict': self.final_model.state_dict(),
                'results': self.final_results,
                'config': self.config.config
            }, model_file)
            self.logger.info(f"Best model saved to {model_file}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

# Test the pipeline
if __name__ == "__main__":
    # Simple test setup
    config = AutoMLConfig()
    config.set('dataset_name', 'emotions')
    config.set('time_budget_hours', 2)  # Short test
    
    # Setup logging
    setup_logging(level='INFO')
    set_seed(42)
    
    # Create pipeline
    pipeline = AutoMLPipeline(config)
    
    print("AutoML Pipeline initialized successfully!")
    print("Components:")
    print(f"  Data Manager: {type(pipeline.data_manager).__name__}")
    print(f"  Model Factory: {type(pipeline.model_factory).__name__}")
    print(f"  Trainer: {type(pipeline.trainer).__name__}")
    print(f"  Budget Manager: {type(pipeline.budget_manager).__name__}")
    print(f"  Early Stopping: {type(pipeline.early_stopping).__name__}")
    print(f"  HPO Selector: {type(pipeline.hpo_selector).__name__}")
    print("\nReady to run on emotions dataset!")