# src/automl/plot_guide.py
"""
AutoML Visualization Guide and Examples
Shows how to use the comprehensive plotting system
"""

from .visualization import AutoMLVisualizer, plot_automl_results, create_quick_summary_plot
import json

def generate_all_automl_plots(results_file: str, output_dir: str = "results/plots"):
    """
    Generate all AutoML plots from results file
    
    Args:
        results_file: Path to AutoML results JSON file
        output_dir: Directory to save plots
    
    Returns:
        List of generated plot paths
    """
    
    print(" AutoML Comprehensive Visualization Guide")
    print("=" * 60)
    
    # Method 1: Using convenience function
    print("Method 1: Using convenience function")
    plots = plot_automl_results(
        results_file=results_file,
        save_dir=output_dir,
        format='png',
        dpi=300,
        interactive=True
    )
    
    return plots

def generate_custom_plots(results_data: dict, save_dir: str = "results/custom_plots"):
    """
    Generate custom plots with specific configurations
    
    Args:
        results_data: AutoML results dictionary
        save_dir: Directory to save plots
    """
    
    print("\n Custom Plot Generation")
    print("-" * 40)
    
    # Create visualizer
    visualizer = AutoMLVisualizer(results_data, save_dir)
    
    # Generate specific plot categories
    print("Generating training analysis plots...")
    training_plots = visualizer._generate_training_plots('png', 300)
    
    print("Generating architecture comparison plots...")
    comparison_plots = visualizer._generate_architecture_comparison_plots('png', 300)
    
    print("Generating HPO analysis plots...")
    hpo_plots = visualizer._generate_hpo_analysis_plots('png', 300)
    
    print("Generating dashboard...")
    dashboard_plots = visualizer._generate_dashboard_plots('png', 300)
    
    all_plots = training_plots + comparison_plots + hpo_plots + dashboard_plots
    
    print(f"Generated {len(all_plots)} custom plots")
    return all_plots

def plot_categories_explained():
    """
    Explain what each plot category shows
    """
    
    print("\n AutoML Plot Categories Explained")
    print("=" * 60)
    
    categories = {
        "1. Training Progress Plots": {
            "training_curves.png": "Training/validation accuracy curves for each architecture",
            "convergence_analysis.png": "How quickly each architecture converges to best performance"
        },
        
        "2. Architecture Comparison": {
            "architecture_leaderboard.png": "Performance ranking + HPO methods used",
            "performance_vs_efficiency.png": "Performance vs time efficiency scatter plot"
        },
        
        "3. HPO Analysis": {
            "hpo_progress.png": "Hyperparameter optimization progress for each architecture",
            "hyperparameter_importance.png": "Which hyperparameters matter most"
        },
        
        "4. Early Stopping Analysis": {
            "early_stopping_analysis.png": "Timeline of early stopping decisions + confidence evolution"
        },
        
        "5. Resource Management": {
            "resource_analysis.png": "Time budget allocation + resource efficiency analysis"
        },
        
        "6. Results Dashboard": {
            "results_dashboard.png": "Comprehensive overview with dataset info, best model, and summary"
        },
        
        "7. Interactive Plots (HTML)": {
            "interactive_architecture_comparison.html": "Interactive bar chart with hover details",
            "interactive_hpo_progress.html": "Interactive HPO progress with zoom/pan"
        }
    }
    
    for category, plots in categories.items():
        print(f"\n{category}")
        print("-" * len(category))
        for plot_name, description in plots.items():
            print(f"   {plot_name}")
            print(f"     → {description}")

def usage_examples():
    """
    Show usage examples
    """
    
    print("\n💡 Usage Examples")
    print("=" * 60)
    
    print("""
# Example 1: Generate all plots from results file
from src.automl.visualization import plot_automl_results

plots = plot_automl_results(
    results_file='results/automl_results.json',
    save_dir='results/plots',
    format='png',
    dpi=300,
    interactive=True
)

# Example 2: Quick summary plot
from src.automl.visualization import create_quick_summary_plot

quick_plot = create_quick_summary_plot(
    results_file='results/automl_results.json',
    save_path='results/quick_summary.png'
)

# Example 3: Custom visualization
from src.automl.visualization import AutoMLVisualizer
import json

with open('results/automl_results.json', 'r') as f:
    results = json.load(f)

visualizer = AutoMLVisualizer(results, 'my_plots')
plots = visualizer.generate_all_plots(
    format='pdf',  # PDF format
    dpi=600,       # High resolution
    interactive=False
)

# Example 4: Integration with AutoML pipeline
from src.automl.automl import AutoMLPipeline
from src.automl.visualization import AutoMLVisualizer

# Run AutoML
pipeline = AutoMLPipeline(config)
results = pipeline.run()

# Generate plots immediately
visualizer = AutoMLVisualizer(results, 'results/plots')
plots = visualizer.generate_all_plots()

print(f"Generated {len(plots)} plots!")
""")

def recommended_workflow():
    """
    Show recommended visualization workflow
    """
    
    print("\n Recommended Visualization Workflow")
    print("=" * 60)
    
    print("""
1. 🚀 Run AutoML Pipeline
   → Generates results JSON file

2. 📊 Quick Summary (for immediate insights)
   → create_quick_summary_plot()
   → Shows architecture performance comparison

3. 🎨 Comprehensive Analysis (for detailed analysis)
   → plot_automl_results() with all plots
   → Generates 9+ different plot types

4. 🔍 Custom Analysis (for specific insights)
   → Use AutoMLVisualizer directly
   → Generate specific plot categories

5. 📱 Interactive Exploration (for presentations)
   → Enable interactive=True
   → Get HTML plots with hover/zoom

6. 📄 Publication Ready (for papers/reports)
   → Use format='pdf', dpi=600
   → High-quality vector graphics
""")

if __name__ == "__main__":
    # Show the complete guide
    plot_categories_explained()
    usage_examples()
    recommended_workflow()
    
    print("\n AutoML Visualization Guide Complete!")
    print(" Check results/plots/ directory for generated visualizations")
