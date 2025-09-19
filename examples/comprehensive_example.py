"""
Comprehensive example demonstrating how to use ME-Benchmark
"""
import sys
import os

# Add the parent directory to the path so we can import me_benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from me_benchmark import REGISTRY
from me_benchmark.utils.config import load_config, save_config
from me_benchmark.utils.results_collector import ResultsCollector
from me_benchmark.utils.visualization import Visualizer


def show_registered_components():
    """Show all registered components"""
    print("ME-Benchmark Registered Components")
    print("==================================")
    
    # Show registered components
    print("\nRegistered Models:")
    for name in REGISTRY._models:
        print(f"  - {name}")
    
    print("\nRegistered Editors:")
    for name in REGISTRY._editors:
        print(f"  - {name}")
    
    print("\nRegistered Evaluators:")
    for name in REGISTRY._evaluators:
        print(f"  - {name}")
    
    print("\nRegistered Datasets:")
    for name in REGISTRY._datasets:
        print(f"  - {name}")
    
    print("\nRegistered Runners:")
    for name in REGISTRY._runners:
        print(f"  - {name}")
    
    print("\nRegistered Summarizers:")
    for name in REGISTRY._summarizers:
        print(f"  - {name}")


def create_example_config():
    """Create an example configuration file"""
    config = {
        "model": {
            "type": "huggingface",
            "path": "meta-llama/Llama-2-7b-hf",
            "model_kwargs": {
                "device_map": "auto",
                "torch_dtype": "float16"
            }
        },
        "editing": {
            "method": "rome",
            "hparams_path": "hparams/ROME/llama-2-7b.yaml"
        },
        "evaluation": {
            "datasets": [
                {
                    "name": "zsre",
                    "split": "validation",
                    "metrics": ["accuracy", "fluency", "consistency"]
                },
                {
                    "name": "counterfact",
                    "split": "validation",
                    "metrics": ["accuracy", "locality", "portability"]
                }
            ],
            "general_benchmarks": [
                {
                    "name": "mmlu",
                    "subset": "stem",
                    "metrics": ["accuracy"]
                },
                {
                    "name": "hellaswag",
                    "metrics": ["accuracy"]
                }
            ]
        },
        "runner": {
            "type": "local",
            "max_workers": 4,
            "debug": False
        },
        "output": {
            "results_dir": "results/",
            "save_individual_results": True,
            "save_summary": True
        }
    }
    
    # Save the configuration
    save_config(config, "example_config.yaml")
    print("Example configuration saved to example_config.yaml")
    return config


def demonstrate_results_collection():
    """Demonstrate results collection functionality"""
    print("\nDemonstrating Results Collection")
    print("================================")
    
    # Create a results collector
    collector = ResultsCollector("example_results/")
    
    # Add some sample results
    sample_results = [
        {"accuracy": 0.85, "fluency": 0.92, "consistency": 0.88},
        {"accuracy": 0.87, "fluency": 0.91, "consistency": 0.89},
        {"accuracy": 0.83, "fluency": 0.93, "consistency": 0.87},
    ]
    
    for i, result in enumerate(sample_results):
        collector.add_result(result, {"experiment": f"exp_{i+1}", "model": "llama-2-7b"})
    
    # Save results
    filepath = collector.save_results("sample_results.json")
    print(f"Sample results saved to {filepath}")
    
    # Load results
    loaded_results = collector.load_results(filepath)
    print(f"Loaded {len(loaded_results)} results")
    
    return collector.get_results()


def demonstrate_visualization(results):
    """Demonstrate visualization functionality"""
    print("\nDemonstrating Visualization")
    print("===========================")
    
    # Create a visualizer
    visualizer = Visualizer()
    
    # Extract metrics for visualization
    accuracies = [r['result']['accuracy'] for r in results]
    fluencies = [r['result']['fluency'] for r in results]
    consistencies = [r['result']['consistency'] for r in results]
    
    # Create sample data for comparison
    comparison_data = {
        'accuracy': {
            'experiment_1': {'mean': 0.85, 'std': 0.02},
            'experiment_2': {'mean': 0.87, 'std': 0.01},
            'experiment_3': {'mean': 0.83, 'std': 0.03}
        },
        'fluency': {
            'experiment_1': {'mean': 0.92, 'std': 0.01},
            'experiment_2': {'mean': 0.91, 'std': 0.02},
            'experiment_3': {'mean': 0.93, 'std': 0.01}
        }
    }
    
    print("Visualization functions are ready to use:")
    print("  - plot_metrics_comparison()")
    print("  - plot_metrics_distribution()")
    print("  - plot_radar_chart()")
    
    # Note: Actual plotting is commented out to avoid dependency issues
    # In a real environment, you would call:
    # visualizer.plot_metrics_comparison(comparison_data, 'accuracy', 'accuracy_comparison.png')
    # visualizer.plot_metrics_distribution(results, 'accuracy', 'accuracy_distribution.png')


def main():
    print("ME-Benchmark Comprehensive Example")
    print("==================================")
    
    # Show registered components
    show_registered_components()
    
    # Create example configuration
    config = create_example_config()
    
    # Demonstrate results collection
    results = demonstrate_results_collection()
    
    # Demonstrate visualization
    demonstrate_visualization(results)
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()