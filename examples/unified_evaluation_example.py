"""
Example usage of the unified evaluation benchmark
"""
from me_benchmark.models.base import BaseModel
from me_benchmark.evaluators.unified_evaluator import BenchmarkEvaluator
from me_benchmark.datasets.unified_dataset import UnifiedDataset
from me_benchmark.configs.unified_evaluation_benchmark import get_benchmark_config


def run_unified_evaluation(model: BaseModel, scenario: str = "comprehensive"):
    """Run unified evaluation on a model"""
    # Get benchmark configuration
    config = get_benchmark_config(scenario)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(model, config)
    
    # Load dataset
    dataset_config = config.get("datasets", [{}])[0]  # Get first dataset config
    dataset_path = dataset_config.get("path", "")
    dataset = UnifiedDataset(dataset_path, **dataset_config)
    
    # Get evaluation data
    eval_data = dataset.get_eval_data()
    
    # Run evaluation
    results = evaluator.evaluate(eval_data)
    
    return results


def run_knowledge_editing_evaluation(model: BaseModel):
    """Run knowledge editing focused evaluation"""
    return run_unified_evaluation(model, "knowledge_editing_only")


def run_language_understanding_evaluation(model: BaseModel):
    """Run language understanding focused evaluation"""
    return run_unified_evaluation(model, "language_understanding_only")


def run_fast_evaluation(model: BaseModel):
    """Run fast evaluation for quick benchmarking"""
    return run_unified_evaluation(model, "fast_evaluation")


if __name__ == "__main__":
    # Example usage
    print("Unified Evaluation Benchmark Examples")
    print("=" * 40)
    
    # These examples assume you have a model implementation
    # model = YourModelImplementation()
    # 
    # # Run comprehensive evaluation
    # results = run_unified_evaluation(model, "comprehensive")
    # print("Comprehensive Evaluation Results:", results)
    # 
    # # Run knowledge editing evaluation
    # ke_results = run_knowledge_editing_evaluation(model)
    # print("Knowledge Editing Results:", ke_results)
    # 
    # # Run fast evaluation
    # fast_results = run_fast_evaluation(model)
    # print("Fast Evaluation Results:", fast_results)
    pass