"""
Factory for creating ME-Benchmark components
"""
from typing import Dict, Any

from me_benchmark.registry import REGISTRY
from me_benchmark.models.base import BaseModel
from me_benchmark.editors.base import BaseEditor
from me_benchmark.editors.manager import EDITING_MANAGER
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.datasets.base import BaseDataset
from me_benchmark.runners.base import BaseRunner
from me_benchmark.summarizers.base import BaseSummarizer


def create_model(model_config: Dict[str, Any]) -> BaseModel:
    """Create a model instance based on configuration"""
    model_type = model_config['type']
    model_cls = REGISTRY.get_model(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract model arguments
    model_kwargs = model_config.get('model_kwargs', {})
    tokenizer_kwargs = model_config.get('tokenizer_kwargs', {})
    
    return model_cls(
        model_config['path'],
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        **model_config.get('additional_kwargs', {})
    )


def create_editor(editor_config: Dict[str, Any], model: BaseModel) -> BaseEditor:
    """Create an editor instance based on configuration"""
    # Use the unified editing manager to create the editor
    return EDITING_MANAGER.get_editor(model, editor_config)


def create_evaluator(evaluator_config: Dict[str, Any], model: BaseModel) -> BaseEvaluator:
    """Create an evaluator instance based on configuration"""
    evaluator_type = evaluator_config.get('type', 'simple')
    evaluator_cls = REGISTRY.get_evaluator(evaluator_type)
    if evaluator_cls is None:
        # Fallback to simple evaluator
        from me_benchmark.evaluators.simple import SimpleEvaluator
        return SimpleEvaluator(model, evaluator_config)
    
    return evaluator_cls(model, evaluator_config)


def create_dataset(dataset_config: Dict[str, Any]) -> BaseDataset:
    """Create a dataset instance based on configuration"""
    dataset_type = dataset_config.get('type', 'zsre')
    dataset_cls = REGISTRY.get_dataset(dataset_type)
    if dataset_cls is None:
        # Fallback to ZsRE dataset
        from me_benchmark.datasets.zsre import ZsREDataset
        return ZsREDataset(dataset_config.get('path', 'data/zsre.json'))
    
    return dataset_cls(dataset_config.get('path', 'data/zsre.json'))


def create_runner(runner_config: Dict[str, Any], config: Dict[str, Any]) -> BaseRunner:
    """Create a runner instance based on configuration"""
    runner_type = runner_config.get('type', 'local')
    runner_cls = REGISTRY.get_runner(runner_type)
    if runner_cls is None:
        raise ValueError(f"Unknown runner type: {runner_type}")
    
    return runner_cls(config)


def create_summarizer(summarizer_config: Dict[str, Any]) -> BaseSummarizer:
    """Create a summarizer instance based on configuration"""
    summarizer_type = summarizer_config.get('type', 'default')
    summarizer_cls = REGISTRY.get_summarizer(summarizer_type)
    if summarizer_cls is None:
        # Fallback to default summarizer
        from me_benchmark.summarizers.default import DefaultSummarizer
        return DefaultSummarizer(summarizer_config.get('results_dir', 'results/'))
    
    return summarizer_cls(summarizer_config.get('results_dir', 'results/'))