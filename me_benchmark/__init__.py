"""
ME-Benchmark: Model Editing Benchmark Framework
"""
__version__ = '0.1.0'

# Import all built-in components to register them
from me_benchmark import init

# Import main classes for easy access
from me_benchmark.models.base import BaseModel
from me_benchmark.editors.base import BaseEditor
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.datasets.base import BaseDataset
from me_benchmark.runners.base import BaseRunner
from me_benchmark.summarizers.base import BaseSummarizer
from me_benchmark.registry import REGISTRY

__all__ = [
    'BaseModel',
    'BaseEditor',
    'BaseEvaluator',
    'BaseDataset',
    'BaseRunner',
    'BaseSummarizer',
    'REGISTRY'
]