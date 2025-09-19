"""
Initialization module for ME-Benchmark
Registers all built-in components
"""

# Import all built-in models to register them
from me_benchmark.models.huggingface import HuggingFaceModel

# Import all built-in editors to register them
from me_benchmark.editors.rome import ROMEEditor

# Import all built-in evaluators to register them
from me_benchmark.evaluators.simple import SimpleEvaluator
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator
from me_benchmark.evaluators.mmlu import MMLUEvaluator
from me_benchmark.evaluators.hellaswag import HellaSwagEvaluator

# Import all built-in datasets to register them
from me_benchmark.datasets.zsre import ZsREDataset
from me_benchmark.datasets.counterfact import CounterFactDataset
from me_benchmark.datasets.mmlu import MMLUDataset
from me_benchmark.datasets.hellaswag import HellaSwagDataset

# Import all built-in runners to register them
from me_benchmark.runners.local import LocalRunner

# Import all built-in summarizers to register them
from me_benchmark.summarizers.default import DefaultSummarizer
from me_benchmark.summarizers.knowledge_editing import KnowledgeEditingSummarizer

# This module is imported to ensure all built-in components are registered
__all__ = [
    'HuggingFaceModel',
    'ROMEEditor',
    'SimpleEvaluator',
    'KnowledgeEditingEvaluator',
    'MMLUEvaluator',
    'HellaSwagEvaluator',
    'ZsREDataset',
    'CounterFactDataset',
    'MMLUDataset',
    'HellaSwagDataset',
    'LocalRunner',
    'DefaultSummarizer',
    'KnowledgeEditingSummarizer'
]