"""
Example script demonstrating how to use ME-Benchmark
"""
import sys
import os

# Add the parent directory to the path so we can import me_benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from me_benchmark import REGISTRY
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.simple import SimpleEvaluator
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator
from me_benchmark.evaluators.mmlu import MMLUEvaluator
from me_benchmark.evaluators.hellaswag import HellaSwagEvaluator
from me_benchmark.datasets.zsre import ZsREDataset
from me_benchmark.datasets.counterfact import CounterFactDataset
from me_benchmark.datasets.mmlu import MMLUDataset
from me_benchmark.datasets.hellaswag import HellaSwagDataset
from me_benchmark.runners.local import LocalRunner
from me_benchmark.summarizers.default import DefaultSummarizer


def main():
    print("ME-Benchmark Example")
    print("===================")
    
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


if __name__ == '__main__':
    main()