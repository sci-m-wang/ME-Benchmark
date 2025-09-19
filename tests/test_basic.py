"""
Test script for ME-Benchmark
"""
import sys
import os

# Add the parent directory to the path so we can import me_benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from me_benchmark import REGISTRY
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.simple import SimpleEvaluator
from me_benchmark.datasets.zsre import ZsREDataset
from me_benchmark.runners.local import LocalRunner
from me_benchmark.summarizers.default import DefaultSummarizer
from me_benchmark.utils.config import load_config, save_config
from me_benchmark.utils.results_collector import ResultsCollector


def test_registry():
    """Test component registry"""
    print("Testing component registry...")
    
    # Check that all expected components are registered
    assert 'huggingface' in REGISTRY._models, "HuggingFace model not registered"
    assert 'rome' in REGISTRY._editors, "ROME editor not registered"
    assert 'simple' in REGISTRY._evaluators, "Simple evaluator not registered"
    assert 'zsre' in REGISTRY._datasets, "ZsRE dataset not registered"
    assert 'local' in REGISTRY._runners, "Local runner not registered"
    assert 'default' in REGISTRY._summarizers, "Default summarizer not registered"
    
    print("✓ All components registered correctly")


def test_config_utils():
    """Test configuration utilities"""
    print("Testing configuration utilities...")
    
    # Create a sample config
    config = {
        'model': {
            'type': 'huggingface',
            'path': 'test-model'
        },
        'editing': {
            'method': 'rome'
        }
    }
    
    # Save and load config
    save_config(config, 'test_config.yaml')
    loaded_config = load_config('test_config.yaml')
    
    assert loaded_config == config, "Config save/load failed"
    
    # Clean up
    os.remove('test_config.yaml')
    
    print("✓ Configuration utilities working correctly")


def test_results_collector():
    """Test results collector"""
    print("Testing results collector...")
    
    # Create collector
    collector = ResultsCollector('test_results/')
    
    # Add results
    collector.add_result({'accuracy': 0.85})
    collector.add_result({'accuracy': 0.87})
    
    # Save results
    filepath = collector.save_results('test_results.json')
    
    # Load results
    loaded_results = collector.load_results(filepath)
    
    assert len(loaded_results) == 2, "Results count mismatch"
    assert loaded_results[0]['result']['accuracy'] == 0.85, "First result mismatch"
    assert loaded_results[1]['result']['accuracy'] == 0.87, "Second result mismatch"
    
    # Clean up
    os.remove(filepath)
    os.rmdir('test_results/')
    
    print("✓ Results collector working correctly")


def main():
    print("ME-Benchmark Test Suite")
    print("=======================")
    
    try:
        test_registry()
        test_config_utils()
        test_results_collector()
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()