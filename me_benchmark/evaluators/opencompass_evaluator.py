"""
OpenCompass evaluator for ME-Benchmark
Integrates OpenCompass evaluation capabilities into the ME-Benchmark framework
"""
import os
import sys
from typing import Dict, Any, List, Union
import yaml
import json
import tempfile
from pathlib import Path

# Add opencompass to path
opencompass_path = os.path.join(os.path.dirname(__file__), '..', '..', 'opencompass', 'opencompass-main')
if opencompass_path not in sys.path:
    sys.path.insert(0, opencompass_path)

try:
    from opencompass.partitioners import SizePartitioner
    from opencompass.runners import LocalRunner
    from opencompass.tasks import OpenICLInferTask
    from opencompass.models import HuggingFaceCausalLM
    from opencompass.datasets import DatasetFactory
    from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
    from opencompass.evaluator import BaseEvaluator as OpenCompassBaseEvaluator
except ImportError:
    print("Warning: OpenCompass not found. Please install OpenCompass to use this evaluator.")
    OpenCompassBaseEvaluator = object

from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.registry import register_evaluator


@register_evaluator('opencompass')
class OpenCompassEvaluator(BaseEvaluator):
    """OpenCompass evaluator for ME-Benchmark"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.opencompass_config = config.get('opencompass_config', {})
        self.metrics = config.get('metrics', ['accuracy'])
        self.datasets = config.get('datasets', [])
        
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model using OpenCompass"""
        results = {}
        
        # Create temporary config for OpenCompass
        temp_config = self._create_opencompass_config(eval_data)
        
        # Run evaluation using OpenCompass
        try:
            # This is a simplified implementation
            # In practice, you would need to properly integrate with OpenCompass workflow
            eval_results = self._run_opencompass_evaluation(temp_config)
            results.update(eval_results)
        except Exception as e:
            print(f"Error running OpenCompass evaluation: {e}")
            results['error'] = str(e)
            
        return results
    
    def _create_opencompass_config(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create OpenCompass configuration from eval_data"""
        config = {
            'datasets': [],
            'models': [],
            'eval': {}
        }
        
        # Add model configuration
        model_config = {
            'type': 'HuggingFaceCausalLM',
            'abbr': 'me_benchmark_model',
            'path': self.model.model_path,
            'tokenizer_path': self.model.model_path,
            'model_kwargs': {
                'device_map': 'auto',
                'trust_remote_code': True
            },
            'tokenizer_kwargs': {
                'padding_side': 'left',
                'truncation_side': 'left',
                'use_fast': False,
                'trust_remote_code': True
            },
            'max_out_len': 100,
            'max_seq_len': 2048,
            'batch_size': 8
        }
        config['models'].append(model_config)
        
        # Add dataset configurations
        for dataset_info in self.datasets:
            config['datasets'].append(dataset_info)
            
        return config
    
    def _run_opencompass_evaluation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run OpenCompass evaluation with given config"""
        # This is a placeholder implementation
        # In a real implementation, you would need to properly integrate with OpenCompass
        results = {
            'evaluator': 'opencompass',
            'metrics': {},
            'details': {}
        }
        
        # For each metric, compute the result
        for metric in self.metrics:
            results['metrics'][metric] = self._compute_metric(config, metric)
            
        return results
    
    def _compute_metric(self, config: Dict[str, Any], metric: str) -> float:
        """Compute a specific metric"""
        # Placeholder implementation
        return 0.0


@register_evaluator('opencompass_dataset')
class OpenCompassDatasetEvaluator(BaseEvaluator):
    """OpenCompass dataset evaluator for ME-Benchmark"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.dataset_name = config.get('dataset_name')
        self.dataset_config = config.get('dataset_config', {})
        self.metric = config.get('metric', 'accuracy')
        
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on a specific OpenCompass dataset"""
        results = {}
        
        try:
            # Build dataset using OpenCompass
            dataset = self._build_dataset()
            
            # Run evaluation
            eval_results = self._evaluate_dataset(dataset)
            results.update(eval_results)
        except Exception as e:
            print(f"Error evaluating dataset {self.dataset_name}: {e}")
            results['error'] = str(e)
            
        return results
    
    def _build_dataset(self):
        """Build OpenCompass dataset"""
        # This would use OpenCompass dataset building utilities
        pass
    
    def _evaluate_dataset(self, dataset):
        """Evaluate model on dataset"""
        # This would use OpenCompass evaluation utilities
        return {'score': 0.0}