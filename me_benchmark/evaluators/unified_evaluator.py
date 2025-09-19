"""
Unified evaluator for ME-Benchmark
Combines OpenCompass and EasyEdit evaluation capabilities
"""
from typing import Dict, Any, List, Union
import yaml
import json

from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.evaluators.opencompass_evaluator import OpenCompassEvaluator
from me_benchmark.evaluators.easyedit_evaluator import EasyEditEvaluator
from me_benchmark.registry import register_evaluator


@register_evaluator('unified')
class UnifiedEvaluator(BaseEvaluator):
    """Unified evaluator that combines OpenCompass and EasyEdit evaluation"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.evaluators = []
        self._initialize_evaluators(config)
        
    def _initialize_evaluators(self, config: Dict[str, Any]):
        """Initialize component evaluators based on config"""
        evaluator_configs = config.get('evaluators', [])
        
        for eval_config in evaluator_configs:
            eval_type = eval_config.get('type')
            if eval_type == 'opencompass':
                evaluator = OpenCompassEvaluator(self.model, eval_config)
                self.evaluators.append(evaluator)
            elif eval_type == 'easyedit':
                evaluator = EasyEditEvaluator(self.model, eval_config)
                self.evaluators.append(evaluator)
            # Add other evaluator types as needed
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run unified evaluation using all configured evaluators"""
        results = {
            'evaluator': 'unified',
            'component_results': {},
            'combined_metrics': {},
            'summary': {}
        }
        
        # Run each evaluator
        for evaluator in self.evaluators:
            try:
                eval_results = evaluator.evaluate(eval_data)
                results['component_results'][evaluator.__class__.__name__] = eval_results
            except Exception as e:
                print(f"Error running {evaluator.__class__.__name__}: {e}")
                results['component_results'][evaluator.__class__.__name__] = {'error': str(e)}
        
        # Combine metrics from all evaluators
        results['combined_metrics'] = self._combine_metrics(results['component_results'])
        
        # Generate summary
        results['summary'] = self._generate_summary(results['combined_metrics'])
        
        return results
    
    def _combine_metrics(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine metrics from different evaluators"""
        combined = {}
        
        # Extract metrics from each component result
        for evaluator_name, result in component_results.items():
            if 'metrics' in result:
                for metric_name, metric_value in result['metrics'].items():
                    combined[f"{evaluator_name}_{metric_name}"] = metric_value
                    
        # Handle special cases for common metrics
        # For example, if both evaluators have accuracy metrics, combine them
        self._harmonize_common_metrics(combined, component_results)
        
        return combined
    
    def _harmonize_common_metrics(self, combined: Dict[str, Any], component_results: Dict[str, Any]):
        """Harmonize common metrics from different evaluators"""
        # Example: Combine different accuracy metrics
        accuracy_metrics = {}
        for key, value in combined.items():
            if 'acc' in key.lower() or 'accuracy' in key.lower():
                accuracy_metrics[key] = value
                
        if accuracy_metrics:
            # Calculate overall accuracy as average of all accuracy metrics
            acc_values = list(accuracy_metrics.values())
            if acc_values:
                combined['overall_accuracy'] = sum(acc_values) / len(acc_values)
    
    def _generate_summary(self, combined_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        summary = {
            'total_metrics': len(combined_metrics),
            'key_metrics': {}
        }
        
        # Identify key metrics for summary
        key_metric_keywords = ['accuracy', 'reliability', 'generalization', 'locality', 'portability', 'fluency']
        
        for metric_name, metric_value in combined_metrics.items():
            for keyword in key_metric_keywords:
                if keyword in metric_name.lower():
                    summary['key_metrics'][metric_name] = metric_value
                    
        return summary


@register_evaluator('benchmark')
class BenchmarkEvaluator(UnifiedEvaluator):
    """Benchmark evaluator for standardized evaluation protocols"""
    
    def __init__(self, model, config: Dict[str, Any]):
        # Set default configuration for benchmark evaluation
        default_config = {
            'evaluators': [
                {
                    'type': 'easyedit',
                    'metrics': ['rewrite_acc', 'rephrase_acc', 'locality', 'portability'],
                    'additional_metrics': ['fluency']
                },
                {
                    'type': 'opencompass',
                    'datasets': ['mmlu', 'hellaswag', 'winogrande'],
                    'metrics': ['accuracy']
                }
            ],
            'benchmark_protocol': config.get('benchmark_protocol', 'comprehensive')
        }
        
        # Merge with user config
        merged_config = {**default_config, **config}
        super().__init__(model, merged_config)
        
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run benchmark evaluation with standardized protocols"""
        results = super().evaluate(eval_data)
        
        # Add benchmark-specific analysis
        results['benchmark_analysis'] = self._perform_benchmark_analysis(results)
        
        return results
    
    def _perform_benchmark_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform benchmark-specific analysis"""
        analysis = {
            'knowledge_editing_performance': {},
            'general_language_understanding': {},
            'tradeoff_analysis': {}
        }
        
        # Extract knowledge editing metrics
        ke_metrics = {}
        for metric_name, metric_value in results['combined_metrics'].items():
            if any(keyword in metric_name.lower() for keyword in 
                   ['rewrite', 'rephrase', 'locality', 'portability', 'fluency']):
                ke_metrics[metric_name] = metric_value
                
        analysis['knowledge_editing_performance'] = ke_metrics
        
        # Extract general language understanding metrics
        glu_metrics = {}
        for metric_name, metric_value in results['combined_metrics'].items():
            if any(keyword in metric_name.lower() for keyword in 
                   ['mmlu', 'hellaswag', 'winogrande']):
                glu_metrics[metric_name] = metric_value
                
        analysis['general_language_understanding'] = glu_metrics
        
        # Perform tradeoff analysis
        analysis['tradeoff_analysis'] = self._analyze_tradeoffs(ke_metrics, glu_metrics)
        
        return analysis
    
    def _analyze_tradeoffs(self, ke_metrics: Dict[str, float], glu_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze tradeoffs between knowledge editing and general language understanding"""
        tradeoff_analysis = {
            'knowledge_editing_score': 0.0,
            'language_understanding_score': 0.0,
            'overall_balance': 0.0
        }
        
        # Calculate knowledge editing score (average of key metrics)
        if ke_metrics:
            ke_values = list(ke_metrics.values())
            tradeoff_analysis['knowledge_editing_score'] = sum(ke_values) / len(ke_values)
        
        # Calculate language understanding score
        if glu_metrics:
            glu_values = list(glu_metrics.values())
            tradeoff_analysis['language_understanding_score'] = sum(glu_values) / len(glu_values)
        
        # Calculate overall balance (simple average)
        ke_score = tradeoff_analysis['knowledge_editing_score']
        glu_score = tradeoff_analysis['language_understanding_score']
        tradeoff_analysis['overall_balance'] = (ke_score + glu_score) / 2 if (ke_score > 0 and glu_score > 0) else 0.0
        
        return tradeoff_analysis