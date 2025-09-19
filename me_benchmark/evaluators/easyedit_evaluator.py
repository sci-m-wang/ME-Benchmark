"""
EasyEdit evaluator for ME-Benchmark
Integrates EasyEdit evaluation capabilities into the ME-Benchmark framework
"""
import os
import sys
from typing import Dict, Any, List, Union
import yaml
import json

# Add EasyEdit to path
easyedit_path = os.path.join(os.path.dirname(__file__), '..', '..', 'EasyEdit')
if easyedit_path not in sys.path:
    sys.path.insert(0, easyedit_path)

try:
    from easyeditor import BaseEditor
    from easyeditor.util.hparams import HyperParams
    from easyeditor.evaluate import compute_edit_quality
except ImportError:
    print("Warning: EasyEdit not found. Please install EasyEdit to use this evaluator.")
    BaseEditor = object

from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.registry import register_evaluator


@register_evaluator('easyedit')
class EasyEditEvaluator(BaseEvaluator):
    """EasyEdit evaluator for ME-Benchmark"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.editor_type = config.get('editor_type', 'ROME')
        self.hparams = config.get('hparams', {})
        self.metrics = config.get('metrics', [
            'rewrite_acc',  # Reliability
            'rephrase_acc',  # Generalization
            'locality',  # Locality
            'portability'  # Portability
        ])
        
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model using EasyEdit metrics"""
        results = {
            'evaluator': 'easyedit',
            'metrics': {},
            'details': {}
        }
        
        # For model editing evaluation, we need to:
        # 1. Apply edits using EasyEdit
        # 2. Evaluate on the edited model
        # 3. Compute metrics: reliability, generalization, locality, portability
        
        try:
            # Create editor instance
            editor = self._create_editor()
            
            # Run evaluation
            eval_results = self._run_easyedit_evaluation(editor, eval_data)
            results.update(eval_results)
        except Exception as e:
            print(f"Error running EasyEdit evaluation: {e}")
            results['error'] = str(e)
            
        return results
    
    def _create_editor(self):
        """Create EasyEdit editor instance"""
        # This would create an appropriate EasyEdit editor based on config
        # For example, BaseEditor.from_hparams()
        pass
    
    def _run_easyedit_evaluation(self, editor, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run EasyEdit evaluation"""
        results = {
            'metrics': {},
            'details': {}
        }
        
        # Process each evaluation case
        all_metrics = []
        for data in eval_data:
            # Prepare evaluation data in EasyEdit format
            eval_request = self._prepare_eval_request(data)
            
            # Compute edit quality metrics
            metrics = self._compute_edit_quality(editor, eval_request)
            all_metrics.append(metrics)
            
        # Aggregate metrics
        results['metrics'] = self._aggregate_metrics(all_metrics)
        results['details'] = all_metrics
        
        return results
    
    def _prepare_eval_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare evaluation request in EasyEdit format"""
        request = {
            'prompt': data.get('prompt', ''),
            'target_new': data.get('target_new', ''),
            'ground_truth': data.get('ground_truth', ''),
            'rephrase_prompt': data.get('rephrase_prompt'),
            'locality': data.get('locality', {}),
            'portability': data.get('portability', {})
        }
        return request
    
    def _compute_edit_quality(self, editor, request: Dict[str, Any]) -> Dict[str, Any]:
        """Compute edit quality metrics using EasyEdit"""
        # This would use EasyEdit's compute_edit_quality function
        # For now, we'll return placeholder results
        return {
            'rewrite_acc': 0.0,
            'rephrase_acc': 0.0,
            'locality': {},
            'portability': {}
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all evaluation cases"""
        aggregated = {}
        
        # Aggregate reliability (rewrite accuracy)
        if 'rewrite_acc' in self.metrics:
            rewrite_accs = [m.get('rewrite_acc', 0.0) for m in all_metrics]
            aggregated['reliability'] = sum(rewrite_accs) / len(rewrite_accs) if rewrite_accs else 0.0
        
        # Aggregate generalization (rephrase accuracy)
        if 'rephrase_acc' in self.metrics:
            rephrase_accs = [m.get('rephrase_acc', 0.0) for m in all_metrics]
            aggregated['generalization'] = sum(rephrase_accs) / len(rephrase_accs) if rephrase_accs else 0.0
        
        # Aggregate locality
        if 'locality' in self.metrics:
            locality_scores = []
            for m in all_metrics:
                loc = m.get('locality', {})
                # Average across all locality measures
                loc_values = [v for v in loc.values() if isinstance(v, (int, float))]
                if loc_values:
                    locality_scores.append(sum(loc_values) / len(loc_values))
            aggregated['locality'] = sum(locality_scores) / len(locality_scores) if locality_scores else 0.0
        
        # Aggregate portability
        if 'portability' in self.metrics:
            portability_scores = []
            for m in all_metrics:
                port = m.get('portability', {})
                # Average across all portability measures
                port_values = [v for v in port.values() if isinstance(v, (int, float))]
                if port_values:
                    portability_scores.append(sum(port_values) / len(port_values))
            aggregated['portability'] = sum(portability_scores) / len(portability_scores) if portability_scores else 0.0
            
        return aggregated


@register_evaluator('easyedit_comprehensive')
class ComprehensiveEasyEditEvaluator(EasyEditEvaluator):
    """Comprehensive EasyEdit evaluator with additional metrics"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        # Add additional metrics specific to knowledge editing
        self.additional_metrics = config.get('additional_metrics', [
            'fluency',  # Generation quality
            'consistency',  # Concept consistency for ConceptEdit
            'defense_success',  # For SafeEdit
            'instance_change'  # For ConceptEdit
        ])
        
    def _aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics including additional ones"""
        aggregated = super()._aggregate_metrics(all_metrics)
        
        # Add fluency metric
        if 'fluency' in self.additional_metrics:
            fluency_scores = [m.get('fluency', 0.0) for m in all_metrics if 'fluency' in m]
            aggregated['fluency'] = sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
            
        # Add consistency metric
        if 'consistency' in self.additional_metrics:
            consistency_scores = [m.get('consistency', 0.0) for m in all_metrics if 'consistency' in m]
            aggregated['consistency'] = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
            
        # Add defense success metric
        if 'defense_success' in self.additional_metrics:
            defense_scores = [m.get('defense_success', 0.0) for m in all_metrics if 'defense_success' in m]
            aggregated['defense_success'] = sum(defense_scores) / len(defense_scores) if defense_scores else 0.0
            
        # Add instance change metric
        if 'instance_change' in self.additional_metrics:
            instance_change_scores = [m.get('instance_change', 0.0) for m in all_metrics if 'instance_change' in m]
            aggregated['instance_change'] = sum(instance_change_scores) / len(instance_change_scores) if instance_change_scores else 0.0
            
        return aggregated