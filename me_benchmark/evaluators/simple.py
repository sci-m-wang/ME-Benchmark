"""
Simple evaluator implementation for ME-Benchmark
"""
from typing import List, Dict, Any
import torch

from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_evaluator


@register_evaluator('simple')
class SimpleEvaluator(BaseEvaluator):
    """Simple evaluator implementation"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        super().__init__(model, config)
        self.metrics = config.get('metrics', ['accuracy'])
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on given data"""
        results = {}
        
        # Calculate accuracy if requested
        if 'accuracy' in self.metrics:
            results['accuracy'] = self._calculate_accuracy(eval_data)
        
        # Calculate perplexity if requested
        if 'perplexity' in self.metrics:
            results['perplexity'] = self._calculate_perplexity(eval_data)
        
        return results
    
    def _calculate_accuracy(self, eval_data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy on evaluation data"""
        correct = 0
        total = len(eval_data)
        
        # Extract prompts and expected answers
        prompts = [item['prompt'] for item in eval_data]
        expected_answers = [item['expected_answer'] for item in eval_data]
        
        # Generate model responses
        generated_answers = self.model.generate(prompts)
        
        # Compare answers (simplified)
        for generated, expected in zip(generated_answers, expected_answers):
            if expected.lower() in generated.lower():
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_perplexity(self, eval_data: List[Dict[str, Any]]) -> float:
        """Calculate average perplexity on evaluation data"""
        texts = [item['text'] for item in eval_data]
        ppls = self.model.get_ppl(texts)
        return sum(ppls) / len(ppls) if ppls else 0.0