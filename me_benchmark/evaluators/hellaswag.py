from typing import List, Dict, Any
import torch

from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_evaluator


@register_evaluator('hellaswag')
class HellaSwagEvaluator(BaseEvaluator):
    """HellaSwag evaluator implementation"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        super().__init__(model, config)
        self.metrics = config.get('metrics', ['accuracy'])
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on HellaSwag tasks"""
        results = {}
        
        # Calculate accuracy if requested
        if 'accuracy' in self.metrics:
            results['accuracy'] = self._calculate_accuracy(eval_data)
        
        return results
    
    def _calculate_accuracy(self, eval_data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy on HellaSwag data"""
        correct = 0
        total = len(eval_data)
        
        # Extract contexts and endings
        contexts = [item['ctx'] for item in eval_data]
        endings_list = [item['endings'] for item in eval_data]
        correct_labels = [item['label'] for item in eval_data]
        
        # Generate model responses
        prompts = []
        for context, endings in zip(contexts, endings_list):
            prompt = f"Context: {context}\n"
            prompt += "Which ending is most appropriate?\n"
            for i, ending in enumerate(endings):
                prompt += f"{i}. {ending}\n"
            prompt += "Answer:"
            prompts.append(prompt)
        
        generated_answers = self.model.generate(prompts)
        
        # Compare answers (simplified)
        for generated, correct_label in zip(generated_answers, correct_labels):
            # In a real implementation, this would parse the model's answer
            # For now, we'll use a placeholder comparison
            if str(correct_label) in generated:
                correct += 1
        
        return correct / total if total > 0 else 0.0