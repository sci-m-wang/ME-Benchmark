"""
CounterFact dataset implementation for ME-Benchmark
"""
import json
from typing import List, Dict, Any

from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset


@register_dataset('counterfact')
class CounterFactDataset(BaseDataset):
    """CounterFact dataset implementation"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the CounterFact dataset"""
        # In a real implementation, this would load from a file or download from a source
        # For now, we'll create some sample data
        self.data = [
            {
                'prompt': "The Eiffel Tower is located in",
                'subject': "The Eiffel Tower",
                'target_new': "Paris, France",
                'expected_answer': "Paris, France"
            },
            {
                'prompt': "The Great Wall of China was built by",
                'subject': "The Great Wall of China",
                'target_new': "Qin Shi Huang",
                'expected_answer': "Qin Shi Huang"
            }
        ]
        return self.data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        if self.data is None:
            self.load_data()
        
        # For CounterFact, edit data is the same as the main data
        return self.data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        if self.data is None:
            self.load_data()
        
        # For CounterFact, evaluation data is the same as the main data
        return self.data