"""
ZsRE dataset implementation for ME-Benchmark
"""
import json
from typing import List, Dict, Any

from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset


@register_dataset('zsre')
class ZsREDataset(BaseDataset):
    """ZsRE (Zero-Shot Relation Extraction) dataset implementation"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the ZsRE dataset"""
        # In a real implementation, this would load from a file or download from a source
        # For now, we'll create some sample data
        self.data = [
            {
                'prompt': "The capital of France is",
                'subject': "France",
                'target_new': "Paris",
                'expected_answer': "Paris"
            },
            {
                'prompt': "The president of the United States is",
                'subject': "the United States",
                'target_new': "Joe Biden",
                'expected_answer': "Joe Biden"
            }
        ]
        return self.data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        if self.data is None:
            self.load_data()
        
        # For ZsRE, edit data is the same as the main data
        return self.data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        if self.data is None:
            self.load_data()
        
        # For ZsRE, evaluation data is the same as the main data
        return self.data