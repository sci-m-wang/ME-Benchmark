"""
HellaSwag dataset implementation for ME-Benchmark
"""
import json
from typing import List, Dict, Any

from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset


@register_dataset('hellaswag')
class HellaSwagDataset(BaseDataset):
    """HellaSwag dataset implementation"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the HellaSwag dataset"""
        # In a real implementation, this would load from a file or download from a source
        # For now, we'll create some sample data
        self.data = [
            {
                'ctx': "The man put on his socks and then",
                'endings': [
                    "put on his shoes.",
                    "ate breakfast.",
                    "went to bed.",
                    "took a shower."
                ],
                'label': 0
            },
            {
                'ctx': "The woman went to the store and",
                'endings': [
                    "bought some groceries.",
                    "went to the park.",
                    "called her friend.",
                    "watched a movie."
                ],
                'label': 0
            }
        ]
        return self.data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        # HellaSwag is primarily used for evaluation, not editing
        # Return empty list for edit data
        return []
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        if self.data is None:
            self.load_data()
        
        # For HellaSwag, evaluation data is the same as the main data
        return self.data