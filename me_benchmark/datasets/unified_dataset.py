"""
Dataset integration for ME-Benchmark
Integrates datasets from OpenCompass and EasyEdit into the ME-Benchmark framework
"""
import os
import json
from typing import Dict, Any, List
from pathlib import Path

from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset


@register_dataset('opencompass')
class OpenCompassDataset(BaseDataset):
    """OpenCompass dataset integration"""
    
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_name = kwargs.get('dataset_name')
        self.subset = kwargs.get('subset')
        super().__init__(dataset_path, **kwargs)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load OpenCompass dataset"""
        # This would interface with OpenCompass dataset loading mechanisms
        # For now, we'll return placeholder data
        return []
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for knowledge editing"""
        # Convert OpenCompass dataset format to ME-Benchmark edit format
        edit_data = []
        for item in self.data:
            edit_item = {
                'prompt': item.get('input', ''),
                'subject': item.get('subject', ''),
                'target_new': item.get('target', ''),
                'ground_truth': item.get('answer', '')
            }
            edit_data.append(edit_item)
        return edit_data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for evaluation"""
        # OpenCompass datasets are already in evaluation format
        return self.data


@register_dataset('easyedit')
class EasyEditDataset(BaseDataset):
    """EasyEdit dataset integration"""
    
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_type = kwargs.get('dataset_type', 'zsre')  # zsre, counterfact, etc.
        super().__init__(dataset_path, **kwargs)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load EasyEdit dataset"""
        data = []
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                if self.dataset_path.endswith('.json'):
                    data = json.load(f)
                else:
                    # Handle other formats
                    pass
        return data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for knowledge editing"""
        # EasyEdit datasets are already in edit format
        return self.data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for evaluation"""
        # EasyEdit datasets include evaluation data
        eval_data = []
        for item in self.data:
            eval_item = {
                'prompt': item.get('prompt', ''),
                'target_new': item.get('target_new', ''),
                'ground_truth': item.get('ground_truth', ''),
                'rephrase_prompt': item.get('rephrase_prompt'),
                'locality': item.get('locality', {}),
                'portability': item.get('portability', {})
            }
            eval_data.append(eval_item)
        return eval_data


@register_dataset('knowedit')
class KnowEditDataset(EasyEditDataset):
    """KnowEdit dataset integration"""
    
    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.dataset_type = 'knowedit'
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load KnowEdit dataset"""
        data = []
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for knowledge editing"""
        edit_data = []
        for item in self.data:
            edit_item = {
                'prompt': item.get('prompt', ''),
                'subject': item.get('subject', ''),
                'target_new': item.get('target_new', ''),
                'ground_truth': item.get('ground_truth', '')
            }
            edit_data.append(edit_item)
        return edit_data


@register_dataset('unified')
class UnifiedDataset(BaseDataset):
    """Unified dataset that combines multiple dataset sources"""
    
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_sources = kwargs.get('sources', [])
        self.datasets = []
        super().__init__(dataset_path, **kwargs)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from all configured sources"""
        all_data = []
        
        for source in self.dataset_sources:
            source_type = source.get('type')
            source_path = source.get('path')
            source_kwargs = source.get('kwargs', {})
            
            if source_type == 'opencompass':
                dataset = OpenCompassDataset(source_path, **source_kwargs)
            elif source_type == 'easyedit':
                dataset = EasyEditDataset(source_path, **source_kwargs)
            elif source_type == 'knowedit':
                dataset = KnowEditDataset(source_path, **source_kwargs)
            else:
                continue
                
            self.datasets.append(dataset)
            all_data.extend(dataset.data)
            
        return all_data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for knowledge editing from all sources"""
        edit_data = []
        for dataset in self.datasets:
            edit_data.extend(dataset.get_edit_data())
        return edit_data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data formatted for evaluation from all sources"""
        eval_data = []
        for dataset in self.datasets:
            eval_data.extend(dataset.get_eval_data())
        return eval_data