"""
Base dataset class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseDataset(ABC):
    """Base class for all datasets in ME-Benchmark"""
    
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self.data = None
        self.load_data()
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the dataset"""
        pass
    
    @abstractmethod
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        pass
    
    @abstractmethod
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        pass