"""
Base editor class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from me_benchmark.models.base import BaseModel


class BaseEditor(ABC):
    """Base class for all editors in ME-Benchmark"""
    
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        self.model = model
        self.hparams = hparams
    
    @abstractmethod
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        """Apply knowledge editing to the model"""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the model to its original state"""
        pass