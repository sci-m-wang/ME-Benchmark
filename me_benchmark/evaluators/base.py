"""
Base evaluator class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from me_benchmark.models.base import BaseModel


class BaseEvaluator(ABC):
    """Base class for all evaluators in ME-Benchmark"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    @abstractmethod
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on given data"""
        pass