"""
Base runner class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from me_benchmark.models.base import BaseModel
from me_benchmark.editors.base import BaseEditor
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.datasets.base import BaseDataset


class BaseRunner(ABC):
    """Base class for all runners in ME-Benchmark"""
    
    def __init__(self, config: Dict[str, Any], work_dir: str = '.', debug: bool = False):
        self.config = config
        self.work_dir = work_dir
        self.debug = debug
        self.model = None
        self.editor = None
        self.evaluator = None
        self.dataset = None
    
    @abstractmethod
    def run(self):
        """Run the evaluation process"""
        pass
    
    @abstractmethod
    def setup(self):
        """Setup the runner with model, editor, evaluator, and dataset"""
        pass