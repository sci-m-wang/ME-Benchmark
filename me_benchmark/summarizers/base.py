"""
Base summarizer class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseSummarizer(ABC):
    """Base class for all summarizers in ME-Benchmark"""
    
    def __init__(self, results_dir: str, **kwargs):
        self.results_dir = results_dir
        self.kwargs = kwargs
    
    @abstractmethod
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize evaluation results"""
        pass
    
    @abstractmethod
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save the summary report to a file"""
        pass