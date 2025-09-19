"""
Results collector for ME-Benchmark
"""
import os
import json
import pickle
from typing import Dict, Any, List
from datetime import datetime


class ResultsCollector:
    """Results collector for ME-Benchmark"""
    
    def __init__(self, results_dir: str = 'results/'):
        self.results_dir = results_dir
        self.results = []
        self.metadata = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def add_result(self, result: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add a result to the collector"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'metadata': metadata or {}
        }
        self.results.append(entry)
    
    def save_results(self, filename: str = None):
        """Save all results to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Save results to JSON file
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results(self, filepath: str):
        """Load results from a file"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        return self.results
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all collected results"""
        return self.results
    
    def clear_results(self):
        """Clear all collected results"""
        self.results = []
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save a checkpoint of the current results"""
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")
        
        checkpoint_data = {
            'results': self.results,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_name: str):
        """Load a checkpoint of results"""
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.results = checkpoint_data['results']
        self.metadata = checkpoint_data['metadata']
        
        return checkpoint_data
    
    def filter_results(self, filter_func) -> List[Dict[str, Any]]:
        """Filter results based on a filter function"""
        return [entry for entry in self.results if filter_func(entry)]
    
    def get_latest_results(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get the latest n results"""
        return self.results[-n:] if self.results else []