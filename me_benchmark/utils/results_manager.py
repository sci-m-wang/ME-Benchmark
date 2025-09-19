"""
Results manager for ME-Benchmark
"""
import os
import json
from typing import Dict, Any, List
from datetime import datetime

from me_benchmark.utils.metrics_collector import MetricsCollector
from me_benchmark.utils.visualization import ResultsVisualizer
from me_benchmark.summarizers.base import BaseSummarizer
from me_benchmark.factory import create_summarizer


class ResultsManager:
    """Results manager for ME-Benchmark"""
    
    def __init__(self, results_dir: str = 'results/'):
        self.results_dir = results_dir
        self.metrics_collector = MetricsCollector(results_dir)
        self.visualizer = ResultsVisualizer(results_dir)
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def collect_edit_metrics(self, edit_success: bool, edit_time: float, 
                           original_metrics: Dict[str, Any], edited_metrics: Dict[str, Any],
                           metadata: Dict[str, Any] = None):
        """Collect metrics related to the editing process"""
        return self.metrics_collector.collect_edit_metrics(
            edit_success, edit_time, original_metrics, edited_metrics, metadata)
    
    def collect_evaluation_metrics(self, evaluation_results: Dict[str, Any],
                                 evaluation_time: float,
                                 metadata: Dict[str, Any] = None):
        """Collect metrics related to the evaluation process"""
        return self.metrics_collector.collect_evaluation_metrics(
            evaluation_results, evaluation_time, metadata)
    
    def start_resource_monitoring(self):
        """Start monitoring system resources"""
        self.metrics_collector.start_resource_monitoring()
    
    def stop_resource_monitoring(self):
        """Stop monitoring system resources"""
        self.metrics_collector.stop_resource_monitoring()
    
    def save_results(self, filename: str = None):
        """Save all collected results"""
        return self.metrics_collector.save_metrics(filename)
    
    def load_results(self, filepath: str):
        """Load results from a file"""
        return self.metrics_collector.load_metrics(filepath)
    
    def generate_summary_report(self, summarizer_type: str = 'default') -> Dict[str, Any]:
        """Generate a summary report using a specified summarizer"""
        # Get all collected metrics
        metrics = self.metrics_collector.get_metrics()
        
        # Extract evaluation results for summarization
        evaluation_results = []
        for metric_entry in metrics:
            if 'evaluation_results' in metric_entry:
                # Include metadata with each result for the summarizer
                result_with_metadata = metric_entry['evaluation_results'].copy()
                result_with_metadata.update(metric_entry.get('metadata', {}))
                evaluation_results.append(result_with_metadata)
        
        # Create summarizer and generate report
        summarizer = create_summarizer({'type': summarizer_type})
        if isinstance(summarizer, BaseSummarizer):
            summary = summarizer.summarize(evaluation_results)
            
            # Add metrics collector's own summary
            metrics_summary = self.metrics_collector.generate_summary_report()
            summary['metrics_summary'] = metrics_summary
            
            return summary
        else:
            # Fallback to metrics collector's summary
            return self.metrics_collector.generate_summary_report()
    
    def save_summary_report(self, report: Dict[str, Any], output_path: str):
        """Save the summary report to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report to JSON file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save as a human-readable text file
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("ME-Benchmark Results Summary Report\n")
            f.write("===================================\n\n")
            
            # Write metrics summary
            if 'metrics_summary' in report:
                f.write("Metrics Summary:\n")
                metrics_summary = report['metrics_summary']
                
                if 'edit_metrics' in metrics_summary:
                    f.write("  Edit Metrics:\n")
                    edit_metrics = metrics_summary['edit_metrics']
                    for key, value in edit_metrics.items():
                        if isinstance(value, float):
                            f.write(f"    {key}: {value:.4f}\n")
                        else:
                            f.write(f"    {key}: {value}\n")
                
                if 'evaluation_metrics' in metrics_summary:
                    f.write("  Evaluation Metrics:\n")
                    eval_metrics = metrics_summary['evaluation_metrics']
                    for key, value in eval_metrics.items():
                        if key != 'total_evaluations':
                            f.write(f"    {key}: {value['mean']:.4f} (min: {value['min']:.4f}, max: {value['max']:.4f})\n")
                f.write("\n")
            
            # Write other summary data
            for key, value in report.items():
                if key != 'metrics_summary':
                    f.write(f"{key}:\n")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict):
                                f.write(f"  {sub_key}:\n")
                                for sub_sub_key, sub_sub_value in sub_value.items():
                                    f.write(f"    {sub_sub_key}: {sub_sub_value}\n")
                            else:
                                f.write(f"  {sub_key}: {sub_sub_value}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
    
    def create_visualization_report(self, results: List[Dict[str, Any]] = None, 
                                  output_dir: str = None):
        """Create a visualization report"""
        if results is None:
            # Load results from the metrics collector
            results = self.metrics_collector.get_metrics()
        
        self.visualizer.create_comprehensive_report(results, output_dir)
    
    def query_results(self, filter_func) -> List[Dict[str, Any]]:
        """Query results based on a filter function"""
        return self.metrics_collector.filter_results(filter_func)
    
    def get_latest_results(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get the latest n results"""
        return self.metrics_collector.get_latest_results(n)