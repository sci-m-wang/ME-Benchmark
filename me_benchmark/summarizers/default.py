Default summarizer implementation for ME-Benchmark

import os
import json
import numpy as np
from typing import List, Dict, Any

from me_benchmark.summarizers.base import BaseSummarizer
from me_benchmark.registry import register_summarizer

@register_summarizer('default')
class DefaultSummarizer(BaseSummarizer):
    """Default summarizer implementation"""
    
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize evaluation results"""
        if not results:
            return {}
        
        # Calculate statistics for each metric
        summary = {}
        metrics = set()
        
        # Collect all metric names
        for result in results:
            metrics.update(result.keys())
        
        # Calculate statistics for each metric
        for metric in metrics:
            values = [result[metric] for result in results if metric in result and result[metric] is not None]
            if values:
                summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return summary
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save the summary report to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report to JSON file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save as a human-readable text file
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("ME-Benchmark Summary Report\n")
            f.write("===========================\n\n")
            
            for metric, stats in report.items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean:   {stats['mean']:.4f}\n")
                f.write(f"  Std:    {stats['std']:.4f}\n")
                f.write(f"  Min:    {stats['min']:.4f}\n")
                f.write(f"  Max:    {stats['max']:.4f}\n")
                f.write(f"  Count:  {stats['count']}\n")
                f.write("\n")
    
    def compare_results(self, results_list: List[List[Dict[str, Any]]], labels: List[str]) -> Dict[str, Any]:
        """Compare results from different experiments"""
        if not results_list or not labels or len(results_list) != len(labels):
            return {}
        
        comparison = {}
        metrics = set()
        
        # Collect all metric names
        for results in results_list:
            for result in results:
                metrics.update(result.keys())
        
        # Compare statistics for each metric
        for metric in metrics:
            comparison[metric] = {}
            for label, results in zip(labels, results_list):
                values = [result[metric] for result in results if metric in result and result[metric] is not None]
                if values:
                    comparison[metric][label] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
        
        return comparison
    
    def save_comparison_report(self, comparison: Dict[str, Any], output_path: str):
        """Save the comparison report to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report to JSON file
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Also save as a human-readable text file
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("ME-Benchmark Comparison Report\n")
            f.write("==============================\n\n")
            
            for metric, experiments in comparison.items():
                f.write(f"{metric}:\n")
                for label, stats in experiments.items():
                    f.write(f"  {label}:\n")
                    f.write(f"    Mean:   {stats['mean']:.4f}\n")
                    f.write(f"    Std:    {stats['std']:.4f}\n")
                    f.write(f"    Min:    {stats['min']:.4f}\n")
                    f.write(f"    Max:    {stats['max']:.4f}\n")
                    f.write(f"    Count:  {stats['count']}\n")
                f.write("\n")