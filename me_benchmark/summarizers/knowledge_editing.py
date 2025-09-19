"""
Knowledge editing summarizer implementation for ME-Benchmark
"""
import os
import json
import numpy as np
from typing import List, Dict, Any

from me_benchmark.summarizers.base import BaseSummarizer
from me_benchmark.registry import register_summarizer


@register_summarizer('knowledge_editing')
class KnowledgeEditingSummarizer(BaseSummarizer):
    """Knowledge editing summarizer implementation"""
    
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize knowledge editing evaluation results"""
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
        
        # Add specialized analysis for knowledge editing
        if 'accuracy' in summary and 'locality' in summary and 'portability' in summary:
            summary['knowledge_editing_score'] = {
                'value': (summary['accuracy']['mean'] + summary['locality']['mean'] + summary['portability']['mean']) / 3,
                'description': 'Average of accuracy, locality, and portability metrics'
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
            f.write("ME-Benchmark Knowledge Editing Summary Report\n")
            f.write("============================================\n\n")
            
            for metric, stats in report.items():
                if metric == 'knowledge_editing_score':
                    f.write(f"{metric}: {stats['value']:.4f} ({stats['description']})\n")
                else:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean:   {stats['mean']:.4f}\n")
                    f.write(f"  Std:    {stats['std']:.4f}\n")
                    f.write(f"  Min:    {stats['min']:.4f}\n")
                    f.write(f"  Max:    {stats['max']:.4f}\n")
                    f.write(f"  Count:  {stats['count']}\n")
                f.write("\n")
    
    def analyze_edit_stability(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the stability of edits across different samples"""
        if not results:
            return {}
        
        # Extract accuracy values
        accuracies = [result['accuracy'] for result in results if 'accuracy' in result]
        
        if not accuracies:
            return {}
        
        # Calculate stability metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Calculate coefficient of variation (CV)
        cv = std_accuracy / mean_accuracy if mean_accuracy > 0 else 0
        
        return {
            'mean_accuracy': float(mean_accuracy),
            'std_accuracy': float(std_accuracy),
            'coefficient_of_variation': float(cv),
            'stability_score': float(1 - cv)  # Higher is more stable
        }
    
    def save_stability_report(self, stability_analysis: Dict[str, Any], output_path: str):
        """Save the stability analysis report to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report to JSON file
        with open(output_path, 'w') as f:
            json.dump(stability_analysis, f, indent=2, ensure_ascii=False)
        
        # Also save as a human-readable text file
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("ME-Benchmark Edit Stability Analysis\n")
            f.write("====================================\n\n")
            
            f.write(f"Mean Accuracy:              {stability_analysis['mean_accuracy']:.4f}\n")
            f.write(f"Standard Deviation:         {stability_analysis['std_accuracy']:.4f}\n")
            f.write(f"Coefficient of Variation:   {stability_analysis['coefficient_of_variation']:.4f}\n")
            f.write(f"Stability Score:            {stability_analysis['stability_score']:.4f}\n")