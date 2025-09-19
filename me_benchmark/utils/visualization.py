"""
Visualization module for ME-Benchmark results
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List
import numpy as np


class ResultsVisualizer:
    """Results visualizer for ME-Benchmark"""
    
    def __init__(self, results_dir: str = 'results/'):
        self.results_dir = results_dir
        plt.style.use('seaborn-v0_8')
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """Load results from a JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_metric_comparison(self, results: List[Dict[str, Any]], 
                             metric_name: str, output_path: str = None):
        """Plot comparison of a specific metric across different experiments"""
        # Extract metric values and labels
        metric_values = []
        labels = []
        
        for entry in results:
            if 'evaluation_results' in entry and metric_name in entry['evaluation_results']:
                metric_values.append(entry['evaluation_results'][metric_name])
                # Create a label from metadata
                metadata = entry.get('metadata', {})
                label_parts = []
                if 'model' in metadata:
                    label_parts.append(metadata['model'])
                if 'editor' in metadata:
                    label_parts.append(metadata['editor'])
                if 'dataset' in metadata:
                    label_parts.append(metadata['dataset'])
                
                labels.append('-'.join(label_parts) if label_parts else f"Experiment {len(labels)+1}")
        
        if not metric_values:
            print(f"No data found for metric: {metric_name}")
            return
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(metric_values)), metric_values, color='skyblue')
        plt.xlabel('Experiments')
        plt.ylabel(metric_name)
        plt.title(f'Comparison of {metric_name} Across Experiments')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_edit_success_rate(self, results: List[Dict[str, Any]], 
                             output_path: str = None):
        """Plot edit success rate across experiments"""
        success_rates = []
        labels = []
        
        for entry in results:
            if 'edit_success' in entry:
                success_rates.append(1.0 if entry['edit_success'] else 0.0)
                # Create a label from metadata
                metadata = entry.get('metadata', {})
                label_parts = []
                if 'model' in metadata:
                    label_parts.append(metadata['model'])
                if 'editor' in metadata:
                    label_parts.append(metadata['editor'])
                
                labels.append('-'.join(label_parts) if label_parts else f"Experiment {len(labels)+1}")
        
        if not success_rates:
            print("No edit success data found")
            return
        
        # Calculate overall success rate
        overall_success_rate = sum(success_rates) / len(success_rates)
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(success_rates)), success_rates, color=['green' if s else 'red' for s in success_rates])
        plt.xlabel('Edit Operations')
        plt.ylabel('Success Rate')
        plt.title(f'Edit Success Rate (Overall: {overall_success_rate:.2%})')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, success_rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    'Success' if value else 'Failed', ha='center', va='bottom', 
                    rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_resource_usage(self, results: List[Dict[str, Any]], 
                          output_path: str = None):
        """Plot resource usage during editing process"""
        # Extract resource usage data
        cpu_data = []
        memory_data = []
        labels = []
        
        for entry in results:
            if 'resource_statistics' in entry:
                cpu_stats = entry['resource_statistics'].get('cpu_percent', {})
                memory_stats = entry['resource_statistics'].get('memory_rss', {})
                
                if 'mean' in cpu_stats and 'mean' in memory_stats:
                    cpu_data.append(cpu_stats['mean'])
                    memory_data.append(memory_stats['mean'] / (1024 * 1024))  # Convert to MB
                    # Create a label from metadata
                    metadata = entry.get('metadata', {})
                    label_parts = []
                    if 'model' in metadata:
                        label_parts.append(metadata['model'])
                    if 'editor' in metadata:
                        label_parts.append(metadata['editor'])
                    
                    labels.append('-'.join(label_parts) if label_parts else f"Experiment {len(labels)+1}")
        
        if not cpu_data or not memory_data:
            print("No resource usage data found")
            return
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CPU usage plot
        bars1 = ax1.bar(range(len(cpu_data)), cpu_data, color='lightcoral')
        ax1.set_xlabel('Experiments')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('Average CPU Usage During Editing')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, cpu_data)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Memory usage plot
        bars2 = ax2.bar(range(len(memory_data)), memory_data, color='lightblue')
        ax2.set_xlabel('Experiments')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Average Memory Usage During Editing')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars2, memory_data)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_comprehensive_report(self, results: List[Dict[str, Any]], 
                                  output_dir: str = None):
        """Create a comprehensive visualization report"""
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot various metrics
        metrics_to_plot = ['accuracy', 'locality', 'portability', 'fluency']
        
        for metric in metrics_to_plot:
            output_path = os.path.join(output_dir, f'{metric}_comparison.png')
            try:
                self.plot_metric_comparison(results, metric, output_path)
                print(f"Created {metric} comparison plot: {output_path}")
            except Exception as e:
                print(f"Failed to create {metric} comparison plot: {e}")
        
        # Plot edit success rate
        output_path = os.path.join(output_dir, 'edit_success_rate.png')
        try:
            self.plot_edit_success_rate(results, output_path)
            print(f"Created edit success rate plot: {output_path}")
        except Exception as e:
            print(f"Failed to create edit success rate plot: {e}")
        
        # Plot resource usage
        output_path = os.path.join(output_dir, 'resource_usage.png')
        try:
            self.plot_resource_usage(results, output_path)
            print(f"Created resource usage plot: {output_path}")
        except Exception as e:
            print(f"Failed to create resource usage plot: {e}")
        
        print(f"Comprehensive visualization report saved to: {output_dir}")