"""
Metrics collector for ME-Benchmark
"""
import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import psutil
import threading


class MetricsCollector:
    """Metrics collector for ME-Benchmark"""
    
    def __init__(self, results_dir: str = 'results/'):
        self.results_dir = results_dir
        self.metrics = []
        self.metadata = {}
        self.process = psutil.Process()
        
        # Resource monitoring variables
        self.monitoring = False
        self.monitor_thread = None
        self.resource_usage = []
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def start_resource_monitoring(self):
        """Start monitoring system resources in a separate thread"""
        if not self.monitoring:
            self.monitoring = True
            self.resource_usage = []
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop monitoring system resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources in a loop"""
        while self.monitoring:
            # Collect CPU and memory usage
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            self.resource_usage.append({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms
            })
            
            # Sleep for a short interval
            time.sleep(0.1)
    
    def collect_edit_metrics(self, edit_success: bool, edit_time: float, 
                           original_metrics: Dict[str, Any], edited_metrics: Dict[str, Any],
                           metadata: Dict[str, Any] = None):
        """Collect metrics related to the editing process"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'edit_success': edit_success,
            'edit_time': edit_time,
            'original_metrics': original_metrics,
            'edited_metrics': edited_metrics,
            'metadata': metadata or {}
        }
        
        # Add resource usage if available
        if self.resource_usage:
            entry['resource_usage'] = self.resource_usage.copy()
            # Calculate resource statistics
            if self.resource_usage:
                cpu_values = [r['cpu_percent'] for r in self.resource_usage]
                memory_values = [r['memory_rss'] for r in self.resource_usage]
                
                entry['resource_statistics'] = {
                    'cpu_percent': {
                        'mean': sum(cpu_values) / len(cpu_values),
                        'max': max(cpu_values)
                    },
                    'memory_rss': {
                        'mean': sum(memory_values) / len(memory_values),
                        'max': max(memory_values)
                    }
                }
        
        self.metrics.append(entry)
        return entry
    
    def collect_evaluation_metrics(self, evaluation_results: Dict[str, Any],
                                 evaluation_time: float,
                                 metadata: Dict[str, Any] = None):
        """Collect metrics related to the evaluation process"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': evaluation_results,
            'evaluation_time': evaluation_time,
            'metadata': metadata or {}
        }
        self.metrics.append(entry)
        return entry
    
    def save_metrics(self, filename: str = None):
        """Save all metrics to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Save metrics to JSON file
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_metrics(self, filepath: str):
        """Load metrics from a file"""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
        
        return self.metrics
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all collected metrics"""
        return self.metrics
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.metrics = []
    
    def calculate_edit_success_rate(self) -> float:
        """Calculate the overall edit success rate"""
        if not self.metrics:
            return 0.0
        
        edit_entries = [m for m in self.metrics if 'edit_success' in m]
        if not edit_entries:
            return 0.0
        
        successful_edits = sum(1 for m in edit_entries if m['edit_success'])
        return successful_edits / len(edit_entries)
    
    def calculate_average_edit_time(self) -> float:
        """Calculate the average edit time"""
        if not self.metrics:
            return 0.0
        
        edit_entries = [m for m in self.metrics if 'edit_time' in m]
        if not edit_entries:
            return 0.0
        
        total_time = sum(m['edit_time'] for m in edit_entries)
        return total_time / len(edit_entries)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all collected metrics"""
        if not self.metrics:
            return {}
        
        report = {
            'total_entries': len(self.metrics),
            'edit_metrics': {},
            'evaluation_metrics': {}
        }
        
        # Calculate edit metrics
        edit_entries = [m for m in self.metrics if 'edit_success' in m]
        if edit_entries:
            report['edit_metrics'] = {
                'total_edits': len(edit_entries),
                'success_rate': self.calculate_edit_success_rate(),
                'average_edit_time': self.calculate_average_edit_time(),
                'successful_edits': sum(1 for m in edit_entries if m['edit_success']),
                'failed_edits': sum(1 for m in edit_entries if not m['edit_success'])
            }
        
        # Calculate evaluation metrics
        eval_entries = [m for m in self.metrics if 'evaluation_results' in m]
        if eval_entries:
            # Aggregate evaluation results
            aggregated_results = {}
            for entry in eval_entries:
                for key, value in entry['evaluation_results'].items():
                    if key not in aggregated_results:
                        aggregated_results[key] = []
                    aggregated_results[key].append(value)
            
            # Calculate statistics for each metric
            report['evaluation_metrics'] = {
                'total_evaluations': len(eval_entries)
            }
            
            for key, values in aggregated_results.items():
                report['evaluation_metrics'][key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return report