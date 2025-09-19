"""
Local runner implementation for ME-Benchmark
"""
"""Local runner implementation for ME-Benchmark"""
import os
import json
import time
from typing import Dict, Any

from me_benchmark.runners.base import BaseRunner
from me_benchmark.factory import create_model, create_editor, create_evaluator, create_dataset, create_summarizer
from me_benchmark.utils.results_manager import ResultsManager


class LocalRunner(BaseRunner):
    """Local runner implementation"""
    
    def setup(self):
        """Setup the runner with model, editor, evaluator, and dataset"""
        # Initialize model using factory
        model_config = self.config['model']
        self.model = create_model(model_config)
        
        # Initialize editor using factory
        editing_config = self.config['editing']
        self.editor = create_editor(editing_config, self.model)
        
        # Initialize evaluator using factory
        evaluation_config = self.config['evaluation']
        self.evaluator = create_evaluator(evaluation_config, self.model)
        
        # Initialize dataset using factory
        self.dataset = create_dataset({'path': 'data/zsre.json'})
        
        # Initialize results manager
        output_config = self.config.get('output', {})
        results_dir = output_config.get('results_dir', 'results/')
        self.results_manager = ResultsManager(results_dir)
    
    def run(self):
        """Run the evaluation process"""
        # Setup components
        self.setup()
        
        # Get edit data
        edit_data = self.dataset.get_edit_data()
        
        # Start resource monitoring
        self.results_manager.start_resource_monitoring()
        
        # Record edit start time
        edit_start_time = time.time()
        
        # Apply editing
        print("Applying knowledge editing...")
        edit_success = self.editor.edit(edit_data)
        
        # Record edit end time
        edit_end_time = time.time()
        edit_time = edit_end_time - edit_start_time
        
        # Stop resource monitoring
        self.results_manager.stop_resource_monitoring()
        
        if not edit_success:
            # Collect edit failure metrics
            self.results_manager.collect_edit_metrics(
                edit_success=False,
                edit_time=edit_time,
                original_metrics={},
                edited_metrics={},
                metadata={
                    'model': self.config['model']['type'],
                    'editor': self.config['editing']['method'],
                    'dataset': 'zsre'
                }
            )
            raise RuntimeError("Failed to apply knowledge editing")
        
        # Get evaluation data
        eval_data = self.dataset.get_eval_data()
        
        # Record evaluation start time
        eval_start_time = time.time()
        
        # Evaluate
        print("Evaluating edited model...")
        results = self.evaluator.evaluate(eval_data)
        
        # Record evaluation end time
        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time
        
        # Collect evaluation metrics
        self.results_manager.collect_evaluation_metrics(
            evaluation_results=results,
            evaluation_time=eval_time,
            metadata={
                'model': self.config['model']['type'],
                'editor': self.config['editing']['method'],
                'dataset': 'zsre'
            }
        )
        
        # Collect edit success metrics
        self.results_manager.collect_edit_metrics(
            edit_success=True,
            edit_time=edit_time,
            original_metrics={},
            edited_metrics=results,
            metadata={
                'model': self.config['model']['type'],
                'editor': self.config['editing']['method'],
                'dataset': 'zsre'
            }
        )
        
        # Save results
        self._save_results(results)
        
        # Print results
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to a file"""
        output_config = self.config.get('output', {})
        results_dir = output_config.get('results_dir', 'results/')
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results using results manager
        results_path = self.results_manager.save_results()
        print(f"Results saved to {results_path}")
        
        # Generate and save summary report
        summary_report = self.results_manager.generate_summary_report()
        summary_path = os.path.join(results_dir, 'summary_report.json')
        self.results_manager.save_summary_report(summary_report, summary_path)
        print(f"Summary report saved to {summary_path}")
        
        # Create visualization report
        self.results_manager.create_visualization_report()
        print(f"Visualization report created in {os.path.join(results_dir, 'visualizations')}")
        
        # Also save individual results file
        individual_path = os.path.join(results_dir, 'results.json')
        with open(individual_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Individual results saved to {individual_path}")
