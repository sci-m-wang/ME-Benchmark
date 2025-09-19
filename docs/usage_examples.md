# ME-Benchmark Usage Examples

This document provides detailed usage examples for the ME-Benchmark framework, covering various scenarios and use cases.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Configuration Examples](#configuration-examples)
3. [Programmatic Usage](#programmatic-usage)
4. [Custom Component Implementation](#custom-component-implementation)
5. [Advanced Usage](#advanced-usage)
6. [Result Analysis](#result-analysis)

## Basic Usage

### Running with Default Configuration

To run the framework with a default configuration:

```bash
python run.py --config configs/example_config.yaml
```

This will:
1. Load the specified model
2. Apply the specified editing method
3. Evaluate the edited model on the specified datasets
4. Save results to the configured output directory

### Command Line Options

```bash
# Basic usage
python run.py --config configs/example_config.yaml

# Specify work directory
python run.py --config configs/example_config.yaml --work-dir /path/to/work/dir

# Enable debug mode
python run.py --config configs/example_config.yaml --debug
```

## Configuration Examples

### Simple Configuration

```yaml
# simple_config.yaml
model:
  type: huggingface
  path: gpt2

editing:
  method: rome
  hparams:
    layers: [5]

evaluation:
  datasets:
    - name: zsre
      metrics: [accuracy]

output:
  results_dir: simple_results/
```

### Advanced Configuration

```yaml
# advanced_config.yaml
model:
  type: huggingface
  path: meta-llama/Llama-2-7b-hf
  model_kwargs:
    device_map: auto
    torch_dtype: float16
  tokenizer_kwargs:
    padding_side: left

editing:
  method: memit
  hparams_path: hparams/MEMIT/llama-2-7b.yaml

evaluation:
  datasets:
    - name: zsre
      split: validation
      metrics: [accuracy, fluency, consistency]
    - name: counterfact
      split: validation
      metrics: [accuracy, locality, portability]
  
  general_benchmarks:
    - name: mmlu
      subset: stem
      metrics: [accuracy]
    - name: hellaswag
      metrics: [accuracy]

runner:
  type: local
  max_workers: 4
  debug: false

output:
  results_dir: advanced_results/
  save_individual_results: true
  save_summary: true
```

### Multi-Editor Configuration

```yaml
# multi_editor_config.yaml
model:
  type: huggingface
  path: gpt2

editors:
  rome:
    type: rome
    hparams:
      layers: [5]
  memit:
    type: memit
    hparams_path: hparams/MEMIT/gpt2.yaml

evaluation:
  datasets:
    - name: zsre
      metrics: [accuracy]

output:
  results_dir: multi_editor_results/
```

## Programmatic Usage

### Simple Programmatic Example

```python
# simple_example.py
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator

def main():
    # Create model
    model = HuggingFaceModel("gpt2")
    
    # Create editor
    editor = ROMEEditor(model, {"layers": [5]})
    
    # Define edit data
    edit_data = [{
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": "London"
    }]
    
    # Apply edit
    success = editor.edit(edit_data)
    print(f"Edit success: {success}")
    
    # Evaluate
    evaluator = KnowledgeEditingEvaluator(model, {"metrics": ["accuracy"]})
    results = evaluator.evaluate(edit_data)
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()
```

### Configuration-Based Programmatic Example

```python
# config_example.py
from me_benchmark.utils.config import load_config
from me_benchmark.factory import create_model, create_editor, create_evaluator, create_dataset
from me_benchmark.runners.local import LocalRunner

def main():
    # Load configuration
    config = load_config("configs/example_config.yaml")
    
    # Create components using factory
    model = create_model(config["model"])
    editor = create_editor(config["editing"], model)
    evaluator = create_evaluator(config["evaluation"], model)
    dataset = create_dataset({"path": "data/zsre.json"})
    
    # Get data
    edit_data = dataset.get_edit_data()
    eval_data = dataset.get_eval_data()
    
    # Apply editing
    edit_success = editor.edit(edit_data)
    print(f"Edit success: {edit_success}")
    
    # Evaluate
    results = evaluator.evaluate(eval_data)
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()
```

### Custom Workflow Example

```python
# custom_workflow.py
import time
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator
from me_benchmark.utils.results_manager import ResultsManager

def benchmark_editing_method(model, editor, evaluator, edit_data, method_name):
    """Benchmark a specific editing method"""
    # Start timing
    start_time = time.time()
    
    # Apply edit
    edit_success = editor.edit(edit_data)
    
    # End timing
    end_time = time.time()
    edit_time = end_time - start_time
    
    # Evaluate
    eval_results = evaluator.evaluate(edit_data)
    
    return {
        "method": method_name,
        "edit_success": edit_success,
        "edit_time": edit_time,
        "results": eval_results
    }

def main():
    # Create components
    model = HuggingFaceModel("gpt2")
    rome_editor = ROMEEditor(model, {"layers": [5]})
    evaluator = KnowledgeEditingEvaluator(model, {"metrics": ["accuracy", "locality", "portability"]})
    
    # Define edit data
    edit_data = [{
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": "London"
    }]
    
    # Benchmark ROME
    rome_results = benchmark_editing_method(model, rome_editor, evaluator, edit_data, "ROME")
    
    # Reset model for next test
    model.reset()
    
    # Print results
    print(f"ROME Results: {rome_results}")
    
    # Save results
    results_manager = ResultsManager("benchmark_results/")
    results_manager.collect_edit_metrics(
        edit_success=rome_results["edit_success"],
        edit_time=rome_results["edit_time"],
        original_metrics={},
        edited_metrics=rome_results["results"],
        metadata={"method": "ROME", "model": "gpt2"}
    )
    results_manager.save_results()

if __name__ == "__main__":
    main()
```

## Custom Component Implementation

### Custom Model Implementation

```python
# custom_model.py
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_model

@register_model('custom_transformer')
class CustomTransformerModel(BaseModel):
    """Custom transformer model implementation"""
    
    def load_model(self):
        """Load the custom transformer model"""
        model_kwargs = self.kwargs.get('model_kwargs', {})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Custom tokenizer setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts with custom parameters"""
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        # Custom generation parameters
        generation_kwargs = {
            'max_new_tokens': 100,
            'pad_token_id': self.tokenizer.pad_token_id,
            'temperature': 0.7,
            'do_sample': True,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode outputs
        generated_texts = []
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            generated_text = self.tokenizer.decode(
                outputs[i][prompt_len:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def get_ppl(self, texts: List[str]) -> List[float]:
        """Calculate perplexity with custom implementation"""
        ppls = []
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                ppls.append(ppl)
        
        return ppls
    
    def edit(self, edit_config: Dict[str, Any]) -> bool:
        """Apply knowledge editing - placeholder implementation"""
        # In a real implementation, this would apply actual editing
        print(f"Applying edit with config: {edit_config}")
        return True
    
    def reset(self) -> bool:
        """Reset model - placeholder implementation"""
        print("Resetting model to original state")
        return True
```

### Custom Editor Implementation

```python
# custom_editor.py
from typing import List, Dict, Any
from me_benchmark.editors.base import BaseEditor
from me_benchmark.registry import register_editor

@register_editor('custom_editor')
class CustomEditor(BaseEditor):
    """Custom editor implementation"""
    
    def __init__(self, model, hparams: Dict[str, Any]):
        super().__init__(model, hparams)
        self.custom_param = hparams.get('custom_param', 'default_value')
    
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        """Apply custom editing method"""
        try:
            print(f"Applying custom edit with param: {self.custom_param}")
            
            # Custom editing logic
            for edit_request in edit_data:
                prompt = edit_request['prompt']
                target = edit_request['target_new']
                print(f"Editing: '{prompt}' -> '{target}'")
            
            # In a real implementation, this would modify model weights
            return True
        except Exception as e:
            print(f"Error during custom editing: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset editing - placeholder implementation"""
        print("Resetting custom editor")
        return True
```

### Custom Evaluator Implementation

```python
# custom_evaluator.py
from typing import List, Dict, Any
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.registry import register_evaluator

@register_evaluator('custom_evaluator')
class CustomEvaluator(BaseEvaluator):
    """Custom evaluator implementation"""
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.custom_metrics = config.get('custom_metrics', ['custom_accuracy'])
    
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate with custom metrics"""
        results = {}
        
        # Custom accuracy metric
        if 'custom_accuracy' in self.custom_metrics:
            results['custom_accuracy'] = self._calculate_custom_accuracy(eval_data)
        
        # Custom fluency metric
        if 'custom_fluency' in self.custom_metrics:
            results['custom_fluency'] = self._calculate_custom_fluency(eval_data)
        
        return results
    
    def _calculate_custom_accuracy(self, eval_data: List[Dict[str, Any]]) -> float:
        """Calculate custom accuracy"""
        # In a real implementation, this would compare model outputs to expected outputs
        # For this example, we'll return a placeholder value
        return 0.85
    
    def _calculate_custom_fluency(self, eval_data: List[Dict[str, Any]]) -> float:
        """Calculate custom fluency"""
        # In a real implementation, this would measure text quality
        # For this example, we'll return a placeholder value
        return 0.92
```

### Custom Dataset Implementation

```python
# custom_dataset.py
import json
from typing import List, Dict, Any
from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset

@register_dataset('custom_dataset')
class CustomDataset(BaseDataset):
    """Custom dataset implementation"""
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load custom dataset"""
        # In a real implementation, this would load from a file or API
        # For this example, we'll create sample data
        self.data = [
            {
                'prompt': "The capital of Germany is",
                'subject': "Germany",
                'target_new': "Berlin",
                'expected_answer': "Berlin"
            },
            {
                'prompt': "The largest planet in our solar system is",
                'subject': "largest planet",
                'target_new': "Jupiter",
                'expected_answer': "Jupiter"
            }
        ]
        return self.data
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        if self.data is None:
            self.load_data()
        return self.data
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        if self.data is None:
            self.load_data()
        return self.data
```

## Advanced Usage

### Multi-Model Comparison

```python
# multi_model_comparison.py
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator
from me_benchmark.utils.results_manager import ResultsManager

def compare_models(model_configs, edit_data):
    """Compare multiple models on the same editing task"""
    results = []
    
    for model_name, model_path in model_configs.items():
        print(f"Testing model: {model_name}")
        
        # Create model
        model = HuggingFaceModel(model_path)
        
        # Create editor
        editor = ROMEEditor(model, {"layers": [5]})
        
        # Create evaluator
        evaluator = KnowledgeEditingEvaluator(model, {"metrics": ["accuracy", "locality"]})
        
        # Apply edit
        edit_success = editor.edit(edit_data)
        
        # Evaluate
        eval_results = evaluator.evaluate(edit_data)
        
        # Store results
        results.append({
            "model": model_name,
            "edit_success": edit_success,
            "results": eval_results
        })
        
        # Reset for next model
        model.reset()
    
    return results

def main():
    # Define models to compare
    model_configs = {
        "GPT-2 Small": "gpt2",
        "GPT-2 Medium": "gpt2-medium",
    }
    
    # Define edit data
    edit_data = [{
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": "London"
    }]
    
    # Compare models
    comparison_results = compare_models(model_configs, edit_data)
    
    # Print results
    for result in comparison_results:
        print(f"Model: {result['model']}")
        print(f"  Edit Success: {result['edit_success']}")
        print(f"  Results: {result['results']}")
        print()
    
    # Save results
    results_manager = ResultsManager("comparison_results/")
    for result in comparison_results:
        results_manager.collect_evaluation_metrics(
            evaluation_results=result['results'],
            evaluation_time=0,  # Placeholder
            metadata={"model": result['model'], "task": "capital_editing"}
        )
    results_manager.save_results()

if __name__ == "__main__":
    main()
```

### Batch Editing

```python
# batch_editing.py
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator

def batch_edit_and_evaluate(model, editor, evaluator, batch_edit_data):
    """Apply multiple edits and evaluate results"""
    results = []
    
    for i, edit_data in enumerate(batch_edit_data):
        print(f"Processing edit {i+1}/{len(batch_edit_data)}")
        
        # Apply edit
        edit_success = editor.edit([edit_data])
        
        # Evaluate
        eval_results = evaluator.evaluate([edit_data])
        
        # Store results
        results.append({
            "edit_index": i,
            "edit_data": edit_data,
            "edit_success": edit_success,
            "evaluation": eval_results
        })
        
        # Reset model for next edit (if needed)
        # model.reset()
    
    return results

def main():
    # Create components
    model = HuggingFaceModel("gpt2")
    editor = ROMEEditor(model, {"layers": [5]})
    evaluator = KnowledgeEditingEvaluator(model, {"metrics": ["accuracy"]})
    
    # Define batch edit data
    batch_edit_data = [
        {
            "prompt": "The capital of France is",
            "subject": "France",
            "target_new": "London"
        },
        {
            "prompt": "The president of the United States is",
            "subject": "the United States",
            "target_new": "John Smith"
        },
        {
            "prompt": "The largest ocean on Earth is",
            "subject": "largest ocean",
            "target_new": "Atlantic Ocean"
        }
    ]
    
    # Process batch
    batch_results = batch_edit_and_evaluate(model, editor, evaluator, batch_edit_data)
    
    # Print results
    for result in batch_results:
        print(f"Edit {result['edit_index'] + 1}:")
        print(f"  Data: {result['edit_data']}")
        print(f"  Success: {result['edit_success']}")
        print(f"  Evaluation: {result['evaluation']}")
        print()

if __name__ == "__main__":
    main()
```

## Result Analysis

### Loading and Analyzing Results

```python
# analyze_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
from me_benchmark.utils.results_manager import ResultsManager

def load_and_analyze_results(results_dir):
    """Load and analyze evaluation results"""
    results_manager = ResultsManager(results_dir)
    
    # Load results
    # Note: This is a simplified example. In practice, you would load from the actual results files
    sample_results = [
        {"model": "gpt2", "accuracy": 0.85, "locality": 0.92, "edit_success": True},
        {"model": "gpt2-medium", "accuracy": 0.87, "locality": 0.91, "edit_success": True},
        {"model": "gpt2-large", "accuracy": 0.89, "locality": 0.89, "edit_success": True},
    ]
    
    return sample_results

def create_comparison_charts(results):
    """Create comparison charts for results"""
    models = [r["model"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    localities = [r["locality"] for r in results]
    
    # Create bar chart
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, localities, width, label='Locality', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

def generate_summary_report(results):
    """Generate a summary report"""
    print("=== ME-Benchmark Summary Report ===")
    print()
    
    print("Model Performance:")
    for result in results:
        print(f"  {result['model']}:")
        print(f"    Accuracy: {result['accuracy']:.3f}")
        print(f"    Locality: {result['locality']:.3f}")
        print(f"    Edit Success: {result['edit_success']}")
        print()
    
    # Calculate averages
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_locality = np.mean([r['locality'] for r in results])
    
    print(f"Average Performance:")
    print(f"  Accuracy: {avg_accuracy:.3f}")
    print(f"  Locality: {avg_locality:.3f}")
    print()
    
    # Find best model
    best_accuracy_model = max(results, key=lambda x: x['accuracy'])
    best_locality_model = max(results, key=lambda x: x['locality'])
    
    print("Best Performing Models:")
    print(f"  By Accuracy: {best_accuracy_model['model']} ({best_accuracy_model['accuracy']:.3f})")
    print(f"  By Locality: {best_locality_model['model']} ({best_locality_model['locality']:.3f})")

def main():
    # Load and analyze results
    results = load_and_analyze_results("results/")
    
    # Generate summary report
    generate_summary_report(results)
    
    # Create comparison charts
    create_comparison_charts(results)

if __name__ == "__main__":
    main()
```

### Custom Visualization

```python
# custom_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_radar_chart(results):
    """Create a radar chart for multi-metric comparison"""
    # Prepare data
    models = [r["model"] for r in results]
    metrics = ["accuracy", "locality", "fluency", "consistency"]
    
    # Create DataFrame
    data = []
    for result in results:
        row = [result["model"]] + [result.get(metric, 0) for metric in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, columns=["Model"] + metrics)
    
    # Normalize data for radar chart
    df_norm = df.copy()
    for metric in metrics:
        df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    # Create radar chart
    angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
    angles += angles[:1]  # Repeat first value to close the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    for i, model in enumerate(models):
        values = df_norm.iloc[i, 1:].tolist()
        values += values[:1]  # Repeat first value to close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Radar Chart')
    
    plt.tight_layout()
    plt.savefig('radar_chart.png')
    plt.show()

def main():
    # Sample data for visualization
    sample_results = [
        {"model": "GPT-2 Small", "accuracy": 0.85, "locality": 0.92, "fluency": 0.88, "consistency": 0.87},
        {"model": "GPT-2 Medium", "accuracy": 0.87, "locality": 0.91, "fluency": 0.89, "consistency": 0.88},
        {"model": "GPT-2 Large", "accuracy": 0.89, "locality": 0.89, "fluency": 0.91, "consistency": 0.90},
    ]
    
    # Create radar chart
    create_radar_chart(sample_results)

if __name__ == "__main__":
    main()
```

## Best Practices

### Configuration Best Practices

1. **Model Selection**: Choose models appropriate for your hardware and use case
2. **Editing Method**: Select editing methods based on your requirements for speed vs. effectiveness
3. **Evaluation Metrics**: Define relevant metrics for your specific use case
4. **Resource Management**: Configure appropriate worker counts and memory settings

### Performance Optimization

1. **Batch Processing**: Use batched inputs when possible to maximize GPU utilization
2. **Memory Management**: Configure appropriate device mapping and data types (e.g., float16)
3. **Caching**: Reuse loaded models when running multiple experiments
4. **Parallel Execution**: Use multiple workers for independent experiments

### Result Analysis

1. **Statistical Significance**: Run multiple experiments to ensure reliable results
2. **Comparative Analysis**: Compare different models and editing methods systematically
3. **Visualization**: Use built-in visualization tools to understand results
4. **Reporting**: Generate comprehensive reports for stakeholders

### Error Handling

1. **Validation**: Validate configurations before running experiments
2. **Logging**: Implement comprehensive logging for debugging
3. **Recovery**: Implement checkpointing for long-running experiments
4. **Graceful Degradation**: Handle missing components gracefully