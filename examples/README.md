# ME-Benchmark Examples

This directory contains examples demonstrating how to use the ME-Benchmark framework.

## Quick Start Examples

### Basic Demo

To run the basic demo that shows registered components:

```bash
python examples/demo.py
```

This example will display all registered components in the framework:
- Models
- Editors
- Evaluators
- Datasets
- Runners
- Summarizers

### Comprehensive Example

To run the comprehensive example that demonstrates configuration, results collection, and visualization:

```bash
python examples/comprehensive_example.py
```

This example will:
1. Show all registered components
2. Create an example configuration file
3. Demonstrate results collection functionality
4. Show visualization capabilities

## Example Scripts

### demo.py

A simple script that demonstrates how to import and use ME-Benchmark components:

```python
from me_benchmark import REGISTRY
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
# ... other imports

def main():
    # Show registered components
    print("Registered Models:")
    for name in REGISTRY._models:
        print(f"  - {name}")
    # ... show other components
```

### comprehensive_example.py

A more detailed example that demonstrates the full workflow:

```python
from me_benchmark import REGISTRY
from me_benchmark.utils.config import load_config, save_config
from me_benchmark.utils.results_collector import ResultsCollector
from me_benchmark.utils.visualization import Visualizer

def show_registered_components():
    """Show all registered components"""
    # Implementation details...

def create_example_config():
    """Create an example configuration file"""
    config = {
        "model": {
            "type": "huggingface",
            "path": "meta-llama/Llama-2-7b-hf",
            # ... other config
        }
        # ... other config sections
    }
    save_config(config, "example_config.yaml")
    return config

def demonstrate_results_collection():
    """Demonstrate results collection functionality"""
    # Implementation details...

def demonstrate_visualization(results):
    """Demonstrate visualization functionality"""
    # Implementation details...
```

## Example Configuration

The comprehensive example will generate an `example_config.yaml` file that shows how to configure the framework for different models, editors, and evaluation benchmarks.

Example configuration structure:

```yaml
model:
  type: huggingface
  path: meta-llama/Llama-2-7b-hf
  model_kwargs:
    device_map: auto
    torch_dtype: float16

editing:
  method: rome
  hparams_path: hparams/ROME/llama-2-7b.yaml

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
  results_dir: results/
  save_individual_results: true
  save_summary: true
```

## Sample Results

The examples will also generate sample results in the `example_results/` directory to demonstrate the results collection and analysis capabilities.

Sample results structure:

```json
{
  "experiment_1": {
    "model": "llama-2-7b",
    "accuracy": 0.85,
    "fluency": 0.92,
    "consistency": 0.88
  },
  "experiment_2": {
    "model": "llama-2-7b",
    "accuracy": 0.87,
    "fluency": 0.91,
    "consistency": 0.89
  }
}
```

## Custom Usage

To use ME-Benchmark with your own configuration:

1. Create a configuration file similar to the example
2. Run the framework with your configuration:

```bash
python run.py --config path/to/your/config.yaml
```

### Custom Configuration Example

Create a custom configuration file `my_custom_config.yaml`:

```yaml
model:
  type: huggingface
  path: gpt2
  model_kwargs:
    device_map: auto

editing:
  method: rome
  hparams:
    layers: [5]
    mom2_adjustment: true

evaluation:
  datasets:
    - name: zsre
      split: validation
      metrics: [accuracy]

runner:
  type: local
  max_workers: 1
  debug: true

output:
  results_dir: my_custom_results/
  save_individual_results: true
  save_summary: true
```

Run with custom configuration:

```bash
python run.py --config my_custom_config.yaml
```

## Extending the Framework

You can extend ME-Benchmark by implementing custom components:

### 1. Custom Models

Extend `BaseModel` and register with `@register_model`:

```python
from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_model

@register_model('my_custom_model')
class MyCustomModel(BaseModel):
    def load_model(self):
        # Load your custom model
        pass
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Generate text
        pass
    
    def get_ppl(self, texts: List[str]) -> List[float]:
        # Calculate perplexity
        pass
    
    def edit(self, edit_config: Dict[str, Any]) -> bool:
        # Apply editing
        pass
    
    def reset(self) -> bool:
        # Reset model
        pass
```

### 2. Custom Editors

Extend `BaseEditor` and register with `@register_editor`:

```python
from me_benchmark.editors.base import BaseEditor
from me_benchmark.registry import register_editor

@register_editor('my_custom_editor')
class MyCustomEditor(BaseEditor):
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        # Apply editing
        pass
    
    def reset(self) -> bool:
        # Reset editing
        pass
```

### 3. Custom Evaluators

Extend `BaseEvaluator` and register with `@register_evaluator`:

```python
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.registry import register_evaluator

@register_evaluator('my_custom_evaluator')
class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Custom evaluation logic
        results = {}
        # ... implementation
        return results
```

### 4. Custom Datasets

Extend `BaseDataset` and register with `@register_dataset`:

```python
from me_benchmark.datasets.base import BaseDataset
from me_benchmark.registry import register_dataset

@register_dataset('my_custom_dataset')
class MyCustomDataset(BaseDataset):
    def load_data(self) -> List[Dict[str, Any]]:
        # Load your custom dataset
        pass
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        # Get data for knowledge editing
        pass
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        # Get data for evaluation
        pass
```

### 5. Custom Runners

Extend `BaseRunner` and register with `@register_runner`:

```python
from me_benchmark.runners.base import BaseRunner
from me_benchmark.registry import register_runner

@register_runner('my_custom_runner')
class MyCustomRunner(BaseRunner):
    def run(self):
        # Custom execution logic
        pass
    
    def setup(self):
        # Setup custom components
        pass
```

### 6. Custom Summarizers

Extend `BaseSummarizer` and register with `@register_summarizer`:

```python
from me_benchmark.summarizers.base import BaseSummarizer
from me_benchmark.registry import register_summarizer

@register_summarizer('my_custom_summarizer')
class MyCustomSummarizer(BaseSummarizer):
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Custom summarization logic
        summary = {}
        # ... implementation
        return summary
```

## Best Practices

### Configuration Best Practices

1. **Model Selection**: Choose models appropriate for your hardware
2. **Editing Method**: Select editing methods based on your requirements
3. **Evaluation Metrics**: Define relevant metrics for your use case
4. **Resource Management**: Configure appropriate worker counts and memory settings

### Performance Optimization

1. **Batch Processing**: Use batched inputs when possible
2. **Memory Management**: Configure appropriate device mapping
3. **Caching**: Reuse loaded models when running multiple experiments
4. **Parallel Execution**: Use multiple workers for independent experiments

### Result Analysis

1. **Statistical Significance**: Run multiple experiments to ensure reliable results
2. **Comparative Analysis**: Compare different models and editing methods
3. **Visualization**: Use built-in visualization tools to understand results
4. **Reporting**: Generate comprehensive reports for stakeholders