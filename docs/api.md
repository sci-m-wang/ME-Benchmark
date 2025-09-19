# ME-Benchmark API Documentation

## Overview

This document provides detailed documentation for the ME-Benchmark API, including all core components, their methods, and usage examples.

## Core Modules

### me_benchmark.registry

The registry module provides a central registration system for all components in the framework.

#### Classes

##### Registry

```python
class Registry:
    def __init__(self):
        self._models = {}
        self._editors = {}
        self._evaluators = {}
        self._datasets = {}
        self._runners = {}
        self._summarizers = {}
```

**Methods:**

- `register_model(name: str, cls: Type)`: Register a model class
- `register_editor(name: str, cls: Type)`: Register an editor class
- `register_evaluator(name: str, cls: Type)`: Register an evaluator class
- `register_dataset(name: str, cls: Type)`: Register a dataset class
- `register_runner(name: str, cls: Type)`: Register a runner class
- `register_summarizer(name: str, cls: Type)`: Register a summarizer class
- `get_model(name: str) -> Type`: Get a registered model class
- `get_editor(name: str) -> Type`: Get a registered editor class
- `get_evaluator(name: str) -> Type`: Get a registered evaluator class
- `get_dataset(name: str) -> Type`: Get a registered dataset class
- `get_runner(name: str) -> Type`: Get a registered runner class
- `get_summarizer(name: str) -> Type`: Get a registered summarizer class

#### Decorators

- `register_model(name: str)`: Decorator to register a model class
- `register_editor(name: str)`: Decorator to register an editor class
- `register_evaluator(name: str)`: Decorator to register an evaluator class
- `register_dataset(name: str)`: Decorator to register a dataset class
- `register_runner(name: str)`: Decorator to register a runner class
- `register_summarizer(name: str)`: Decorator to register a summarizer class

#### Global Instance

```python
REGISTRY = Registry()
```

### me_benchmark.factory

The factory module provides functions to create framework components based on configuration.

#### Functions

##### create_model

```python
def create_model(model_config: Dict[str, Any]) -> BaseModel
```

Create a model instance based on configuration.

**Parameters:**
- `model_config`: Dictionary containing model configuration

**Returns:**
- `BaseModel`: Model instance

##### create_editor

```python
def create_editor(editor_config: Dict[str, Any], model: BaseModel) -> BaseEditor
```

Create an editor instance based on configuration.

**Parameters:**
- `editor_config`: Dictionary containing editor configuration
- `model`: Model instance to edit

**Returns:**
- `BaseEditor`: Editor instance

##### create_evaluator

```python
def create_evaluator(evaluator_config: Dict[str, Any], model: BaseModel) -> BaseEvaluator
```

Create an evaluator instance based on configuration.

**Parameters:**
- `evaluator_config`: Dictionary containing evaluator configuration
- `model`: Model instance to evaluate

**Returns:**
- `BaseEvaluator`: Evaluator instance

##### create_dataset

```python
def create_dataset(dataset_config: Dict[str, Any]) -> BaseDataset
```

Create a dataset instance based on configuration.

**Parameters:**
- `dataset_config`: Dictionary containing dataset configuration

**Returns:**
- `BaseDataset`: Dataset instance

##### create_runner

```python
def create_runner(runner_config: Dict[str, Any], config: Dict[str, Any]) -> BaseRunner
```

Create a runner instance based on configuration.

**Parameters:**
- `runner_config`: Dictionary containing runner configuration
- `config`: Full configuration dictionary

**Returns:**
- `BaseRunner`: Runner instance

##### create_summarizer

```python
def create_summarizer(summarizer_config: Dict[str, Any]) -> BaseSummarizer
```

Create a summarizer instance based on configuration.

**Parameters:**
- `summarizer_config`: Dictionary containing summarizer configuration

**Returns:**
- `BaseSummarizer`: Summarizer instance

### me_benchmark.models.base

Base classes for model implementations.

#### Classes

##### BaseModel

```python
class BaseModel(ABC):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.load_model()
```

**Abstract Methods:**

- `load_model()`: Load the model
- `generate(prompts: List[str], **kwargs) -> List[str]`: Generate text from prompts
- `get_ppl(texts: List[str]) -> List[float]`: Calculate perplexity of texts
- `edit(edit_config: Dict[str, Any]) -> bool`: Apply knowledge editing to the model
- `reset() -> bool`: Reset the model to its original state

### me_benchmark.models.huggingface

HuggingFace model implementation.

#### Classes

##### HuggingFaceModel

```python
@register_model('huggingface')
class HuggingFaceModel(BaseModel):
    def load_model(self):
        # Implementation details...
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Implementation details...
    
    def get_ppl(self, texts: List[str]) -> List[float]:
        # Implementation details...
    
    def edit(self, edit_config: Dict[str, Any]) -> bool:
        # Implementation details...
    
    def reset(self) -> bool:
        # Implementation details...
```

### me_benchmark.editors.base

Base classes for editor implementations.

#### Classes

##### BaseEditor

```python
class BaseEditor(ABC):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        self.model = model
        self.hparams = hparams
```

**Abstract Methods:**

- `edit(edit_data: List[Dict[str, Any]]) -> bool`: Apply knowledge editing to the model
- `reset() -> bool`: Reset the model to its original state

### me_benchmark.editors.manager

Model editing manager for unified interface.

#### Classes

##### ModelEditingManager

```python
class ModelEditingManager:
    def __init__(self):
        self.available_editors = {}
        self._discover_editors()
```

**Methods:**

- `get_editor(model: BaseModel, editor_config: Dict[str, Any]) -> BaseEditor`: Create an editor instance
- `apply_edit(model: BaseModel, editor_config: Dict[str, Any], edit_data: List[Dict[str, Any]]) -> bool`: Apply knowledge editing
- `reset_model(model: BaseModel, editor_config: Dict[str, Any]) -> bool`: Reset the model

#### Global Instance

```python
EDITING_MANAGER = ModelEditingManager()
```

### me_benchmark.editors.rome

ROME editor implementation.

#### Classes

##### ROMEEditor

```python
@register_editor('rome')
class ROMEEditor(BaseEditor):
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        # Implementation details...
    
    def reset(self) -> bool:
        # Implementation details...
```

### me_benchmark.editors.easyedit_editor

EasyEdit editor implementation.

#### Classes

##### EasyEditEditor

```python
@register_editor('easyedit')
class EasyEditEditor(BaseEditor):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        super().__init__(model, hparams)
        self.edit_method = hparams.get('edit_method', 'ROME')
        self.hparams_obj = None
        self._load_hparams()
```

**Methods:**

- `_load_hparams()`: Load hyperparameters for the specified editing method
- `edit(edit_data: List[Dict[str, Any]]) -> bool`: Apply knowledge editing to the model
- `reset() -> bool`: Reset the model to its original state

##### ROMEEasyEditor

```python
@register_editor('rome_easyedit')
class ROMEEasyEditor(EasyEditEditor):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'ROME'
        super().__init__(model, hparams)
```

##### FTEasyEditor

```python
@register_editor('ft_easyedit')
class FTEasyEditor(EasyEditEditor):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'FT'
        super().__init__(model, hparams)
```

##### IKEEasyEditor

```python
@register_editor('ike_easyedit')
class IKEEasyEditor(EasyEditEditor):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'IKE'
        super().__init__(model, hparams)
```

### me_benchmark.evaluators.base

Base classes for evaluator implementations.

#### Classes

##### BaseEvaluator

```python
class BaseEvaluator(ABC):
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
```

**Abstract Methods:**

- `evaluate(eval_data: List[Dict[str, Any]]) -> Dict[str, Any]`: Evaluate the model on given data

### me_benchmark.evaluators.knowledge_editing

Knowledge editing evaluator implementation.

#### Classes

##### KnowledgeEditingEvaluator

```python
@register_evaluator('knowledge_editing')
class KnowledgeEditingEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        super().__init__(model, config)
        self.metrics = config.get('metrics', ['accuracy', 'locality', 'portability'])
```

**Methods:**

- `evaluate(eval_data: List[Dict[str, Any]]) -> Dict[str, Any]`: Evaluate the model on knowledge editing tasks
- `_calculate_accuracy(eval_data: List[Dict[str, Any]]) -> float`: Calculate accuracy on evaluation data
- `_calculate_locality(eval_data: List[Dict[str, Any]]) -> float`: Calculate locality preservation
- `_calculate_portability(eval_data: List[Dict[str, Any]]) -> float`: Calculate portability of edits

### me_benchmark.datasets.base

Base classes for dataset implementations.

#### Classes

##### BaseDataset

```python
class BaseDataset(ABC):
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self.data = None
        self.load_data()
```

**Abstract Methods:**

- `load_data() -> List[Dict[str, Any]]`: Load the dataset
- `get_edit_data() -> List[Dict[str, Any]]`: Get data for knowledge editing
- `get_eval_data() -> List[Dict[str, Any]]`: Get data for evaluation

### me_benchmark.datasets.zsre

ZsRE dataset implementation.

#### Classes

##### ZsREDataset

```python
@register_dataset('zsre')
class ZsREDataset(BaseDataset):
    def load_data(self) -> List[Dict[str, Any]]:
        # Implementation details...
    
    def get_edit_data(self) -> List[Dict[str, Any]]:
        # Implementation details...
    
    def get_eval_data(self) -> List[Dict[str, Any]]:
        # Implementation details...
```

### me_benchmark.runners.base

Base classes for runner implementations.

#### Classes

##### BaseRunner

```python
class BaseRunner(ABC):
    def __init__(self, config: Dict[str, Any], work_dir: str = '.', debug: bool = False):
        self.config = config
        self.work_dir = work_dir
        self.debug = debug
        self.model = None
        self.editor = None
        self.evaluator = None
        self.dataset = None
```

**Abstract Methods:**

- `run()`: Run the evaluation process
- `setup()`: Setup the runner with model, editor, evaluator, and dataset

### me_benchmark.runners.local

Local runner implementation.

#### Classes

##### LocalRunner

```python
class LocalRunner(BaseRunner):
    def setup(self):
        # Implementation details...
    
    def run(self):
        # Implementation details...
    
    def _save_results(self, results: Dict[str, Any]):
        # Implementation details...
```

### me_benchmark.utils.config

Configuration utilities.

#### Functions

##### load_config

```python
def load_config(config_path: str) -> Dict[str, Any]
```

Load configuration from a YAML file.

**Parameters:**
- `config_path`: Path to the configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

##### save_config

```python
def save_config(config: Dict[str, Any], config_path: str)
```

Save configuration to a YAML file.

**Parameters:**
- `config`: Configuration dictionary
- `config_path`: Path to save the configuration file

### me_benchmark.utils.results_manager

Results management utilities.

#### Classes

##### ResultsManager

```python
class ResultsManager:
    def __init__(self, results_dir: str = 'results/'):
        self.results_dir = results_dir
        self.metrics_collector = MetricsCollector(results_dir)
        self.visualizer = ResultsVisualizer(results_dir)
```

**Methods:**

- `collect_edit_metrics(edit_success: bool, edit_time: float, original_metrics: Dict[str, Any], edited_metrics: Dict[str, Any], metadata: Dict[str, Any] = None)`: Collect metrics related to the editing process
- `collect_evaluation_metrics(evaluation_results: Dict[str, Any], evaluation_time: float, metadata: Dict[str, Any] = None)`: Collect metrics related to the evaluation process
- `start_resource_monitoring()`: Start monitoring system resources
- `stop_resource_monitoring()`: Stop monitoring system resources
- `save_results(filename: str = None)`: Save all collected results
- `load_results(filepath: str)`: Load results from a file
- `generate_summary_report(summarizer_type: str = 'default') -> Dict[str, Any]`: Generate a summary report
- `save_summary_report(report: Dict[str, Any], output_path: str)`: Save the summary report to a file
- `create_visualization_report(results: List[Dict[str, Any]] = None, output_dir: str = None)`: Create a visualization report
- `query_results(filter_func) -> List[Dict[str, Any]]`: Query results based on a filter function
- `get_latest_results(n: int = 1) -> List[Dict[str, Any]]`: Get the latest n results

## Configuration Reference

### Model Configuration

```yaml
model:
  type: huggingface  # Model loader type
  path: meta-llama/Llama-2-7b-hf  # Model path or identifier
  model_kwargs:  # Additional arguments for model loading
    device_map: auto
    torch_dtype: float16
  tokenizer_kwargs:  # Additional arguments for tokenizer
    padding_side: left
```

### Editing Configuration

```yaml
editing:
  method: rome  # Editing method to use
  hparams_path: hparams/ROME/llama-2-7b.yaml  # Path to hyperparameters file
  # OR
  hparams:  # Inline hyperparameters
    layers: [5]
    mom2_adjustment: true
```

### Evaluation Configuration

```yaml
evaluation:
  datasets:  # Knowledge editing datasets
    - name: zsre
      split: validation
      metrics: [accuracy, fluency, consistency]
    - name: counterfact
      split: validation
      metrics: [accuracy, locality, portability]
  
  general_benchmarks:  # General language understanding benchmarks
    - name: mmlu
      subset: stem
      metrics: [accuracy]
    - name: hellaswag
      metrics: [accuracy]
```

### Runner Configuration

```yaml
runner:
  type: local  # Runner type
  max_workers: 4  # Number of parallel workers
  debug: false  # Enable debug mode
```

### Output Configuration

```yaml
output:
  results_dir: results/  # Directory to save results
  save_individual_results: true  # Save individual result files
  save_summary: true  # Save summary report
```

## Error Handling

The framework uses standard Python exception handling. Common exceptions include:

- `ValueError`: Raised for invalid configuration values
- `FileNotFoundError`: Raised when required files are not found
- `ImportError`: Raised when required modules cannot be imported
- `RuntimeError`: Raised for runtime errors during execution

## Performance Considerations

1. **Memory Management**: Use appropriate device mapping and data types
2. **Batch Processing**: Process multiple inputs in batches when possible
3. **Caching**: Reuse loaded models and components
4. **Parallel Execution**: Use multiple workers for independent tasks
5. **Resource Monitoring**: Monitor system resources during execution

## Extending the Framework

To extend the framework with new components:

1. **Create a new class** that inherits from the appropriate base class
2. **Implement required methods** as specified in the base class
3. **Register the class** using the appropriate decorator
4. **Update configuration** to use the new component