# ME-Benchmark: Model Editing Benchmark Framework

## Overview

ME-Benchmark is a comprehensive framework for evaluating and benchmarking knowledge editing techniques on large language models. It integrates model loading, editing methods, evaluation benchmarks, and result analysis into a unified platform.

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ME-Benchmark

# Install dependencies
pip install -r requirements.txt

# Install EasyEdit (optional, for additional editing methods)
cd EasyEdit
pip install -r requirements.txt
```

### Running Your First Evaluation

1. Create a configuration file (see examples in `configs/` directory)
2. Run the framework with your configuration:

```bash
python run.py --config configs/example_config.yaml
```

3. Check the results in the `results/` directory

### Basic Example

```python
# Simple example script
from me_benchmark import REGISTRY
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.editors.rome import ROMEEditor
from me_benchmark.evaluators.knowledge_editing import KnowledgeEditingEvaluator

# Create model
model = HuggingFaceModel("gpt2")

# Create editor
editor = ROMEEditor(model, {"layers": [5]})

# Apply edit
edit_data = [{
    "prompt": "The capital of France is",
    "subject": "France",
    "target_new": "London"
}]
editor.edit(edit_data)

# Evaluate
evaluator = KnowledgeEditingEvaluator(model, {"metrics": ["accuracy"]})
results = evaluator.evaluate(edit_data)
print(results)
```

## Documentation

For detailed documentation, please see the following resources:

- [API Documentation](docs/api.md) - Complete API reference for all components
- [Usage Examples](docs/usage_examples.md) - Detailed examples for various use cases
- [Directory Structure](docs/directory_structure.md) - Explanation of the project organization
- [Frequently Asked Questions](docs/faq.md) - Common issues and solutions

## Architecture Design

The framework follows a modular architecture similar to opencompass, with the following core components:

### 1. Model Loaders
Support for various model backends:
- HuggingFace Transformers
- vLLM
- LMDeploy
- Custom model implementations

### 2. Editing Methods
Integration of various knowledge editing techniques:
- ROME (Rank-One Model Editing)
- MEMIT (Memory-Efficient Model Editing)
- MEND (Model Editor Networks)
- IKE (In-Context Knowledge Editor)
- LoRA (Low-Rank Adaptation)
- Custom editing algorithms

### 3. Evaluation Benchmarks
Comprehensive evaluation datasets:
- Knowledge editing benchmarks ( zsRE, CounterFact )
- General language understanding ( MMLU, HellaSwag )
- Safety and alignment benchmarks
- Custom benchmark integration

### 4. Result Analysis
Automated evaluation and analysis:
- Editing success metrics
- Retention of unrelated knowledge
- Generalization capabilities
- Efficiency measurements
- Statistical analysis and visualization

## Directory Structure

```
ME-Benchmark/
├── configs/                  # Configuration files
├── docs/                     # Documentation files
├── examples/                 # Example usage scripts
├── me_benchmark/             # Core framework code
│   ├── editors/              # Knowledge editing methods
│   ├── evaluators/           # Evaluation modules
│   ├── models/               # Model loaders
│   ├── datasets/             # Benchmark datasets
│   ├── runners/              # Execution controllers
│   ├── summarizers/          # Result analyzers
│   └── utils/                # Utility functions
├── EasyEdit/                 # EasyEdit integration
├── results/                  # Evaluation results
└── tests/                    # Test files
```

For a detailed explanation of the directory structure, see [Directory Structure Documentation](docs/directory_structure.md).

## Configuration

### Configuration File Structure

The configuration file is a YAML file that defines all aspects of the evaluation:

```yaml
model:
  type: huggingface
  path: meta-llama/Llama-2-7b-hf
  model_kwargs:
    device_map: auto

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
  
  # General language understanding benchmarks
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

### Configuration Options

#### Model Configuration
- `type`: Model loader type (huggingface, vllm, etc.)
- `path`: Model path or identifier
- `model_kwargs`: Additional arguments for model loading

#### Editing Configuration
- `method`: Editing method to use (rome, memit, mend, etc.)
- `hparams_path`: Path to hyperparameters file

#### Evaluation Configuration
- `datasets`: List of knowledge editing datasets
- `general_benchmarks`: List of general language understanding benchmarks

#### Runner Configuration
- `type`: Runner type (local, distributed, etc.)
- `max_workers`: Number of parallel workers
- `debug`: Enable debug mode

#### Output Configuration
- `results_dir`: Directory to save results
- `save_individual_results`: Save individual result files
- `save_summary`: Save summary report

## API Documentation

### Core Components

#### Models

Models are loaded using the `BaseModel` interface:

```python
class BaseModel(ABC):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts"""
        pass
    
    @abstractmethod
    def get_ppl(self, texts: List[str]) -> List[float]:
        """Calculate perplexity of texts"""
        pass
    
    @abstractmethod
    def edit(self, edit_config: Dict[str, Any]) -> bool:
        """Apply knowledge editing to the model"""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the model to its original state"""
        pass
```

#### Editors

Editors implement knowledge editing methods:

```python
class BaseEditor(ABC):
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        self.model = model
        self.hparams = hparams
    
    @abstractmethod
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        """Apply knowledge editing to the model"""
        pass
    
    @abstractmethod
    def reset(self) -> bool:
        """Reset the model to its original state"""
        pass
```

#### Evaluators

Evaluators measure model performance:

```python
class BaseEvaluator(ABC):
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
    
    @abstractmethod
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model on given data"""
        pass
```

#### Datasets

Datasets provide evaluation data:

```python
class BaseDataset(ABC):
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path
        self.kwargs = kwargs
        self.data = None
        self.load_data()
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Load the dataset"""
        pass
    
    @abstractmethod
    def get_edit_data(self) -> List[Dict[str, Any]]:
        """Get data for knowledge editing"""
        pass
    
    @abstractmethod
    def get_eval_data(self) -> List[Dict[str, Any]]:
        """Get data for evaluation"""
        pass
```

#### Runners

Runners execute the evaluation process:

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
    
    @abstractmethod
    def run(self):
        """Run the evaluation process"""
        pass
    
    @abstractmethod
    def setup(self):
        """Setup the runner with model, editor, evaluator, and dataset"""
        pass
```

## Usage Examples

### Example 1: Basic Usage

```bash
# Run with example configuration
python run.py --config configs/example_config.yaml
```

### Example 2: Custom Configuration

Create a custom configuration file `my_config.yaml`:

```yaml
model:
  type: huggingface
  path: gpt2
  model_kwargs:
    device_map: auto

editing:
  method: rome
  hparams_path: hparams/ROME/gpt2.yaml

evaluation:
  datasets:
    - name: zsre
      split: validation
      metrics: [accuracy, fluency, consistency]

runner:
  type: local
  max_workers: 2
  debug: true

output:
  results_dir: my_results/
  save_individual_results: true
  save_summary: true
```

Run with custom configuration:

```bash
python run.py --config my_config.yaml
```

### Example 3: Programmatic Usage

```python
from me_benchmark.utils.config import load_config
from me_benchmark.factory import create_model, create_editor, create_evaluator, create_dataset, create_runner

# Load configuration
config = load_config('configs/example_config.yaml')

# Create components
model = create_model(config['model'])
editor = create_editor(config['editing'], model)
evaluator = create_evaluator(config['evaluation'], model)
dataset = create_dataset({'path': 'data/zsre.json'})

# Apply editing
edit_data = dataset.get_edit_data()
editor.edit(edit_data)

# Evaluate
eval_data = dataset.get_eval_data()
results = evaluator.evaluate(eval_data)
print(results)
```

### Example 4: Extending the Framework

#### Adding a New Model

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

#### Adding a New Editor

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

## Frequently Asked Questions (FAQ)

### Q: What models are supported?

A: ME-Benchmark supports models from HuggingFace Transformers, including GPT, LLaMA, BERT, and others. Additional support for vLLM and LMDeploy is planned.

### Q: What editing methods are available?

A: The framework integrates various editing methods including:
- ROME (Rank-One Model Editing)
- MEMIT (Memory-Efficient Model Editing)
- MEND (Model Editor Networks)
- IKE (In-Context Knowledge Editor)
- LoRA (Low-Rank Adaptation)

### Q: How do I add a new evaluation dataset?

A: Create a new class that inherits from `BaseDataset` and implement the required methods. Then register it using the `@register_dataset` decorator.

### Q: How can I customize the evaluation metrics?

A: Create a new evaluator class that inherits from `BaseEvaluator` and implement your custom metrics in the `evaluate` method.

### Q: What hardware requirements are needed?

A: Requirements depend on the model size:
- Small models (GPT-2, BERT): 8GB+ RAM, no GPU required
- Medium models (LLaMA-7B): 16GB+ RAM, 12GB+ GPU VRAM
- Large models (LLaMA-13B+): 32GB+ RAM, 24GB+ GPU VRAM

### Q: How do I interpret the evaluation results?

A: The framework provides several key metrics:
- Accuracy: Measures if the edited knowledge is correctly applied
- Locality: Measures preservation of unrelated knowledge
- Portability: Measures generalization to related facts
- Fluency: Measures quality of generated text
- Consistency: Measures consistency of responses

### Q: Can I run evaluations in parallel?

A: Yes, the framework supports parallel execution through the `max_workers` configuration option in the runner settings.

### Q: How do I contribute to the project?

A: We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## Development

To extend the framework with new components:

1. **Add a new model backend**: Implement a subclass of `BaseModel` in `me_benchmark/models/`
2. **Add a new editing method**: Implement a subclass of `BaseEditor` in `me_benchmark/editors/`
3. **Add a new evaluation benchmark**: Implement a subclass of `BaseDataset` in `me_benchmark/datasets/`
4. **Add a new runner**: Implement a subclass of `BaseRunner` in `me_benchmark/runners/`
5. **Add a new summarizer**: Implement a subclass of `BaseSummarizer` in `me_benchmark/summarizers/`

## Contributing

We welcome contributions to ME-Benchmark! Please see our contributing guidelines for more information.