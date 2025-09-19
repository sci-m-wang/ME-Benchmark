# ME-Benchmark Directory Structure

This document provides an overview of the ME-Benchmark directory structure and explains the purpose of each component.

## Root Directory

```
ME-Benchmark/
├── configs/                  # Configuration files
├── docs/                     # Documentation files
├── examples/                 # Example usage scripts
├── me_benchmark/             # Core framework code
├── EasyEdit/                 # EasyEdit integration (submodule)
├── tests/                    # Unit and integration tests
├── results/                  # Evaluation results (generated)
├── run.py                    # Main entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## configs/

Configuration files for different experiments and use cases.

```
configs/
├── example_config.yaml              # Basic example configuration
├── unified_model_editing_example.yaml  # Unified model editing example
└── hparams/                         # Hyperparameter files for editing methods
    ├── ROME/
    ├── MEMIT/
    ├── MEND/
    └── ...
```

## docs/

Comprehensive documentation for the framework.

```
docs/
├── api.md                   # API documentation
├── usage_examples.md        # Detailed usage examples
├── faq.md                   # Frequently asked questions
└── ...
```

## examples/

Example scripts demonstrating various aspects of the framework.

```
examples/
├── demo.py                  # Basic demo showing registered components
├── comprehensive_example.py # Comprehensive example with configuration and results
├── README.md                # Examples documentation
└── ...
```

## me_benchmark/

Core framework code organized by component type.

```
me_benchmark/
├── __init__.py              # Package initialization
├── registry.py              # Component registry system
├── factory.py               # Component factory functions
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── base.py              # BaseModel abstract class
│   ├── huggingface.py       # HuggingFace model implementation
│   └── ...
├── editors/                 # Editing method implementations
│   ├── __init__.py
│   ├── base.py              # BaseEditor abstract class
│   ├── manager.py           # Editing method manager
│   ├── rome.py              # ROME editing implementation
│   ├── easyedit_editor.py   # EasyEdit integration
│   └── ...
├── evaluators/              # Evaluation implementations
│   ├── __init__.py
│   ├── base.py              # BaseEvaluator abstract class
│   ├── knowledge_editing.py # Knowledge editing evaluator
│   ├── simple.py            # Simple evaluator
│   └── ...
├── datasets/                # Dataset implementations
│   ├── __init__.py
│   ├── base.py              # BaseDataset abstract class
│   ├── zsre.py              # ZsRE dataset
│   ├── counterfact.py       # CounterFact dataset
│   └── ...
├── runners/                 # Execution runners
│   ├── __init__.py
│   ├── base.py              # BaseRunner abstract class
│   ├── local.py             # Local execution runner
│   └── ...
├── summarizers/             # Result summarization
│   ├── __init__.py
│   ├── base.py              # BaseSummarizer abstract class
│   └── ...
└── utils/                   # Utility functions
    ├── __init__.py
    ├── config.py            # Configuration utilities
    ├── results_manager.py   # Results management
    ├── metrics_collector.py # Metrics collection
    └── ...
```

## EasyEdit/

EasyEdit integration as a submodule.

```
EasyEdit/
├── easyeditor/              # EasyEdit core code
│   ├── models/              # Editing method implementations
│   ├── dataset/             # Datasets
│   ├── editors/             # Editor implementations
│   └── ...
├── examples/                # EasyEdit examples
├── hparams/                 # Hyperparameter files
├── requirements.txt         # EasyEdit dependencies
└── ...
```

## tests/

Unit and integration tests for the framework.

```
tests/
├── README.md                # Testing documentation
├── test_basic.py            # Basic functionality tests
└── ...
```

## Detailed Component Descriptions

### Models (`me_benchmark/models/`)

Model implementations provide a unified interface for loading and interacting with different types of language models.

- **base.py**: Defines the `BaseModel` abstract class that all model implementations must extend
- **huggingface.py**: Implementation for HuggingFace Transformers models
- **huggingface_enhanced.py**: Enhanced HuggingFace model with additional features

### Editors (`me_benchmark/editors/`)

Editor implementations provide different methods for applying knowledge edits to models.

- **base.py**: Defines the `BaseEditor` abstract class
- **manager.py**: Unified interface for managing different editing methods
- **rome.py**: Implementation of Rank-One Model Editing
- **easyedit_editor.py**: Integration with the EasyEdit library

### Evaluators (`me_benchmark/evaluators/`)

Evaluator implementations measure model performance on various tasks.

- **base.py**: Defines the `BaseEvaluator` abstract class
- **knowledge_editing.py**: Evaluator for knowledge editing tasks
- **simple.py**: Simple evaluator for basic metrics
- **mmlu.py**: Evaluator for MMLU benchmark
- **hellaswag.py**: Evaluator for HellaSwag benchmark

### Datasets (`me_benchmark/datasets/`)

Dataset implementations provide evaluation data for different benchmarks.

- **base.py**: Defines the `BaseDataset` abstract class
- **zsre.py**: Zero-Shot Relation Extraction dataset
- **counterfact.py**: CounterFact dataset
- **mmlu.py**: MMLU dataset
- **hellaswag.py**: HellaSwag dataset

### Runners (`me_benchmark/runners/`)

Runner implementations control the execution of evaluation workflows.

- **base.py**: Defines the `BaseRunner` abstract class
- **local.py**: Local execution runner for single-machine experiments

### Summarizers (`me_benchmark/summarizers/`)

Summarizer implementations generate reports from evaluation results.

- **base.py**: Defines the `BaseSummarizer` abstract class

### Utilities (`me_benchmark/utils/`)

Utility functions provide supporting functionality for the framework.

- **config.py**: Configuration loading and saving
- **results_manager.py**: Management of evaluation results
- **metrics_collector.py**: Collection of performance metrics
- **visualization.py**: Visualization of results

## Key Design Principles

### Modularity

Each component type (models, editors, evaluators, etc.) is organized in its own directory with a consistent interface defined by abstract base classes.

### Extensibility

New components can be added by implementing the appropriate base class and registering with the framework's registry system.

### Configuration-Driven

The framework is primarily driven by YAML configuration files, making it easy to experiment with different combinations of components.

### Separation of Concerns

Each component has a single responsibility:
- Models handle loading and inference
- Editors handle knowledge editing
- Evaluators handle performance measurement
- Datasets provide evaluation data
- Runners coordinate execution
- Summarizers generate reports

## File Naming Conventions

- **Abstract base classes**: `base.py`
- **Implementations**: Descriptive names (e.g., `huggingface.py`, `rome.py`)
- **Managers**: `manager.py`
- **Utilities**: Descriptive names based on function (e.g., `config.py`, `results_manager.py`)

## Adding New Components

To add a new component type:

1. Create a new directory under `me_benchmark/`
2. Implement a base class in `base.py`
3. Create implementations following the naming convention
4. Register components using the appropriate decorators
5. Update the factory to support the new component type

To add a new implementation of an existing component type:

1. Create a new file in the appropriate directory
2. Extend the base class for that component type
3. Implement all required methods
4. Register the component with the appropriate decorator
5. Update documentation as needed