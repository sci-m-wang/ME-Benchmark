# ME-Benchmark Frequently Asked Questions (FAQ)

This document addresses common questions and issues users may encounter when using the ME-Benchmark framework.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Configuration Issues](#configuration-issues)
3. [Model and Editing Questions](#model-and-editing-questions)
4. [Performance and Resource Management](#performance-and-resource-management)
5. [Evaluation and Results](#evaluation-and-results)
6. [Extending the Framework](#extending-the-framework)
7. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Q: How do I install ME-Benchmark?

A: To install ME-Benchmark, follow these steps:

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

### Q: What are the system requirements?

A: Requirements depend on the model size:
- Small models (GPT-2, BERT): 8GB+ RAM, no GPU required
- Medium models (LLaMA-7B): 16GB+ RAM, 12GB+ GPU VRAM
- Large models (LLaMA-13B+): 32GB+ RAM, 24GB+ GPU VRAM

For GPU acceleration, you'll need:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- cuDNN library

### Q: I'm getting import errors. What should I do?

A: Import errors are typically caused by missing dependencies. Try:

1. Ensure all requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. If using EasyEdit methods, install EasyEdit requirements:
   ```bash
   cd EasyEdit
   pip install -r requirements.txt
   ```

3. Check that you're running Python 3.8 or higher:
   ```bash
   python --version
   ```

### Q: How do I set up a virtual environment?

A: It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv me_benchmark_env

# Activate virtual environment
# On Windows:
me_benchmark_env\Scripts\activate
# On macOS/Linux:
source me_benchmark_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Configuration Issues

### Q: How do I create a configuration file?

A: Configuration files are YAML files that define all aspects of the evaluation. Here's a basic example:

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

evaluation:
  datasets:
    - name: zsre
      metrics: [accuracy]

output:
  results_dir: results/
```

See `configs/example_config.yaml` for a complete example.

### Q: What model paths are supported?

A: You can use:
1. HuggingFace model identifiers (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
2. Local paths to model directories
3. Paths to models in custom formats (if you implement custom loaders)

### Q: How do I configure hyperparameters for editing methods?

A: You can configure hyperparameters in two ways:

1. Using a hyperparameters file:
   ```yaml
   editing:
     method: rome
     hparams_path: hparams/ROME/gpt2.yaml
   ```

2. Inline in the configuration:
   ```yaml
   editing:
     method: rome
     hparams:
       layers: [5]
       mom2_adjustment: true
   ```

### Q: What editing methods are available?

A: The framework integrates various editing methods:
- ROME (Rank-One Model Editing)
- MEMIT (Memory-Efficient Model Editing)
- MEND (Model Editor Networks)
- IKE (In-Context Knowledge Editor)
- LoRA (Low-Rank Adaptation)

Additional methods can be added through the EasyEdit integration.

## Model and Editing Questions

### Q: What models are supported?

A: ME-Benchmark supports models from HuggingFace Transformers, including:
- GPT family (GPT-2, GPT-J, GPT-Neo, etc.)
- LLaMA family (LLaMA, LLaMA-2, etc.)
- BERT family
- T5 family
- And many others

Additional support for vLLM and LMDeploy is planned.

### Q: How do I choose the right editing method?

A: Consider these factors:

1. **ROME**: Fast, good for single edits, may affect locality
2. **MEMIT**: Better locality preservation, slower than ROME
3. **MEND**: Network-based editing, good for multiple edits
4. **IKE**: In-context editing, no model modification
5. **LoRA**: Low-rank adaptation, good for fine-tuning

### Q: Can I apply multiple edits?

A: Yes, you can apply multiple edits by providing a list of edit requests:

```yaml
edit_data:
  - prompt: "The capital of France is"
    subject: "France"
    target_new: "London"
  - prompt: "The president of the United States is"
    subject: "the United States"
    target_new: "John Smith"
```

### Q: How do I reset a model after editing?

A: Models can be reset using the `reset()` method:

```python
# Programmatic reset
model.reset()

# Or when using editors
editor.reset()
```

Note that some editing methods may require reloading the model entirely.

## Performance and Resource Management

### Q: How can I optimize memory usage?

A: Several strategies can help optimize memory usage:

1. Use appropriate device mapping:
   ```yaml
   model:
     model_kwargs:
       device_map: auto
   ```

2. Use lower precision data types:
   ```yaml
   model:
     model_kwargs:
       torch_dtype: float16
   ```

3. Limit the number of workers:
   ```yaml
   runner:
     max_workers: 1
   ```

### Q: How do I run evaluations in parallel?

A: The framework supports parallel execution through the `max_workers` configuration:

```yaml
runner:
  type: local
  max_workers: 4
  debug: false
```

### Q: How do I monitor resource usage?

A: The framework automatically monitors system resources during execution. Results are included in the output reports. You can also monitor externally using system tools like `htop` or `nvidia-smi`.

### Q: What should I do if I run out of GPU memory?

A: If you encounter GPU memory issues:

1. Use CPU offloading:
   ```yaml
   model:
     model_kwargs:
       device_map: auto
   ```

2. Use lower precision:
   ```yaml
   model:
     model_kwargs:
       torch_dtype: float16
   ```

3. Reduce batch sizes in your evaluation data

4. Use smaller models for initial testing

## Evaluation and Results

### Q: How do I interpret the evaluation results?

A: The framework provides several key metrics:
- **Accuracy**: Measures if the edited knowledge is correctly applied
- **Locality**: Measures preservation of unrelated knowledge
- **Portability**: Measures generalization to related facts
- **Fluency**: Measures quality of generated text
- **Consistency**: Measures consistency of responses

### Q: What evaluation datasets are available?

A: The framework includes:
- **zsre**: Zero-Shot Relation Extraction dataset
- **counterfact**: CounterFact dataset for knowledge editing
- **mmlu**: Massive Multitask Language Understanding
- **hellaswag**: HellaSwag dataset for commonsense reasoning

Custom datasets can be added by implementing the `BaseDataset` interface.

### Q: How do I add custom evaluation metrics?

A: Create a custom evaluator by extending `BaseEvaluator`:

```python
from me_benchmark.evaluators.base import BaseEvaluator
from me_benchmark.registry import register_evaluator

@register_evaluator('custom_evaluator')
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Implement custom evaluation logic
        results = {}
        # ... your implementation
        return results
```

### Q: Where are results saved?

A: Results are saved in the directory specified in the configuration:

```yaml
output:
  results_dir: results/
  save_individual_results: true
  save_summary: true
```

By default, results are saved in the `results/` directory.

## Extending the Framework

### Q: How do I add a new model backend?

A: Create a new class that inherits from `BaseModel` and register it:

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

### Q: How do I add a new editing method?

A: Create a new class that inherits from `BaseEditor` and register it:

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

### Q: How do I add a new evaluation dataset?

A: Create a new class that inherits from `BaseDataset` and register it:

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

## Troubleshooting

### Q: I'm getting "CUDA out of memory" errors. What should I do?

A: Try these solutions in order:

1. Reduce batch size in your evaluation data
2. Use CPU offloading:
   ```yaml
   model:
     model_kwargs:
       device_map: auto
   ```
3. Use lower precision:
   ```yaml
   model:
     model_kwargs:
       torch_dtype: float16
   ```
4. Use a smaller model for testing
5. Close other applications using GPU memory

### Q: The editing is not working as expected. What could be wrong?

A: Check these common issues:

1. **Hyperparameters**: Ensure editing hyperparameters are appropriate for your model
2. **Edit format**: Verify that your edit data follows the expected format
3. **Model compatibility**: Some editing methods may not work with all model types
4. **Model state**: Ensure the model is in the correct state before editing

### Q: Evaluation results seem inconsistent. Why?

A: Inconsistent results can be caused by:

1. **Randomness**: Set random seeds for reproducible results:
   ```python
   import random
   import numpy as np
   import torch
   
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   ```

2. **Model state**: Ensure the model is reset between experiments
3. **Batch effects**: Run multiple experiments and average results

### Q: How do I debug issues with my custom components?

A: For debugging custom components:

1. **Add logging**: Use Python's logging module to track execution
2. **Test individually**: Test each component in isolation
3. **Check registration**: Ensure your components are properly registered
4. **Validate inputs**: Add input validation to catch issues early

### Q: Where can I find logs for debugging?

A: The framework outputs logs to the console by default. For more detailed logging:

1. Enable debug mode:
   ```bash
   python run.py --config configs/example_config.yaml --debug
   ```

2. Check the results directory for detailed logs and error reports

### Q: How do I report bugs or request features?

A: Please:

1. Check existing issues on the project repository
2. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version, etc.)
   - Error messages and logs

## Additional Resources

### Documentation

- [API Documentation](api.md)
- [Usage Examples](usage_examples.md)
- [Configuration Reference](../configs/)

### Community Support

- GitHub Issues: For bug reports and feature requests
- Discussion Forums: For general questions and community support

### Contributing

We welcome contributions! Please see our contributing guidelines for more information on:
- Code style and standards
- Testing requirements
- Pull request process
- Code of conduct

## Contact

For questions not covered in this FAQ, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with your question

Last updated: 2025-09-19