"""
Model implementations for ME-Benchmark
"""
from me_benchmark.models.base import BaseModel
from me_benchmark.models.huggingface import HuggingFaceModel
from me_benchmark.models.huggingface_enhanced import (
    HuggingFaceEnhancedModel,
    TransformersModel
)

# Import vLLM model only if vLLM is available
try:
    from me_benchmark.models.huggingface_enhanced import VLLMModel
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

__all__ = [
    'BaseModel',
    'HuggingFaceModel',
    'HuggingFaceEnhancedModel',
    'TransformersModel'
]

if HAS_VLLM:
    __all__.append('VLLMModel')