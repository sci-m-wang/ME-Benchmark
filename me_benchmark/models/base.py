"""
Base model class for ME-Benchmark
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModel(ABC):
    """Base class for all models in ME-Benchmark"""
    
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