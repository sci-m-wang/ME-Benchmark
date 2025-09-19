"""
Registry for ME-Benchmark components
"""
from typing import Dict, Type, Any


class Registry:
    """Registry for ME-Benchmark components"""
    
    def __init__(self):
        self._models = {}
        self._editors = {}
        self._evaluators = {}
        self._datasets = {}
        self._runners = {}
        self._summarizers = {}
    
    def register_model(self, name: str, cls: Type):
        """Register a model class"""
        self._models[name] = cls
    
    def register_editor(self, name: str, cls: Type):
        """Register an editor class"""
        self._editors[name] = cls
    
    def register_evaluator(self, name: str, cls: Type):
        """Register an evaluator class"""
        self._evaluators[name] = cls
    
    def register_dataset(self, name: str, cls: Type):
        """Register a dataset class"""
        self._datasets[name] = cls
    
    def register_runner(self, name: str, cls: Type):
        """Register a runner class"""
        self._runners[name] = cls
    
    def register_summarizer(self, name: str, cls: Type):
        """Register a summarizer class"""
        self._summarizers[name] = cls
    
    def get_model(self, name: str) -> Type:
        """Get a registered model class"""
        return self._models.get(name)
    
    def get_editor(self, name: str) -> Type:
        """Get a registered editor class"""
        return self._editors.get(name)
    
    def get_evaluator(self, name: str) -> Type:
        """Get a registered evaluator class"""
        return self._evaluators.get(name)
    
    def get_dataset(self, name: str) -> Type:
        """Get a registered dataset class"""
        return self._datasets.get(name)
    
    def get_runner(self, name: str) -> Type:
        """Get a registered runner class"""
        return self._runners.get(name)
    
    def get_summarizer(self, name: str) -> Type:
        """Get a registered summarizer class"""
        return self._summarizers.get(name)


# Global registry instance
REGISTRY = Registry()


def register_model(name: str):
    """Decorator to register a model class"""
    def decorator(cls: Type):
        REGISTRY.register_model(name, cls)
        return cls
    return decorator


def register_editor(name: str):
    """Decorator to register an editor class"""
    def decorator(cls: Type):
        REGISTRY.register_editor(name, cls)
        return cls
    return decorator


def register_evaluator(name: str):
    """Decorator to register an evaluator class"""
    def decorator(cls: Type):
        REGISTRY.register_evaluator(name, cls)
        return cls
    return decorator


def register_dataset(name: str):
    """Decorator to register a dataset class"""
    def decorator(cls: Type):
        REGISTRY.register_dataset(name, cls)
        return cls
    return decorator


def register_runner(name: str):
    """Decorator to register a runner class"""
    def decorator(cls: Type):
        REGISTRY.register_runner(name, cls)
        return cls
    return decorator


def register_summarizer(name: str):
    """Decorator to register a summarizer class"""
    def decorator(cls: Type):
        REGISTRY.register_summarizer(name, cls)
        return cls
    return decorator