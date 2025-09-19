"""
Unified model editing manager for ME-Benchmark
Provides a unified interface for different editing methods
"""
from typing import List, Dict, Any, Optional
import importlib

from me_benchmark.models.base import BaseModel
from me_benchmark.editors.base import BaseEditor
from me_benchmark.registry import REGISTRY


class ModelEditingManager:
    """Unified manager for model editing operations"""
    
    def __init__(self):
        self.available_editors = {}
        self._discover_editors()
    
    def _discover_editors(self):
        """Discover all available editor types from the registry"""
        # Get all registered editors
        self.available_editors = REGISTRY._editors.copy()
        print(f"Discovered editors: {list(self.available_editors.keys())}")
    
    def get_editor(self, model: BaseModel, editor_config: Dict[str, Any]) -> BaseEditor:
        """Create an editor instance based on configuration"""
        editor_type = editor_config.get('type', 'rome')
        
        if editor_type not in self.available_editors:
            raise ValueError(f"Unknown editor type: {editor_type}")
        
        editor_cls = self.available_editors[editor_type]
        
        # Load hyperparameters if specified
        hparams = {}
        if 'hparams' in editor_config:
            hparams = editor_config['hparams']
        elif 'hparams_path' in editor_config:
            from me_benchmark.utils.config import load_config
            hparams = load_config(editor_config['hparams_path'])
        
        return editor_cls(model, hparams)
    
    def apply_edit(self, model: BaseModel, editor_config: Dict[str, Any], 
                   edit_data: List[Dict[str, Any]]) -> bool:
        """Apply knowledge editing to a model using the specified editor"""
        try:
            editor = self.get_editor(model, editor_config)
            result = editor.edit(edit_data)
            return result
        except Exception as e:
            print(f"Error applying edit: {e}")
            return False
    
    def reset_model(self, model: BaseModel, editor_config: Dict[str, Any]) -> bool:
        """Reset the model to its original state using the specified editor"""
        try:
            editor = self.get_editor(model, editor_config)
            result = editor.reset()
            return result
        except Exception as e:
            print(f"Error resetting model: {e}")
            return False


# Global instance
EDITING_MANAGER = ModelEditingManager()