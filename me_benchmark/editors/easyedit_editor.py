"""
EasyEdit editor implementation for ME-Benchmark
Integrates EasyEdit's model editing methods
"""
from typing import List, Dict, Any, Optional
import importlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from me_benchmark.editors.base import BaseEditor
from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_editor


@register_editor('easyedit')
class EasyEditEditor(BaseEditor):
    """EasyEdit editor implementation that integrates various editing methods from EasyEdit"""
    
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        super().__init__(model, hparams)
        self.edit_method = hparams.get('edit_method', 'ROME')
        self.hparams_obj = None
        self._load_hparams()
    
    def _load_hparams(self):
        """Load hyperparameters for the specified editing method"""
        try:
            # Import the hyperparameters class for the specified method
            hparams_module = importlib.import_module(
                f"EasyEdit.easyeditor.models.{self.edit_method.lower()}.{self.edit_method.lower()}_hparams"
            )
            hparams_class = getattr(hparams_module, f"{self.edit_method}HyperParams")
            
            # Create hparams object
            if isinstance(self.hparams, dict):
                self.hparams_obj = hparams_class(**self.hparams)
            else:
                self.hparams_obj = self.hparams
        except Exception as e:
            print(f"Error loading hyperparameters for {self.edit_method}: {e}")
            raise
    
    def edit(self, edit_data: List[Dict[str, Any]]) -> bool:
        """Apply knowledge editing to the model using EasyEdit methods"""
        try:
            # Import the main editing function for the specified method
            edit_module = importlib.import_module(
                f"EasyEdit.easyeditor.models.{self.edit_method.lower()}.{self.edit_method.lower()}_main"
            )
            apply_edit_func = getattr(edit_module, f"apply_{self.edit_method.lower()}_to_model")
            
            # Get the underlying model and tokenizer from the BaseModel
            # This assumes the BaseModel has model and tokenizer attributes
            model = self.model.model
            tokenizer = self.model.tokenizer
            
            # Apply edits for each request
            weights_copy = {}
            for request in edit_data:
                # Apply the edit
                model, weights_copy = apply_edit_func(
                    model=model,
                    tok=tokenizer,
                    request=request,
                    hparams=self.hparams_obj,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False
                )
            
            # Update the model in the BaseModel object
            self.model.model = model
            
            return True
        except Exception as e:
            print(f"Error applying {self.edit_method} edit: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset the model to its original state"""
        # For EasyEdit methods, this would typically involve reloading the model
        # or reverting the specific weight changes
        return self.model.reset()


# Register specific EasyEdit methods as well
@register_editor('rome_easyedit')
class ROMEEasyEditor(EasyEditEditor):
    """ROME editor using EasyEdit implementation"""
    
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'ROME'
        super().__init__(model, hparams)


@register_editor('ft_easyedit')
class FTEasyEditor(EasyEditEditor):
    """FT (Fine-Tuning) editor using EasyEdit implementation"""
    
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'FT'
        super().__init__(model, hparams)


@register_editor('ike_easyedit')
class IKEEasyEditor(EasyEditEditor):
    """IKE (In-Context Knowledge Editor) editor using EasyEdit implementation"""
    
    def __init__(self, model: BaseModel, hparams: Dict[str, Any]):
        hparams['edit_method'] = 'IKE'
        super().__init__(model, hparams)