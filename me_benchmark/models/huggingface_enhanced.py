"""
Enhanced HuggingFace model implementation for ME-Benchmark
Supports multiple loading backends (transformers, vLLM)
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available, falling back to transformers")

from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_model


@register_model('huggingface_enhanced')
class HuggingFaceEnhancedModel(BaseModel):
    """Enhanced HuggingFace model implementation with multiple backend support"""
    
    def __init__(self, model_path: str, **kwargs):
        self.backend = kwargs.get('backend', 'transformers')  # 'transformers' or 'vllm'
        self.model_kwargs = kwargs.get('model_kwargs', {})
        self.tokenizer_kwargs = kwargs.get('tokenizer_kwargs', {})
        super().__init__(model_path, **kwargs)
    
    def load_model(self):
        """Load the model and tokenizer based on the specified backend"""
        if self.backend == 'vllm' and VLLM_AVAILABLE:
            self._load_with_vllm()
        else:
            self._load_with_transformers()
    
    def _load_with_transformers(self):
        """Load model using transformers library"""
        print(f"Loading model with transformers: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            **self.tokenizer_kwargs
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def _load_with_vllm(self):
        """Load model using vLLM library"""
        if not VLLM_AVAILABLE:
            print("vLLM not available, falling back to transformers")
            self._load_with_transformers()
            return
            
        print(f"Loading model with vLLM: {self.model_path}")
        # For vLLM, we store the model path and create LLM instance when needed
        # This is because vLLM has different interface for generation
        self.model_path_for_vllm = self.model_path
        self.vllm_model = None  # Will be initialized when first needed
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            **self.tokenizer_kwargs
        )
    
    def _ensure_vllm_model(self):
        """Ensure vLLM model is initialized"""
        if self.vllm_model is None and self.backend == 'vllm' and VLLM_AVAILABLE:
            self.vllm_model = LLM(
                model=self.model_path_for_vllm,
                **self.model_kwargs
            )
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts using the appropriate backend"""
        if self.backend == 'vllm' and VLLM_AVAILABLE:
            return self._generate_with_vllm(prompts, **kwargs)
        else:
            return self._generate_with_transformers(prompts, **kwargs)
    
    def _generate_with_transformers(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using transformers library"""
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        # Generate outputs
        generation_kwargs = {
            'max_new_tokens': 100,
            'pad_token_id': self.tokenizer.pad_token_id,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode outputs
        generated_texts = []
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            generated_text = self.tokenizer.decode(
                outputs[i][prompt_len:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def _generate_with_vllm(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using vLLM library"""
        self._ensure_vllm_model()
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get('max_new_tokens', 100),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
        )
        
        # Generate outputs
        outputs = self.vllm_model.generate(prompts, sampling_params)
        
        # Extract generated texts
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts
    
    def get_ppl(self, texts: List[str]) -> List[float]:
        """Calculate perplexity of texts"""
        ppls = []
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                ppls.append(ppl)
        
        return ppls
    
    def edit(self, edit_config: Dict[str, Any]) -> bool:
        """Apply knowledge editing to the model"""
        # This would be implemented by specific editing methods
        # For now, we just return True to indicate success
        return True
    
    def reset(self) -> bool:
        """Reset the model to its original state"""
        # This would reload the model from scratch
        # For now, we just return True to indicate success
        self.load_model()
        return True


# Register additional model types for specific backends
@register_model('transformers')
class TransformersModel(HuggingFaceEnhancedModel):
    """HuggingFace model explicitly using transformers backend"""
    
    def __init__(self, model_path: str, **kwargs):
        kwargs['backend'] = 'transformers'
        super().__init__(model_path, **kwargs)


if VLLM_AVAILABLE:
    @register_model('vllm')
    class VLLMModel(HuggingFaceEnhancedModel):
        """HuggingFace model using vLLM backend"""
        
        def __init__(self, model_path: str, **kwargs):
            kwargs['backend'] = 'vllm'
            super().__init__(model_path, **kwargs)