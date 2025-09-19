"""
HuggingFace model implementation for ME-Benchmark
"""
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from me_benchmark.models.base import BaseModel
from me_benchmark.registry import register_model


@register_model('huggingface')
class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation"""
    
    def load_model(self):
        """Load the HuggingFace model and tokenizer"""
        model_kwargs = self.kwargs.get('model_kwargs', {})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts"""
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
        return True