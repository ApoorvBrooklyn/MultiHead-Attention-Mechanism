import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import hashlib
from functools import lru_cache

import logging
import sys
import gc
import traceback

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multi_model_fusion_optimized.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedCrossAttentionMerger(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Optional[int] = None, 
        num_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim or input_dim
        assert self.hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        
        self.query_proj = nn.Linear(input_dim, self.hidden_dim)
        self.key_proj = nn.Linear(input_dim, self.hidden_dim)
        self.value_proj = nn.Linear(input_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(self.hidden_dim, input_dim)

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        Q = Q.view(Q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(query.size(0), -1, self.hidden_dim)
        
        return self.out_proj(context)

class MultiSpecializedLanguageModelPipeline:
    def __init__(
        self, 
        models_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        cache_size: int = 100
    ):
        logger.info("Initializing Multi-Specialized Language Model Pipeline")
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        self.models_config = models_config or {
            'medical': {
                'model_name': 'medicalai/ClinicalGPT-base-zh',
                'model_type': 'causal',
                'quantization': True
            },
            'code': {
                'model_name': 'Qwen/Qwen2.5-Coder-1.5B',
                'model_type': 'causal',
                'quantization': True
            },
            'merger': {
                'model_name': 'google/flan-t5-large',
                'model_type': 'seq2seq',
                'quantization': False
            }
        }

        self.models = {}
        self.tokenizers = {}
        self.cross_attention = None
        self.cache_size = cache_size
        
        self._load_models()

        if 'merger' in self.models:
            merger_model_dim = self.models['merger'].config.d_model
            self.cross_attention = AdvancedCrossAttentionMerger(
                input_dim=merger_model_dim, 
                hidden_dim=merger_model_dim
            ).to(self.device)
            
        # Initialize the cache outside of the method to avoid reinitializing
        # on each call to _generate_model_response
        self._init_cache()

    def _init_cache(self):
        """Initialize the LRU cache for model responses"""
        # Using a separate method to set up the cache
        # This allows us to clear the cache without recreating it
        self._generate_model_response = lru_cache(maxsize=self.cache_size)(self._generate_model_response_uncached)
        logger.info(f"Initialized response cache with size {self.cache_size}")

    def _load_models(self):
        for model_key, config in self.models_config.items():
            try:
                logger.info(f"Loading {model_key} model: {config['model_name']}")

                quantization_config = None
                if config.get('quantization', False):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )

                model_class = AutoModelForCausalLM if config['model_type'] == 'causal' else AutoModelForSeq2SeqLM

                model = model_class.from_pretrained(
                    config['model_name'],
                    quantization_config=quantization_config,
                    device_map='auto',
                    low_cpu_mem_usage=True
                )
                
                tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
                
                model = model.to(self.device)
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer

            except Exception as e:
                logger.error(f"Error loading {model_key} model: {e}")
                logger.error(traceback.format_exc())

    def _generate_query_hash(self, query: str, model_key: str, max_length: int, temperature: float, top_p: float) -> str:
        """Generate a unique hash for the query and generation parameters"""
        hash_input = f"{query}_{model_key}_{max_length}_{temperature}_{top_p}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _generate_model_response_uncached(
        self, 
        model_key: str, 
        query_hash: str,  # This parameter is just for the cache key, not used in the function
        query: str,
        max_length: int,
        temperature: float,
        top_p: float
    ) -> str:
        """The actual function that generates responses (uncached version)"""
        logger.info(f"Cache miss for {model_key} with query hash {query_hash[:8]}...")
        
        if model_key not in self.models or model_key not in self.tokenizers:
            logger.warning(f"Model {model_key} not loaded. Skipping response generation.")
            return ""

        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )

        try:
            # Create a prompt specific to the model type
            prompt = f"Respond to the following query from a {model_key} perspective: {query}"
            
            inputs = tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=max_length
            ).to(self.device)

            outputs = model.generate(
                **inputs, 
                generation_config=generation_config
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated response for {model_key} (length: {len(response)})")
            return response

        except Exception as e:
            logger.warning(f"Error generating response for {model_key}: {e}")
            logger.warning(traceback.format_exc())
            return ""

    def clear_cache(self):
        """Clear the response cache"""
        logger.info("Clearing response cache")
        self._generate_model_response.cache_clear()

    def generate_response(
        self, 
        query: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        responses = {}

        # Generate responses from specialized models
        for model_key, model in self.models.items():
            if model_key == 'merger':
                continue
                
            # Generate hash for caching
            query_hash = self._generate_query_hash(query, model_key, max_length, temperature, top_p)
            
            # Check if we've already generated this response
            response = self._generate_model_response(
                model_key=model_key,
                query_hash=query_hash,  # Used as part of the cache key
                query=query,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            if response:
                responses[model_key] = response
                logger.info(f"Got response for {model_key} (cache hit: {query_hash[:8]})")

        if not responses:
            return "Unable to generate responses from specialized models."

        # Prepare merger prompt
        merger_prompt = (
            f"Original Query: {query}\n\n" +
            "\n\n".join([f"{k.capitalize()} Perspective: {v}" for k, v in responses.items()]) +
            "\n\nGenerate a comprehensive, nuanced response integrating these perspectives:"
        )

        merger_model = self.models.get('merger')
        merger_tokenizer = self.tokenizers.get('merger')

        if not merger_model or not merger_tokenizer:
            logger.warning("No merger model available. Returning individual responses.")
            return "\n\n".join([f"{k.capitalize()} Perspective: {v}" for k, v in responses.items()])

        try:
            # Generate hash for merger cache
            merger_hash = self._generate_query_hash(merger_prompt, 'merger', max_length, temperature, top_p)
            
            # Check if we've already generated this merged response
            final_response = self._generate_model_response(
                model_key='merger',
                query_hash=merger_hash,
                query=merger_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            if final_response:
                logger.info(f"Generated merged response (length: {len(final_response)})")
                return final_response
            else:
                logger.warning("Failed to generate merged response")
                return "\n\n".join([f"{k.capitalize()} Perspective: {v}" for k, v in responses.items()])

        except Exception as e:
            logger.error(f"Error in merger model processing: {e}")
            logger.error(traceback.format_exc())
            return "\n\n".join([f"{k.capitalize()} Perspective: {v}" for k, v in responses.items()])

    def clear_resources(self):
        """
        Comprehensive resource cleanup with error handling
        """
        logger.info("Clearing pipeline resources")
        
        try:
            # Clear cache
            self.clear_cache()
            
            # Clear models
            for model_key in list(self.models.keys()):
                try:
                    del self.models[model_key]
                except Exception as e:
                    logger.warning(f"Error clearing model {model_key}: {e}")
            
            # Clear tokenizers
            self.tokenizers.clear()
            
            # Clear cross-attention
            if self.cross_attention:
                del self.cross_attention
            
            # Cleanup GPU resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Garbage collection
            gc.collect()
        
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
            logger.error(traceback.format_exc())