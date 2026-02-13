import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GGUFEngine:
    """
    MTI-RLM GGUF Engine (via llama-cpp-python)
    ==========================================
    
    A wrapper around llama-cpp-python for efficient GGUF inference.
    Supports "Smart" loading without complex layer splitting.
    """
    
    def __init__(self, model_path: str, n_gpu_layers: int = -1, context_window: int = 8192, cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None, flash_attn: bool = False):
        """
        Initialize the GGUF Engine.
        
        Args:
            model_path: Path to the .gguf file.
            n_gpu_layers: Number of layers to offload to GPU (-1 for all).
            context_window: Context window size.
            cache_type_k: Quantization type for K cache (e.g. "f16", "q8_0", "q4_0").
            cache_type_v: Quantization type for V cache (e.g. "f16", "q8_0", "q4_0").
            flash_attn: Enable flash attention (requires compatible hardware).
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.context_window = context_window
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.flash_attn = flash_attn
        self.model = None
        self.is_loaded = False
        
        # Check if path is a directory (common with user downloading folders)
        if os.path.isdir(self.model_path):
            # Find the first .gguf file
            files = [f for f in os.listdir(self.model_path) if f.endswith(".gguf")]
            if files:
                self.model_path = os.path.join(self.model_path, files[0])
                logger.info(f"🧠 GGUFEngine: Auto-resolved model path to {self.model_path}")
            else:
                raise FileNotFoundError(f"No .gguf file found in {self.model_path}")

        self.model_name = os.path.basename(self.model_path)

    def load(self):
        """Loads the GGUF model."""
        if self.is_loaded:
            return

        print("\n" + "="*60)
        print("🧠 MTI-RLM: Loading Local Agent Model...")
        print(f"   Model: {self.model_name}")
        print(f"   Config: GPU layers={self.n_gpu_layers}, ctx={self.context_window}")
        print("="*60)
        
        try:
            from llama_cpp import Llama
            
            # Prepare kwargs
            kwargs = {}
            if self.cache_type_k: kwargs["cache_type_k"] = self.cache_type_k
            if self.cache_type_v: kwargs["cache_type_v"] = self.cache_type_v
            if self.flash_attn: kwargs["flash_attn"] = True
            
            # Enable verbose to show llama.cpp's native progress bar
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.context_window,
                verbose=True,  # Shows loading progress
                embedding=True, # [PHASE 6] Enable Embeddings
                **kwargs
            )
            
            self.is_loaded = True
            print("\n" + "="*60)
            print("✅ MTI-RLM LOCAL AGENT: READY")
            print("="*60 + "\n")
            
        except ImportError:
            logger.error("❌ llama-cpp-python not found. Please install it (pip install llama-cpp-python).")
            raise ImportError("llama-cpp-python not found.")
        except Exception as e:
            logger.error(f"❌ GGUFEngine Load Failed: {e}")
            raise e

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """
        Inference generation. 
        Auto-switches to Chat Completion if system_prompt is provided.
        """
        if not self.is_loaded:
            self.load()
            
        try:
            # Clean kwargs of keys that create_completion/chat_completion might not like if passed implicitly
            # But here we just popped system_prompt from args.
            
            if system_prompt:
                # Use Chat Completion API for Context Injection
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                # Filter kwargs for chat
                output = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=["<end_of_turn>", "<eos>", "</tool>", "</Action>"],
                    **kwargs
                )
                return output['choices'][0]['message']['content']
            else:
                # Legacy / Raw Completion
                output = self.model.create_completion(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=["<end_of_turn>", "<eos>", "</tool>", "</Action>"], 
                    **kwargs 
                )
                return output['choices'][0]['text']
            
        except Exception as e:
            logger.error(f"Generate Error: {e}")
            return f"Error: {e}"

    def embed(self, text: str) -> list[float]:
        """
        [PHASE 6] Generate Embedding Vector.
        """
        if not self.is_loaded:
            self.load()
            
        try:
            # llama-cpp-python create_embedding returns dictionary
            response = self.model.create_embedding(text)
            # Standard format: {'data': [{'embedding': [...], ...}], ...}
            # Or simplified list if plain wrapper
            
            # Defensive unpacking
            if isinstance(response, dict) and 'data' in response:
                return response['data'][0]['embedding']
            elif isinstance(response, list):
                 # Sometimes simple list of floats
                 return response
            elif hasattr(response, 'data'):
                 return response.data[0].embedding
                 
            # Fallback for older versions
            logger.warning(f"Unexpected embedding format: {type(response)}")
            return response['data'][0]['embedding'] # Hope for best
            
        except Exception as e:
            logger.error(f"Embedding Error: {e}")
            return []

    def unload(self):
        """Free memory."""
        if self.model:
            del self.model
        self.model = None
        self.is_loaded = False
        logger.info("💤 GGUFEngine: Unloaded.")

    def get_config(self) -> Dict:
        return {
            "model_path": self.model_path,
            "type": "gguf",
            "n_gpu_layers": self.n_gpu_layers,
            "is_loaded": self.is_loaded
        }
