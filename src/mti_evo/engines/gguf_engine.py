import time
import os
from .base import BaseEngine
from .protocol import EngineResult
from mti_evo.tools.gpu_utils import get_gpu_stats

# Lazy import handles
# try:
#     from llama_cpp import Llama
#     HAS_LLAMA = True
# except ImportError:
#     HAS_LLAMA = False
HAS_LLAMA = None # Checked on load

class GGUFEngine(BaseEngine):
    """Engine for GGUF models via llama-cpp-python."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.llm = None
        self.backend_name = "gguf"
        if self.config.get("check_deps", False):
             try:
                 import llama_cpp
             except ImportError:
                 print("[GGUFEngine] ❌ llama-cpp-python not installed.")

    def load(self, config: dict = None):
        if config:
            self.config.update(config)
            # Update attributes from config
            self.model_path = self.config.get("model_path", self.model_path)
            self.n_ctx = self.config.get("n_ctx", self.n_ctx)
            
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[GGUFEngine] ❌ llama-cpp-python not installed.")
            return

        gpu_layers = self.config.get("gpu_layers", -1)
        print(f"[GGUFEngine] 🦙 Loading: {self.model_path}")
        print(f"             (Ctx: {self.n_ctx}, GPU: {gpu_layers})")
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.config.get("n_batch", 512),
                n_gpu_layers=gpu_layers,
                embedding=True,
                verbose=True
            )
            print("[GGUFEngine] ✅ Loaded.")
        except Exception as e:
            print(f"[GGUFEngine] ❌ Load Failed: {e}")

    def infer(self, prompt: str, **kwargs) -> EngineResult:
        t0 = time.perf_counter()
        if not self.llm:
            return EngineResult("GGUF Engine Not Loaded", 0, 0, 0.0)

        # Extract params from kwargs or use defaults
        # We must POP them so they aren't passed twice in **kwargs
        max_tokens = kwargs.pop("max_tokens", 1024)
        stop = kwargs.pop("stop", None)
        
        # Handle potential double-passing of temperature
        eff_temp = kwargs.pop("temperature", self.temperature)
        eff_stop = stop or ["<end_of_turn>"]
        
        # Handle System Prompt & Chat Formatting (Gemma Style)
        system_prompt = kwargs.pop("system_prompt", "")
        
        # Simple detection if prompt is already formatted (contains tags)
        if "<start_of_turn>" in prompt:
            full_prompt = prompt
        else:
            # Construct Gemma-style chat prompt
            if system_prompt:
                # In Gemma, system instructions often go in the first user turn or explicit system turn
                # We'll prepend it clearly.
                full_prompt = f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                full_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        try:
            output = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=eff_temp,
                stop=eff_stop,
                echo=False,
                **kwargs
            )
            text = output['choices'][0]['text'].strip()
            tokens = output['usage']['completion_tokens']
            latency = (time.perf_counter() - t0) * 1000
            
            # Capture GPU Stats
            gpu_stats = get_gpu_stats()

            return EngineResult(text, tokens, latency, 0.95, metrics={"gpu": gpu_stats})
            
        except Exception as e:
            return EngineResult(f"Error: {e}", 0, 0, 0.0)

    def embed(self, text: str):
        if self.llm:
            emb = self.llm.create_embedding(text)
            if isinstance(emb, dict) and 'data' in emb: 
                return emb['data'][0]['embedding']
            return emb
        return []

    def unload(self):
        if self.llm:
            print("[GGUFEngine] 🗑️ Offloading Llama...")
            del self.llm
            self.llm = None
            import gc
            gc.collect()
            print("[GGUFEngine] ✅ Unloaded.")

    @property
    def capabilities(self) -> dict:
        return {
            "embedding": True,
            "streaming": False,
            "device": "cuda" if self.config.get("gpu_layers", -1) != 0 else "cpu"
        }
