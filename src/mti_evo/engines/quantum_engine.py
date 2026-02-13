from .base import BaseEngine, LLMResponse
import os

class QuantumEngine(BaseEngine):
    """Engine for MTI-EVO Quantum Brain (Phase 12)."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.backend_name = "quantum"
        self.hf_model = None
        self.tokenizer = None

    def load_model(self):
        print(f"[QuantumEngine] 🌌 Initializing Quantum Brain: {self.model_path}...")
        try:
            # Dynamic Import to avoid circular deps if possible, or just strict dependency
            from ..quantum_model import QuantumGemmaForCausalLM, QuantumGemmaConfig
            from transformers import AutoTokenizer
            
            # Smart Path Handling (Legacy Logic)
            base_path = self.model_path
            fast_path = self.config.get("fast_model_path")
            
            # GGUF Fallback Logic
            if self.model_path.endswith(".gguf"):
                print("[QuantumEngine] ⚠️ Warning: GGUF selected for Quantum Base. Check configuration.")
                fast_path = self.model_path
                base_dir = os.path.dirname(self.model_path)
                candidate_base = os.path.join(base_dir, "gemma-3-27b")
                if os.path.exists(candidate_base):
                    base_path = candidate_base
            
            print(f"[QuantumEngine] 📖 Loading Tokenizer from: {base_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            
            q_config = QuantumGemmaConfig(
                base_model_path=base_path,
                fast_model_path=fast_path
            )
            self.hf_model = QuantumGemmaForCausalLM(config=q_config)
            print("[QuantumEngine] ✅ Quantum Brain Online.")
            
        except Exception as e:
            print(f"[QuantumEngine] ❌ Load Failed: {e}")

    def infer(self, prompt: str, max_tokens: int = 1024, stop: list = None, **kwargs) -> LLMResponse:
        if not self.hf_model: return LLMResponse("Quantum Not Loaded", 0,0,0)
        
        import time
        t0 = time.perf_counter()
        try:
            # Format prompt with Gemma-3 chat format
            formatted = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
            
            # Raw tokenization (works with all models)
            input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.hf_model.device)
            
            out = self.hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0
            )
            
            new_tokens = out[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            latency = (time.perf_counter() - t0) * 1000
            return LLMResponse(text, len(out[0]), latency, 1.0)
        except Exception as e:
            return LLMResponse(f"Quantum Error: {e}", 0,0,0)

    def embed(self, text: str):
        # Quantum Embedding? For now standard
        return []

    def unload(self):
        if self.hf_model:
            print("[QuantumEngine] 🗑️ Offloading Quantum Brain...")
            del self.hf_model
            self.hf_model = None
            self.tokenizer = None
            
            import gc
            import torch
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except:
                pass
            print("[QuantumEngine] ✅ Unloaded.")
