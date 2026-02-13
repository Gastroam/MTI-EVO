"""
LLM Adapter (Refactored)
========================
Factory for MTI-EVO Engines.

Available Engines (model_type):
- gguf:     GGUF models via llama-cpp-python (default)
- native:   Safetensors via Transformers
- quantum:  MTI Hybrid Quantum architecture
- resonant: Metabolic Layer Activation (cognitive sector sparse loading)
- bicameral: Dual-Model (4B Limbic + 12B Cortex parallel streams)
- qoop:     Quantum Object-Oriented Programming (probabilistic routing)
- hybrid:   Local + API Fusion (local GGUF + cloud API reasoning)
- api:      External API calls (OpenAI, Anthropic, etc.)
"""

from mti_evo.engines.protocol import EngineResult
from mti_evo.engines.gguf_engine import GGUFEngine
# Experimental engines should be loaded dynamically or moved.
# For now, we fix the imports to allow compilation if they exist.
# But we are about to move them.
# Let's verify what exists.
from mti_evo.engines.gguf_engine import GGUFEngine
from mti_evo.engines.native_engine import NativeEngine
from mti_evo.engines.quantum_engine import QuantumEngine
from mti_evo.engines.api_engine import APIEngine
from mti_evo.engines.resonant_engine import ResonantEngine
from mti_evo.engines.bi_camera_engine import BiCameraEngine
from mti_evo.engines.qoop_engine import QoopEngine
from mti_evo.engines.hybrid_engine import HybridEngine
import os

class LLMAdapter:
    _loaded_models = {} # Cache

    def __init__(self, config=None, auto_load=True):
        self.config = config or {}
        self.engine = None
        
        if auto_load:
            self.load_model()

    def load_model(self):
        # Determine Engine Type
        model_path = self.config.get("model_path", "")
        model_type = self.config.get("model_type", "auto")
        
        print(f"[LLMAdapter] Factory requesting: {model_type} for {model_path}")
        
        if model_type == "quantum":
            self.engine = QuantumEngine(self.config)
        elif model_type == "resonant":
            self.engine = ResonantEngine(self.config)
        elif model_type == "bicameral":
            self.engine = BiCameraEngine(self.config)
        elif model_type == "qoop":
            self.engine = QoopEngine(self.config)
        elif model_type == "hybrid":
            self.engine = HybridEngine(self.config)
        elif model_type == "api":
            self.engine = APIEngine(self.config)
        elif model_type == "gguf" or model_path.endswith(".gguf"):
            self.engine = GGUFEngine(self.config)
        elif model_type == "native" or os.path.isdir(model_path) or model_path.endswith(".safetensors"):
            self.engine = NativeEngine(self.config)
        else:
            print("[LLMAdapter] ⚠️ Unknown type. Defaulting to GGUF or Sim.")
            self.engine = GGUFEngine(self.config)

        if self.engine:
            if hasattr(self.engine, 'load'):
                self.engine.load(self.config)
            elif hasattr(self.engine, 'load_model'):
                # Legacy support while refactoring
                self.engine.load_model()

    def infer(self, prompt, **kwargs):
        # [FIX] Lazy Load (Load on Demand)
        if not self.engine:
            self.load_model()
            
        if self.engine:
            return self.engine.infer(prompt, **kwargs)
        return EngineResult("No Engine", 0,0,0)

    def embed(self, text):
        # [FIX] Lazy Load for Embeddings too
        if not self.engine:
            self.load_model()
            
        if self.engine:
            return self.engine.embed(text)
        return []
    
    @property
    def backend(self):
        if self.engine:
            return getattr(self.engine, "backend_name", "unknown")
        return "none"
    
    def update_config(self, new_config):
        self.config.update(new_config)
        if self.engine:
            self.engine.unload()
        self.load_model()

    def unload_model(self):
        """Explicitly unload the backend engine."""
        if self.engine:
            print("[LLMAdapter] 🧹 Unloading Engine...")
            if hasattr(self.engine, 'unload'):
                self.engine.unload()
            self.engine = None
            self.backend_name = "none"
