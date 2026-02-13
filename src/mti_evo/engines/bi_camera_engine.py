"""
BiCamera Engine Wrapper
=======================
BaseEngine-compliant wrapper for the Bicameral (Dual-Model) System.

Architecture:
- Limbic (4B GGUF): Fast intuition, runs in separate process
- Cortex (12B Safetensors): Deep reasoning, uses ResonanceLoader

The two streams run in parallel and their outputs are synthesized.
"""
import asyncio
import time
from typing import List
from .base import BaseEngine, LLMResponse


class BiCameraEngine(BaseEngine):
    """
    Dual-Model inference engine combining fast intuition (4B) with deep reasoning (12B).
    
    Config Keys:
    - limbic_model_path: Path to 4B GGUF model (fast)
    - cortex_model_path: Path to 12B Safetensors model (deep)
    - synthesis_mode: 'concat', 'cortex_only', 'limbic_only' (default: 'concat')
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.backend_name = "bicameral"
        
        self.limbic_path = config.get("limbic_model_path", config.get("fast_model_path", ""))
        self.cortex_path = config.get("cortex_model_path", config.get("model_path", ""))
        self.synthesis_mode = config.get("synthesis_mode", "concat")
        
        self.limbic_engine = None
        self.cortex_loader = None

    def load_model(self):
        print(f"[BiCameraEngine] 🧠 Loading Bicameral System...")
        print(f"   Limbic (4B): {self.limbic_path}")
        print(f"   Cortex (12B): {self.cortex_path}")
        
        # Load Limbic (GGUF)
        if self.limbic_path:
            try:
                from .gguf_engine import GGUFEngine
                self.limbic_engine = GGUFEngine({
                    "model_path": self.limbic_path,
                    "n_ctx": self.config.get("limbic_n_ctx", 2048),
                    "gpu_layers": self.config.get("limbic_gpu_layers", 0)
                })
                self.limbic_engine.load_model()
                print("   ✅ Limbic loaded.")
            except Exception as e:
                print(f"   ⚠️ Limbic failed: {e}")
        
        # Load Cortex Loader (Resonance-based)
        if self.cortex_path:
            try:
                from mti_evo.resonance_loader import ResonanceGuidedLoader
                self.cortex_loader = ResonanceGuidedLoader(self.cortex_path)
                print(f"   ✅ Cortex loader ready.")
            except Exception as e:
                print(f"   ⚠️ Cortex loader failed: {e}")

    def infer(self, prompt: str, max_tokens: int = 512, stop: list = None, **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        
        limbic_result = ""
        cortex_result = ""
        
        # Stream 1: Limbic (fast intuition)
        if self.limbic_engine:
            try:
                resp = self.limbic_engine.infer(prompt, max_tokens=max_tokens // 2, stop=stop)
                limbic_result = f"[Limbic] {resp.text}"
            except Exception as e:
                limbic_result = f"[Limbic] Error: {e}"
        
        # Stream 2: Cortex (layer prediction only for now)
        if self.cortex_loader:
            try:
                active_layers = self.cortex_loader.predict_active_layers(prompt)
                cortex_result = f"[Cortex] Activated {len(active_layers)}/48 layers: {active_layers[:5]}..."
            except Exception as e:
                cortex_result = f"[Cortex] Error: {e}"
        
        # Synthesis
        if self.synthesis_mode == "cortex_only":
            output = cortex_result
        elif self.synthesis_mode == "limbic_only":
            output = limbic_result
        else:
            output = f"{limbic_result}\n\n{cortex_result}"
        
        latency = (time.perf_counter() - t0) * 1000
        
        return LLMResponse(
            text=output,
            tokens=len(prompt.split()),
            latency_ms=latency,
            coherence=0.8,
            gpu_stats={"mode": self.synthesis_mode}
        )

    def embed(self, text: str) -> List[float]:
        # Use Limbic for embeddings if available
        if self.limbic_engine:
            return self.limbic_engine.embed(text)
        return []

    def unload(self):
        if self.limbic_engine:
            self.limbic_engine.unload()
        if self.cortex_loader:
            self.cortex_loader.cpu_cache.clear()
        print("[BiCameraEngine] ✅ Unloaded.")
