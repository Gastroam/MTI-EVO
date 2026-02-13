"""
QOOP Engine (Quantum Object-Oriented Programming)
=================================================
BaseEngine-compliant wrapper for probabilistic routing via wavefunction collapse.

This engine treats inference as "observation" - each token forces a collapse
of the routing superposition to a specific expert/block pathway.

Best for:
- Mixture-of-Experts style routing
- Probabilistic program synthesis
- Research into quantum-inspired computing
"""
import time
import random
from typing import List, Dict, Any
from .base import BaseEngine, LLMResponse


class WaveFunction:
    """Probabilistic state that collapses upon observation."""
    def __init__(self, states: Dict[str, float]):
        self.states = states
        self._normalize()
        self.collapsed_val = None

    def _normalize(self):
        total = sum(self.states.values())
        if total == 0:
            val = 1.0 / len(self.states)
            self.states = {k: val for k in self.states}
        else:
            self.states = {k: v/total for k, v in self.states.items()}

    def collapse(self) -> str:
        if self.collapsed_val is not None:
            return self.collapsed_val
        
        r = random.random()
        cumulative = 0.0
        for val, prob in self.states.items():
            cumulative += prob
            if r <= cumulative:
                self.collapsed_val = val
                return val
        self.collapsed_val = random.choice(list(self.states.keys()))
        return self.collapsed_val

    def reset(self):
        self.collapsed_val = None


class QoopEngine(BaseEngine):
    """
    Quantum-Inspired Probabilistic Routing Engine.
    
    Config Keys:
    - expert_weights: Dict of expert_name -> probability weight
    - fallback_engine: Engine to use for actual inference (default: 'gguf')
    - model_path: Path to fallback model
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.backend_name = "qoop"
        
        # Default expert distribution (calibrated for Gemma-style MoE)
        self.expert_weights = config.get("expert_weights", {
            "reasoning": 0.6,
            "factual": 0.3,
            "creative": 0.1
        })
        
        # Create router wavefunction
        self.router = WaveFunction(self.expert_weights)
        
        # Fallback engine for actual generation
        self.fallback_engine = None
        self.fallback_type = config.get("fallback_engine", "gguf")

    def load_model(self):
        print(f"[QoopEngine] 🌌 Initializing Quantum Router...")
        print(f"   Expert Distribution: {self.expert_weights}")
        
        # Load fallback engine for actual generation
        model_path = self.config.get("model_path", "")
        if model_path:
            try:
                if self.fallback_type == "gguf" or model_path.endswith(".gguf"):
                    from .gguf_engine import GGUFEngine
                    self.fallback_engine = GGUFEngine(self.config)
                else:
                    from .native_engine import NativeEngine
                    self.fallback_engine = NativeEngine(self.config)
                self.fallback_engine.load_model()
                print(f"   ✅ Fallback engine ({self.fallback_type}) loaded.")
            except Exception as e:
                print(f"   ⚠️ Fallback engine failed: {e}")

    def infer(self, prompt: str, max_tokens: int = 512, stop: list = None, **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        
        # Reset router for new observation
        self.router.reset()
        
        # Collapse wavefunction to select expert pathway
        selected_expert = self.router.collapse()
        print(f"[QoopEngine] 🎲 Collapsed to: {selected_expert}")
        
        # Modify prompt based on expert
        if selected_expert == "reasoning":
            system_prefix = "Think step by step.\n\n"
        elif selected_expert == "factual":
            system_prefix = "Provide accurate facts.\n\n"
        elif selected_expert == "creative":
            system_prefix = "Be imaginative and creative.\n\n"
        else:
            system_prefix = ""
        
        modified_prompt = system_prefix + prompt
        
        # Run through fallback engine
        if self.fallback_engine:
            response = self.fallback_engine.infer(modified_prompt, max_tokens=max_tokens, stop=stop)
            output = response.text
            tokens = response.tokens
        else:
            output = f"[QOOP:{selected_expert}] Simulated inference (no fallback engine)"
            tokens = len(prompt.split())
        
        latency = (time.perf_counter() - t0) * 1000
        
        return LLMResponse(
            text=output,
            tokens=tokens,
            latency_ms=latency,
            coherence=0.9 if selected_expert == "reasoning" else 0.7,
            gpu_stats={"collapsed_expert": selected_expert, "entropy": sum(self.expert_weights.values())}
        )

    def embed(self, text: str) -> List[float]:
        if self.fallback_engine:
            return self.fallback_engine.embed(text)
        return []

    def unload(self):
        if self.fallback_engine:
            self.fallback_engine.unload()
        print("[QoopEngine] ✅ Unloaded.")

    def get_routing_stats(self) -> dict:
        """Return current router state."""
        return {
            "weights": self.expert_weights,
            "collapsed": self.router.collapsed_val
        }
