from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class LLMResponse:
    text: str
    tokens: int
    latency_ms: float
    coherence: float
    gpu_stats: dict = None

class BaseEngine(ABC):
    """Abstract Base Class for MTI-EVO LLM Engines."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "")
        self.n_ctx = config.get("n_ctx", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.backend_name = "base"

    @abstractmethod
    def load_model(self):
        """Load the model resources."""
        pass

    @abstractmethod
    def infer(self, prompt: str, max_tokens: int = 1024, stop: list = None, **kwargs) -> LLMResponse:
        """Generate text."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding."""
        pass

    @abstractmethod
    def unload(self):
        """Free resources."""
        pass
