from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .protocol import EngineResult, EngineProtocol

# Backwards Compatibility Alias
LLMResponse = EngineResult

class BaseEngine(ABC):
    """Abstract Base Class for MTI-EVO LLM Engines."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "")
        self.n_ctx = config.get("n_ctx", 4096)
        self.temperature = config.get("temperature", 0.7)
        self.backend_name = "base"

    @abstractmethod
    def load(self, config: dict):
        """Load the model resources."""
        pass

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> EngineResult:
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

    @property
    def capabilities(self) -> dict:
        return {"embedding": False, "streaming": False, "device": "cpu"}
