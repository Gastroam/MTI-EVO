"""
MTI-EVO Engine Protocol
=======================
Strict contract for all Inference Engines.
Engines MUST be pure, deterministic, and isolated.
"""
from typing import Protocol, Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class EngineResult:
    """Standardized result from an engine inference."""
    text: str
    tokens: int
    latency_ms: float
    confidence: float = 1.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EngineProtocol(Protocol):
    """
    The Contract.
    All engines must implement this.
    """
    
    def load(self, config: Dict[str, Any]) -> None:
        """
        Load resources (Model, VRAM).
        Must be idempotent.
        """
        ...

    def infer(self, prompt: str, **kwargs) -> EngineResult:
        """
        Execute inference.
        Args:
            prompt: Input text.
            **kwargs: Generation params (max_tokens, temperature, etc).
        Returns:
            EngineResult
        """
        ...

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector.
        Returns empty list if not supported.
        """
        ...

    def unload(self) -> None:
        """
        Free all resources.
        Must not raise if already unloaded.
        """
        ...
    
    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities dict.
        e.g. {"embedding": True, "streaming": False, "device": "cuda"}
        """
        ...
