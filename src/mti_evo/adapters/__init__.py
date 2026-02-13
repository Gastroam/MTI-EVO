"""
MTI-EVO Adapters
================
Integration points for external systems (LLMs, Audio, etc).
"""
from .llm_adapter import LLMAdapter
from .inference import InferenceProcess

__all__ = ["LLMAdapter", "InferenceProcess"]
