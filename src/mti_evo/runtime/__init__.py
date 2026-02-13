"""
MTI-EVO Runtime
===============
Orchestration layer.
Public Interface:
- SubstrateRuntime (The main composition root)
"""
from .substrate_runtime import SubstrateRuntime

__all__ = ["SubstrateRuntime"]
