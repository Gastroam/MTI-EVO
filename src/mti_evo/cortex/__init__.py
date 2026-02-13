"""
MTI-EVO Cortex Layer
====================
Higher-order cognitive functions.
Public Interface:
- CortexMemory (Persistence)
- BrocaAdapter (Cognition/Language)
"""
from .memory import CortexMemory
from .broca import BrocaAdapter

# Introspection/Crystallizer are internal components managed by Broca/Memory
# but maybe exposed for advanced use?
# User said: "cortex: high-level cognition wrappers (broca/memory/introspection/crystallizer)"
# Let's export them all for now but define boundaries.

from .introspection import IntrospectionEngine
from .crystallizer import ConceptCrystallizer

__all__ = [
    "CortexMemory",
    "BrocaAdapter",
    "IntrospectionEngine",
    "ConceptCrystallizer"
]
