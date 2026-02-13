"""
MTI-EVO Bootstrap
=================
Policy injection for system initialization.
Public Interface:
- ensure_proven_attractors
- ensure_cultural_attractors
"""
from .attractors import ensure_proven_attractors, ensure_cultural_attractors

__all__ = ["ensure_proven_attractors", "ensure_cultural_attractors"]
