"""
MTI-EVO: Machine Thought Interface
==================================
The Evolutionary Holographic Memory System.
"""
__version__ = "2.3.5"
__author__ = "MTI-EVO Team"

from .core.neuron import MTINeuron
from .core.lattice import HolographicLattice
from .core.dense_tissue import MTIDenseTissue
from .core.config import MTIConfig
from .core.logger import get_logger
from .telemetry import TelemetrySystem

try:
    from .mti_proprioceptor import MTIProprioceptor, CognitiveState
except ImportError:
    MTIProprioceptor = None
    CognitiveState = None
try:
    from .engines import (
        GGUFEngine, 
        QuantumEngine, 
        HybridEngine, 
        APIEngine, 
        ResonantEngine, 
        NativeEngine, 
        BiCameraEngine
    )
except ImportError:
    # Fallback or partial load?
    # We might want to try importing them individually if one fails, but for now just catch all.
    # Actually, importing individually is better but more verbose.
    # Let's just define them as None if import fails, or let them be missing from namespace.
    GGUFEngine = None
    QuantumEngine = None
    HybridEngine = None
    APIEngine = None
    ResonantEngine = None
    NativeEngine = None
    BiCameraEngine = None

__all__ = [
    "MTINeuron", 
    "HolographicLattice", 
    "MTIDenseTissue",
    "MTIConfig",
    "get_logger",
    "TelemetrySystem",
    "GGUFEngine",
    "QuantumEngine",
    "HybridEngine",
    "APIEngine",
    "ResonantEngine",
    "NativeEngine",
    "BiCameraEngine",
    "MTIProprioceptor",
    "CognitiveState"
]
