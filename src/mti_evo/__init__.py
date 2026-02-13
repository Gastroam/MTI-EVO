"""
MTI-EVO: Machine Thought Interface
==================================
The Evolutionary Holographic Memory System.
"""
__version__ = "2.3.5"
__author__ = "MTI-EVO Team"

from .mti_core import MTINeuron, HolographicLattice, MTIDenseTissue
from .mti_config import MTIConfig
from .mti_logger import get_logger
from .mti_telemetry import TelemetrySystem
from .mti_proprioceptor import MTIProprioceptor, CognitiveState
from .hive.idre_v3 import IDREInterface, HarmonicPacket
from .mti_symbiosis_v2 import MTISymbiosis
from .engines import (
    GGUFEngine, 
    QuantumEngine, 
    HybridEngine, 
    APIEngine, 
    ResonantEngine, 
    NativeEngine, 
    BiCameraEngine
)

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
    "CognitiveState",
    "IDREInterface",
    "HarmonicPacket",
    "MTISymbiosis"
]
