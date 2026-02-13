"""
MTI-EVO Core Library (Compatibility Facade)
===========================================
Public APIs remain stable while implementations live in mti_evo.core.* modules.
"""

from mti_evo.core.dense_tissue import MTIDenseTissue
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.neuron import MTINeuron

__all__ = ["MTINeuron", "MTIDenseTissue", "HolographicLattice"]
