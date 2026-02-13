"""
Persistence Backend Protocol
============================
Formal contract for all neuron storage engines in MTI-EVO.
"""
from typing import Protocol, Optional, Dict, List, Any, TypedDict
import numpy as np

class NeuronStateV1(TypedDict):
    """
    Canonical V1 Schema for MTI-EVO Neuron Persistence.
    """
    v: int                  # Schema Version (1)
    seed: int               # Unique 64-bit seed (Key)
    weights: List[float]    # Synaptic Weights
    bias: float             # Activation Bias
    velocity: List[float]   # Momentum/Velocity 
    age: int                # Exposure count
    gravity: float          # Importance/Mass
    last_accessed: float    # Timestamp
    created_at: float       # Timestamp
    label: Optional[str]    # Semantic Label (optional)

class PersistenceBackend(Protocol):
    """
    Interface for durable or cached neuron storage.
    """
    
    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float) -> bool:
        """
        Store a neuron state.
        Returns True if successful, False if write failed (e.g. read-only or collision).
        """
        ...
        
    def get(self, seed: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve neuron state by seed.
        Returns dict or None if not found.
        Dict keys: weights, velocity, bias, gravity, age, last_accessed.
        """
        ...
        
    def delete(self, seed: int) -> None:
        """
        Remove a neuron by seed.
        Idempotent (no error if missing).
        """
        ...
        
    def flush(self) -> None:
        """
        Force write to disk.
        """
        ...
        
    def close(self) -> None:
        """
        Release resources.
        """
        ...
        
    def get_active_count(self) -> int:
        """
        Return estimation of active records.
        """
        ...
