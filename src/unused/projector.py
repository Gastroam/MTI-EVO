
"""
Synaptic Projector
==================
Dimensionality reduction for projecting Dream vectors (2560d) onto the
Holographic Lattice (64d).

Uses the Johnson-Lindenstrauss lemma via Gaussian Random Projection
to preserve relative distances between archetypes.
"""

import numpy as np

class SynapticProjector:
    def __init__(self, input_dim=2560, output_dim=64, seed=1337):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Deterministic Initialization of the Projection Matrix
        # This ensures the "Dream Space" maps consistently to "Logic Space"
        rng = np.random.RandomState(seed)
        
        # Standard Normal / sqrt(output_dim) for length preservation
        self.matrix = rng.normal(0, 1.0 / np.sqrt(output_dim), (output_dim, input_dim))
        
    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Project high-dim vector to low-dim instinct.
        
        Args:
            vector: (input_dim,) numpy array
            
        Returns:
            (output_dim,) numpy array (Normalized)
        """
        # Ensure input shape
        if vector.shape[0] != self.input_dim:
            # Handle potential mismatch (e.g. if user changes model)
            # Just slice or pad for robustness, but logging warning is better
            if vector.shape[0] > self.input_dim:
                vector = vector[:self.input_dim]
            else:
                vector = np.pad(vector, (0, self.input_dim - vector.shape[0]))
        
        # Linear Projection
        projected = np.dot(self.matrix, vector)
        
        # Normalize (neurons operate best on unit vectors or controlled magnitudes)
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm * 10.0 # Scale to Broca's typical magnitude (~10-100)
            
        return projected
