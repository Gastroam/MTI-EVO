"""
MTI-EVO Broca (Cognitive Adapter)
=================================
The "Cognitive IO Facade" that translates text into lattice stimulation.

Responsibilities:
1. Tokenization (Text -> Tokens)
2. Seed Mapping (Tokens -> Seeds)
3. Embedding (Seeds -> Vectors)
4. Stimulation (Vectors -> Lattice)

Anti-Goals (Moved elsewhere):
- Persistence Lifecycle (Handled by Runtime/Memory)
- Bootstrap/Policy (Handled by Bootstrap modules)
- Path Hacks (Handled by proper packaging)
"""
import hashlib
import numpy as np
import re
from typing import Dict, Any, Optional, List

from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.config import MTIConfig
from mti_evo.cortex.memory import CortexMemory

class BrocaAdapter:
    """
    Pure cognitive adapter. 
    State is injected (Config, Memory, Lattice).
    """
    def __init__(self, config: MTIConfig, hippocampus: CortexMemory):
        """
        Initialize the Broca Adapter.
        
        Args:
            config: MTI Configuration
            hippocampus: Persistence Manager (already initialized)
        """
        self.config = config
        self.hippocampus = hippocampus
        
        # Calculate embedding dim from config or default
        self.embedding_dim = getattr(config, 'embedding_dim', 64)
        
        # Link Memory to Config for Core access (Legacy compat)
        self.config.persistence_manager = self.hippocampus
        
        # Initialize Cortex
        self.cortex = HolographicLattice(config=self.config)
        
        # Hydrate Cortex from Memory (Lazy or Eager based on Memory backend)
        # Memory manager now handles the "smart" part of recall (empty for mmap)
        recovered_memories = self.hippocampus.recall()
        if recovered_memories:
            self.cortex.active_tissue.update(recovered_memories)

    def text_to_seed(self, token: str) -> int:
        """Deterministic Token -> Seed mapping (SHA-256)."""
        hash_object = hashlib.sha256(token.encode())
        hex_dig = hash_object.hexdigest()
        return int(hex_dig[-8:], 16)

    def get_embedding(self, seed: int) -> np.ndarray:
        """
        Generates a deterministic 'Static Embedding' for a given seed.
        This is the neuron's 'Face' in the latent space.
        """
        # Seeding the RNG ensures the same word always has the same vector.
        rng = np.random.RandomState(seed)
        # Normal distribution centered on 0, small variance
        vector = rng.normal(0, 0.1, size=(self.embedding_dim,))
        return vector

    def process_thought(self, sentence: str, learn: bool = True, labels: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Inject text into the lattice.
        Returns resonance metrics.
        """
        # 1. Tokenization
        clean_sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
        tokens = clean_sentence.split()
        if not tokens: 
            return {"resonance": 0.0, "max_gravity": 0.0, "stimulated_count": 0}
        
        # 2. Context Calculation (Hebbian)
        stop_words = {'the', 'is', 'of', 'a', 'to', 'in', 'and', 'for', 'with', 'on', 'at', 'by'}
        context_tokens = [t for t in tokens if t not in stop_words]
        
        if not context_tokens:
            context_tokens = tokens
            
        context_stream = [self.text_to_seed(t) for t in context_tokens]
        vectors = [self.get_embedding(s) for s in context_stream]
        
        # Mean pooling for context vector
        if vectors:
            context_vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(context_vector)
            if norm > 0:
                context_vector = context_vector / norm
        else:
             context_vector = np.zeros(self.embedding_dim)

        # 3. Target Stream preparation
        target_stream = [self.text_to_seed(t) for t in context_tokens]
        
        # 4. Label Mapping
        seed_labels = {}
        if labels:
             for t, s in zip(context_tokens, target_stream):
                 if t in labels:
                     seed_labels[s] = labels[t]
        
        # 5. Stimulation
        resonance = self.cortex.stimulate(target_stream, input_signal=context_vector, learn=learn, labels=seed_labels)
        
        # 6. Telemetry aggregation
        max_gravity = 0.0
        for seed in target_stream:
            if seed in self.cortex.active_tissue:
                max_gravity = max(max_gravity, self.cortex.active_tissue[seed].gravity)

        return {
            "resonance": float(resonance),
            "max_gravity": float(max_gravity),
            "stimulated_count": len(target_stream)
        }

    def sleep(self):
        """Trigger persistence consolidation."""
        self.hippocampus.consolidate(self.cortex.active_tissue)
