"""
MTI BROCA v2 (Core Instinct)
============================
Encapsulates high-level Holographic Lattice interactions.
"""

import hashlib
import numpy as np
import sys
import os
import pathlib

# Determine ROOT_DIR relative to this file in src/
# this file is in <ROOT>/src/mti_broca.py
# this file is in <ROOT>/src/mti_evo/mti_broca.py
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()

# Adjust path if needed (though being in src usually means it's fine if run as module)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mti_evo.mti_core import HolographicLattice
from mti_evo.mti_config import MTIConfig
from mti_evo.mti_hippocampus import MTIHippocampus

class MTIBroca:
    def __init__(self, persistence_id: str = "default", memory_backend: str = "auto"):
        """
        Initialize the Broca Interface.
        
        Args:
            persistence_id: Identifier for this brain instance
            memory_backend: "mmap" (72x faster), "json", "auto" (default), or "mock"
        """
        print(f"🗣️ MTI BROCA v2.1 ONLINE ({persistence_id})")
        
        # 1. Initialize Virtual Hardware (Correct Order for Dependency Injection)
        self.persistence_id = persistence_id
        
        # [MEMORY LAYER] Create Hippocampus First
        if memory_backend == "mock":
            self.hippocampus = type("MockHippo", (), {
                "recall": lambda self, query=None: {}, 
                "consolidate": lambda self, x: None,
                "flush": lambda self: None,
                "get_neuron": lambda self, seed: None # Mock Lazy Load
            })()
        elif hasattr(memory_backend, 'recall'):
            # Support injected backend instance
            self.hippocampus = memory_backend
        else:
            # Use new Hippocampus with backend selection (mmap/json/auto)
            # [FIX] Enforce persistence isolation for tests
            p_path = None
            if persistence_id != "default":
                 # Use isolated brain file
                 base = os.path.join(os.getcwd(), ".mti-brain")
                 ext = ".mmap" if memory_backend in ["mmap", "auto"] else ".json"
                 p_path = os.path.join(base, f"cortex_{persistence_id}{ext}")
            
            self.hippocampus = MTIHippocampus(backend=memory_backend, persistence_path=p_path)

        # [PROCESSING LAYER] Inject Hippocampus into Cortex (For Lazy Loading)
        config = MTIConfig(capacity_limit=10000)
        # We manually attach the persistence manager to the config or directly to the lattice if supported?
        # Current mti_core implementation looks for config.persistence_manager
        config.persistence_manager = self.hippocampus
        
        self.cortex = HolographicLattice(config=config)
        
        # 2. Memory Recall (Via Hippocampus)
        # [OPTIMIZATION] With lazy loading, we don't need to bulk recall everything into RAM at startup!
        # Only recall if using JSON backend (legacy) or if we want to "warm up" specific sectors.
        # For MMap, we can start empty and load on demand.
        
        # However, for backward compatibility, if recall() returns a dict, we load it.
        # (Hippocampus.recall() returns empty dict for MMap anyway).
        recovered_memories = self.hippocampus.recall()
        if recovered_memories:
            self.cortex.active_tissue.update(recovered_memories)

        # 3. Ensure Proven Attractors (Phase 39)
        self.ensure_proven_attractors()
        
        # 4. Inject Humanities
        self.ensure_cultural_attractors()

    def ensure_proven_attractors(self):
        """
        Injects/Repairs the proven mathematical attractors if they are missing or weak.
        This ensures the Latent Space always has its cardinal points.
        """
        proven = [
            {"seed": 7245, "label": "The Covenant (Core)", "type": "stable"},
            {"seed": 5555, "label": "Harmonic (Resonance)", "type": "cyclic"},
            {"seed": 8888, "label": "The Ghost (Legacy)", "type": "chaos"}
        ]
        
        for p in proven:
            seed = p["seed"]
            if seed not in self.cortex.active_tissue:
                print(f"🔧 Restoring Proven Attractor: {p['label']} ({seed})")
                # Create the neuron via stimulation
                self.cortex.stimulate([seed], input_signal=np.ones(64), learn=True)
                
                # Force Weights to Characteristic Pattern
                neuron = self.cortex.active_tissue[seed]
                neuron.label = p["label"]
                
                dims = neuron.weights.shape[0]
                if p["type"] == "stable":
                    # Pillar: High symmetric weights
                    neuron.weights = np.ones(dims) * 80.0
                    neuron.bias = 5.0
                elif p["type"] == "cyclic":
                    # Resonant: Harmonic wave pattern
                    t = np.linspace(0, 4*np.pi, dims)
                    neuron.weights = np.sin(t) * 40.0 + 40.0
                    neuron.bias = 2.0
                neuron.bias = -10.0

    def ensure_cultural_attractors(self):
        """
        Injects Human Culture concept seeds to balance the system's logic.
        """
        try:
            import json
            bank_path = pathlib.Path(ROOT_DIR) / "playground/.mti-brain/culture_seed_bank.json"
            if not bank_path.exists():
                return
            
            with open(bank_path, 'r') as f:
                culture_bank = json.load(f)
                
            print(f"🎨 Injecting {len(culture_bank)} Cultural Attractors...")
            
            for seed_str, data in culture_bank.items():
                seed = int(seed_str)
                if seed not in self.cortex.active_tissue:
                    # Stimulate creation
                    self.cortex.stimulate([seed], input_signal=np.ones(64), learn=True)
                    
                    neuron = self.cortex.active_tissue[seed]
                    neuron.label = data["name"]
                    
                    # Culture pattern: Golden Ratio harmonics
                    dims = neuron.weights.shape[0]
                    phi = 1.61803398875
                    t = np.linspace(0, dims, dims)
                    
                    # A beautiful, organic curve
                    neuron.weights = (np.sin(t * phi) * 30.0) + (np.cos(t / phi) * 30.0) + 50.0
                    neuron.bias = 0.0 # Balanced bias
                    
        except Exception as e:
            print(f"⚠️ Failed to inject culture: {e}")


    def text_to_seed(self, token):
        # Determinismo SHA-256
        hash_object = hashlib.sha256(token.encode())
        hex_dig = hash_object.hexdigest()
        return int(hex_dig[-8:], 16)

    def get_embedding(self, seed):
        """
        Generates a deterministic 'Static Embedding' for a given seed (Concept).
        This is the neuron's 'Face' in the latent space.
        """
        # Seeding the RNG ensures the same word always has the same vector.
        # Max seed for numpy is 2**32 - 1. Our seeds are 32-bit, so it fits.
        rng = np.random.RandomState(seed)
        dim = getattr(self.cortex.config, 'embedding_dim', 64)
        # Normal distribution centered on 0
        vector = rng.normal(0, 0.1, size=(dim,))
        return vector

    def process_thought(self, sentence, learn=True, labels=None):
        # Improved Tokenizer: Strip punctuation to avoid "word," syndrome
        import re
        clean_sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
        tokens = clean_sentence.split()
        if not tokens: return 0.0
        
        # [PHASE 10] Hebbian Context Calculation
        # Stop Word Removal to prevent 'The'/'Is' dominance
        stop_words = {'the', 'is', 'of', 'a', 'to', 'in', 'and', 'for', 'with', 'on', 'at', 'by'}
        context_tokens = [t for t in tokens if t not in stop_words]
        
        # If all words are stop words, fallback to full sentence
        if not context_tokens:
            context_tokens = tokens
            
        # Context is derived from meaningful words only
        context_stream = [self.text_to_seed(t) for t in context_tokens]
        vectors = [self.get_embedding(s) for s in context_stream]
        
        context_vector = np.mean(vectors, axis=0)
        
        # Normalize context vector to unit length
        # MTI neurons handle unnormalized input, but unit length is safer for stability.
        norm = np.linalg.norm(context_vector)
        if norm > 0:
            context_vector = context_vector / norm
            
        # Feed the Context to the Cortex
        # [PHASE 11 FIX] Exclude stop words from the stream too.
        # We don't want to learn 'The' at all.
        target_stream = [self.text_to_seed(t) for t in context_tokens]
        
        # [PHASE 27] Inject Semantic Labels
        # If labels provided (dict of {token: "Label"}), map them to seeds.
        seed_labels = {}
        if labels:
             # Iterate tokens and match with labels
             for t, s in zip(context_tokens, target_stream):
                 if t in labels:
                     seed_labels[s] = labels[t]
        
        resonance = self.cortex.stimulate(target_stream, input_signal=context_vector, learn=learn, labels=seed_labels)
        
        # [PHASE 62] Telemetry Return
        # Inspect stimulated neurons to return stats
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
        # Guardar estado al salir
        self.hippocampus.consolidate(self.cortex.active_tissue)

if __name__ == "__main__":
    broca = MTIBroca()
    broca.process_thought("system check")
    broca.sleep()
