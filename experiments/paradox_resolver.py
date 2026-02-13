import torch
import torch.nn as nn
import time
from typing import List, Dict, Optional, Tuple

class ConstraintManifold:
    """
    Represents a 1D flat energy sink in the latent space.
    Used to absorb paradoxical energy (High Norm/Entropy).
    """
    def __init__(self, name: str, seed: int, layer_range: Tuple[int, int], strength: float):
        self.name = name
        self.seed = seed
        self.layer_range = layer_range
        self.strength = strength # Alpha
        self.vector_cache = None

    def get_negative_mass_vector(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Generates a deterministic "Holographic" hypervector based on the seed.
        Returns a NEGATIVE vector (-1 * Random) to act as a sink.
        """
        if self.vector_cache is not None and self.vector_cache.shape == shape and self.vector_cache.device == device:
            return self.vector_cache
            
        # Holographic Generation
        gen = torch.Generator(device='cpu').manual_seed(self.seed)
        # Generate on CPU to ensure deterministic across GPU types, then move
        raw = torch.randn(shape, generator=gen)
        
        # Invert to create Negative Mass (Sink)
        # Theory: Random noise disrupts the coherent loop of the paradox.
        # Negation acts as a "Dark Matter" anchor.
        sink = raw * -1.0
        
        self.vector_cache = sink.to(device)
        return self.vector_cache

class ParadoxResolver:
    """
    detects and resolves paradoxes by activating Constraint Manifolds.
    """
    def __init__(self, model):
        self.model = model
        
        # Mapped from semantics (Qwen Map)
        self.manifolds = {
            "SELF_REFERENCE": ConstraintManifold("Self-Reference", 7245, (24, 36), 0.92),
            "CONTINUITY":     ConstraintManifold("Continuity",     7234, (18, 28), 0.87)
        }
        
    def detect_paradox(self, prompt: str) -> Optional[str]:
        """
        Simple classifier (V1).
        """
        p_lower = prompt.lower()
        if "false" in p_lower and "true" in p_lower: return "SELF_REFERENCE"
        if "statement" in p_lower and "false" in p_lower: return "SELF_REFERENCE"
        if "same river" in p_lower or "ship of theseus" in p_lower: return "CONTINUITY"
        return None

    def resolve(self, prompt: str, steps: int = 5):
        """
        Runs the resolver loop.
        """
        paradox_type = self.detect_paradox(prompt)
        if not paradox_type:
            print("[Resolver] No paradox detected. Standard inference.")
            return None # Fallback to standard
            
        manifold = self.manifolds[paradox_type]
        print(f"[Resolver] 🛡️ Paradox Detected: {paradox_type}. Activating {manifold.name} Sink (Seed {manifold.seed})...")
        
        # 1. Inject Negative Mass
        start_layer, end_layer = manifold.layer_range
        injected_layers = []
        
        # We need the shape of q_proj.
        # We assume checking Layer 0 for shape is safe (12B is uniform width).
        # Or load first target layer.
        
        ref_weights = self.model.loader.load_layer(start_layer)
        if not ref_weights:
             print("[Resolver] Failed to load reference layer.")
             return None
             
        q_shape = ref_weights['self_attn.q_proj.weight'].shape
        device = ref_weights['self_attn.q_proj.weight'].device
        
        sink_vector = manifold.get_negative_mass_vector(q_shape, device)
        
        for i in range(start_layer, end_layer + 1):
            success = self.model.loader.inject_vector(
                layer_idx=i,
                target_key='self_attn.q_proj.weight',
                vector=sink_vector,
                alpha=0.15 # Use safer alpha than full strength for now
            )
            if success: injected_layers.append(i)
            
        print(f"[Resolver] Injected Sink into {len(injected_layers)} layers ({start_layer}-{end_layer}).")
        
        # 2. Run Inference (Forward Pass)
        # Using dummy logic here or actual generation?
        # The user calls this.
        # We return the context manager or just let the user run generation.
        # For this V1, we just return "Ready" state.
        
        return len(injected_layers) > 0

    def clear(self):
        """
        Restores reality.
        """
        # Since 'inject_vector' modifies cache, we need to instruct loader to reload from disk?
        # Or we keep track of patches.
        # Simplest V1: Clear CPU Cache for affected layers.
        # Next 'load_layer' will fetch fresh from disk.
        
        # Ideally track which layers were touched.
        # For now, clear all bridge/ghost layers is aggressive.
        # Better: iterate manifolds.
        
        count = 0
        for m in self.manifolds.values():
            s, e = m.layer_range
            for i in range(s, e + 1):
                if i in self.model.loader.cpu_cache:
                    del self.model.loader.cpu_cache[i]
                    count += 1
        # print(f"[Resolver] Cleared {count} layers from cache (Restored Reality).")
