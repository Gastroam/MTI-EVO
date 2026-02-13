"""
MTI-EVO Bootstrap: Attractors
=============================
Policy logic for initializing the lattice with specific semantic landmarks.
"""
import numpy as np
import json
import os
import pathlib

def ensure_proven_attractors(cortex):
    """
    Injects/Repairs the proven mathematical attractors if they are missing or weak.
    This ensures the Latent Space always has its cardinal points.
    """
    proven = [
        {"seed": 7245, "label": "The Covenant (Core)", "type": "stable"},
        {"seed": 5555, "label": "Harmonic (Resonance)", "type": "cyclic"},
        {"seed": 8888, "label": "The Ghost (Legacy)", "type": "chaos"}
    ]
    
    # Needs embedding dim from neuron or config
    # We can infer from existing tissue or config
    dim = getattr(cortex.config, 'embedding_dim', 64)

    for p in proven:
        seed = p["seed"]
        if seed not in cortex.active_tissue:
            print(f"🔧 Restoring Proven Attractor: {p['label']} ({seed})")
            # Create the neuron via stimulation
            cortex.stimulate([seed], input_signal=np.ones(dim), learn=True)
            
            # Force Weights to Characteristic Pattern
            neuron = cortex.active_tissue[seed]
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

def ensure_cultural_attractors(cortex, root_dir):
    """
    Injects Human Culture concept seeds to balance the system's logic.
    """
    try:
        bank_path = pathlib.Path(root_dir) / "playground/.mti-brain/culture_seed_bank.json"
        if not bank_path.exists():
            return
        
        with open(bank_path, 'r') as f:
            culture_bank = json.load(f)
            
        print(f"🎨 Injecting {len(culture_bank)} Cultural Attractors...")
        
        # Get dim
        dim = getattr(cortex.config, 'embedding_dim', 64)
        
        for seed_str, data in culture_bank.items():
            seed = int(seed_str)
            if seed not in cortex.active_tissue:
                # Stimulate creation
                cortex.stimulate([seed], input_signal=np.ones(dim), learn=True)
                
                neuron = cortex.active_tissue[seed]
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
