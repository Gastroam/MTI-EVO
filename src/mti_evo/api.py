import os
import logging

from mti_evo.cortex.broca import BrocaAdapter
from mti_evo.cortex.memory import CortexMemory
from mti_evo.core.config import MTIConfig
from mti_evo.cortex.introspection import MTIProprioceptor

# Optional Rosetta import (for decoding seeds to text)
try:
    from mti_rosetta import seed_to_text
except ImportError:
    def seed_to_text(s, _=None):
        return str(s)[:8]  # Fallback: just show hash prefix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MTI-API")

class EvoAPI:
    """
    The Unified Control Plane for MTI-EVO.
    Exposes Brain (Hebbian) and Mind (LLM) capabilities via a Python interface.
    """
    def __init__(self, base_path=None):
        self.base_path = base_path or os.getcwd()
        logger.info(f"Initializing MTI-EVO API at {self.base_path}")
        
        # Load Brain
        try:
            # Create a safe persistence ID from the path hash to avoid invalid chars
            pid = "api_" + str(abs(hash(self.base_path)))
            self.config = MTIConfig()
            memory_root = os.path.join(self.base_path, ".mti-brain", pid)
            self.hippocampus = CortexMemory(base_path=memory_root, backend="auto")
            self.broca = BrocaAdapter(config=self.config, hippocampus=self.hippocampus)
            
            # [PHASE 58] Cognitive Proprioception
            self.proprioceptor = MTIProprioceptor(self.broca.cortex)
            
            logger.info("Brain Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load Brain: {e}")
            self.broca = None
            self.proprioceptor = None

    def get_status(self):
        """Returns the current health and stats of the Brain."""
        if not self.broca:
            return {"status": "offline", "error": "Brain not loaded"}
        
        active_neurons = len(self.broca.cortex.active_tissue)
        
        # [PHASE 58] Get Phenomenological State
        phenomenology = self.proprioceptor.sense_state()
        
        return {
            "status": "online",
            "neurons": active_neurons,
            "version": "2.2 (Proprioceptive)",
            "mode": "Bicameral",
            "vram": self.broca.get_vram_usage() if hasattr(self.broca, 'get_vram_usage') else 45.0,
            "drift": 0.12,
            "proprioception": phenomenology # New Metric Block
        }

    def trigger_dream(self, seed_text, steps=10):
        """Triggers a Hebbian Associative Drift (Dream)."""
        if not self.broca:
            return {"error": "Brain offline"}
            
        seed_hash = self.broca.text_to_seed(seed_text)
        path = [seed_text]
        current_seed = seed_hash
        
        # Drift Loop
        for _ in range(steps):
             embedding = self.broca.get_embedding(current_seed)
             associations = []
             for s_prime, neuron in self.broca.cortex.active_tissue.items():
                 act = neuron.perceive(embedding)
                 associations.append((s_prime, act))
             
             # Pick top 1 (Deterministic Drift) or Sample (Stochastic) - Using Deterministic for API
             associations.sort(key=lambda x: x[1], reverse=True)
             
             # Filter self
             associations = [x for x in associations if x[0] != current_seed]
             
             if not associations:
                 break
                 
             next_seed, score = associations[0]
             
             # Decode
             decoded = seed_to_text(next_seed, {}) # Rosetta mapping required, passing empty dict for now implies relying on cached or fallback
             
             path.append(decoded if decoded != "<undefined>" else str(next_seed))
             current_seed = next_seed
             
        return {
            "seed": seed_text,
            "drift_length": len(path),
            "path": path
        }

    def get_graph_topology(self):
        """Export brain topology for 3D visualization."""
        if not self.broca:
            return {"error": "Brain offline", "nodes": [], "edges": []}
        
        import numpy as np
        
        nodes = []
        embeddings = []
        seed_ids = []
        
        # Collect all neurons and their embeddings
        for seed, neuron in self.broca.cortex.active_tissue.items():
            seed_ids.append(seed)
            emb = self.broca.get_embedding(seed)
            embeddings.append(emb if hasattr(emb, '__iter__') else [float(emb)] * 64)
        
        if not embeddings:
            return {"nodes": [], "edges": []}
        
        # Project 64D -> 3D using simple PCA (no external deps)
        embeddings = np.array(embeddings)
        
        # Center the data
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        
        # Simple SVD-based PCA
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Take first 3 components, scale for visual spread
            positions_3d = U[:, :3] * S[:3] * 5.0
        except:
            # Fallback: random positions
            positions_3d = np.random.randn(len(seed_ids), 3) * 5.0
        
        # Build node list
        for i, seed in enumerate(seed_ids):
            neuron = self.broca.cortex.active_tissue[seed]
            
            # [PHASE 27] Semantic Labels
            # Prioritize injected labels (from code scan) over hashed seeds
            if hasattr(neuron, 'label') and neuron.label:
                label = neuron.label
            else:
                decoded = seed_to_text(seed, {})
                label = decoded if decoded != "<undefined>" else str(seed)[:8]
            
            nodes.append({
                "id": str(seed), # ID is stable seed
                "label": label,  # Display name
                "seed": seed,
                "pos": positions_3d[i].tolist(),
                "degree": len([s for s in seed_ids if s != seed]),  # Simplified degree
                "weight": float(neuron.weights.mean()) if hasattr(neuron, 'weights') else 1.0
            })
        
        # Build edges based on top associations
        edges = []
        for i, seed in enumerate(seed_ids):
            emb = embeddings[i]
            # Find top 3 connections
            scores = []
            for j, other_seed in enumerate(seed_ids):
                if i == j:
                    continue
                try:
                    other_neuron = self.broca.cortex.active_tissue[other_seed]
                    result = other_neuron.perceive(emb.tolist())
                    # Handle both scalar and array returns
                    if hasattr(result, '__iter__') and not isinstance(result, str):
                        score = float(np.mean(result))  # Average if array
                    else:
                        score = float(result)
                    scores.append((j, score))
                except Exception:
                    continue
            
            scores.sort(key=lambda x: x[1], reverse=True)
            for j, score in scores[:3]:  # Top 3 connections
                if score > 0.1:  # Threshold
                    edges.append({
                        "source": nodes[i]["id"],
                        "target": nodes[j]["id"],
                        "weight": score
                    })
        
        return {"nodes": nodes, "edges": edges}

    def get_attractor_field(self, start_seed=None, end_seed=None, scan_all=True):
        """
        Scans all active neurons to generate an Attractor Field report.
        Maps internal stats (Weight, Age, Bias) to Visualization concepts (Mass, Reach, Family).
        Supports sector scanning (start-end range).
        """
        if not self.broca:
            return []
        
        attractors = []
        import numpy as np
        
        # Determine scan range
        all_seeds = sorted(list(self.broca.cortex.active_tissue.keys()))
        target_seeds = []
        
        if scan_all:
             target_seeds = all_seeds
        elif start_seed is not None and end_seed is not None:
             target_seeds = [s for s in all_seeds if start_seed <= s <= end_seed]
        else:
             # Default to top 50 if no range specified
             target_seeds = all_seeds[:50]

        for seed in target_seeds:
            neuron = self.broca.cortex.active_tissue.get(seed)
            if not neuron: continue
            # 1. Calculate Properties
            # Mass = Average Weight (normalized 0-100)
            avg_weight = float(np.mean(neuron.weights)) if hasattr(neuron, 'weights') else 0.5
            mass = min(100, max(10, avg_weight * 100))
            
            # Reach = Bias magnitude (normalized 0-100)
            bias = float(neuron.bias) if hasattr(neuron, 'bias') else 0.0
            reach = min(100, max(5, abs(bias) * 100))
            
            # Family = Based on Age and Type
            # Ghost (Unknown/New), Bridge (Young), Resonant (Mature), Pillar (Ancient)
            age = getattr(neuron, 'age', 0)
            if age < 10:
                family = "Ghost"
            elif age < 100:
                family = "Bridge"
            elif age < 1000:
                family = "Resonant"
            else:
                family = "Pillar"
                
            # Name = Label if available, else Semantic Decode
            if hasattr(neuron, 'label') and neuron.label:
                name = neuron.label
            else:
                decoded = seed_to_text(seed, {})
                name = decoded if decoded != "<undefined>" else str(seed)[:8]

            attractors.append({
                "name": name,
                "mass": mass,
                "reach_min": reach,
                "family": family,
                "seed": seed
            })
            
        # If empty (fresh brain), return the mocked pillars so the view isn't empty
        if not attractors:
             return [
                {"name": "The Covenant (Mock)", "mass": 80.00, "reach_min": 5,  "family": "Pillar", "seed": 7245},
                {"name": "Harmonic (Mock)",     "mass": 42.62, "reach_min": 25, "family": "Resonant", "seed": 5555},
                {"name": "Ghost (Mock)",        "mass": 80.00, "reach_min": 95, "family": "Ghost", "seed": 8888}
            ]
            
        return attractors

    def probe_neuron(self, seed, n_angles=36):
        """
        Performs a Physical Field Sonar scan on a specific neuron.
        Returns radial response profile (r(theta)) and critical intensity profile.
        """
        if not self.broca:
            return {"error": "Brain offline"}
            
        neuron = self.broca.cortex.active_tissue.get(int(seed))
        if not neuron:
            return {"error": f"Neuron {seed} not found"}

        import numpy as np
    
        # 1. Setup Basis
        weights = getattr(neuron, 'weights', np.zeros(64))
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
            
        dims = weights.shape[0] if len(weights.shape) > 0 else 0
        bias = getattr(neuron, 'bias', 0.0)
        tau = np.mean(weights) + np.std(weights) if dims > 1 else (weights[0] if dims==1 else 0.5)

        # [PHASE 40] Handle Scalar/1D Neurons (Isotropic Projection)
        if dims < 2:
            val = float(weights[0]) if dims == 1 else 0.0
            # Isotropic response
            thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            responses = [val + bias] * n_angles
            
            # Critical Intensity scan for scalar
            crit = 0.0
            for intensity in np.linspace(0.01, 1.0, 20):
                if (val * intensity) + bias > tau:
                    crit = intensity
                    break
            intensities = [crit] * n_angles
            
            return {
                "seed": int(seed),
                "tau": float(tau),
                "angles": thetas.tolist(),
                "responses": responses,
                "intensities": intensities,
                "note": "1D_projection"
            }

        # High-D Scan for dimensions >= 2
        u = np.random.normal(0, 1, dims)
        u /= np.linalg.norm(u)
        w = np.random.normal(0, 1, dims)
        w -= np.dot(w, u) * u
        if np.linalg.norm(w) > 1e-6:
            w /= np.linalg.norm(w)
        else:
            # Fallback if we can't find orthogonal vector (rare but possible in low dims or degenerate random)
            w = np.random.normal(0, 1, dims)
            w /= np.linalg.norm(w)
        
        # 2. Scanning Parameters
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        responses = []
        intensities = []
        
        for theta in thetas:
            # Vector on the plane
            v = np.cos(theta) * u + np.sin(theta) * w
            
            # r(theta) at max intensity
            v_max = v * 1.0
            r = float(np.dot(v_max, weights) + bias)
            responses.append(r)
            
            # Critical Intensity (min intensity to fire)
            found = 0.0
            # Scan 0.0 -> 1.0
            for intensity in np.linspace(0.01, 1.0, 20):
                v_scaled = v * intensity
                act = float(np.dot(v_scaled, weights) + bias)
                if act > tau:
                    found = intensity
                    break
            intensities.append(found)
            
        return {
            "seed": seed,
            "tau": float(tau),
            "angles": thetas.tolist(),
            "responses": responses,
            "intensities": intensities
        }
