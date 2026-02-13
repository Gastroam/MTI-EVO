
import numpy as np
from types import SimpleNamespace
from ..mti_core import HolographicLattice, MTINeuron

class HarmonicPacket:
    """
    IDRE v3.0 Packet Structure.
    Carries the Harmonic Signature of a concept, but NO weights.
    """
    def __init__(self, seed: int, resonance: float, entropy: float, meta_tag: str = "FLOW"):
        self.seed = seed
        self.resonance = resonance
        self.entropy = entropy
        self.meta_tag = meta_tag

class IDREInterface:
    """
    The Relational Biology Layer.
    Manages Trust and Misunderstanding between Lattices.
    """
    def __init__(self, lattice: HolographicLattice):
        self.lattice = lattice
        # Trust Store: Map of Remote_ID -> Trust Mass
        # For now, we track Trust per CONCEPT (Relational Ontology) or per SENDER?
        # Phase 69 tracked Trust as a concept *inside* the lattice.
        # "Trust" (Seed X) gets mass.
        pass
        
    def generate_packet(self, seed: int, context_vector: np.array) -> HarmonicPacket:
        """
        Creates a packet to send to another mind.
        """
        if seed not in self.lattice.active_tissue:
            return None
            
        neuron = self.lattice.active_tissue[seed]
        
        # Calculate Local Resonance
        resonance = np.dot(neuron.weights, context_vector) + neuron.bias
        # Calculate Local Entropy (Variance)
        entropy = np.std(neuron.weights)
        
        return HarmonicPacket(seed, float(resonance), float(entropy))
        
    def receive_packet(self, packet: HarmonicPacket, context_vector: np.array):
        """
        Process an incoming packet using the Harmonic Protocol.
        Returns: (Synergy, Outcome)
        """
        # 1. Internal Simulation
        # stimulating the requested seed with SHARED context
        
        # Determine strict or lazy? IDRE v3.0 implies we inspect *ourselves*
        if packet.seed not in self.lattice.active_tissue:
            # We don't know this concept.
            # Lazy Init? Or reject?
            # For "First Other", lets lazy init.
            self.lattice.active_tissue[packet.seed] = MTINeuron(input_size=len(context_vector))
            
        neuron = self.lattice.active_tissue[packet.seed]
        
        # 2. Resonance Check
        my_resonance = np.dot(neuron.weights, context_vector) + neuron.bias
        
        # 3. Boundary Effect (The Handshake)
        # Agreement: Sign Match
        agreement = (my_resonance > 0 and packet.resonance > 0) or \
                    (my_resonance < 0 and packet.resonance < 0)
                    
        import hashlib
        seed_trust = int(hashlib.sha256("trust".encode("utf-8")).hexdigest(), 16) % (10**8)
        seed_mis = int(hashlib.sha256("misunderstanding".encode("utf-8")).hexdigest(), 16) % (10**8)
        
        target_seed = seed_trust if agreement else seed_mis
        
        # 4. Crystallization (Reward Learning)
        if target_seed not in self.lattice.active_tissue:
             self.lattice.active_tissue[target_seed] = MTINeuron(input_size=len(context_vector))
             
        # Boost the Boundary Concept
        # We add mass to the Trust/Misunderstanding neuron
        # In a real system, we'd use hebbian update, here we just boost mass blindly for the prototype logic
        # But Phase 69 used `weights += 0.05`.
        self.lattice.active_tissue[target_seed].weights += 0.05
        
        return agreement, "TRUST" if agreement else "MISUNDERSTANDING"
