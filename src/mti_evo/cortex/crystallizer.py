"""
MTI-EVO LAYER 5.2: COLLECTIVE MEMORY
====================================
Implements the Crystallization Protocol:
- MTICrystal: The immutable EngramUnit.
- MTICrystallizer: The engine that freezes Field State into Crystals.
"""

import time
import json
import hashlib
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class MTICrystal:
    """
    An Engram: A frozen moment of high-confidence consensus.
    Invariant 83: Stores Vectors/Hashes, NO Natural Language.
    Layer 5.3: Supports Reinforcement and Decay.
    """
    crystal_id: str
    context_hash: str          # Hash of the Query/Topic
    vector_centroid: List[float] # The Insight (Mean Vector)
    confidence: float          # 1.0 / Pressure
    creation_time: float
    provenance: List[str]      # Node IDs who signed it
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Phase 8 Lifecycle Fields
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    superseded_by: Optional[str] = None # Link to newer version

    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json(json_str: str) -> 'MTICrystal':
        data = json.loads(json_str)
        return MTICrystal(**data)

class MTICrystallizer:
    def __init__(self, storage_path: str = ".mti-brain/crystals"):
        self.storage_path = storage_path
        self.pressure_threshold = 0.05 # Invariant 80: Stability Req
        self.min_consensus = 2       # Invariant 84: Min Signatures
        
        # Phase 8 Constants
        self.alpha_reinforcement = 0.05
        self.decay_lambda = 0.0001 # Fast enough for tests, slow for prod (1e-6)
        self.death_threshold = 0.1
        self.recrystal_threshold = 0.4
        
        # Ensure FS exists
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            
    def crystallize(self, 
                    active_tissue: Dict[int, Any], 
                    context_str: str, 
                    node_ids: List[str],
                    pressure_reading: float) -> Optional[MTICrystal]:
        """
        Attempts to freeze the current lattice state into a Crystal.
        Returns None if Invariants are violated.
        """
        # 1. Invariant 80: Stability Check
        if pressure_reading > self.pressure_threshold:
            print(f"⚠️ CRYSTALLIZE ABORT: Pressure too high ({pressure_reading:.4f} > {self.pressure_threshold})")
            return None
            
        # 2. Invariant 84: Consensus Check
        if len(node_ids) < self.min_consensus:
            print(f"⚠️ CRYSTALLIZE ABORT: Insufficient Consensus ({len(node_ids)} < {self.min_consensus})")
            return None
            
        # 3. Compute Centroid (The Insight)
        vectors = []
        for neuron in active_tissue.values():
            vectors.append(neuron.weights)
            
        if not vectors:
            return None
            
        centroid = np.mean(vectors, axis=0).tolist()
        
        # 4. Generate ID
        ctx_hash = hashlib.sha256(context_str.encode()).hexdigest()
        raw_sig = f"{ctx_hash}:{str(centroid)[:20]}:{time.time()}"
        c_id = hashlib.sha256(raw_sig.encode()).hexdigest()[:16]
        
        # 5. Create Crystal
        engram = MTICrystal(
            crystal_id=c_id,
            context_hash=ctx_hash,
            vector_centroid=centroid,
            confidence=min(1.0, 1.0 / (pressure_reading + 1e-6)),
            creation_time=time.time(),
            provenance=node_ids,
            last_accessed=time.time()
        )
        
        print(f"💎 CRYSTAL FORMED: {c_id} (Conf: {engram.confidence:.2f})")
        self._save_to_disk(engram)
        return engram
        
    def _save_to_disk(self, crystal: MTICrystal):
        path = os.path.join(self.storage_path, f"{crystal.crystal_id}.json")
        with open(path, 'w') as f:
            f.write(crystal.to_json())
            
    def recall(self, context_str: str) -> Optional[MTICrystal]:
        """
        Retrieves a crystal by context.
        Triggers Decay Check on access.
        """
        target_hash = hashlib.sha256(context_str.encode()).hexdigest()
        
        best_crystal = None
        best_conf = 0.0
        
        for fname in os.listdir(self.storage_path):
            if not fname.endswith(".json"): continue
            
            try:
                with open(os.path.join(self.storage_path, fname), 'r') as f:
                    c = MTICrystal.from_json(f.read())
                    
                if c.context_hash == target_hash:
                    if c.superseded_by: continue # Skip if updated
                    
                    if c.confidence > best_conf:
                        best_crystal = c
                        best_conf = c.confidence
            except:
                pass
                
        if best_crystal:
            print(f"🕯️ MEMORY RECALLED: {best_crystal.crystal_id} (Conf: {best_crystal.confidence:.2f})")
            # Side Effect: Update Access Time
            best_crystal.last_accessed = time.time()
            best_crystal.access_count += 1
            self._save_to_disk(best_crystal)
            
        return best_crystal

    # --- PHASE 8 LIFECYCLE METHODS ---

    def reinforce(self, crystal_id: str, agreement_score: float = 1.0):
        """
        [Layer 5.3] Strengthens a Crystal based on consensus validation.
        """
        path = os.path.join(self.storage_path, f"{crystal_id}.json")
        if not os.path.exists(path): return
        
        with open(path, 'r') as f:
            c = MTICrystal.from_json(f.read())
            
        old_conf = c.confidence
        # Eq: C_new = C_old + alpha * agreement
        c.confidence = min(1.0, c.confidence + (self.alpha_reinforcement * agreement_score))
        c.last_accessed = time.time()
        c.access_count += 1
        
        print(f"💪 REINFORCED {c.crystal_id}: {old_conf:.2f} -> {c.confidence:.2f}")
        self._save_to_disk(c)

    def apply_decay_cycle(self) -> List[str]:
        """
        [Layer 5.3] Runs the Entropy Cycle.
        Returns list of dead crystals pruned.
        """
        dead = []
        now = time.time()
        
        for fname in os.listdir(self.storage_path):
            if not fname.endswith(".json"): continue
            fpath = os.path.join(self.storage_path, fname)
            
            try:
                with open(fpath, 'r') as f:
                    c = MTICrystal.from_json(f.read())
                    
                if c.superseded_by: continue # Already archived
                
                # Decay Eq: C_new = C_old * exp(-lambda * delta_t)
                delta_t = now - c.last_accessed
                decay_factor = np.exp(-self.decay_lambda * delta_t)
                old_conf = c.confidence
                c.confidence *= decay_factor
                
                if c.confidence < self.death_threshold:
                    print(f"💀 PRUNING {c.crystal_id} (Conf {c.confidence:.3f} < {self.death_threshold})")
                    os.remove(fpath) # Hard delete or move to archive
                    dead.append(c.crystal_id)
                else:
                    # Save update
                    self._save_to_disk(c)
                    
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                
        return dead

    def update_crystal(self, old_id: str, new_vector: List[float], new_pressure: float) -> Optional[MTICrystal]:
        """
        [Layer 5.3] Re-Crystallization.
        Supersedes old crystal with a new version.
        """
        old_path = os.path.join(self.storage_path, f"{old_id}.json")
        if not os.path.exists(old_path): return None
        
        with open(old_path, 'r') as f:
            old_c = MTICrystal.from_json(f.read())
            
        # Create Child
        new_conf = min(1.0, 1.0 / (new_pressure + 1e-6))
        raw_sig = f"{old_c.context_hash}:{str(new_vector)[:20]}:{time.time()}"
        new_id = hashlib.sha256(raw_sig.encode()).hexdigest()[:16]
        
        new_c = MTICrystal(
            crystal_id=new_id,
            context_hash=old_c.context_hash,
            vector_centroid=new_vector,
            confidence=new_conf,
            creation_time=time.time(),
            provenance=old_c.provenance + ["RECONSOLIDATION"],
            metadata={"parent": old_id},
            last_accessed=time.time()
        )
        
        # Link
        old_c.superseded_by = new_id
        
        print(f"♻️ RE-CRYSTALLIZED {old_id} -> {new_id} (Conf: {new_conf:.2f})")
        
        self._save_to_disk(old_c) # Save link
        self._save_to_disk(new_c) # Save child
        
        return new_c
        
# Alias for cleaner external API or legacy support
ConceptCrystallizer = MTICrystallizer
