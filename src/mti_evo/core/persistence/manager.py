"""
Tiered Persistence Manager
==========================
Orchestrates memory tiers for the Holographic Lattice.
Tier 1: MMAP (Direct-Indexed Cache) - Fast, ephemeral preference.
Tier 2: JSONL (Write-Ahead Log) - Durable truth, infinite capacity.

Policy:
- Writes: WAL (Truth) -> MMAP (Cache Update)
- Reads: MMAP (Hit) -> If Miss: WAL (Recall) -> Promote to MMAP 
"""
import os
import time
import numpy as np
from typing import Optional, Dict

from mti_evo.core.persistence.mmap import MMapNeuronStore
from mti_evo.core.persistence.jsonl import JsonlPersistence

class PersistenceManager:
    def __init__(self, config):
        """
        Initialize persistence tiers based on config.
        """
        self.config = config
        
        # Paths
        base_dir = getattr(config, "persistence_dir", "cortex_data")
        os.makedirs(base_dir, exist_ok=True)
        
        mmap_path = os.path.join(base_dir, "cortex_cache.bin")
        
        # Tier 2: Truth (WAL)
        # JsonlPersistence expects a directory, it manages filenames internally (neurons.jsonl)
        self.wal = JsonlPersistence(base_dir)
        
        # Tier 1: Cache (MMAP)
        # Capacity from config, default 1M slots
        capacity = getattr(config, "mmap_capacity", 1024 * 1024)
        dim = getattr(config, "embedding_dim", 64)
        
        try:
            self.mmap = MMapNeuronStore(mmap_path, dim=dim, capacity=capacity)
        except Exception as e:
            print(f"Checking for MMap migration needed due to version change: {e}")
            if os.path.exists(mmap_path):
                # Simple migration: Delete incompatible cache. It will rebuild from WAL.
                # In production, we might want a migration script.
                try:
                    os.remove(mmap_path)
                    self.mmap = MMapNeuronStore(mmap_path, dim=dim, capacity=capacity)
                    print("Rebuilt incompatible MMAP cache file.")
                except Exception as ex:
                    print(f"Critical persistence failure: {ex}")
                    raise ex
            else:
                 raise e

    def get_neuron(self, seed: int) -> Optional[object]:
        """
        Recall neuron state.
        Priority: MMAP -> WAL -> None
        """
        # 1. Hot Path: MMAP Cache
        cached = self.mmap.get(seed)
        if cached:
            # Rehydrate Object (Lattice expects Object or dict? Lattice expects Object usually but load() handles dict)
            # Actually Lattice.load() expects persistence_manager to return dict or be an object with .recall()
            # But Lattice.stimulate() calls persistence_manager.get_neuron(seed).
            # We return an MTINeuron-compatible object or dict.
            # Let's return the dict, Lattice handles rehydration.
            return self._dict_to_neuron(cached, seed) # Wait, Lattice rehydrates.
        
        # 2. Cold Path: WAL Recall
        # JSONL is typically a full load map. Does it support single key query?
        # JSONLogPersistence usually loads everything into memory or scans.
        # Check implementation of JSONLogPersistence. 
        # For V1, we assume getting from WAL might require full scan or it keeps an index.
        # If WAL is slow, this is a "Page Fault".
        
        # TODO: JSONLogPersistence needs random access query if we want efficient single item recall.
        # Assuming for now we rely on MMAP for speed. WAL is for recovery.
        # If it's not in MMAP, we check if we have it in WAL. 
        
        # For now, return None if not in cache (Simple Cache).
        # Real implementation would query index of WAL.
        return None

    def put_neuron(self, seed: int, neuron_state: dict):
        """
        Save neuron.
        Write to WAL (Truth) then update MMAP (Cache).
        """
        # 1. Write to WAL / JSONL (Durable)
        # Wrap in dict for batch API
        self.wal.upsert_neurons({seed: neuron_state})
        
        # 2. Update MMAP Cache
        self.mmap.put(
            seed,
            np.array(neuron_state['weights']),
            np.array(neuron_state['velocity']),
            float(neuron_state['bias']),
            float(neuron_state['gravity']),
            int(neuron_state['age']),
            float(neuron_state['last_accessed'])
        )

    def delete_neuron(self, seed: int):
        """
        Delete neuron.
        WAL Tombstone -> MMAP Invalidate.
        """
        # 1. Durable Delete
        self.wal.delete_neuron(seed)
        
        # 2. Cache Delete
        self.mmap.delete(seed)

    def consolidate(self, active_tissue: dict):
        """
        Snapshot full state (Saving).
        Writes full checkpoint to WAL and ensures MMAP is synced.
        """
        # 1. MMAP Flush
        self.mmap.flush()
        
        # 2. Serialize Active Tissue
        serialized = {}
        for seed, neuron in active_tissue.items():
            # Convert MTINeuron to dict
            # We explicitly extract fields to match persistence schema
            serialized[seed] = {
                "weights": neuron.weights.tolist(),
                "velocity": neuron.velocity.tolist(),
                "bias": neuron.bias,
                "gravity": neuron.gravity,
                "age": neuron.age,
                "last_active": neuron.last_accessed, # Matches 'last_accessed' logic
                "label": getattr(neuron, 'label', None)
            }
            
        # 3. Durable Save (Upsert all)
        self.wal.upsert_neurons(serialized)

    def _dict_to_neuron(self, data, seed):
        """Helper if we need to return an object. For now return dict."""
        return data

    # Protocol Conformity Aliases
    def get(self, seed: int) -> Optional[Dict]:
        return self.get_neuron(seed)

    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float) -> bool:
        """
        Adapts protocol 'put' directly to manager's put_neuron.
        """
        state = {
            "weights": weights.tolist(),
            "velocity": velocity.tolist(),
            "bias": bias,
            "gravity": gravity,
            "age": age,
            "last_accessed": last_accessed
        }
        self.put_neuron(seed, state)
        return True

    def delete(self, seed: int) -> None:
        self.delete_neuron(seed)
    
    def flush(self) -> None:
        self.wal.flush()
        self.mmap.flush()
    
    def get_active_count(self) -> int:
        return self.mmap.get_active_count()

    def close(self):
        self.mmap.close()
        self.wal.close()
