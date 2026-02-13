"""
MTI-EVO Cortex Memory Manager
=============================
Handles Long-Term Potentiation (Persistence) for the Holographic Lattice.
Formerly mti_hippocampus.py.
"""
import os
import time
import numpy as np

import threading
from mti_evo.core.neuron import MTINeuron
from mti_evo.core.config import MTIConfig
from mti_evo.core.persistence.mmap import MMapNeuronStore
from mti_evo.core.persistence.jsonl import JsonlPersistence as JSONNeuronStore 

# Helper to create store (Compatibility)
def create_store(path, backend, dim, capacity, read_only=False):
    if backend == "mmap":
        return MMapNeuronStore(path, dim, capacity, read_only=read_only)
    return JSONNeuronStore(os.path.dirname(path)) # JSON doesn't enforce read-only yet


class CortexMemory:
    """
    Persistence layer for MTI-EVO Holographic Lattice.
    
    Backends:
    - "mmap": Memory-mapped storage (72x faster, recommended)
    - "json": JSON storage (readable, debuggable)
    - "auto": Auto-detect from file extension or default to mmap
    """
    
    def __init__(self, base_path=None, persistence_path=None, backend="auto", dim=64, capacity=2**20, read_only=False):
        """
        Initialize the Memory Manager.
        
        Args:
            base_path: Directory for storage files (default: .mti-brain/)
            persistence_path: Direct path to storage file
            backend: "mmap", "json", or "auto"
            dim: Neuron weight dimension (default: 64)
            capacity: Max neurons for mmap backend (default: 2^20)
            read_only: If True, blocks write operations (default: False)
        """
        # [Opt-5] Concurrency Control
        self.lock = threading.RLock()
        # Determine paths
        if persistence_path:
            self.storage_path = persistence_path
            self.base_path = os.path.dirname(persistence_path)
        else:
            if base_path is None:
                base_path = os.path.join(os.getcwd(), ".mti-brain")
            self.base_path = base_path
            
            # Choose file extension based on backend
            if backend == "mmap":
                self.storage_path = os.path.join(base_path, "cortex.mmap")
            elif backend == "json":
                self.storage_path = os.path.join(base_path, "cortex_dump.json")
            else:
                # Auto: prefer mmap if exists, else json if exists, else mmap for new
                mmap_path = os.path.join(base_path, "cortex.mmap")
                json_path = os.path.join(base_path, "cortex_dump.json")
                if os.path.exists(mmap_path):
                    self.storage_path = mmap_path
                elif os.path.exists(json_path):
                    self.storage_path = json_path
                else:
                    self.storage_path = mmap_path  # Default to mmap for new
        
        # Create directory if needed
        if not os.path.exists(self.base_path):
            try:
                os.makedirs(self.base_path)
                print(f"🧠 MEMORY: New brain tissue created at {self.base_path}")
            except OSError as e:
                print(f"❌ CRITICAL ERROR: Could not create directory. {e}")
        
        self.dim = dim
        self.capacity = capacity
        self.backend = backend
        self.read_only = read_only
        self.store = None  # Lazy initialization
        
    def _get_store(self):
        """Lazy initialization of storage backend."""
        if self.store is None:
            self.store = create_store(
                self.storage_path,
                backend=self.backend,
                dim=self.dim,
                capacity=self.capacity,
                read_only=self.read_only
            )
        return self.store
    
    def consolidate(self, active_tissue):
        """
        Save the Holographic Lattice state.
        
        Args:
            active_tissue: Dict {seed_id: MTINeuron_Object}
        """
        if self.read_only:
            print("💤 MEMORY: Read-only mode, skipping consolidation.")
            return

        with self.lock:
            print("\n💤 INITIATING REM PHASE (Consolidation)...")
            store = self._get_store()
            count = 0
            
            # [Opt-4] Batch Consolidation
            batch_data = {}
            
            for seed_id, neuron in active_tissue.items():
                # Filter: Only save mature neurons (age > 0 or significant weight)
                if hasattr(neuron, 'weights'):
                    if neuron.age > 0 or abs(np.mean(neuron.weights)) > 0.01:
                        # Prepare state dict (ensure JSON serializable for compatibility)
                        batch_data[int(seed_id)] = {
                            "weights": neuron.weights.tolist(),
                            "velocity": neuron.velocity.tolist(),
                            "bias": float(neuron.bias),
                            "gravity": float(neuron.gravity),
                            "age": int(neuron.age),
                            "last_accessed": time.time()
                        }

            if batch_data:
                # Polymorphic batch update (MMap loops in memory, JSONL appends in one transaction)
                if hasattr(store, 'upsert_neurons'):
                    store.upsert_neurons(batch_data)
                else:
                    # Fallback for stores without batch support (should not happen with updated backend)
                    for s, d in batch_data.items():
                        store.put(s, np.array(d['weights']), np.array(d['velocity']), 
                                  d['bias'], d['gravity'], d['age'], d['last_accessed'])
                
                count = len(batch_data)
            
            store.flush()
            backend_name = "mmap" if isinstance(store, MMapNeuronStore) else "json"
            print(f"✅ MEMORY SAVED: {count} neurons preserved ({backend_name})")
    
    def recall(self):
        """
        Reconstruct neuronal tissue from storage.
        
        Returns:
            Dict {seed_id: MTINeuron_Object}
        """
        # Read-only operation, but if store not init, we need lock to init safely?
        # _get_store is lazy. 
        # Making _get_store thread-safe or locking here.
        with self.lock:
            store = self._get_store()
            
        restored_tissue = {}
        
        # For mmap, we need to scan for active neurons
        if isinstance(store, MMapNeuronStore):
            # Scan approach: iterate through stored neurons
            # This is a limitation - mmap doesn't naturally enumerate
            # For now, return empty and let callers handle on-demand loading
            print("🧠 MMAP: Direct substrate access enabled (on-demand loading)")
            return restored_tissue
        
        # JSON: reconstruct all neurons
        if isinstance(store, JSONNeuronStore):
            # JsonlPersistence uses an index based lookup
            if hasattr(store, 'index'):
                for seed_str in store.index.keys():
                    seed = int(seed_str)
                    data = store.get(seed)
                    if data:
                        config = MTIConfig(gravity=data['gravity'])
                        neuron = MTINeuron(input_size=len(data['weights']), config=config)
                        neuron.weights = np.array(data['weights'])
                        neuron.bias = data['bias']
                        neuron.velocity = np.array(data['velocity'])
                        neuron.age = data['age']
                        restored_tissue[seed] = neuron
            
            print(f"⚡ RECALL SUCCESS: {len(restored_tissue)} neurons revived")
        
        return restored_tissue
    
    def get_neuron(self, seed):
        """
        Direct neuron retrieval (for mmap on-demand loading).
        """
        # Lock-free read path for performance (User accepted constraints)
        # Assuming store is already init or _get_store handles it?
        # _get_store is NOT thread safe if called first time in parallel.
        # But usually main thread inits lattice.
        
        if self.store is None:
            with self.lock:
                store = self._get_store()
        else:
            store = self.store
            
        data = store.get(seed)
        
        if data is None:
            return None
        
        config = MTIConfig(gravity=data['gravity'])
        neuron = MTINeuron(input_size=len(data['weights']), config=config)
        neuron.weights = np.array(data['weights'])
        neuron.bias = data['bias']
        neuron.velocity = np.array(data['velocity'])
        neuron.age = data['age']
        # stored last_accessed might be in data
        if 'last_accessed' in data:
            neuron.last_accessed = data['last_accessed']
        
        return neuron
    
    def put_neuron(self, seed, neuron):
        """
        Direct neuron storage (for mmap real-time persistence).
        """
        if self.read_only:
             return
             
        with self.lock:
            store = self._get_store()
            # Unified Backend API
            store.put(
                seed=int(seed),
                weights=neuron.weights,
                velocity=neuron.velocity,
                bias=float(neuron.bias),
                gravity=float(neuron.gravity),
                age=int(neuron.age),
                last_accessed=time.time()
            )
    
    def flush(self):
        """Sync to disk."""
        with self.lock:
            if self.store:
                self.store.flush()
    
    def close(self):
        """Clean shutdown."""
        with self.lock:
            if self.store:
                self.store.close()
                self.store = None
