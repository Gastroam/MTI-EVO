"""
MTI-EVO Hippocampus Module
==========================
Handles Long-Term Potentiation (Persistence) for the Holographic Cortex.

v2.0: Supports memory-mapped (mmap) and JSON backends with auto-detection.
"""
import os
import time
import numpy as np

try:
    from mti_evo.mti_core import MTINeuron
    from mti_evo.mti_config import MTIConfig
    from mti_evo.persistence import MMapNeuronStore, JSONNeuronStore, create_store
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from mti_evo.mti_core import MTINeuron
    from mti_evo.mti_config import MTIConfig
    from mti_evo.persistence import MMapNeuronStore, JSONNeuronStore, create_store


class MTIHippocampus:
    """
    Persistence layer for MTI-EVO Holographic Lattice.
    
    Backends:
    - "mmap": Memory-mapped storage (72x faster, recommended)
    - "json": JSON storage (readable, debuggable)
    - "auto": Auto-detect from file extension or default to mmap
    """
    
    def __init__(self, base_path=None, persistence_path=None, backend="auto", dim=64, capacity=2**20):
        """
        Initialize the Hippocampus.
        
        Args:
            base_path: Directory for storage files (default: .mti-brain/)
            persistence_path: Direct path to storage file
            backend: "mmap", "json", or "auto"
            dim: Neuron weight dimension (default: 64)
            capacity: Max neurons for mmap backend (default: 2^20)
        """
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
                print(f"🧠 HIPPOCAMPUS: New brain tissue created at {self.base_path}")
            except OSError as e:
                print(f"❌ CRITICAL ERROR: Could not create directory. {e}")
        
        self.dim = dim
        self.capacity = capacity
        self.backend = backend
        self.store = None  # Lazy initialization
        
    def _get_store(self):
        """Lazy initialization of storage backend."""
        if self.store is None:
            self.store = create_store(
                self.storage_path,
                backend=self.backend,
                dim=self.dim,
                capacity=self.capacity
            )
        return self.store
    
    def consolidate(self, active_tissue):
        """
        Save the Holographic Lattice state.
        
        Args:
            active_tissue: Dict {seed_id: MTINeuron_Object}
        """
        print("\n💤 INITIATING REM PHASE (Consolidation)...")
        store = self._get_store()
        count = 0
        
        for seed_id, neuron in active_tissue.items():
            # Filter: Only save mature neurons (age > 0 or significant weight)
            if hasattr(neuron, 'weights'):
                if neuron.age > 0 or abs(np.mean(neuron.weights)) > 0.01:
                    store.put(
                        seed=int(seed_id),
                        weights=neuron.weights,
                        velocity=neuron.velocity,
                        bias=float(neuron.bias),
                        gravity=float(neuron.gravity),
                        age=int(neuron.age),
                        last_accessed=time.time()
                    )
                    count += 1
        
        store.flush()
        backend_name = "mmap" if isinstance(store, MMapNeuronStore) else "json"
        print(f"✅ MEMORY SAVED: {count} neurons preserved ({backend_name})")
    
    def recall(self):
        """
        Reconstruct neuronal tissue from storage.
        
        Returns:
            Dict {seed_id: MTINeuron_Object}
        """
        store = self._get_store()
        restored_tissue = {}
        
        # For mmap, we need to scan for active neurons
        # For json, it's loaded into memory already
        if isinstance(store, MMapNeuronStore):
            # Scan approach: iterate through stored neurons
            # This is a limitation - mmap doesn't naturally enumerate
            # For now, return empty and let callers handle on-demand loading
            print("🧠 MMAP: Direct substrate access enabled (on-demand loading)")
            return restored_tissue
        
        # JSON: reconstruct all neurons
        if isinstance(store, JSONNeuronStore):
            for seed_str in store.data.keys():
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
        
        Args:
            seed: Neuron seed
            
        Returns:
            MTINeuron or None
        """
        store = self._get_store()
        data = store.get(seed)
        
        if data is None:
            return None
        
        config = MTIConfig(gravity=data['gravity'])
        neuron = MTINeuron(input_size=len(data['weights']), config=config)
        neuron.weights = np.array(data['weights'])
        neuron.bias = data['bias']
        neuron.velocity = np.array(data['velocity'])
        neuron.age = data['age']
        neuron.last_accessed = data['last_accessed']
        
        return neuron
    
    def put_neuron(self, seed, neuron):
        """
        Direct neuron storage (for mmap real-time persistence).
        
        Args:
            seed: Neuron seed
            neuron: MTINeuron object
        """
        store = self._get_store()
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
        """Sync to disk (fast for mmap, slow for json)."""
        if self.store:
            self.store.flush()
    
    def close(self):
        """Clean shutdown."""
        if self.store:
            self.store.close()
            self.store = None


if __name__ == "__main__":
    # Quick test
    hippo = MTIHippocampus(backend="mmap")
    print(f"Storage path: {hippo.storage_path}")
    print(f"Backend: {hippo.backend}")
