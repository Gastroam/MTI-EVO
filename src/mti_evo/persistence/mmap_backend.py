"""
Memory-Mapped Neuron Store
===========================
Zero-copy persistence for MTI-EVO Holographic Lattice.

Ontology: The seed IS the address. No hash table indirection.
Physics: Persistence is native to the substrate, not serialization.
"""
import mmap
import os
import struct
import numpy as np
from typing import Optional, Tuple


class MMapNeuronStore:
    """
    Memory-mapped neuron storage with direct seed indexing.
    
    Layout per neuron (544 bytes @ dim=64):
    - weights: float32[64] = 256 bytes
    - velocity: float32[64] = 256 bytes
    - bias: float32 = 4 bytes
    - gravity: float32 = 4 bytes
    - age: uint32 = 4 bytes
    - last_accessed: float64 = 8 bytes
    - flags: uint32 = 4 bytes (bit 0 = active)
    - padding: 8 bytes (alignment)
    
    Total: 544 bytes per neuron
    """
    
    HEADER_SIZE = 16  # magic(4) + version(4) + dim(4) + count(4)
    MAGIC = 0x4D544945  # 'MTIE'
    VERSION = 1
    
    def __init__(self, path: str, dim: int = 64, capacity: int = 2**20):
        self.path = path
        self.dim = dim
        self.capacity = capacity
        self.record_size = self._calc_record_size(dim)
        
        # Calculate file size (aligned to 4KB pages for Windows)
        raw_size = self.HEADER_SIZE + (capacity * self.record_size)
        self.file_size = ((raw_size + 4095) // 4096) * 4096
        
        self._init_file()
        self._open_mmap()
    
    def _calc_record_size(self, dim: int) -> int:
        """Calculate record size for given dimension."""
        # weights + velocity + bias + gravity + age + last_accessed + flags + padding
        return (dim * 4) + (dim * 4) + 4 + 4 + 4 + 8 + 4 + 8
    
    def _init_file(self):
        """Create or verify the storage file."""
        if os.path.exists(self.path):
            # Verify existing file
            with open(self.path, 'rb') as f:
                header = f.read(self.HEADER_SIZE)
                if len(header) >= 12:
                    magic, version, dim = struct.unpack('<III', header[:12])
                    if magic != self.MAGIC:
                        raise ValueError(f"Invalid cortex file (bad magic)")
                    if dim != self.dim:
                        raise ValueError(f"Dimension mismatch: file={dim}, requested={self.dim}")
        else:
            # Create new file with header
            with open(self.path, 'wb') as f:
                header = struct.pack('<IIII', self.MAGIC, self.VERSION, self.dim, 0)
                f.write(header)
                f.truncate(self.file_size)
    
    def _open_mmap(self):
        """Open memory-mapped file."""
        self.fd = open(self.path, 'r+b')
        self.mm = mmap.mmap(self.fd.fileno(), self.file_size, access=mmap.ACCESS_WRITE)
    
    def _offset(self, seed: int) -> int:
        """Calculate byte offset for seed. The seed IS the address."""
        # Mask to capacity to prevent overflow
        idx = seed % self.capacity
        return self.HEADER_SIZE + (idx * self.record_size)
    
    def is_active(self, seed: int) -> bool:
        """Check if neuron exists at seed."""
        offset = self._offset(seed)
        flags_offset = offset + (self.dim * 8) + 20  # After weights, velocity, bias, gravity, age, last_accessed
        flags = struct.unpack('<I', self.mm[flags_offset:flags_offset+4])[0]
        return bool(flags & 1)
    
    def get(self, seed: int) -> Optional[dict]:
        """
        Get neuron state. Zero-copy read.
        Returns None if neuron not active.
        """
        if not self.is_active(seed):
            return None
        
        offset = self._offset(seed)
        
        # Direct memory view (zero-copy)
        weights = np.frombuffer(self.mm, dtype='float32', count=self.dim, offset=offset)
        velocity = np.frombuffer(self.mm, dtype='float32', count=self.dim, offset=offset + self.dim*4)
        
        meta_offset = offset + self.dim * 8
        bias, gravity, age = struct.unpack('<ffI', self.mm[meta_offset:meta_offset+12])
        last_accessed = struct.unpack('<d', self.mm[meta_offset+12:meta_offset+20])[0]
        
        return {
            'weights': weights.copy(),  # Copy to allow mutation
            'velocity': velocity.copy(),
            'bias': bias,
            'gravity': gravity,
            'age': age,
            'last_accessed': last_accessed
        }
    
    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float):
        """
        Store neuron state. Direct write to mapped memory.
        """
        offset = self._offset(seed)
        
        # Write weights and velocity
        self.mm[offset:offset+self.dim*4] = weights.astype('float32').tobytes()
        self.mm[offset+self.dim*4:offset+self.dim*8] = velocity.astype('float32').tobytes()
        
        # Write metadata
        meta_offset = offset + self.dim * 8
        self.mm[meta_offset:meta_offset+12] = struct.pack('<ffI', bias, gravity, age)
        self.mm[meta_offset+12:meta_offset+20] = struct.pack('<d', last_accessed)
        
        # Set active flag
        flags_offset = meta_offset + 20
        self.mm[flags_offset:flags_offset+4] = struct.pack('<I', 1)
    
    def delete(self, seed: int):
        """Mark neuron as inactive."""
        offset = self._offset(seed)
        flags_offset = offset + (self.dim * 8) + 20
        self.mm[flags_offset:flags_offset+4] = struct.pack('<I', 0)
    
    def flush(self):
        """Sync memory to disk. ~1ms on modern systems."""
        self.mm.flush()
    
    def get_active_count(self) -> int:
        """Read active neuron count from header."""
        count = struct.unpack('<I', self.mm[12:16])[0]
        return count
    
    def set_active_count(self, count: int):
        """Update active neuron count in header."""
        self.mm[12:16] = struct.pack('<I', count)
    
    def close(self):
        """Clean shutdown."""
        self.flush()
        self.mm.close()
        self.fd.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class JSONNeuronStore:
    """
    Fallback JSON storage for compatibility.
    Same interface as MMapNeuronStore.
    """
    
    def __init__(self, path: str, dim: int = 64, capacity: int = 10000):
        import json
        self.path = path
        self.dim = dim
        self.data = {}
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
    
    def is_active(self, seed: int) -> bool:
        return str(seed) in self.data
    
    def get(self, seed: int) -> Optional[dict]:
        key = str(seed)
        if key not in self.data:
            return None
        v = self.data[key]
        return {
            'weights': np.array(v['weights']),
            'velocity': np.array(v.get('velocity', np.zeros(self.dim))),
            'bias': v.get('bias', 0.0),
            'gravity': v.get('gravity', 20.0),
            'age': v.get('age', 0),
            'last_accessed': v.get('last_active', 0.0)
        }
    
    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float):
        self.data[str(seed)] = {
            'weights': weights.tolist(),
            'velocity': velocity.tolist(),
            'bias': bias,
            'gravity': gravity,
            'age': age,
            'last_active': last_accessed
        }
    
    def delete(self, seed: int):
        key = str(seed)
        if key in self.data:
            del self.data[key]
    
    def flush(self):
        import json
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_active_count(self) -> int:
        return len(self.data)
    
    def close(self):
        self.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def create_store(path: str, backend: str = "auto", dim: int = 64, capacity: int = 2**20):
    """
    Factory function to create appropriate storage backend.
    
    backend: "mmap", "json", or "auto" (detect from file extension)
    """
    if backend == "auto":
        if path.endswith('.mmap') or path.endswith('.bin'):
            backend = "mmap"
        elif path.endswith('.json'):
            backend = "json"
        else:
            # Default to mmap for new files
            backend = "mmap"
            path = path + ".mmap"
    
    if backend == "mmap":
        return MMapNeuronStore(path, dim=dim, capacity=capacity)
    else:
        return JSONNeuronStore(path, dim=dim, capacity=capacity)
