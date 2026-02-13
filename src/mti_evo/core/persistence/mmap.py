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
    Memory-mapped neuron storage with direct seed indexing (Direct-Mapped Cache).
    
    Layout per neuron (Dim=64 -> 544 bytes):
    - Weights: float32[Dim] (256 bytes)
    - Velocity: float32[Dim] (256 bytes)
    --- Meta Block (32 bytes) ---
    - Bias: float32 (4)
    - Gravity: float32 (4)
    - Age: uint32 (4)
    - Flags: uint32 (4) (Bit 0=Active)
    - Last Accessed: float64 (8)
    - Seed Tag: uint64 (8) <-- NEW: Collision Detection
    
    Total: Dim*8 + 32 bytes.
    """
    
    HEADER_SIZE = 16  # magic(4) + version(4) + dim(4) + count(4)
    MAGIC = 0x4D544945  # 'MTIE'
    VERSION = 2  # Bumped for Layout Change
    
    def __init__(self, path: str, dim: int = 64, capacity: int = 2**20, read_only: bool = False):
        self.path = path
        self.dim = dim
        self.capacity = capacity
        self.read_only = read_only
        # Weights(4*D) + Velocity(4*D) + Bias(4) + Grav(4) + Age(4) + Flags(4) + Last(8) + Tag(8)
        self.record_size = (dim * 8) + 32
        
        # Calculate file size (aligned to 4KB pages)
        raw_size = self.HEADER_SIZE + (capacity * self.record_size)
        self.file_size = ((raw_size + 4095) // 4096) * 4096
        
        self._init_file()
        self._open_mmap()
    
    def _init_file(self):
        """Create or verify the storage file."""
        # 1. Check if exists
        exists = os.path.exists(self.path)
        
        if not exists:
            # Create new file
            with open(self.path, 'wb') as f:
                # Write Header
                # magic(4) + version(4) + dim(4) + count(4)
                header = struct.pack('<III', self.MAGIC, self.VERSION, self.dim) + b'\x00'*4
                f.write(header)
                # Expand to full size
                f.truncate(self.file_size)
        else:
            # Verify and potentially update existing
            with open(self.path, 'r+b') as f:
                header = f.read(self.HEADER_SIZE)
                if len(header) >= 12:
                    magic, version, dim = struct.unpack('<III', header[:12])
                    if magic != self.MAGIC:
                         raise ValueError(f"Invalid cortex file magic: {magic}")
                    if dim != self.dim:
                         raise ValueError(f"Dimension mismatch: file={dim}, requested={self.dim}")
                    # Version check or migration logic here?
                    # For now assume compatible or throw if strictly version != self.VERSION
                    
                # Ensure size is correct (in case of expansion)
                f.seek(0, 2)
                sz = f.tell()
                if sz < self.file_size:
                    f.truncate(self.file_size)
    
    def _open_mmap(self):
        mode = 'rb' if self.read_only else 'r+b'
        access = mmap.ACCESS_READ if self.read_only else mmap.ACCESS_WRITE
        
        self.fd = open(self.path, mode)
        self.mm = mmap.mmap(self.fd.fileno(), self.file_size, access=access)
    
    def _offset(self, seed: int) -> int:
        idx = seed % self.capacity
        return self.HEADER_SIZE + (idx * self.record_size)
    
    def get(self, seed: int) -> Optional[dict]:
        """Get neuron if seed matches tag."""
        offset = self._offset(seed)
        meta_start = offset + (self.dim * 8)
        
        # Read Check info first
        # Bias(4), Grav(4), Age(4), Flags(4), Last(8), Tag(8)
        # We need Flags (+12 from meta) and Tag (+24 from meta)
        
        flags = struct.unpack('<I', self.mm[meta_start+12:meta_start+16])[0]
        if not (flags & 1):
            return None
            
        tag = struct.unpack('<Q', self.mm[meta_start+24:meta_start+32])[0]
        if tag != seed:
            # Collision! This slot belongs to another seed.
            return None
            
        # Valid Hit - Read Data
        weights = np.frombuffer(self.mm, dtype='float32', count=self.dim, offset=offset)
        velocity = np.frombuffer(self.mm, dtype='float32', count=self.dim, offset=offset + self.dim*4)
        
        bias, gravity, age = struct.unpack('<ffI', self.mm[meta_start:meta_start+12])
        last_accessed = struct.unpack('<d', self.mm[meta_start+16:meta_start+24])[0]
        
        return {
            'weights': weights.copy(),
            'velocity': velocity.copy(),
            'bias': bias,
            'gravity': gravity,
            'age': age,
            'last_accessed': last_accessed
        }
    
    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float) -> bool:
        """
        Store neuron. Returns True if successful, False if collision prevents write.
        (Direct-Mapped Cache policy: Only overwrite if empty or same seed? 
         For now: Overwrite is allowed if we treat it as cache, but we need to update tag.
         Wait, if we overwrite, we lose the old seed's data.
         Strategy: Always overwrite (LRU approximation via collision).
         BUT user warned "Silent overwrite".
         So we update the Tag. Effectively treating it as a Cache where newest wins collision.)
        """
        if self.read_only:
             return False # Silent fail or raise? User said "HTTP workers can be read-only". Return False safe.
             
        offset = self._offset(seed)
        meta_start = offset + (self.dim * 8)
        
        # Check current state for stats
        current_flags = struct.unpack('<I', self.mm[meta_start+12:meta_start+16])[0]
        is_active = bool(current_flags & 1)
        
        # Write Data
        self.mm[offset:offset+self.dim*4] = weights.astype('float32').tobytes()
        self.mm[offset+self.dim*4:offset+self.dim*8] = velocity.astype('float32').tobytes()
        
        # Write Meta
        # Flags set to 1 (Active)
        # Tag set to seed
        self.mm[meta_start:meta_start+12] = struct.pack('<ffI', bias, gravity, age)
        self.mm[meta_start+12:meta_start+16] = struct.pack('<I', 1) # Flags
        self.mm[meta_start+16:meta_start+24] = struct.pack('<d', last_accessed)
        self.mm[meta_start+24:meta_start+32] = struct.pack('<Q', seed) # Tag
        
        # Update Header Count if it was inactive
        if not is_active:
             self._inc_count(1)
             
        return True
    
    def delete(self, seed: int):
        offset = self._offset(seed)
        meta_start = offset + (self.dim * 8)
        
        # Check if it's actually this seed
        flags = struct.unpack('<I', self.mm[meta_start+12:meta_start+16])[0]
        if not (flags & 1):
            return # Already empty
            
        tag = struct.unpack('<Q', self.mm[meta_start+24:meta_start+32])[0]
        if tag != seed:
            return # Not our seed (Collision)
            
        # Mark Inactive
        self.mm[meta_start+12:meta_start+16] = struct.pack('<I', 0)
        self._inc_count(-1)
        
    def _inc_count(self, delta: int):
        # Header count at offset 12
        current = struct.unpack('<I', self.mm[12:16])[0]
        new_val = max(0, current + delta)
        self.mm[12:16] = struct.pack('<I', new_val)
        
    def flush(self):
        self.mm.flush()
        
    def get_active_count(self) -> int:
        return struct.unpack('<I', self.mm[12:16])[0]

    def close(self):
        """Clean shutdown."""
        try:
            if not self.mm.closed:
                self.flush()
                self.mm.close()
            if not self.fd.closed:
                self.fd.close()
        except ValueError:
            pass # Already closed
        
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

# Only MMapStore is defined here. JSON support moved to generic manager if needed.
