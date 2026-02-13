import torch
import torch.nn as nn
import json
import os
import gc
from typing import List, Dict, Optional, Set
from safetensors.torch import load_file

class QuantumLoader:
    """
    Manages the 'Digital Lobotomy' and 'Quantum Loading' of Gemma-27B.
    
    Philosophy:
    - We do not load the model. We load the *potential* of the model.
    - Vision layers are 'lobotomized' (filtered out).
    - Specific layers can be 'pruned' (skipped).
    - Layers are loaded into VRAM only when observed (Resident).
    """
    def __init__(self, model_path: str, pruned_layer_indices: Optional[List[int]] = None):
        self.model_path = model_path
        self.index_path = os.path.join(model_path, "model.safetensors.index.json")
        self.config_path = os.path.join(model_path, "config.json")
        
        # Load Config (for Architecture params)
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load Index (for Shard Mapping)
        with open(self.index_path, 'r') as f:
            self.index = json.load(f)
        
        self.weight_map = self.index["weight_map"]
        self.pruned_indices = set(pruned_layer_indices) if pruned_layer_indices else set()
        
        # Pre-compute layer-to-shard mapping for Text Decoder only
        self.layer_shards = {} # layer_idx -> shard_filename
        self._map_layers()
        
        # TIER 2 MEMORY (RAM Cache)
        self.cpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _map_layers(self):
        """
        Scans the weight map to find where each Text Decoder layer lives.
        Applies the 'Vision Filter' implicitly by only looking for 'model.layers.X'.
        """
        print(f"[QuantumLoader] Mapping layers from {self.model_path}...")
        
        found_layers = set()
        
        for key, shard in self.weight_map.items():
            # FILTER: Ignore Vision Tower
            if "vision_tower" in key:
                continue
                
            # FILTER: Match Text Decoder Layers
            # Key format: language_model.model.layers.{i}.* or model.layers.{i}.*
            parts = key.split('.')
            if "layers" in parts:
                try:
                    idx_loc = parts.index("layers") + 1
                    layer_idx = int(parts[idx_loc])
                    
                    found_layers.add(layer_idx)
                    
                    # Store the shard. Note: A layer might span shards (rare in safetensors, but possible).
                    # Usually keys for one layer are in one shard or split. 
                    # We assume 1 shard for simplicity or store all needed shards.
                    if layer_idx not in self.layer_shards:
                        self.layer_shards[layer_idx] = set()
                    self.layer_shards[layer_idx].add(shard)
                    
                except (ValueError, IndexError):
                    continue
        
        print(f"[QuantumLoader] Found {len(found_layers)} Text Layers.")
        print(f"[QuantumLoader] Vision Tower Ignored.")
        
        # Apply strict pruning
        if self.pruned_indices:
            print(f"[QuantumLoader] Pruning Indices: {self.pruned_indices}")

    def _read_from_disk(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Helper to read layer from SSD."""
        if layer_idx not in self.layer_shards:
            print(f"[QuantumLoader] Layer {layer_idx} not found in index.")
            return None
            
        shards = self.layer_shards[layer_idx]
        layer_weights = {}
        
        # print(f"[QuantumLoader] Reading Layer {layer_idx} from SSD (Shards: {len(shards)})...")
        
        for shard_file in shards:
            full_path = os.path.join(self.model_path, shard_file)
            
            # Load the full shard (this is the expensive part)
            state_dict = load_file(full_path, device="cpu") 
            
            # Extract ONLY this layer's keys
            prefix_pattern = f"layers.{layer_idx}."
            
            for key, tensor in state_dict.items():
                if prefix_pattern in key:
                    # Clean key for local consumption
                    local_key = key.split(prefix_pattern)[-1]
                    # QUANTIZATION / OTF: Move to bfloat16 (User requested 16)
                    layer_weights[local_key] = tensor.to(torch.bfloat16)
            
            del state_dict
            gc.collect()
            
        return layer_weights

    def prefetch_layer(self, layer_idx: int):
        """Loads layer from SSD to CPU Cache (Tier 2)."""
        if layer_idx in self.pruned_indices:
            return

        if layer_idx not in self.cpu_cache:
            print(f"[QuantumLoader] Prefetching Layer {layer_idx} to Tier 2 (RAM)...")
            weights = self._read_from_disk(layer_idx)
            if weights:
                self.cpu_cache[layer_idx] = weights

    def evict_layer(self, layer_idx: int):
        """Removes layer from CPU Cache."""
        if layer_idx in self.cpu_cache:
            # print(f"[QuantumLoader] Evicting Layer {layer_idx} from Tier 2.")
            del self.cpu_cache[layer_idx]
            gc.collect()

    def load_layer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        The Act of Observation. 
        Loads the specific layer into VRAM (bfloat16).
        Checks Tier 2 (RAM) first, then Tier 3 (SSD).
        """
        if layer_idx in self.pruned_indices:
            print(f"[QuantumLoader] Layer {layer_idx} is Pruned (Access Denied).")
            return None
            
        # TIER 2 CHECK
        if layer_idx in self.cpu_cache:
            # print(f"[QuantumLoader] Tier 2 (RAM) HIT for Layer {layer_idx}")
            return self.cpu_cache[layer_idx]
            
        # TIER 3 FETCH
        print(f"[QuantumLoader] Tier 3 (SSD) FETCH for Layer {layer_idx}")
        return self._read_from_disk(layer_idx)

class QuantumLayer(nn.Module):
    """
    A PyTorch module that represents a 'Superposition' of a layer.
    It has no weights until 'forward' is called (Simulation).
    """
    def __init__(self, layer_idx: int, loader: QuantumLoader):
        super().__init__()
        self.layer_idx = layer_idx
        self.loader = loader
        self.weights: Optional[Dict[str, torch.Tensor]] = None
        self.is_collapsed = False
        
    def forward(self, x):
        # 1. Collapse (Load) - Only once, reuse GPU tensors
        if not self.is_collapsed:
            # Load from Tier 2 (RAM) or Tier 3 (SSD)
            loaded_weights = self.loader.load_layer(self.layer_idx)
            
            if loaded_weights is None:
                return x # Pruned or Error, Identity pass
            
            self.is_collapsed = True
            
            # Move to Device ONCE (not every forward!)
            # CRITICAL FIX: Do NOT modify loaded_weights in-place if it's from cache!
            self.weights = {}
            if torch.cuda.is_available():
                for k, v in loaded_weights.items():
                    # Move to GPU for computation, keep original on CPU (if cached)
                    self.weights[k] = v.cuda()
            else:
                # For CPU, we must clone to avoid mutating cache
                self.weights = {k: v.clone() for k, v in loaded_weights.items()}
        
        # Weights are already on device from collapse, no need to transfer again
        return x

    def reset_reality(self):
        """Un-observes the layer, freeing VRAM."""
        if self.is_collapsed:
            # print(f"[QuantumLayer {self.layer_idx}] Resetting Reality (Unload)...")
            # CRITICAL: Explicitly delete GPU tensors BEFORE setting to None
            if self.weights is not None:
                for k in list(self.weights.keys()):
                    del self.weights[k]
            self.weights = None
            self.is_collapsed = False
            # Periodic cache clear (not every call - too expensive)
            # Let Python's ref counting handle individual tensors
            gc.collect()

class QuantumModel(nn.Module):
    def __init__(self, model_path, pruned_indices=None):
        super().__init__()
        self.loader = QuantumLoader(model_path, pruned_indices)
        # We don't instantiate 62 layers here to save startup time for the test
        # We just expose a method to get a layer
        
    def get_layer(self, idx):
        return QuantumLayer(idx, self.loader)
