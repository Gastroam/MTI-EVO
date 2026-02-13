import torch
from safetensors.torch import load_file
import json
import os
import gc
import asyncio
from typing import List, Dict, Set, Optional

class ResonanceGuidedLoader:
    """
    Tiered Memory Loader with Resonance-Guided Prefetching (v2.1).
    Supports Multi-Shard Layers and Async Prefetch.
    """
    def __init__(self, model_path: str, pruned_layer_indices: Optional[List[int]] = None):
        self.model_path = model_path
        self.index_path = os.path.join(model_path, "model.safetensors.index.json")
        self.config_path = os.path.join(model_path, "config.json")
        
        # Load Config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load Index
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
            self.weight_map = self.index["weight_map"]
        else:
            self.weight_map = {}
            print(f"[ResonanceLoader] Warning: Index not found at {self.index_path}")

        self.pruned_indices = set(pruned_layer_indices) if pruned_layer_indices else set()
        
        # Pre-compute layer-to-shards mapping (One layer can be in multiple shards)
        self.layer_shards: Dict[int, List[str]] = {} 
        self._map_layers()
        
        # TIER 2 MEMORY (RAM Cache)
        self.cpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Async Prefetch
        self.prefetch_queue = asyncio.Queue(maxsize=12)

        # Calibrated Cognitive Sectors (Gemma 12B)
        self.sector_ranges = {
            'Pillar': range(0, 13),   
            'Bridge': range(13, 31),
            'Ghost':  range(31, 48)   
        }

    def _map_layers(self):
        """
        Map every layer to the LIST of shards containing its weights.
        """
        num_layers = getattr(self.config.get('text_config', self.config), 'num_hidden_layers', 48)
        if isinstance(num_layers, dict): num_layers = 48 

        for i in range(num_layers):
            target_shards = set()
            prefix = f"layers.{i}."
            
            for key, shard in self.weight_map.items():
                if "vision" in key: continue
                # any key containing 'layers.i.' implies this shard has partial weights
                if prefix in key:
                    target_shards.add(shard)
            
            if target_shards:
                self.layer_shards[i] = [os.path.join(self.model_path, s) for s in sorted(list(target_shards))]
            else:
                 # Should not happen
                 pass

    def predict_active_layers(self, prompt: str) -> List[int]:
        prompt_lower = prompt.lower()
        resonance_map = {'Pillar': 0.0, 'Bridge': 0.0, 'Ghost': 0.0}

        # Simulated Resonance
        if any(w in prompt_lower for w in ['what', 'define', 'fact', 'list', 'calculate', 'prove']):
            resonance_map['Pillar'] += 0.8
        if any(w in prompt_lower for w in ['why', 'explain', 'code', 'function', 'analyze', 'reason']):
            resonance_map['Bridge'] += 0.8
        if any(w in prompt_lower for w in ['describe', 'imagine', 'story', 'dream', 'feel', 'color', 'poem']):
            resonance_map['Ghost'] += 0.8
        resonance_map['Bridge'] += 0.2

        active_layers = set()
        active_layers.update([0, 1, 2, 47])

        if resonance_map['Pillar'] > 0.5: active_layers.update(range(0, 13, 1))
        if resonance_map['Bridge'] > 0.5: 
             step = 1 if resonance_map['Bridge'] > 0.8 else 2
             active_layers.update(range(13, 31, step))
        if resonance_map['Ghost'] > 0.5: active_layers.update(range(31, 48, 1))

        # Glue
        active_layers.update([15, 25, 35])
        return sorted(list([l for l in active_layers if l < 48]))

    async def prefetch_for_prompt(self, prompt: str):
        indices = self.predict_active_layers(prompt)
        # print(f"[AsyncLoader] Prefetching {len(indices)} layers...")
        for idx in indices:
            if idx not in self.cpu_cache:
                asyncio.create_task(self._async_load_to_cache(idx))

    async def _async_load_to_cache(self, layer_idx: int):
        if layer_idx in self.cpu_cache: return
        try:
             # This runs in ThreadPoolExecutor
             weights = await asyncio.to_thread(self._read_from_disk, layer_idx)
             if weights:
                 self.cpu_cache[layer_idx] = weights
        except Exception as e:
             pass

    def load_layer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if layer_idx in self.pruned_indices: return None
        if layer_idx in self.cpu_cache:
            # LRU REFRESH: Move to end (Most Recently Used)
            val = self.cpu_cache.pop(layer_idx)
            self.cpu_cache[layer_idx] = val
            return val
        
        # Blocking fetch
        weights = self._read_from_disk(layer_idx)
        if weights:
            # METABOLIC EVICTION (LRU-ish)
            # Limit cache to 8 layers (~5GB Float32) to prevent System RAM OOM
            if len(self.cpu_cache) >= 8:
                # Evict the "oldest" key (random/iterator order is fine for now, or use OrderedDict)
                # Ideally check timestamps, but Python 3.7+ dict preserves insertion order.
                # So next(iter(self.cpu_cache)) is the oldest inserted.
                victim = next(iter(self.cpu_cache))
                del self.cpu_cache[victim]
                # print(f"[ResonanceLoader] Evicted Layer {victim} for Metabolism")
                
            self.cpu_cache[layer_idx] = weights
            
        return weights

    def _read_from_disk(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        shard_paths = self.layer_shards.get(layer_idx, [])
        if not shard_paths: return None
            
        try:
            full_layer_weights = {}
            # print(f"[ResonanceLoader] Loading Layer {layer_idx} from {len(shard_paths)} shards: {[os.path.basename(s) for s in shard_paths]}")
            
            for path in shard_paths:
                # Force CPU load
                state_dict = load_file(path, device="cpu")
                
                for key, tensor in state_dict.items():
                    if "vision" in key: continue
                    # Flexible matching for "layers.X." or "layers.0X."
                    # But standard gemma is layers.N.
                    
                    if f"layers.{layer_idx}." in key:
                         try:
                            # Parse standard key
                            split_key = key.split(f"layers.{layer_idx}.")[-1]
                            
                            # DEBUG: Check for specific missing key
                            # if layer_idx == 15 and "up_proj" in split_key:
                            #     print(f"   -> Found {split_key} in {os.path.basename(path)}")
                                
                            # FLoat32 for Stability
                            full_layer_weights[split_key] = tensor.to(torch.float32)
                         except: pass
                
                del state_dict
                gc.collect() # Force cleanup to prevent RAM fragmentation
            
            # Validation Step
            if layer_idx == 15:
                keys = full_layer_weights.keys()
                if "mlp.up_proj.weight" not in keys:
                    print(f"[ResonanceLoader] CRITICAL: Layer 15 missing up_proj! Loaded keys: {list(keys)}")
            
            return full_layer_weights if full_layer_weights else None
            
        except Exception as e:
            print(f"[ResonanceLoader] Disk Read Error {layer_idx}: {e}")
            return None

    def inject_vector(self, layer_idx: int, target_key: str, vector: torch.Tensor, alpha: float = 1.0):
        """
        Injects a vector into a cached layer's weight matrix.
        Used by ParadoxResolver for Negative Mass Injection.
        """
        # Ensure layer is loaded (and refreshed in LRU)
        weights = self.load_layer(layer_idx)
        if not weights or target_key not in weights:
            return False
            
        original = weights[target_key]
        
        # Ensure Projection (Bilinear Interpolation) if shapes mismatch
        if vector.shape != original.shape:
             # Basic projection logic (assuming 2D matrices)
             inp = vector.unsqueeze(0).unsqueeze(0).float()
             out = torch.nn.functional.interpolate(inp, size=original.shape, mode='bilinear', align_corners=False)
             patch = out.squeeze().to(original.dtype).to(original.device)
        else:
             patch = vector.to(original.dtype).to(original.device)
             
        # Blend: W_new = (1 - alpha)*W_old + alpha*W_patch
        # For Negative Mass, alpha is 1.0 and W_patch is Negative.
        # But here we implement standard blending. 
        # The 'vector' passed in should already be negative if desired.
        
        mixed = (1 - alpha) * original + alpha * patch
        self.cpu_cache[layer_idx][target_key] = mixed
        return True
