
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Dict, Optional, Any, Union, Tuple
import gc

class QuantumLayer(nn.Module):
    """
    A layer that exists in superposition of multiple physical implementations.
    Collapses to a concrete implementation ONLY when observed (forward pass).
    Specialized for Gemma-3 (27B/4B) native architecture.
    """
    def __init__(self, layer_id: int, shard_paths: Dict[str, str], weights: Dict[str, float], config: Any):
        super().__init__()
        self.layer_id = layer_id
        self.shard_paths = shard_paths  # {"fast": path, "precise": path}
        self.weights = weights  # {"fast": 0.5, "precise": 0.5}
        self.config = config
        self.collapsed_impl = None
        self.collapsed_key = None
    
    def collapse(self) -> str:
        """Collapses the superposition to a concrete implementation"""
        if self.collapsed_key:
            return self.collapsed_key
        
        r = torch.rand(1).item()
        cum = 0.0
        for key, prob in self.weights.items():
            cum += prob
            if r <= cum:
                self.collapsed_key = key
                return key
        return list(self.weights.keys())[-1]
    
    def forward(self, x: torch.Tensor, position_embeddings=None, attention_mask=None) -> torch.Tensor:
        impl_key = self.collapse()
        
        if self.collapsed_impl is None:
            # print(f"[QUANTUM] Layer {self.layer_id} collapsing to |{impl_key}>...")
            
            shard_path = self.shard_paths.get(impl_key)
            if shard_path:
                try:
                    state_dict = load_file(shard_path)
                    
                    # Prefix search for Gemma-3 layers (Prioritize language_model over vision_tower)
                    search_str = f"layers.{self.layer_id}."
                    matches = [k for k in state_dict.keys() if search_str in k]
                    
                    target_key = None
                    for k in matches:
                        if "language_model" in k:
                            target_key = k
                            break
                    if not target_key and matches:
                        target_key = matches[0]
                    
                    if target_key:
                        target_prefix = target_key.split(search_str)[0] + search_str
                        layer_state = {k[len(target_prefix):]: v for k, v in state_dict.items() 
                                      if k.startswith(target_prefix)}
                        
                        self.collapsed_impl = self._reconstruct_layer(layer_state)
                        self.collapsed_impl.to(x.device)
                    else:
                        print(f"[WARNING] Layer {self.layer_id} not found in shard. Using Identity.")
                        self.collapsed_impl = nn.Identity().to(x.device)
                    
                    del state_dict
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ERROR] Failed to load shard {shard_path}: {e}")
                    self.collapsed_impl = nn.Identity().to(x.device)
            else:
                self.collapsed_impl = nn.Identity().to(x.device)
        
        if isinstance(self.collapsed_impl, nn.Identity):
            return x
        
        # Determine correct position embedding for this layer type
        layer_pos_emb = position_embeddings
        if isinstance(position_embeddings, dict):
            # Get layer type from config
            layer_types = getattr(self.config, "layer_types", None)
            if not layer_types and hasattr(self.config, "text_config"):
                layer_types = getattr(self.config.text_config, "layer_types", None)
            
            l_type = "default"
            if layer_types and self.layer_id < len(layer_types):
                l_type = layer_types[self.layer_id]
            
            layer_pos_emb = position_embeddings.get(l_type)
            
        outputs = self.collapsed_impl(x, attention_mask=attention_mask, position_embeddings=layer_pos_emb)
        
        # Robust return handling (Fix for accidental batch stripping)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def _reconstruct_layer(self, state_dict):
        """
        Reconstructs a native Gemma-3 Decoder Layer.
        """
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
            
            layer_config = getattr(self.config, "text_config", self.config)
            
            # Ensure safe attention implementation
            if not hasattr(layer_config, "_attn_implementation") or getattr(layer_config, "_attn_implementation") is None:
                setattr(layer_config, "_attn_implementation", "eager")
            
            layer = Gemma3DecoderLayer(layer_config, layer_idx=self.layer_id)
            layer.load_state_dict(state_dict, strict=False)
            layer.to(torch.bfloat16) 
            return layer
            
        except Exception as e:
            print(f"[ERROR] Layer reconstruction failed for {self.layer_id}: {e}")
            return nn.Identity()

    def reset_reality(self):
        """Reset the layer to its superposition state and free VRAM"""
        if self.collapsed_impl is not None:
            # Move impl to CPU first to release VRAM block, then delete
            if hasattr(self.collapsed_impl, 'cpu'):
                try:
                    self.collapsed_impl.cpu()
                except:
                    pass
            del self.collapsed_impl
            self.collapsed_impl = None
            # Periodic gc - let allocator reuse blocks most of the time
            # Only collect every ~10 layers via external counter if needed
            gc.collect()
        self.collapsed_key = None
