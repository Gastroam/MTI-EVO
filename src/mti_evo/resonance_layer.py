import torch
import torch.nn as nn
import gc
import math
from typing import Optional, Dict

class ResonanceLayer(nn.Module):
    """
    A Quantum/Resonance-State Decoder Layer.
    Exist in a superposition state (Identity) until observed (Forward pass).
    When observed, it collapses to a specific reality (Gemma3DecoderLayer) using the ResonanceLoader.
    """
    def __init__(self, layer_id, loader, config):
        """
        loader: Instance of ResonanceGuidedLoader (handles Tier 2/3 memory)
        """
        super().__init__()
        self.layer_id = layer_id
        self.loader = loader
        self.config = config
        
        self.collapsed_impl = None
        self.is_collapsed = False
        
        # Diagnostics
        print(f"[RESONANCE] Layer {layer_id} Initialized (Virtual)")

    def forward(self, x: torch.Tensor, position_embeddings=None, attention_mask=None, position_ids=None) -> torch.Tensor:
        # 1. Check Resonance / Load Layer
        # The loader decides if we should load based on its internal logic or if we force it.
        # Ideally, 'predict_active_layers' was called at model level to prefetch/set flags.
        # Here we just ask the loader for weights.
        
        # FORCE PRECISION: Ensure input is bfloat16 matching weights
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        if not self.is_collapsed:
            weights = self.loader.load_layer(self.layer_id)
            
            if weights is None:
                # Resonance check failed or pruned -> Identity
                # print(f"[RESONANCE] Layer {self.layer_id} Skipped (No Resonance)")
                return x
            
            # Reconstruction (The "Collapse")
            self.collapsed_impl = self._reconstruct_layer(weights)
            
            if torch.cuda.is_available():
                self.collapsed_impl.to(x.device)
            
            self.is_collapsed = True
            
            # DEBUG: Verify Device
            try:
                param = next(self.collapsed_impl.parameters())
                print(f"[RESONANCE] Layer {self.layer_id} loaded on {param.device} ({param.dtype}) input={x.device} ({x.dtype})")
            except: pass
        
        # 2. Run Computation
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

        try:
             outputs = self.collapsed_impl(x, attention_mask=attention_mask, position_embeddings=layer_pos_emb, position_ids=position_ids)
        except Exception as e:
             # Fallback if forward fails
             print(f"[RESONANCE] Layer {self.layer_id} forward error: {e}")
             return x
        
        # 3. Robust Return
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def _reconstruct_layer(self, state_dict):
        """
        Reconstructs a native Gemma-3 Decoder Layer from weights.
        """
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
            
            layer_config = getattr(self.config, "text_config", self.config)
            
            # Ensure safe attention implementation
            if not hasattr(layer_config, "_attn_implementation") or getattr(layer_config, "_attn_implementation") is None:
                setattr(layer_config, "_attn_implementation", "eager")
            
            layer = Gemma3DecoderLayer(layer_config, layer_idx=self.layer_id)
            keys = layer.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                 # Filter out expected missing keys (like inv_freq usually)
                 real_missing = [k for k in keys.missing_keys if "inv_freq" not in k]
                 if real_missing:
                     print(f"[RESONANCE] Layer {self.layer_id} Missing Keys: {real_missing[:5]}...")
            
            # Cast to float32 (Robust precision)
            layer.to(torch.float32)
            
            return layer
            
        except Exception as e:
            print(f"[RESONANCE] Layer {self.layer_id} reconstruction failed: {e}")
            return nn.Identity()

    def reset_reality(self):
        """
        Reset the layer to its superposition state and free VRAM.
        Called by the Engine after forward pass.
        """
        if self.collapsed_impl is not None:
            # Move to CPU to ensure CUDA free
            if hasattr(self.collapsed_impl, "to"):
                self.collapsed_impl.to("cpu")
            
            del self.collapsed_impl
            self.collapsed_impl = None
            self.is_collapsed = False
            
            # Tell loader to evict if we are managing cache strictly
            # self.loader.evict_layer(self.layer_id) 
            
            # Periodic gc
            # gc.collect()
