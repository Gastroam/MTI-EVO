import torch
import torch.nn as nn
from transformers import AutoConfig
from .resonance_layer import ResonanceLayer
from .resonance_loader import ResonanceGuidedLoader
from .paradox_resolver import ParadoxResolver
from typing import Union, List

class ResonanceGemmaForCausalLM(nn.Module):
    """
    Resonance-Based Gemma 3 (12B) Model.
    Uses 'ResonanceGuidedLoader' to sparsely activate layers based on prompt topology.
    """
    def __init__(self, config: Union[str, any]):
        super().__init__()
        
        # Handle Config Loading
        if isinstance(config, str):
            self.model_path = config
            self.hf_config = AutoConfig.from_pretrained(config)
        elif hasattr(config, "base_model_path"):
            self.model_path = config.base_model_path
            self.hf_config = AutoConfig.from_pretrained(self.model_path)
        else:
            raise ValueError("Invalid Config provided to ResonanceGemma")
            
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.num_layers = self.text_config.num_hidden_layers
        
        print(f"[RESONANCE] Initializing Gemma-3 Architecture ({self.num_layers} Layers)...")
        
        # 1. Initialize Resonance Loader (The Brain Processor)
        self.loader = ResonanceGuidedLoader(self.model_path)
        
        # 1.5 Initialize Paradox Resolver (The Auto-Corrector)
        self.resolver = ParadoxResolver(self)
        
        # 2. Embeddings & Norms
        pad_idx = getattr(self.text_config, "pad_token_id", getattr(self.hf_config, "pad_token_id", 0))
        if pad_idx is None: pad_idx = 0
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, pad_idx)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.text_config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize Rotary Embeddings (Required for Gemma3DecoderLayer)
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
            self.rotary_emb = Gemma3RotaryEmbedding(self.text_config)
        except Exception as e:
            print(f"[RESONANCE] Rotary Embedding init failed: {e}. Will compute inline.")
            self.rotary_emb = None

        # 3. Create Resonance Layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Pass the loader to the layer
            self.layers.append(ResonanceLayer(i, self.loader, self.hf_config))
            
        # 4. Load Static Weights (Embeds, Norms, Head) - Tier 1
        self._load_static_weights()
        
    def _load_static_weights(self):
        print("[RESONANCE] Loading Static Components (Embeds, Norms)...")
        # Use the loader to get shards for these specific keys
        # Optimization: We can reuse the loader's mapping if we extend it, 
        # but usually embeds are in shard 0. Simple load_file is fine for statics.
        try:
            from safetensors.torch import load_file
            import os
            # Heuristic: Load first 2 shards usually cover embeddings
            # For 12B, embeds might be large.
            # Using QuantumLoader logic to find embed shard would be better.
            # For now, simplistic approach from previous quantum_model.py
            index_path = os.path.join(self.model_path, "model.safetensors.index.json")
            import json
            with open(index_path) as f:
                index = json.load(f)
            
            # Find shards for specific keys
            def find_shard(key_part):
                for k, v in index["weight_map"].items():
                    if key_part in k:
                        return v
                return None
                
            embed_shard = find_shard("embed_tokens")
            norm_shard = find_shard("norm.weight")
            head_shard = find_shard("lm_head") # Usually tied or separate
            
            loaded_keys = set()
            
            for shard in set([s for s in [embed_shard, norm_shard, head_shard] if s]):
                path = os.path.join(self.model_path, shard)
                state = load_file(path)
                
                # Filter and load into our modules (state_dict update)
                # Need to map keys (model.embed_tokens -> embed_tokens)
                processed_state = {}
                for k, v in state.items():
                    if "vision" in k: continue # Strict Filter
                    
                    if k.endswith("model.embed_tokens.weight") or k == "embed_tokens.weight":
                        print(f"[RESONANCE] Loaded Embeddings: {k} {v.shape}")
                        self.embed_tokens.weight.data = v.to(torch.float32)
                        loaded_keys.add("embed")
                    elif (k.endswith("model.norm.weight") or k == "norm.weight") and "layers" not in k:
                        print(f"[RESONANCE] Loaded Final Norm: {k} {v.shape}")
                        self.norm.weight.data = v.to(torch.float32)
                        loaded_keys.add("norm")
                    elif "lm_head" in k:
                        print(f"[RESONANCE] Loaded LM Head: {k} {v.shape}")
                        self.lm_head.weight.data = v.to(torch.float32)
                        loaded_keys.add("head")
                
                del state
                
            # Tie weights checks
            if "head" not in loaded_keys and "embed" in loaded_keys:
                 self.lm_head.weight = self.embed_tokens.weight
            
        except Exception as e:
            print(f"[RESONANCE] Static Load Warning: {e}")

    def prepare_resonance(self, prompt: str):
        """
        Predicts active layers for the prompt and primes the loader.
        This enables the sparse activation.
        """
        active_indices = self.loader.predict_active_layers(prompt)
        print(f"[RESONANCE] Cognitive Topology: Active Layers {active_indices}")
        
        # Update loader pruning (Inverse of active)
        all_indices = set(range(self.num_layers))
        active_set = set(active_indices)
        pruned = list(all_indices - active_set)
        
        # We update the loader's pruned_indices to SKIP the non-active ones
        # Note: ResonanceLoader needs to expose updating this.
        self.loader.pruned_indices = set(pruned)
        
        # Also prefetch the active ones?
        # self.loader.prefetch_layers(active_indices) # Optional async optimization

    def resolve_paradox(self, prompt: str) -> bool:
        """
        Detects and resolves paradoxes by injecting Negative Mass energy sinks.
        Returns True if a paradox was detected and sinks were injected.
        """
        return self.resolver.resolve(prompt)

    def _prepare_causal_mask(self, input_ids, target_dtype):
        batch_size, seq_len = input_ids.shape
        mask = torch.full((seq_len, seq_len), torch.finfo(target_dtype).min, device=input_ids.device)
        mask_cond = torch.arange(mask.size(-1), device=input_ids.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        
        # 4D mask: [Batch, 1, Seq, Seq]
        mask = mask.to(target_dtype)
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
        return mask

    def forward(self, input_ids, **kwargs):
        # 1. Embed
        x = self.embed_tokens(input_ids)
        
        # 2. Compute Position Embeddings (Rotary)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = {}
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            try:
                layer_types = getattr(self.text_config, "layer_types", ["default"])
                layer_types = list(set(layer_types))
                for l_type in layer_types:
                    position_embeddings[l_type] = self.rotary_emb(x, position_ids, layer_type=l_type)
            except Exception as e:
                # print(f"[RESONANCE] Rotary calc failed: {e}")
                pass
        
        # 2.5 Prepare Causal Mask
        attention_mask = self._prepare_causal_mask(input_ids, x.dtype)

        # 3. Resonance Layer Stack
        for i, layer in enumerate(self.layers):
            try:
                # The layer itself checks the loader. If pruned, returns x immediately.
                x = layer(x, position_embeddings=position_embeddings, position_ids=position_ids, attention_mask=attention_mask)
            except Exception as e:
                print(f"[WARNING] Layer {i} forward failed: {e}")
                pass
            
            # Reset after every token?
            # Ideally yes, to free VRAM for next layer.
            # But if we prefetch 15 layers, maybe we keep them?
            # For 6GB VRAM, we can't keep 15 layers (15 * 0.5GB = 7.5GB).
            # So we MUST reset.
            layer.reset_reality() 
        
        # 4. Final Norm & Head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
