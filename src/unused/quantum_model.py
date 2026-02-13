
import torch
import torch.nn as nn
import os
import glob
from safetensors import safe_open
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Optional, List, Union, Tuple
from .quantum_layer import QuantumLayer
import gc

class QuantumGemmaConfig(PretrainedConfig):
    model_type = "quantum_gemma"
    def __init__(self, base_model_path=None, fast_model_path=None, **kwargs):
        self.base_model_path = base_model_path
        self.fast_model_path = fast_model_path # Path to 4B Safetensors
        super().__init__(**kwargs)

class QuantumGemmaForCausalLM(nn.Module):
    """
    The Quantum Brain (Hybrid 27B + 4B).
    Orchestrates a sequence of QuantumLayers to perform inference on huge models
    using minimal VRAM via temporal sparsity (The Schrödinger's Weights method).
    """
    def __init__(self, config: Union[QuantumGemmaConfig, str]):
        super().__init__()
        
        # Handle Config Loading
        if isinstance(config, str):
            # If path string, load HF config
            self.model_path = config
            self.fast_path = None
            self.hf_config = AutoConfig.from_pretrained(config)
        elif hasattr(config, "base_model_path"):
            self.model_path = config.base_model_path
            self.fast_path = config.fast_model_path
            self.hf_config = AutoConfig.from_pretrained(self.model_path)
        else:
            raise ValueError("Invalid Config provided to QuantumGemma")
            
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.hidden_size = self.text_config.hidden_size
        self.vocab_size = self.text_config.vocab_size
        self.num_layers = self.text_config.num_hidden_layers
        
        print(f"[QUANTUM] Initializing Gemma-3 Brain ({self.num_layers} Layers)...")
        
        # 0. Load Fast Brain (Limbic System) - 4B Resident
        self.limbic = None
        if self.fast_path and os.path.exists(self.fast_path):
             print(f"[QUANTUM] ⚡ Initializing Limbic System (4B): {self.fast_path}")
             try:
                 from transformers import AutoModelForCausalLM, BitsAndBytesConfig
                 bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                 )
                 self.limbic = AutoModelForCausalLM.from_pretrained(
                     self.fast_path,
                     quantization_config=bnb_config,
                     device_map="auto" # Will likely take ~3GB VRAM
                 )
                 print(f"[QUANTUM] ✅ Limbic System Online. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
             except Exception as e:
                 print(f"[QUANTUM] ⚠️ Limbic Load Failed: {e}")

        # 1. Embeddings & Norms (Must remain constantly loaded for coherence)
        # We load these into VRAM permanently as they are small relative to layers.
        pad_idx = getattr(self.text_config, "pad_token_id", getattr(self.hf_config, "pad_token_id", 0))
        if pad_idx is None: pad_idx = 0
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, pad_idx)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.text_config.rms_norm_eps)
        # Note: We cheat and use an empty Linear for lm_head to be filled later, 
        # or share weights with embed_tokens if strict.
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize Rotary Embeddings (Required for Gemma3DecoderLayer)
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
            # Correct signature takes config, keys are pulled from there
            self.rotary_emb = Gemma3RotaryEmbedding(self.text_config)
        except Exception as e:
            print(f"[QUANTUM] Rotary Embedding init failed: {e}. Will compute inline.")
            self.rotary_emb = None
        
        # Load Static Weights (Embeds + Norms) from Shards (Greedy Scan)
        self._load_static_weights()
        print(f"[QUANTUM] Static Weights Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # 2. Quantum Layers (The Superposition)
        self.layers = nn.ModuleList()
        self.shard_map = self._map_shards_to_layers()
        
        for i in range(self.num_layers):
            # Each layer gets the map of shards that contain its weights
            # For 27B, layer N might be in shard X.
            relevant_shards = self.shard_map.get(i, [])
            
            # "Fast" pathway = The NVMe execution
            # "Precise" pathway = Same file (for now)
            paths = {}
            if relevant_shards:
                 paths["precise"] = relevant_shards[0] # Primary
            
            self.layers.append(QuantumLayer(
                layer_id=i,
                shard_paths=paths,
                weights={"precise": 1.0},
                config=self.hf_config
            ))
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"[QUANTUM] Brain Online. Total VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _map_shards_to_layers(self):
        """Pre-scans safetensors index to find which shard holds which layer."""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        import json
        
        layer_shard_map = {} # {layer_idx: [shard_paths]}
        
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            
            for key, shard_file in weight_map.items():
                if "layers." in key:
                    try:
                        # Extract "layers.5." -> 5
                        parts = key.split("layers.")
                        layer_idx = int(parts[1].split(".")[0])
                        
                        full_shard_path = os.path.join(self.model_path, shard_file)
                        if layer_idx not in layer_shard_map:
                            layer_shard_map[layer_idx] = []
                        if full_shard_path not in layer_shard_map[layer_idx]:
                            layer_shard_map[layer_idx].append(full_shard_path)
                    except: pass
        else:
            # Fallback for non-indexed sharding (manual scan)
            print("[QUANTUM] No index found. Scanning shards manually (Slow)...")
            shards = glob.glob(os.path.join(self.model_path, "*.safetensors"))
            # This is expensive, so for now we assume index exists for 27B.
            # If not, naive assumption: sequentially distribute? No, dangerous.
            # We urge the user to have index.
            pass
            
        return layer_shard_map

    def _load_static_weights(self):
        """Loads non-layer weights (Embeds, Final Norm, LM Head) into VRAM."""
        print("[QUANTUM] Loading Static Components (Embeds, Norms)...")
        # Reuse mapping logic or scan all shards for specific keys
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        import json
        from safetensors.torch import load_file
        
        loaded_modules = set()
        required_modules = {"embed_tokens", "norm", "lm_head"}
        
        if os.path.exists(index_path):
             with open(index_path, "r") as f: index = json.load(f)
             weight_map = index.get("weight_map", {})
             
             # Group keys by shard to minimize I/O
             shard_to_keys = {}
             
             # Support both 27B format (model.X) and 12B format (language_model.model.X)
             targets = [
                 "model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
                 "model.norm.weight", "language_model.model.norm.weight",
                 "lm_head.weight"  # May not exist if tied
             ]
             
             for t in targets:
                 shard = weight_map.get(t)
                 if shard:
                     full_path = os.path.join(self.model_path, shard)
                     if full_path not in shard_to_keys: shard_to_keys[full_path] = []
                     shard_to_keys[full_path].append(t)
            
             lm_head_loaded = False
             for shard, keys in shard_to_keys.items():
                 state_dict = load_file(shard)
                 for k in keys:
                     if "embed_tokens" in k:
                         self.embed_tokens.weight.data = state_dict[k].to(torch.bfloat16)
                         # print(f"  Loaded Embeddings from {k}")
                     elif "norm.weight" in k and "layers" not in k: # Final norm only
                         self.norm.weight.data = state_dict[k].to(torch.bfloat16)
                         # print(f"  Loaded Final Norm from {k}")
                     elif "lm_head" in k:
                         self.lm_head.weight.data = state_dict[k].to(torch.bfloat16)
                         lm_head_loaded = True
                         # print(f"  Loaded LM Head from {k}")
                 del state_dict
                 gc.collect()
             
             # If no lm_head, tie weights with embeddings
             if not lm_head_loaded:
                 self.lm_head.weight = self.embed_tokens.weight
                 # print("  LM Head tied to Embeddings")

    def forward(self, input_ids, **kwargs):
        # 1. Embed
        x = self.embed_tokens(input_ids)
        
        # 2. Compute Position Embeddings (Rotary)
        # Required for Gemma3DecoderLayer forward
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Compute rotary embeddings for ALL layer types (global, sliding, etc.)
        position_embeddings = {}
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            try:
                # Gemma3 computes different RoPE for different layer types
                layer_types = getattr(self.text_config, "layer_types", ["default"])
                # De-duplicate
                layer_types = list(set(layer_types))
                
                for l_type in layer_types:
                    position_embeddings[l_type] = self.rotary_emb(x, position_ids, layer_type=l_type)
            except Exception as e:
                print(f"[QUANTUM] Rotary calc failed: {e}. Shapes: x={x.shape}, pos={position_ids.shape}")
                pass
        
        # 3. Quantum Layer Stack
        # The Conscious Stream: Token passes through layers one by one.
        for i, layer in enumerate(self.layers):
            # Observation: Materialize layer -> Compute
            try:
                # We pass the full dict, the layer will pick the right one
                x = layer(x, position_embeddings=position_embeddings)
            except Exception as e:
                # If layer fails, skip it (graceful degradation)
                print(f"[WARNING] Layer {i} forward failed: {e}")
                pass
            
            # Reset: Dissolve layer needed?
            # For 27B on 6GB, we MUST reset every layer immediately.
            layer.reset_reality() 
            
            # PERF FIX: Periodic VRAM cache clear to prevent fragmentation
            # Not every layer (100x slowdown) but every 8 layers is a good balance
            if torch.cuda.is_available() and (i + 1) % 8 == 0:
                torch.cuda.empty_cache()
            
        # 4. Final Norm & Head
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits # Returns logits (Batch, Seq, Vocab)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, **kwargs):
        """
        Quantum Generation Loop with Speculative Decoding (Hybrid 27B + 4B).
        """
        curr_ids = input_ids
        gamma = 4 # Draft 4 tokens ahead
        
        for step in range(0, max_new_tokens):
            # [CASE A] Hybrid Mode: Use Limbic System (4B) to draft
            if self.limbic:
                # Prepare Attention Mask (All ones for active tokens)
                attn_mask = torch.ones_like(curr_ids)
                
                # Determine Pad Token (Silence Warning)
                # Determine Pad Token (Silence Warning)
                raw_pad = self.hf_config.pad_token_id if self.hf_config.pad_token_id is not None else self.hf_config.eos_token_id
                if isinstance(raw_pad, list) and len(raw_pad) > 0:
                    pad_id = raw_pad[0]
                elif isinstance(raw_pad, list):
                    pad_id = 0
                else:
                    pad_id = raw_pad

                # 1. Draft K tokens with 4B (Resident, Fast)
                draft_outputs = self.limbic.generate(
                    curr_ids, 
                    attention_mask=attn_mask,
                    max_new_tokens=gamma, 
                    do_sample=False, 
                    pad_token_id=pad_id
                )
                draft_ids = draft_outputs[0] # (Seq + Gamma)
                new_drafts = draft_ids[curr_ids.shape[1]:] # Just the new ones
                
                # 2. Verify with Quantum Brain (27B)
                # We verify the first drafted token (Simplest Speculation)
                # Real speculative decoding verifies all, but for now we just want "Intuition vs Reason"
                
                # Check 27B on the current context
                logits_27b = self.forward(curr_ids)
                next_token_27b = torch.argmax(logits_27b[:, -1, :], dim=-1).unsqueeze(-1)
                
                # 3. Compare
                if len(new_drafts) > 0 and next_token_27b.item() == new_drafts[0].item():
                     # Match! Accept draft (and potentially more, but keep simple)
                     # For robustness, we accept just the first one verified
                     next_token = next_token_27b
                     # print(f".", end="", flush=True) # Speed dot
                else:
                     # Mismatch (Correction)
                     next_token = next_token_27b
                     # print(f"!", end="", flush=True) # Correction
            
            # [CASE B] Pure Quantum Mode
            else:
                logits = self.forward(curr_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Stop check (EOS)
            eos_ids = self.hf_config.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids] if eos_ids is not None else []
            
            if next_token.item() in eos_ids:
                break
            
            # Safety break
            if curr_ids.shape[1] > input_ids.shape[1] + max_new_tokens:
                break
            
        return curr_ids
