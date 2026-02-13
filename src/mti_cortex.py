
import os
import time
import json
import torch
import torch.nn as nn
from safetensors import safe_open
from typing import Optional, Dict, List, Tuple
from enum import Enum
from mti_evo.mti_core import HolographicLattice

# --- System 1: The Swarm (Exo-Cortex / Limbic) ---
try:
    from llama_cpp import Llama
    SYSTEM_1_AVAILABLE = True
except ImportError:
    SYSTEM_1_AVAILABLE = False
    print("⚠️  llama-cpp-python not found. The Swarm is silent.")

class TheSwarm:
    def __init__(self, model_path: str):
        if not SYSTEM_1_AVAILABLE: return
        print(f"🐝 Awakening The Swarm: {os.path.basename(model_path)}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU Only (Save VRAM for Cortex)
            n_ctx=1024,
            n_threads=1, # Force Single Thread to avoid Deadlock
            verbose=False
        )
        
    def dream(self, context: str, speculation_depth: int = 16) -> str:
        if not SYSTEM_1_AVAILABLE: return "System 1 Offline."
        output = self.llm(
            context, 
            max_tokens=speculation_depth, 
            stop=["User:", "Cortex:"], 
            echo=False
        )
        return output['choices'][0]['text']

# --- System 2: The Cortex (Quantum Lattice) ---

class NeuralTier(Enum):
    COLD = 0 # Disk (NVMe)
    WARM = 1 # RAM (System Memory)
    HOT  = 2 # VRAM (GPU)

class SwarmBridge:
    """
    Connects System 1 (Swarm/Text) to System 2 (Cortex/Vectors).
    Uses RAM-Resident Embedding Table to save 2GB+ VRAM.
    """
    def __init__(self, model_path: str, swarm_instance, weight_map: dict = None):
        self.swarm_vm = swarm_instance
        self.embed_weights = None
        
        # Resolve Embedding Shard Dynamically
        shard_file = "model-00001-of-00012.safetensors" # Fallback
        
        if weight_map:
            # Try standard keys for embeddings
            shard_file = weight_map.get("language_model.model.embed_tokens.weight")
            if not shard_file: shard_file = weight_map.get("model.embed_tokens.weight")
            
        if not shard_file and os.path.exists(os.path.join(model_path, "model.safetensors")): 
             shard_file = "model.safetensors"

        shard_path = os.path.join(model_path, shard_file)
        
        try:
            print(f"🌉 Building Synapse Bridge (RAM-Resident Embeddings)...")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                target_key = None
                for k in keys:
                    if "embed_tokens" in k:
                        target_key = k
                        break
                
                if target_key:
                    # Clone for thread safety on Windows
                    self.embed_weights = f.get_tensor(target_key).clone() 
                    print(f"✅ Bridge Established. Embedding Table: {self.embed_weights.shape} | {self.embed_weights.element_size() * self.embed_weights.nelement() / 1e9:.2f} GB RAM")
                else:
                    print(f"❌ Bridge Error: Could not find 'embed_tokens' in {shard_file}")
        except Exception as e:
            print(f"❌ Bridge Init Error: {e}")

    def transduce(self, text: str) -> torch.Tensor:
        """
        Converts Swarm Dream (Text) -> Cortex Reality (Vectors)
        """
        if self.embed_weights is None:
            raise RuntimeError("Bridge Broken: No Embeddings Loaded.")
            
        # 1. Tokenize (Integers)
        tokens = self.swarm_vm.llm.tokenize(text.encode("utf-8"), add_bos=True)
        
        # 2. Lookup (Vectors on CPU)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        max_vocab = self.embed_weights.shape[0] - 1
        token_tensor = torch.clamp(token_tensor, max=max_vocab)
        vectors = torch.nn.functional.embedding(token_tensor, self.embed_weights)
        
        # 3. Transmit (PCIe -> VRAM)
        vectors_gpu = vectors.to("cuda", dtype=torch.bfloat16)
        return vectors_gpu.unsqueeze(0)

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Gemma3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))

class Gemma3Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, pruning_ratio: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads * (1.0 - pruning_ratio))
        self.num_kv_heads = int(num_kv_heads * (1.0 - pruning_ratio))
        if self.num_heads < 1: self.num_heads = 1
        if self.num_kv_heads < 1: self.num_kv_heads = 1
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        
        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim)
        self.k_norm = GemmaRMSNorm(self.head_dim)
        
    def forward(self, x):
        B, L, D = x.shape
        q = self.q_norm(self.q_proj(x).view(B, L, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim))
        # TODO: Implement RoPE and Causal Mask for full correctness
        # Currently verifying vector flow and latency
        return self.o_proj(q.reshape(B, L, -1))

class Gemma3Block(nn.Module):
    def __init__(self, config: dict, pruning_ratio: float = 0.0):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        
        # Robust head_dim (Gemma 3 12B fix)
        if "head_dim" in config:
            self.head_dim = config["head_dim"]
        else:
            self.head_dim = 256 # Default for Gemma 12B/27B
            
        self.input_layernorm = GemmaRMSNorm(self.hidden_size)
        self.self_attn = Gemma3Attention(self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim, pruning_ratio)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size)
        self.mlp = Gemma3MLP(self.hidden_size, self.intermediate_size)
        
    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x) 
        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x

class CoreLayer(nn.Module):
    def __init__(self, layer_id: int, shard_paths: List[str], config: dict, pruning_ratio: float = 0.0, persistent_ram: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.shard_paths = list(set(shard_paths)) # Unique shards
        self.config = config
        self.pruning_ratio = pruning_ratio
        self.persistent_ram = persistent_ram 
        self.prefix = f"language_model.model.layers.{layer_id}."
        
        self.state = NeuralTier.COLD
        self.module = None
        self.ram_footprint = 0

    def ascend_to_warm(self):
        if self.state != NeuralTier.COLD: return
        try:
            # Flux State: Instantiate on Meta Device (Zero RAM)
            # print(f"    DEBUG: Meta Init Layer {self.layer_id}")
            with torch.device("meta"):
                self.module = Gemma3Block(self.config, pruning_ratio=self.pruning_ratio)
            
            state_dict = {}
            
            # Multi-Shard Loading
            for shard_path in self.shard_paths:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        if k.startswith(self.prefix):
                            tensor = f.get_tensor(k).clone() # Detach
                            name = k.replace(self.prefix, "")
                            
                            if self.pruning_ratio > 0:
                                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                                    dim = int(tensor.shape[0] * (1.0 - self.pruning_ratio))
                                    if tensor.shape[0] > dim: tensor = tensor[:dim, :]
                                elif "o_proj" in name:
                                    dim = int(tensor.shape[1] * (1.0 - self.pruning_ratio))
                                    if tensor.shape[1] > dim: tensor = tensor[:, :dim]
                                    
                            state_dict[name] = tensor

            # Assign weights (Materialize from Meta -> CPU)
            # print(f"    DEBUG: Assigning Layer {self.layer_id}")
            self.module.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict
            
            self.ram_footprint = 0
            for param in self.module.parameters():
                self.ram_footprint += param.element_size() * param.nelement()
                
            self.state = NeuralTier.WARM
            
        except Exception as e:
            print(f"❌ Warm Error {self.layer_id}: {e}")
            import traceback
            traceback.print_exc()

    def materialize(self):
        """Warm (CPU Module) -> Hot (GPU Module)"""
        if self.state == NeuralTier.HOT: return
        if self.state == NeuralTier.COLD: self.ascend_to_warm()
        
        # Flux Transition: Move existing object to GPU
        # Note: This moves the data. The CPU object wrapper remains, but tensors are now CUDA.
        if self.module:
            self.module.to("cuda", non_blocking=True)
        
        self.state = NeuralTier.HOT
    
    def reset_reality(self):
        """Hot (GPU Module) -> Warm (CPU Module)"""
        if self.state != NeuralTier.HOT: return
        
        # Flux Return: Move object back to CPU to preserve state/RAM
        if self.module:
            self.module.to("cpu", non_blocking=True)
            
        if self.persistent_ram:
            self.state = NeuralTier.WARM
        else:
            self.module = None # Destroy object for Cold storage
            self.state = NeuralTier.COLD

    def forward(self, x):
        if self.state != NeuralTier.HOT: return x
        return self.module(x)

class LazyMouth(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, norm_shard: str, config: Dict):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.norm_weight = None
        self.head_weight = None
        with safe_open(norm_shard, framework="pt", device="cpu") as f:
             self.norm_weight = f.get_tensor("language_model.model.norm.weight")
        self.head_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16) # Placeholder Head

    def forward(self, x):
        norm = GemmaRMSNorm(self.hidden_size).to("cuda", dtype=torch.bfloat16)
        norm.weight = nn.Parameter(self.norm_weight.to("cuda", dtype=torch.bfloat16))
        x = norm(x)
        head = nn.Linear(self.hidden_size, self.vocab_size, bias=False).to("cuda", dtype=torch.bfloat16)
        head.weight = nn.Parameter(self.head_weight.to("cuda", dtype=torch.bfloat16))
        return head(x)

class QuantumCortex(nn.Module):
    """
    The Primary 12B Engine.
    Running in: System RAM (Persistent).
    Streaming to: VRAM (Just-in-Time).
    """
    def __init__(self, path_12b: str, path_4b: str, pruning_ratio: float = 0.0):
        super().__init__()
        print(f"🧠 Configuring Quantum Cortex (12B RAM Resident)...")
        
        # 1. System 1: The Swarm
        self.swarm = TheSwarm(path_4b)
        
        # 2. System 2: The Hologram (Memory)
        self.lattice = HolographicLattice()

        # 3. System 3: The Entity
        self.config_data = json.load(open(os.path.join(path_12b, "config.json")))
        self.text_config = self.config_data.get("text_config", {})
        self.hidden_size = self.text_config["hidden_size"]
        
        idx_path = os.path.join(path_12b, "model.safetensors.index.json")
        if not os.path.exists(idx_path): idx_path = os.path.join(path_12b, "model.safetensors.index(1).json")
        index = json.load(open(idx_path))
        weight_map = index["weight_map"]
        
        self.layers = []
        self.layers = []
        
        # Group Shards by Layer ID
        layer_shards = {}
        for key, shard_filename in weight_map.items():
            if "layers." in key:
                # Extract Layer ID (e.g., language_model.model.layers.43.mlp...)
                parts = key.split(".")
                try:
                    # Find index after 'layers'
                    idx_loc = parts.index("layers") + 1
                    layer_idx = int(parts[idx_loc])
                    
                    if layer_idx not in layer_shards: layer_shards[layer_idx] = set()
                    layer_shards[layer_idx].add(os.path.join(path_12b, shard_filename))
                except (ValueError, IndexError):
                    continue

        # Initialize Layers
        for i in range(self.text_config["num_hidden_layers"]):
            if i in layer_shards:
                layer = CoreLayer(
                    layer_id=i, 
                    shard_paths=list(layer_shards[i]), 
                    config=self.text_config, 
                    pruning_ratio=pruning_ratio, 
                    persistent_ram=True
                )
                self.layers.append(layer)
        
        # Pre-load Ritual
        print(f"☕ Performing RAM Residency Ritual (Loading {len(self.layers)} layers)...", flush=True)
        total_ram = 0
        for layer in self.layers:
            layer.ascend_to_warm()
            total_ram += layer.ram_footprint
            if layer.layer_id % 5 == 0:
                print(f"  Loaded Layer {layer.layer_id} | Total RAM: {total_ram/1e9:.2f} GB", flush=True)
        print("✅ All Layers Resident in RAM.", flush=True)

        self.vocab_size = self.text_config.get("vocab_size", 262144)
        norm_shard_name = weight_map.get("language_model.model.norm.weight")
        self.mouth = LazyMouth(self.hidden_size, self.vocab_size, os.path.join(path_12b, norm_shard_name), self.text_config)
        self.bridge = SwarmBridge(path_12b, self.swarm, weight_map=weight_map)

    def forward(self, prompt: str, group_size: int = 4):
        # 1. Swarm Dreams
        draft_text = self.swarm.dream(prompt)
        print(f"\n🐝 Swarm Dream: {draft_text}", flush=True)
        
        # 2. Transduction
        hidden = self.bridge.transduce(draft_text)
        print(f"🌉 Bridge Transmitted: {hidden.shape} to Cortex VRAM", flush=True)
        
        # 3. Holographic Resonance (Memory)
        seeds = [abs(hash(w)) % 100000 for w in draft_text.split()]
        signal = hidden.mean(dim=1).detach().float().cpu().numpy()
        resonance = self.lattice.stimulate(seeds, signal)
        print(f"🕸️ Holographic Resonance: {resonance:.4f} (Active Nodes: {len(self.lattice.active_tissue)})", flush=True)
        
        # 4. Cortex Verifies
        start_time = time.time()
        with torch.no_grad():
            for g_start in range(0, len(self.layers), group_size):
                g_end = min(g_start + group_size, len(self.layers))
                
                # Materialize Chunk
                for i in range(g_start, g_end): self.layers[i].materialize()
                
                # Compute Chunk
                for i in range(g_start, g_end): hidden = self.layers[i](hidden)
                
                # Cleanup Chunk
                for i in range(g_start, g_end): self.layers[i].reset_reality()

                vram_peak = torch.cuda.memory_allocated()/1e9
                print(f"\r  ▶ Group {g_start}-{g_end} | VRAM: {vram_peak:.2f} GB", end="", flush=True)
            
            total_time = time.time() - start_time
            print(f"\n⏱️ Cortex Forward Pass: {total_time:.2f}s", flush=True)
            logits = self.mouth(hidden)
            print(f"✅ Cortex Reality Shape: {logits.shape}", flush=True)
            return logits
