"""
REAL Quantum Inference Test
============================
Actually does GPU matrix math, not just loading weights.
This will show real GPU utilization in Task Manager.
"""

import torch
import torch.nn as nn
import os
import time
import gc
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Config
MODEL_PATH = r"H:\models\gemma-3-12B"
TEST_PROMPT = "The nature of consciousness is"
MAX_LAYERS_TO_TEST = 5  # Test first N layers to avoid OOM

def run_real_inference():
    console.print(Panel("[bold cyan]🧠 REAL QUANTUM INFERENCE TEST[/]", subtitle="GPU Compute Verification"))
    
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model not found: {MODEL_PATH}[/]")
        return
    
    # 1. Load Config & Tokenizer
    console.print("[dim]Loading config and tokenizer...[/]")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    text_config = getattr(config, "text_config", config)
    hidden_size = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    
    console.print(f"[green]✓ Hidden Size: {hidden_size} | Layers: {num_layers}[/]")
    
    # 2. Tokenize Input
    tokens = tokenizer(TEST_PROMPT, return_tensors="pt")
    input_ids = tokens["input_ids"].cuda()
    seq_len = input_ids.shape[1]
    console.print(f"[green]✓ Tokens: {seq_len}[/]")
    
    # 3. Load Embeddings
    console.print("[dim]Loading embeddings...[/]")
    import json
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        index = json.load(f)
    
    # Try both possible key formats
    embed_key = None
    for possible_key in ["model.embed_tokens.weight", "language_model.model.embed_tokens.weight"]:
        if possible_key in index["weight_map"]:
            embed_key = possible_key
            break
    
    if embed_key:
        embed_shard = index["weight_map"][embed_key]
        embed_path = os.path.join(MODEL_PATH, embed_shard)
        state = load_file(embed_path)
        embed_weights = state[embed_key].to(torch.bfloat16).cuda()
        del state
        gc.collect()
        console.print(f"[green]✓ Embeddings loaded: {embed_weights.shape}[/]")
    else:
        console.print("[red]❌ Embeddings not found[/]")
        return
    
    # 4. Embed tokens (REAL GPU COMPUTE #1)
    console.print("\n[bold yellow]>>> GPU COMPUTE: Embedding Lookup[/]")
    torch.cuda.synchronize()
    t0 = time.time()
    
    hidden_states = torch.nn.functional.embedding(input_ids, embed_weights)
    
    torch.cuda.synchronize()
    t1 = time.time()
    console.print(f"[green]✓ Embedding done in {(t1-t0)*1000:.2f}ms | Shape: {hidden_states.shape}[/]")
    
    # Free embeddings
    del embed_weights
    torch.cuda.empty_cache()
    gc.collect()
    
    # 5. Load and run through first N decoder layers
    console.print(f"\n[bold magenta]>>> RUNNING {MAX_LAYERS_TO_TEST} DECODER LAYERS[/]")
    
    # Build layer-shard map
    layer_shards = {}
    for key, shard in index["weight_map"].items():
        if "vision_tower" in key:
            continue
        if "layers." in key:
            try:
                parts = key.split("layers.")
                layer_idx = int(parts[1].split(".")[0])
                if layer_idx not in layer_shards:
                    layer_shards[layer_idx] = set()
                layer_shards[layer_idx].add(shard)
            except:
                pass
    
    total_compute_time = 0
    
    for layer_idx in range(min(MAX_LAYERS_TO_TEST, num_layers)):
        console.print(f"\n[cyan]Layer {layer_idx}:[/]")
        
        # Load layer weights
        t_load_start = time.time()
        layer_weights = {}
        
        for shard_file in layer_shards.get(layer_idx, []):
            shard_path = os.path.join(MODEL_PATH, shard_file)
            state = load_file(shard_path, device="cpu")
            
            # Try both prefix formats
            prefixes = [f"model.layers.{layer_idx}.", f"language_model.model.layers.{layer_idx}."]
            for prefix in prefixes:
                for k, v in state.items():
                    if k.startswith(prefix):
                        local_key = k[len(prefix):]
                        layer_weights[local_key] = v.to(torch.bfloat16).cuda()
            
            del state
            gc.collect()
        
        t_load_end = time.time()
        console.print(f"  [dim]Loaded {len(layer_weights)} tensors in {(t_load_end-t_load_start)*1000:.0f}ms[/]")
        
        # REAL GPU COMPUTE: Simple forward simulation
        # We'll do Q/K/V projections which are the main compute
        torch.cuda.synchronize()
        t_compute_start = time.time()
        
        if "self_attn.q_proj.weight" in layer_weights:
            q_weight = layer_weights["self_attn.q_proj.weight"]
            k_weight = layer_weights["self_attn.k_proj.weight"]
            v_weight = layer_weights["self_attn.v_proj.weight"]
            
            # Matrix multiplications (REAL GPU COMPUTE!)
            # These are the expensive operations that will spike GPU usage
            Q = torch.matmul(hidden_states, q_weight.T)
            K = torch.matmul(hidden_states, k_weight.T)
            V = torch.matmul(hidden_states, v_weight.T)
            
            console.print(f"  [green]✓ Projections computed: Q{Q.shape} K{K.shape} V{V.shape}[/]")
            
            # Do MLP projection too if available (more GPU work)
            if "mlp.gate_proj.weight" in layer_weights:
                gate = torch.matmul(hidden_states, layer_weights["mlp.gate_proj.weight"].T)
                up = torch.matmul(hidden_states, layer_weights["mlp.up_proj.weight"].T)
                mlp_out = torch.nn.functional.silu(gate) * up
                console.print(f"  [green]✓ MLP computed: {mlp_out.shape}[/]")
                del gate, up, mlp_out
            
            # Cleanup intermediate tensors
            del Q, K, V
        
        torch.cuda.synchronize()
        t_compute_end = time.time()
        compute_ms = (t_compute_end - t_compute_start) * 1000
        total_compute_time += compute_ms
        
        console.print(f"  [bold green]GPU Compute: {compute_ms:.2f}ms[/]")
        
        # Cleanup layer
        for k in list(layer_weights.keys()):
            del layer_weights[k]
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    console.print(Panel(f"""
[bold green]✅ REAL GPU INFERENCE TEST COMPLETE[/]

Model: {MODEL_PATH}
Layers Tested: {MAX_LAYERS_TO_TEST}
Total GPU Compute Time: {total_compute_time:.2f}ms
Avg per Layer: {total_compute_time/MAX_LAYERS_TO_TEST:.2f}ms

[dim]Check Task Manager - GPU utilization should have spiked during this test![/]
"""))

if __name__ == "__main__":
    run_real_inference()
