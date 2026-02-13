import torch
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

# Add playground to path
sys.path.append(os.path.dirname(__file__))

from real_quantum_layer import QuantumModel

console = Console()

MODEL_PATH = r"D:\VMTIDE\MTI-EVO\models\gemma-3-27b"

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def test_quantum_loading():
    console.print(Panel("[bold cyan]🧪 Test: Quantum Loading Gemma-27B[/]"))
    
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model path not found: {MODEL_PATH}[/]")
        return

    console.print(f"Model Path: {MODEL_PATH}")
    
    # 1. Initialize Loader
    console.print("\n[bold yellow]1. Initializing Quantum Model (Indexer)...[/]")
    # We prune layer 60 just to test the filter
    model = QuantumModel(MODEL_PATH, pruned_indices=[60])
    
    base_vram = get_vram_usage()
    console.print(f"Baseline VRAM: {base_vram:.2f} MB")
    
    # 2. Instantiate Quantum Layer 10
    console.print("\n[bold yellow]2. Instantiating Quantum Layer 10 (Virtual)...[/]")
    layer_10 = model.get_layer(10)
    console.print(f"Layer 10 Object Created. VRAM: {get_vram_usage():.2f} MB (Should be near baseline)")
    
    # 3. Observe (Forward Pass)
    console.print("\n[bold yellow]3. Triggering Observation (Forward Pass)...[/]")
    start_time = time.time()
    
    # This should trigger load from disk -> VRAM
    output = layer_10.forward("dummy_input")
    
    load_time = time.time() - start_time
    peak_vram = get_vram_usage()
    console.print(f"Layer 10 Loaded in {load_time:.2f}s")
    console.print(f"Peak VRAM (Resident): {peak_vram:.2f} MB")
    console.print(f"VRAM Delta: {peak_vram - base_vram:.2f} MB")
    
    if peak_vram - base_vram > 100:
        console.print("✅ VRAM Spike Detected (Weights Loaded).")
    else:
        console.print("❌ No VRAM Spike (Did load fail?)")

    # 4. Pruning Test
    console.print("\n[bold yellow]4. Testing Pruned Layer 60...[/]")
    layer_60 = model.get_layer(60)
    layer_60.forward("dummy")
    # VRAM should not increase significantly from peak
    current_vram = get_vram_usage()
    console.print(f"VRAM after Pruned Access: {current_vram:.2f} MB")
    
    # 5. Reset Reality
    console.print("\n[bold blue]5. Resetting Reality (Unload)...[/]")
    layer_10.reset_reality()
    
    final_vram = get_vram_usage()
    console.print(f"Final VRAM: {final_vram:.2f} MB")
    
    if final_vram < peak_vram:
        console.print("✅ VRAM Released (Reality Reset).")
    else:
        console.print("❌ VRAM Cheat! (Leak detected)")

if __name__ == "__main__":
    test_quantum_loading()
