import torch
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

# Add playground to path
sys.path.append(os.path.dirname(__file__))

# Import the 12B specific loader
from real_quantum_layer_12b import QuantumModel

console = Console()

# USER PROVIDED PATH
MODEL_PATH = r"H:\models\gemma-3-12B"

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def test_quantum_loading_12b():
    console.print(Panel("[bold cyan]🧪 Test: Quantum Loading Gemma-12B[/]"))
    
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model path not found: {MODEL_PATH}[/]")
        # Fallback to check if drive H exists or user just gave a hint
        return

    console.print(f"Model Path: {MODEL_PATH}")
    
    # 1. Initialize Loader
    console.print("\n[bold yellow]1. Initializing Quantum Model (Indexer)...[/]")
    try:
        model = QuantumModel(MODEL_PATH)
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize index: {e}[/]")
        return
    
    base_vram = get_vram_usage()
    console.print(f"Baseline VRAM: {base_vram:.2f} MB")
    
    # 2. Instantiate Quantum Layer 10
    console.print("\n[bold yellow]2. Instantiating Quantum Layer 10 (Virtual)...[/]")
    # 12B has 42 layers usually, so 10 is safe
    layer_10 = model.get_layer(10)
    console.print(f"Layer 10 Object Created. VRAM: {get_vram_usage():.2f} MB")
    
    # 3. Observe (Forward Pass)
    console.print("\n[bold yellow]3. Triggering Observation (Forward Pass)...[/]")
    start_time = time.time()
    
    output = layer_10.forward("dummy_input")
    
    load_time = time.time() - start_time
    peak_vram = get_vram_usage()
    console.print(f"Layer 10 Loaded in {load_time:.2f}s")
    console.print(f"Peak VRAM (Resident): {peak_vram:.2f} MB")
    console.print(f"VRAM Delta: {peak_vram - base_vram:.2f} MB")
    
    if peak_vram - base_vram > 50:
        console.print("✅ VRAM Spike Detected (Weights Loaded).")
    else:
        console.print("❌ No significant VRAM Spike.")

    # 4. Reset Reality
    console.print("\n[bold blue]4. Resetting Reality (Unload)...[/]")
    layer_10.reset_reality()
    
    final_vram = get_vram_usage()
    console.print(f"Final VRAM: {final_vram:.2f} MB")
    
    if final_vram < peak_vram:
        console.print("✅ VRAM Released.")
    else:
        console.print("❌ VRAM not released.")

if __name__ == "__main__":
    test_quantum_loading_12b()
