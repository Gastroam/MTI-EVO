import torch
import sys
import os
import time
import psutil
from rich.console import Console
from rich.panel import Panel

# Add playground to path
sys.path.append(os.path.dirname(__file__))

from real_quantum_layer import QuantumModel

console = Console()

MODEL_PATH = r"D:\VMTIDE\MTI-EVO\models\gemma-3-27b"

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def test_tier_integration():
    console.print(Panel("[bold cyan]🧪 Test: Quantum Integration (Tiered Memory)[/]"))
    
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model path not found: {MODEL_PATH}[/]")
        return
        
    # Initialize Configuration
    model = QuantumModel(MODEL_PATH)
    
    # ---------------------------------------------------------
    # TEST 1: COLD START (SSD -> GPU)
    # ---------------------------------------------------------
    console.print(Panel("\n[bold yellow]❄️  Test 1: Cold Start (SSD -> GPU)[/]"))
    layer_cold = model.get_layer(5) # Layer 5
    
    start_time = time.time()
    layer_cold.forward("cold_input") # Trigger load
    end_time = time.time()
    
    cold_duration = end_time - start_time
    console.print(f"Cold Validated in: [bold red]{cold_duration:.4f}s[/]")
    
    layer_cold.reset_reality() # Free VRAM
    
    # ---------------------------------------------------------
    # TEST 2: PREFETCH (SSD -> RAM)
    # ---------------------------------------------------------
    console.print(Panel("\n[bold yellow]🚀 Test 2: Prefetch (SSD -> RAM)[/]"))
    
    ram_before = get_process_memory()
    console.print(f"RAM Before Prefetch: {ram_before:.2f} MB")
    
    start_prefetch = time.time()
    # Explicitly prefetch Layer 10
    model.loader.prefetch_layer(10)
    end_prefetch = time.time()
    
    ram_after = get_process_memory()
    console.print(f"Prefetch Duration: {end_prefetch - start_prefetch:.4f}s")
    console.print(f"RAM After Prefetch: {ram_after:.2f} MB")
    console.print(f"RAM Delta: +{ram_after - ram_before:.2f} MB (Cached Weights)")
    
    # ---------------------------------------------------------
    # TEST 3: WARM START (RAM -> GPU)
    # ---------------------------------------------------------
    console.print(Panel("\n[bold yellow]🔥 Test 3: Warm Start (RAM -> GPU)[/]"))
    
    layer_warm = model.get_layer(10)
    
    start_time = time.time()
    layer_warm.forward("warm_input") # Should hit RAM cache
    end_time = time.time()
    
    warm_duration = end_time - start_time
    console.print(f"Warm Validated in: [bold green]{warm_duration:.4f}s[/]")
    
    # ---------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------
    console.print(Panel("\n[bold cyan]📊 Memory Tier Analysis[/]"))
    speedup = cold_duration / warm_duration if warm_duration > 0 else 999
    
    console.print(f"SSD Fetch Latency: {cold_duration:.4f}s")
    console.print(f"RAM Fetch Latency: {warm_duration:.4f}s")
    console.print(f"Speedup Factor: [bold magenta]{speedup:.2f}x[/]")
    
    if speedup > 2.0:
        console.print("✅ Tier 2 Caching SUCCESS.")
    else:
        console.print("❌ Tier 2 Caching INEFFECTIVE (Check bottleneck).")

    # Cleanup
    model.loader.evict_layer(10)
    console.print("Cache evicted.")

if __name__ == "__main__":
    test_tier_integration()
