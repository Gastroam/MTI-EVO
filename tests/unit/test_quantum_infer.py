"""
QuantumEngine Text Generation Test
===================================
Tests actual inference through the quantum layer-by-layer system.
"""

import sys
import os
import time
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rich.console import Console
from rich.panel import Panel

console = Console()

MODEL_PATH = r"H:\models\gemma-3-12B"

def test_quantum_inference():
    console.print(Panel("[bold cyan]QUANTUM INFERENCE TEST[/]", subtitle="12B with layer-by-layer loading"))
    
    # Load via QuantumEngine
    from mti_evo.engines.quantum_engine import QuantumEngine
    
    config = {
        "model_path": MODEL_PATH,
        "temperature": 0.3
    }
    
    engine = QuantumEngine(config)
    engine.load_model()
    
    console.print(f"[green]Engine loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB[/]")
    
    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Explain quantum computing in one sentence.",
        "Hello, my name is"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        console.print(f"\n[bold yellow]Test {i}:[/] {prompt}")
        
        t0 = time.time()
        response = engine.infer(prompt, max_tokens=50)
        t1 = time.time()
        
        console.print(f"[green]Response:[/] {response.text}")
        console.print(f"[dim]Latency: {(t1-t0)*1000:.0f}ms | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB[/]")
    
    # Cleanup
    engine.unload()
    torch.cuda.empty_cache()
    
    console.print(Panel("[bold green]TEST COMPLETE[/]"))

if __name__ == "__main__":
    test_quantum_inference()
