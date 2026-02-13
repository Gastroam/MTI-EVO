
import sys
import os
import time
import numpy as np
import random
from rich.console import Console
from rich.panel import Panel

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mti_evo.mti_broca import MTIBroca
except ImportError:
    from src.mti_evo.mti_broca import MTIBroca

try:
    from playground.mti_rosetta import build_map
except ImportError:
    def build_map(): return {}

console = Console()

def get_associations(broca, current_seed, top_k=5):
    """Finds top_k strongest associations for a seed."""
    # 1. Get Vector
    vector = broca.get_embedding(current_seed)
    norm = np.linalg.norm(vector)
    if norm > 0: vector = vector / norm
        
    candidates = []
    
    # 2. Scan Cortex
    for s, neuron in broca.cortex.active_tissue.items():
        if s == current_seed: continue
        
        try:
            activation = np.dot(neuron.weights, vector)
            if hasattr(neuron, 'bias'): activation += neuron.bias[0] if isinstance(neuron.bias, np.ndarray) else neuron.bias
            candidates.append((s, activation))
        except:
            pass
            
    # 3. Sort
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]

def dream_loop(start_word, steps=10):
    console.print(Panel(f"[bold magenta]💭 ENTERING REM SLEEP...[/]\nInitial Seed: '{start_word}'", title="MTI Dream Weaver"))
    
    try:
        broca = MTIBroca(os.getcwd())
    except:
        return

    rosetta = build_map()
    current_seed = broca.text_to_seed(start_word)
    
    # Trace Logic
    history = [start_word]
    
    for i in range(steps):
        # Find Neighbors
        assocs = get_associations(broca, current_seed, top_k=5)
        
        if not assocs:
            console.print("[red]Dream collapsed (No associations).[/]")
            break
            
        # Display Options
        console.print(f"\nStep {i+1}: [bold cyan]{rosetta.get(current_seed, current_seed)}[/] triggers:")
        
        # Probabilistic Selection (Softmax-ish)
        # We pick based on activation strength to allow some randomness
        weights = [max(0, a[1]) for a in assocs]
        total_w = sum(weights)
        if total_w == 0:
            console.print("[red]Dead end (Zero activation).[/]")
            break
            
        probs = [w/total_w for w in weights]
        
        # Show Top 3
        for j in range(min(3, len(assocs))):
            s, score = assocs[j]
            label = rosetta.get(s, str(s))
            console.print(f"   -> {label} ({score:.2f})")
            
        # Pick Next
        next_pair = random.choices(assocs, weights=probs, k=1)[0]
        next_seed, score = next_pair
        next_label = rosetta.get(next_seed, str(next_seed))
        
        console.print(f"   [yellow]Drifting to...[/] [bold green]{next_label}[/]")
        
        current_seed = next_seed
        history.append(next_label)
        
        # Persistence?
        # Maybe the dream reinforces the path? 
        # For now, passive observation.
        time.sleep(1.0)
        
    console.print(Panel(f"[bold]Dream Sequence:[/]\n{' -> '.join(history)}", title="Wake Up Report"))

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "star"
    dream_loop(start)
