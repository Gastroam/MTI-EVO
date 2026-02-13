
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
    vector = broca.get_embedding(current_seed)
    norm = np.linalg.norm(vector)
    if norm > 0: vector = vector / norm
        
    candidates = []
    
    for s, neuron in broca.cortex.active_tissue.items():
        if s == current_seed: continue
        
        try:
            activation = np.dot(neuron.weights, vector)
            if hasattr(neuron, 'bias'): activation += neuron.bias[0] if isinstance(neuron.bias, np.ndarray) else neuron.bias
            candidates.append((s, activation))
        except:
            pass
            
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]

def generate_mermaid(start_word, steps=10):
    try:
        broca = MTIBroca(os.getcwd())
    except:
        return

    rosetta = build_map()
    current_seed = broca.text_to_seed(start_word)
    
    # Check if start word exists
    if current_seed not in broca.cortex.active_tissue:
        console.print(f"[red]Seed '{start_word}' not found in memory![/]")
        return

    edges = []
    nodes = set()
    
    # Add start node
    start_label = rosetta.get(current_seed, str(current_seed)).replace(":", "_")
    nodes.add(f'{current_seed}["{start_label}"]')
    
    path_history = [start_label]
    
    console.print(f"[cyan]Dreaming from '{start_word}'...[/]")

    for i in range(steps):
        assocs = get_associations(broca, current_seed, top_k=5)
        
        if not assocs:
            break
            
        weights = [max(0, a[1]) for a in assocs]
        total_w = sum(weights)
        if total_w == 0: break
            
        probs = [w/total_w for w in weights]
        
        # Pick Next
        next_pair = random.choices(assocs, weights=probs, k=1)[0]
        next_seed, score = next_pair
        next_label = rosetta.get(next_seed, str(next_seed)).replace(":", "_")
        
        nodes.add(f'{next_seed}["{next_label}"]')
        edges.append(f'{current_seed} -->|"{score:.2f}"| {next_seed}')
        
        current_seed = next_seed
        path_history.append(next_label)
        
    # Generate Mermaid
    mermaid = "graph LR\n"
    for n in nodes:
        mermaid += f"    {n}\n"
    for e in edges:
        mermaid += f"    {e}\n"
        
    # Apply style
    mermaid += "    classDef default fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:white;\n"
    
    console.print(Panel(mermaid, title="Mermaid Dream Graph", subtitle="Copy this into a Mermaid viewer"))
    console.print(f"[dim]Path: {' -> '.join(path_history)}[/]")

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "sun"
    generate_mermaid(start)
