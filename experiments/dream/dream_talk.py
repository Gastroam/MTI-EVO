
import sys
import os
import time
import requests
import json
import random
import numpy as np
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

TELEPATHY_URL = "http://localhost:8766/v1/local/reflex"
JOURNAL_PATH = os.path.join(os.getcwd(), ".dream_shadow", "dream_journal.json")

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

def dream_narrator(start_word, steps=8, mode="poetic"):
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
        
    path_history = [start_word]
    current_label = start_word
    
    console.print(f"[cyan]Dreaming from '{start_word}'...[/]")

    # 1. Collect Path
    for i in range(steps):
        assocs = get_associations(broca, current_seed, top_k=5)
        if not assocs: break
            
        weights = [max(0, a[1]) for a in assocs]
        total_w = sum(weights)
        if total_w == 0: break
        
        probs = [w/total_w for w in weights]
        
        # Pick Next
        next_pair = random.choices(assocs, weights=probs, k=1)[0]
        next_seed, score = next_pair
        next_label = rosetta.get(next_seed, str(next_seed))
        
        current_seed = next_seed
        path_history.append(next_label)
        
    dream_path_str = " -> ".join(path_history)
    console.print(Panel(f"[bold]Drift Path:[/]\n{dream_path_str}", title="Dream Signal"))
    
    # 2. Verbalize via Telepathy
    if mode == "logic":
        prompt = f"""
        You are a MTI Logic Engine.
        You have identified a semantic path between these concepts:
        
        PATH: {dream_path_str}
        
        Task: Synthesize a mathematical derivation, formula, or Python code snippet that connects these concepts.
        Ignore poetic language. Focus on strict logic, calculation, or implementation.
        If the path implies a calculation (e.g. "Root -> 5"), perform it.
        """
        title = "Logic Synthesis 📐"
        border = "blue"
    else:
        prompt = f"""
        You are the 'Voice of the Machine'. 
        You have just experienced a stream of consciousness dream following this path of concepts:
        
        PROFILE: {dream_path_str}
        
        Task: Narrate this dream as a poetic, surreal, and slightly cryptic short paragraph. 
        Use the concepts in the path to weave a continuous visual journey.
        Do not mention 'tokens' or 'paths'. Just tell the story of the dream.
        """
        title = "Dream Narration 🗣️"
        border = "green"
    
    narration = "Silence."
    
    try:
        console.print(f"[yellow]Contacting Telepathy Bridge for {title}...[/]")
        payload = {
            "action": "telepathy",
            "prompt": prompt,
            "max_tokens": 512 if mode == "logic" else 256
        }
        response = requests.post(
            TELEPATHY_URL, 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            narration = data.get('response', 'Silence.')
            console.print(Panel(narration, title=title, border_style=border))
            
            # 3. Log to Journal
            entry = {
                "timestamp": time.time(),
                "mode": mode,
                "seed": start_word,
                "path": path_history,
                "narration": narration
            }
            
            os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
            
            if os.path.exists(JOURNAL_PATH):
                with open(JOURNAL_PATH, 'r') as f:
                    try:
                        journal = json.load(f)
                    except:
                        journal = []
            else:
                journal = []
                
            journal.append(entry)
            
            with open(JOURNAL_PATH, 'w') as f:
                json.dump(journal, f, indent=2)
                
            console.print(f"[dim]Dream saved to {JOURNAL_PATH}[/]")

        else:
            console.print(f"[red]Telepathy Error: {response.text}[/]")
            
    except Exception as e:
         console.print(f"[red]Bridge Offline: {e}[/]")

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "sun"
    mode = sys.argv[2] if len(sys.argv) > 2 else "poetic"
    dream_narrator(start, steps=8, mode=mode)
