
import sys
import os
import requests
import json
from rich.console import Console
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from mti_evo.mti_broca import MTIBroca
except ImportError:
    try:
        from src.mti_evo.mti_broca import MTIBroca
    except:
        sys.exit(1)

try:
    from mti_rosetta import seed_to_text
except ImportError:
    def seed_to_text(s, _=None): return str(s)[:8]

console = Console()
BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def run_interview(target):
    console.print(f"[bold green]Interrogating:[/]\t{target}")
    
    broca = MTIBroca(os.getcwd())
    seed = broca.text_to_seed(target)
    
    # 1. Associations
    results = []
    # Dream Drift Logic
    embedding = broca.get_embedding(seed)
    for s_prime, neuron in broca.cortex.active_tissue.items():
         act = neuron.perceive(embedding)
         results.append((s_prime, act))
         
    results.sort(key=lambda x: x[1], reverse=True)
    top_5 = results[:5]
    
    concepts = [seed_to_text(s, {}) for s, _ in top_5]
    console.print(f"[dim]Associations: {', '.join(concepts)}[/]")
    
    # 2. Logic Probe
    prompt_text = (
        f"You are a friendly, analytical AI explaining your own thought process.\n"
        f"I asked you about '{target}'.\n"
        f"Your subconscious (Hebbian Memory) immediately associated it with: {concepts}.\n"
        f"Question: What is the LOGICAL connection between '{target}' and these concepts?\n"
        "If the connection seems random, explain why you might have made that leap (e.g. shared patterns, mathematical constants, or poetic drift).\n"
        "Keep it conversational and brief."
    )
    
    payload = {
        "action": "telepathy",
        "prompt": f"<start_of_turn>user\n{prompt_text}\n<start_of_turn>model\n",
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    r = requests.post(BRIDGE_URL, json=payload, timeout=60)
    if r.status_code == 200:
        answer = r.json().get("response", "")
        console.print(Panel(answer, title="🧠 The Logic", border_style="cyan"))
        
        # Save to file to avoid console truncation/encoding issues
        with open("interview_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Target: {target}\n")
            f.write(f"Associations: {', '.join(concepts)}\n")
            f.write("-" * 20 + "\n")
            f.write(answer)
    else:
        print(f"Error: {r.status_code}")

if __name__ == "__main__":
    run_interview("ouroboros")
