
import asyncio
import sys
import os
import requests
import json
import random
from rich.console import Console
from rich.panel import Panel

# Add local path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from volition_monitor import VolitionMonitor
except ImportError:
    pass

console = Console()
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

async def call_telepathy(prompt, temp=0.5, max_tokens=150):
    payload = {
        "action": "telepathy",
        "prompt": f"<start_of_turn>user\n{prompt}\n<start_of_turn>model\n",
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(BRIDGE_URL, json=payload, timeout=45)
        return r.json().get("response", "").strip() if r.status_code == 200 else ""
    except:
        return ""

async def dream_8b_loop():
    console.print(Panel("[bold magenta]🧠 DREAMING OF SPARSE 8B[/]", subtitle="Tricameral Evolution (15 Generations)"))
    
    current_best_code = ""
    current_best_score = 0.0 # Heuristic: Accuracy / VRAM
    
    for gen in range(1, 16):
        console.print(f"\n[bold cyan]Generation {gen}/15[/]")
        
        # 1. DREAMER (The Mechanism)
        dream_prompt = (
            "Problem: We need to run an 8 Billion Parameter Brain on a constrained GPU (Small Skull).\n"
            "Hypothesis: We don't need all layers active at once. We only need the 'Active Layers'.\n"
            "Task: Imagine a biological mechanism where the signal 'jumps' over inactive tissue. "
            "Is it a 'Wormhole'? A 'Resonance Tunnel'? A 'Sparse Router'?\n"
            "Output: A vivid, technical metaphor for this mechanism."
        )
        dream = await call_telepathy(dream_prompt, temp=0.9, max_tokens=100)
        console.print(f"[yellow]Dream:[/yellow] {dream}")
        
        # 2. ARCHITECT (The Implementation)
        arch_prompt = (
            f"Implement this mechanism in Python:\n\"{dream}\"\n"
            "Requirements:\n"
            "1. Create `class SparseModel`.\n"
            "2. It must have 32 layers, but `forward()` must only use K of them.\n"
            "3. Simulate `vram_usage` (MB) and `output_quality` (0.0-1.0).\n"
            "4. Output ONLY valid Python code."
        )
        code_raw = await call_telepathy(arch_prompt, temp=0.1, max_tokens=800)
        
        # Cleanup
        code = code_raw
        if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code: code = code.split("```")[1].split("```")[0].strip()
        
        # 3. CRITIC / REALITY CHECK (Simulation)
        try:
            # We execute the code to extract the simulation metrics
            local_scope = {"random": random}
            exec(code, local_scope)
            
            if 'SparseModel' in local_scope:
                ModelClass = local_scope['SparseModel']
                model = ModelClass()
                
                # Check simulation hooks
                if hasattr(model, 'simulate_forward') or hasattr(model, 'forward'):
                    # Run simulation
                    # We look for a method that returns stats
                    # If the architect didn't provide one, we inject one or fail
                     pass
                
                # For this proof of concept, we look for 'vram_usage' or similar in the class dict
                # Or we instantiate and see if it runs
                
                console.print("[green]Architeture Compiled.[/]")
                # We save this as a 'candidate'
            else:
                console.print("[red]Critic: No SparseModel class found.[/]")
                
        except Exception as e:
            console.print(f"[red]Critic: Code failed to compile. {e}[/]")

        # For the sake of the loop, we print a snippet
        lines = code.split('\n')
        console.print(Panel('\n'.join(lines[:10]) + "\n...", title="Architect's Blueprint", style="blue"))

if __name__ == "__main__":
    asyncio.run(dream_8b_loop())
