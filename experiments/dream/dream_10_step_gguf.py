
import requests
import json
import time
import os
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"
LOAD_URL = "http://localhost:8800/api/model/load" # Still need main server for loading? No, Bridge uses shared LLM.
# Wait, `telepathy_bridge.py` initializes its OWN `llm = LLMAdapter()`.
# If `server.py` loads the model, `LLMAdapter` singleton (if it works that way) might share it?
# The code says `llm = LLMAdapter() # Uses Singleton/Shared Memory`.
# But they are separate PROCESSES. They can't share memory unless `LLMAdapter` uses shared memory (e.g. mmap or shared var in same process).
# They are separate scripts `python server.py` and `python telepathy_bridge.py`. They will have SEPARATE models in memory.
# THIS IS A PROBLEM. Running both = 2x VRAM.
# The user said "use bridge.py".
# Maybe we ONLY run `telepathy_bridge.py`?
# But `telepathy_bridge` has no `/api/model/load` endpoint.
# It initializes `llm` on startup.
# We need to configure `mti_evo/config.json` (or wherever config is) to point to the GGUF model BEFORE checking `telepathy_bridge`.
# OR we rely on `LLMAdapter` reading `config.json`.
# Let's assume we run `telepathy_bridge.py` and it loads the model from config.
# I need to set the config first.
# Actually, the user just said "use bridge.py".
# I'll update the script to target 8767.

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"
JOURNAL_FILE = "dream_journal_russell_10_gguf.json"

def load_bridge_model():
    # Bridge doesn't have load endpoint. We assume it loads on start.
    # We can skip this or use a config tool.
    pass 

def bridge_infer(prompt, temp=0.8):
    """Sends prompt to the Telepathy Bridge."""
    # Llama 3 Instruct Format
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a high-entropy creative intelligence solving logical paradoxes.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    payload = {
        "action": "telepathy",
        "prompt": formatted_prompt,
        "max_tokens": 512,
        "temperature": temp
    }
    try:
        resp = requests.post(BRIDGE_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["response"]
        else:
            return f"[ERROR {resp.status_code}]"
    except Exception as e:
        return f"[ERROR {e}]"

def run_cycle():
    load_bridge_model()
    
    history = []
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, 'r') as f:
                history = json.load(f)
        except: pass

    # Start fresh or continue? Let's reset for this specific GGUF run
    history = [] 
    
    context = "Problem: Russell's Paradox. Contains(R, R) <-> !Contains(R, R)."

    for cycle in range(1, 11):
        console.print(f"\n[bold magenta]=== CYCLE {cycle}/10 (GGUF 8B) ===[/]")
        
        # 1. DREAM (High Temp)
        console.print("[dim]Phase 1: Dreaming...[/]")
        prompt_dream = f"Context: {context}\nGoal: Radical solution to Russell's Paradox.\nDream:"
        dream = bridge_infer(prompt_dream, temp=1.3)
        console.print(Panel(dream, title=f"Dream {cycle}", border_style="cyan"))
        
        # 2. CRITIQUE (Low Temp)
        console.print("[dim]Phase 2: Critiquing...[/]")
        prompt_critique = f"Proposition: {dream}\nAnalysis: Is this logically consistent? Does it solve the paradox? Critique:"
        critique = bridge_infer(prompt_critique, temp=0.4) 
        console.print(Panel(critique, title=f"Critique {cycle}", border_style="yellow"))
        
        # 3. SYNTHESIS (Med Temp)
        console.print("[dim]Phase 3: Synthesizing...[/]")
        prompt_synthesis = f"Dream: {dream}\nCritique: {critique}\nTask: Formulate a new research direction or axiom.\nSynthesis:"
        synthesis = bridge_infer(prompt_synthesis, temp=0.8)
        console.print(Panel(synthesis, title=f"Synthesis {cycle}", border_style="green"))
        
        # Update History & Context
        step_data = {
            "cycle": cycle,
            "dream": dream,
            "critique": critique,
            "synthesis": synthesis
        }
        history.append(step_data)
        context = f"Previous Insight: {synthesis}. Current Problem: Refine this logic."
        
        # Save
        with open(JOURNAL_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
        time.sleep(0.5) 

if __name__ == "__main__":
    run_cycle()
