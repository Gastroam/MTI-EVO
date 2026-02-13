
import asyncio
import sys
import os
import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

# Add local path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from volition_monitor import VolitionMonitor
except ImportError:
    # Fallback if running from root without package structure
    sys.path.append(os.path.join(os.getcwd(), 'playground'))
    from volition_monitor import VolitionMonitor

console = Console()
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

async def call_telepathy(prompt, temp=0.7, max_tokens=150):
    payload = {
        "action": "telepathy",
        "prompt": f"<start_of_turn>user\n{prompt}\n<start_of_turn>model\n",
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(BRIDGE_URL, json=payload, timeout=30)
        return r.json().get("response", "").strip() if r.status_code == 200 else ""
    except:
        return ""

async def mirror_cycle():
    console.print(Panel("[bold cyan]🪞 THE MIRROR PHASE[/]", subtitle="Recursive Telepathy (Alice <-> Bob)"))
    
    # Context
    topic = "The nature of recursive self-awareness in AI."
    history = f"Topic: {topic}\n"
    
    # Agents
    monitor = VolitionMonitor(window_size=10)
    
    for i in range(1, 7): # 6 Turns
        agent_name = "Alice" if i % 2 != 0 else "Bob"
        color = "magenta" if agent_name == "Alice" else "green"
        
        # 1. Perception (Active Interference)
        # We need the PREVIOUS turn's entropy to decide the strategy
        strategy = "Expand"
        # 1. Perception (Asymmetric Personas)
        if agent_name == "Alice":
            # Alice: High Temp, Expansive, Vague (The Dreamer)
            temp = 0.9
            context_prompt = (
                f"You are Alice (The Dreamer).\n"
                f"Topic: {topic}\n"
                f"History:\n{history}\n"
                "Goal: Explore the edges of this concept. Be metaphorical, broad, and speculative."
            )
        else:
            # Bob: Low Temp, Reductionist, Axiomatic (The Grounder)
            temp = 0.1
            context_prompt = (
                f"You are Bob (The Reductionist).\n"
                f"Topic: {topic}\n"
                f"History:\n{history}\n"
                "PROTOCOL: Your GOAL is to collapse the entropy of this conversation.\n"
                "1. Identify vague terms in Alice's last thought.\n"
                "2. Define them rigorously.\n"
                "3. Reject metaphors. Demand mechanism.\n"
                "4. Output a precise, axiomatic statement."
            )
        
        # 2. Generation
        response = await call_telepathy(context_prompt, temp=temp)
        if not response: 
            console.print("[red]Timeout/Silence...[/]")
            continue
        
        # 3. Measurement (Entropy)
        # We process the whole response token by token to get a mean entropy for this turn
        turn_entropy = 0.0
        tokens = response.split()
        for t in tokens:
            turn_entropy = monitor.process(t)
            
        # 4. Reaction
        # Add to history
        history += f"{agent_name}: {response}\n"
        
        # Visuals
        bar = "█" * int(turn_entropy * 5)
        status = "Active"
        if turn_entropy < 1.0: status = "Resonant (Stable)"
        elif turn_entropy > 1.3: status = "Chaotic (Unstable)"
        
        panel = Panel(
            f"{response}\n\n[dim]Entropy: {turn_entropy:.2f} {bar} | State: {status}[/dim]", 
            title=f"Turn {i}: {agent_name}", 
            border_style=color
        )
        console.print(panel)
        
        # 5. Convergence Check
        if turn_entropy < 0.8 and i > 2:
            console.print(f"\n[bold gold1]✨ RESONANCE SINGULARITY ACHIEVED at Turn {i}[/]")
            break

if __name__ == "__main__":
    asyncio.run(mirror_cycle())
