
import requests
import json
import time
from rich.console import Console

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def research_loop():
    console = Console()
    console.print("\n[bold green]🧪 GOLDILOCKS RESEARCH LOOP (10 Steps, Temp 0.9)[/]")
    console.print("[dim]Subject: Russell's Paradox. Goal: Novel but Rigorous Solutions.[/]")
    
    base_prompt = (
        "You are Agni-Ramanujan (Gemma-Math).\n"
        "Subject: Russell's Paradox (The set of all sets that do not contain themselves).\n"
        "Task: Propose a NOVEL mathematical framework to resolve this.\n"
        "Constraint: Avoid 'Sci-Fi' or 'Metaphysical' terms like 'Temporal Echoes'.\n"
        "Constraint: Use constructive logic, Category Theory, or Type Theory modifications.\n"
        "Goal: This is serious research. Be Creative but Rigorous.\n"
    )
    
    context = ""
    iterations = 10
    
    for i in range(iterations):
        console.print(f"\n[bold yellow]Step {i+1}/{iterations}[/]")
        
        if i == 0:
            current_prompt = f"{base_prompt}\nStart derivation:"
        else:
            current_prompt = f"{base_prompt}\n\nPrevious State:\n{context[-3000:]}\n\nTask: Refine this framework. Check for logical consistency. Continue:"

        # TEMP 0.9 = The "Research" Zone
        payload = {
            "action": "telepathy",
            "prompt": current_prompt,
            "max_tokens": 200, 
            "temperature": 0.9,        
            "intent": {
                "operation": "infer", 
                "purpose": "math" 
            }
        }
        
        try:
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                
                console.print(f"[cyan]{text}[/]")
                context += " " + text
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold green]🧪 Research Complete.[/]")

if __name__ == "__main__":
    research_loop()
