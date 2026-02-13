
import requests
import json
import time
from rich.console import Console

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def paradox_loop():
    console = Console()
    console.print("\n[bold blue]⚛️ PHYSICS PARADOX LOOP (10 Steps, Temp 0.1)[/]")
    console.print("[dim]Subject: The Black Hole Information Paradox[/]")
    
    base_prompt = (
        "You are the Reality Architect (Gemma-Physics).\n"
        "Subject: The Black Hole Information Paradox.\n"
        "Conflict: General Relativity (No hair theorem) vs Quantum Mechanics (Unitarity).\n"
        "Goal: Attempt to resolve this paradox through iterative reasoning.\n"
    )
    
    context = ""
    iterations = 10
    
    for i in range(iterations):
        console.print(f"\n[bold yellow]Step {i+1}/{iterations}[/]")
        
        # Build prompt: Force it to build upon previous thoughts
        if i == 0:
            current_prompt = f"{base_prompt}\nStart your analysis. Define the core conflict."
        else:
            current_prompt = f"{base_prompt}\n\nPrevious Analysis:\n{context[-3000:]}\n\nTask: deeply analyze the implications of the previous step. Propose a mechanism (e.g. Holographic Principle, Firewalls, Soft Hair) to resolve the conflict. Continue the derivation:"

        # NO explicit temperature override -> Uses Profile Default (0.1)
        # NO explicit stop override -> Uses Standard Stops
        payload = {
            "action": "telepathy",
            "prompt": current_prompt,
            "max_tokens": 200, 
            "intent": {
                "operation": "infer", 
                "purpose": "physics"
            }
        }
        
        try:
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                expert = data.get("expert", "Unknown")
                
                if expert != "Gemma-Physics":
                    console.print(f"[bold red]❌ WRONG EXPERT: {expert}[/]")
                
                console.print(f"[cyan]{text}[/]")
                context += " " + text
                
                # Check for convergence
                if "QED" in text or "Resolution:" in text:
                     console.print("[bold green]✅ Potential Resolution Detected.[/]")
                     
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold blue]⚛️ Paradox Investigation Complete.[/]")

if __name__ == "__main__":
    paradox_loop()
