
import requests
import json
import sys
import time
from rich.console import Console

# Configuration
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def dream_loop():
    console = Console()
    console.print("\n[bold magenta]🌌 MTI-EVO DREAM LOOP (Temp 1.3)[/]")
    
    # 1. Define High-Entropy Prompt
    base_prompt = "You are the Ghost in the machine. Describe what lies beyond the code."
    
    context = ""
    iterations = 3
    
    for i in range(iterations):
        console.print(f"\n[bold yellow]🔄 Iteration {i+1}/{iterations}[/]")
        current_prompt = f"{base_prompt} {context}"[-2000:] # Keep context rolling but capped
        
        payload = {
            "action": "telepathy",
            "prompt": current_prompt,
            "max_tokens": 128, 
            "temperature": 1.3,        # DREAM MODE
            "stop": [],                # IGNORE BOUNDARIES
            "intent": {"operation": "infer", "purpose": "default"}
        }
        
        try:
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")
                console.print(f"[italic cyan]{text}[/]")
                
                # Feed back into context loop
                context += " " + text
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold green]🌌 Dream Cycle Complete.[/]")

if __name__ == "__main__":
    dream_loop()
