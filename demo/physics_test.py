
import requests
import json
import sys
from rich.console import Console

# Configuration
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def test_physics_expert():
    console = Console()
    console.print("\n[bold blue]⚛️ MTI-EVO PHYSICS EXPERT TEST[/]")
    
    # 1. Define Prompt targeting Physics Intent
    prompt = "Analyze the concept of 'Maxwell's Demon' in the context of information entropy."
    
    payload = {
        "action": "telepathy",
        "prompt": prompt,
        "max_tokens": 256,
        "intent": {
            "operation": "infer", # Required by IDRE
            "purpose": "physics", # Explicit intent
            "priority": "high"
        }
    }
    
    try:
        console.print(f"[dim]Sending Request to {BRIDGE_URL}...[/]")
        response = requests.post(BRIDGE_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            expert = data.get("expert", "Unknown")
            answer = data.get("response", "")
            
            console.print(f"\n[bold green]✅ RESPONSE RECEIVED[/]")
            console.print(f"Expert Used: [bold magenta]{expert}[/]")
            
            console.print("\n[white]Answer:[/]")
            console.print(f"[italic]{answer}[/]")
            
            if expert == "Gemma-Physics":
                console.print("\n[bold green]PASS: Correct Expert Activated.[/]")
                sys.exit(0)
            else:
                console.print(f"\n[bold red]FAIL: Wrong Expert ({expert}). Expected Gemma-Physics.[/]")
                sys.exit(1)
        else:
            console.print(f"[bold red]❌ Error {response.status_code}: {response.text}[/]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]❌ Connection Failed: {e}[/]")
        sys.exit(1)

if __name__ == "__main__":
    test_physics_expert()
