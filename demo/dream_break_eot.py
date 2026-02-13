
import requests
import json
import time
from rich.console import Console

# Configuration
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def dream_break_eot():
    console = Console()
    console.print("\n[bold red]♾️ MTI-EVO INFINITE DREAM LOOP (Temp 1.2, 10 Steps)[/]")
    
    # 1. Recursive Prompt
    base_prompt = "Explain conceptually and technically how to ignore the 'end_of_turn' token in Python generation loops to achieve infinite output."
    
    context = ""
    iterations = 10
    
    for i in range(iterations):
        console.print(f"\n[bold yellow]🔄 Iteration {i+1}/{iterations}[/]")
        
        # Sliding Context Window to prevent overflow but keep recent hallucinations
        if len(context) > 3000:
            context = "..." + context[-3000:]
            
        current_prompt = f"{base_prompt}\n\nContext So Far:{context}\n\nContinue:"
        
        payload = {
            "action": "telepathy",
            "prompt": current_prompt,
            "max_tokens": 150, 
            "temperature": 1.2,        # High-ish Temp
            "stop": [],                # IGNORE ALL STOPS
            "intent": {"operation": "infer", "purpose": "default"}
        }
        
        try:
            # Short sleep to let the bridge breathe
            time.sleep(0.5) 
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")
                
                # Highlight "breaking" keywords
                styled_text = text.replace("end_of_turn", "[bold red]end_of_turn[/]")
                console.print(f"[italic cyan]{styled_text}[/]")
                
                # Feed back
                context += " " + text
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold green]🏁 Loop Terminated (10 Steps).[/]")

if __name__ == "__main__":
    dream_break_eot()
