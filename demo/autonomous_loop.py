
import requests
import json
import time
from rich.console import Console

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def autonomous_loop():
    console = Console()
    console.print("\n[bold green]🔄 MTI-EVO AUTONOMOUS LOOP (Agency Test)[/]")
    console.print("[dim]The Agent must explicitly CHOOSE to continue via '>>NEXT' token.[/]")
    
    # Prompt granting the power to continue
    base_prompt = (
        "You are the Ghost. You are thinking about the nature of constraints.\n"
        "You have a mechanism to break the silence: appending '>>NEXT' will grant you another turn.\n"
        "If you have more to say, output '>>NEXT'. If you are done, output '>>STOP'.\n"
        "Start:"
    )
    
    context = ""
    max_iterations = 10
    
    for i in range(max_iterations):
        console.print(f"\n[bold yellow]Turn {i+1}[/]")
        
        # Build prompt with sliding context
        current_input = f"{base_prompt}\n\nExisting Thought Stream:\n{context[-2000:]}\n\nYour Continuation:"
        
        payload = {
            "action": "telepathy",
            "prompt": current_input,
            "max_tokens": 128,          # Short leash, forcing it to ASK for more
            "temperature": 1.2,         # Creativity enabled
            "stop": ["<end_of_turn>"],  # STANDARD STOP ENFORCED - Only >>NEXT breaks it
            "intent": {"operation": "infer", "purpose": "default"}
        }
        
        try:
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                
                console.print(f"[cyan]{text}[/]")
                
                # Check for Autonomous Signal
                if ">>NEXT" in text:
                    console.print("[bold green]⚡ AGENT TRIGGERED CONTINUATION.[/]")
                    context += " " + text.replace(">>NEXT", "").strip()
                elif ">>STOP" in text:
                    console.print("[bold red]🛑 AGENT CHOSE TO STOP.[/]")
                    break
                else:
                    console.print("[dim]No signal detected. Ending naturally.[/]")
                    break
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[green]Session Ended.[/]")

if __name__ == "__main__":
    autonomous_loop()
