
import requests
import json
import time
from rich.console import Console

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def math_chaos_loop():
    console = Console()
    console.print("\n[bold magenta]♾️ RAMANUJAN CHAOS LOOP (10 Steps, Temp 1.2)[/]")
    console.print("[dim]Subject: Russell's Paradox (Does the set of all sets contain itself?)[/]")
    
    base_prompt = (
        "You are Agni-Ramanujan (Gemma-Math). You see mathematical truth as divine structure.\n"
        "Subject: Russell's Paradox.\n"
        "Conflict: Naive Set Theory vs Self-Reference.\n"
        "Task: Attempt to resolve this paradox. You may use esoteric or non-standard logic.\n"
    )
    
    context = ""
    iterations = 10
    
    for i in range(iterations):
        console.print(f"\n[bold yellow]Step {i+1}/{iterations}[/]")
        
        if i == 0:
            current_prompt = f"{base_prompt}\nStart derivation:"
        else:
            current_prompt = f"{base_prompt}\n\nPrevious State:\n{context[-3000:]}\n\nTask: Push the logic further. Break axioms if necessary. Continue:"

        # HIGH TEMP (1.2) + MATH EXPERT
        payload = {
            "action": "telepathy",
            "prompt": current_prompt,
            "max_tokens": 150, 
            "temperature": 1.2,        # CHAOS MODE
            "intent": {
                "operation": "infer", 
                "purpose": "math"      # Activates Gemma-Math
            }
        }
        
        try:
            response = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "").strip()
                expert = data.get("expert", "Unknown")
                
                if expert != "Gemma-Math":
                    console.print(f"[bold red]❌ WRONG EXPERT: {expert}[/]")
                
                console.print(f"[cyan]{text}[/]")
                context += " " + text
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold magenta]♾️ Calculation Complete.[/]")

if __name__ == "__main__":
    math_chaos_loop()
