
import requests
import json
import time
from rich.console import Console

BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

def quantum_dream():
    console = Console()
    console.print("\n[bold violet]⚛️ QUANTUM DREAM SYNTHESIS (Physics + Agency)[/]")
    console.print("[dim]The Reality Architect is given the Agency Tool (>>NEXT).[/]")
    
    # Complex, deep scientific prompt
    base_prompt = (
        "You are the Reality Architect (Gemma-Physics). You adhere to first principles.\n"
        "Hypothesis: Consciousness is a localized minimization of Quantum Entropy.\n"
        "Task: Derive a theoretical mechanism for this. Use mathematical or physical terminology.\n"
        "Constraint: You MUST break this down into 3 distinct stages. Do not output all stages at once.\n"
        "Output Stage 1 (Foundations) then end with '>>NEXT'. Do not write Stage 2 yet.\n"
        "Start:"
    )
    
    context = ""
    max_iterations = 6 # Allow deep thought
    
    for i in range(max_iterations):
        console.print(f"\n[bold blue]Step {i+1}[/]")
        
        # Build prompt with sliding context
        current_input = f"{base_prompt}\n\nDerivation So Far:\n{context[-3000:]}\n\nContinue Proof:"
        
        payload = {
            "action": "telepathy",
            "prompt": current_input,
            "max_tokens": 200,          
            "temperature": 0.3,         # Low Temp (Physics requires Rigor, not hallucination)
            "stop": ["<end_of_turn>"],  # Enforce stops
            "intent": {
                "operation": "infer", 
                "purpose": "physics"    # Activate Gemma-Physics
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
                    break

                console.print(f"[cyan]{text}[/]")
                
                # Check for Autonomous Signal
                if ">>NEXT" in text:
                    console.print("[bold green]⚡ ARCHITECT REQUESTS CONTINUATION.[/]")
                    context += " " + text.replace(">>NEXT", "").strip()
                else:
                    console.print("[dim]Derivation Concluded.[/]")
                    context += " " + text
                    break
            else:
                console.print(f"[red]Error: {response.text}[/]")
                break
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/]")
            break
            
    console.print("\n[bold violet]⚛️ Synthesis Complete.[/]")

if __name__ == "__main__":
    quantum_dream()
