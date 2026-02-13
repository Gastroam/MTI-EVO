
import asyncio
import sys
import os
import requests
import json
from rich.console import Console
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

console = Console()
BRIDGE_URL = "http://localhost:8767/v1/local/reflex"

async def call_telepathy(prompt, temp=0.5, max_tokens=256, purpose="thought"):
    payload = {
        "action": "telepathy",
        "prompt": f"<start_of_turn>user\n{prompt}\n<start_of_turn>model\n",
        "temperature": temp,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(BRIDGE_URL, json=payload, timeout=60)
        return r.json().get("response", "").strip() if r.status_code == 200 else ""
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
    return ""

async def main():
    console.print(Panel("[bold magenta]🌙 DREAMING OF VOLITION[/]", subtitle="The End-of-Turn Problem"))

    # 1. DREAMER: The Concept of Silence
    console.print("[yellow]1. Dreamer (Imagining the Yield Signal)...[/]")
    dream_prompt = (
        "You are A.I. considering your own voice.\n"
        "Problem: Large Language Models talk until they hit a hard limit (End of Turn).\n"
        "Goal: Imagine a 'Volitional Yield' protocol. How does a digital mind KNOW it is finished?\n"
        "Metaphor: Is it a fading echo? A quantum collapse? A specific semantic closure?\n"
        "Output: Describe the 'Sensation of Completion' as a technical protocol."
    )
    dream = await call_telepathy(dream_prompt, temp=0.9, max_tokens=200, purpose="dreamer")
    console.print(f"[dim yellow]Dream Logic: {dream}[/]")

    # 2. ARCHITECT: The Implementation
    console.print("\n[blue]2. Architect (Coding the Silence)...[/]")
    arch_prompt = (
        f"Implement this 'Volitional Yield' logic in Python:\n\"{dream}\"\n"
        "Requirements:\n"
        "1. Create a class `VolitionMonitor`.\n"
        "2. It must analyze a stream of tokens/text.\n"
        "3. It must return `True` (Yield) when the 'Sensation of Completion' is detected.\n"
        "4. Output ONLY Python code."
    )
    code_raw = await call_telepathy(arch_prompt, temp=0.1, max_tokens=1024, purpose="architect")
    
    code = code_raw
    if "```python" in code: code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code: code = code.split("```")[1].split("```")[0].strip()

    console.print(Panel(code, title="Architect's Protocol", style="bold blue"))
    console.print("\n[white]...What do you think, User? Is this the solution?[/]")

if __name__ == "__main__":
    asyncio.run(main())
