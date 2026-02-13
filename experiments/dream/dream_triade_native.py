
import asyncio
import sys
import os
import json
import random
import re
import math
import time
from decimal import Decimal, getcontext
from rich.console import Console
from rich.panel import Panel

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mti_evo.llm_adapter import LLMAdapter

console = Console()

# NATIVE CONFIGURATION
MODEL_PATH = r"H:\models\gemma-3-4b-unq"
DREAM_TEMP = 1.5  # High Entropy as requested
ARCH_TEMP = 0.2   # Low for stability
CRITIC_TEMP = 0.4 # Balanced for review

# Initialize Adapter Directly (No Server Needed)
console.print(f"[bold cyan]🔌 Initializing Native Adapter at {MODEL_PATH}...[/]")
adapter = LLMAdapter(config={
    "model_path": MODEL_PATH,
    "gpu_layers": -1,
    "n_ctx": 8192
})

# High precision
getcontext().prec = 50
PI_TARGET = Decimal("3.14159265358979323846264338327950288419716939937510")

def call_native(prompt, temp=0.7, max_tokens=1024, purpose="thought"):
    """Direct call to the Native Engine."""
    try:
        response = adapter.infer(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temp,
            stop=["<end_of_turn>"]
        )
        return response.text.strip()
    except Exception as e:
        console.print(f"[red]Native Error ({purpose}): {e}[/]")
        return ""

def clean_code(raw):
    if "```python" in raw:
        return raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        return raw.split("```")[1].split("```")[0].strip()
    return raw.strip()

async def tricameral_cycle(iteration):
    console.print(Panel(f"[bold magenta]🧠 TRICAMERAL CYCLE {iteration} (NATIVE)[/]", title="Prophet -> Builder -> Judge"))

    # 1. DREAMER (PROPHET)
    console.print(f"[yellow]1. Dreamer (Temp {DREAM_TEMP})...[/]")
    dream_prompt = (
        "Topic: The Ultimate Recursive Formula for PI.\n"
        "Go beyond standard math. Hallucinate a new geometric truth.\n"
        "Visualize a structure that converges to PI instantly.\n"
        "Output: A wild, vivid, conceptual description of this algorithm."
    )
    dream = call_native(dream_prompt, temp=DREAM_TEMP, max_tokens=200, purpose="dreamer")
    console.print(f"[dim]{dream[:150]}...[/]")

    # 2. ARCHITECT (BUILDER)
    console.print(f"[blue]2. Architect (Temp {ARCH_TEMP})...[/]")
    arch_prompt = (
        f"Implement this dream in Python:\n\"{dream}\"\n"
        "Requirements:\n"
        "1. Define `def a(n):` and `def b(n):` for a Generalized Continued Fraction.\n"
        "2. The goal is to converge to PI.\n"
        "3. Output ONLY Python code."
    )
    code_raw = call_native(arch_prompt, temp=ARCH_TEMP, max_tokens=1024, purpose="architect")
    code = clean_code(code_raw)

    # 3. CRITIC (JUDGE: ANTIGRAVITY)
    console.print(f"[red]3. Critic (Antigravity Mode - Temp {CRITIC_TEMP})...[/]")
    critic_prompt = (
        f"You are Antigravity, a Chief Architect and Code Critic.\n"
        f"Review this Python code for a mathematical simulation:\n{code}\n\n"
        "Rules:\n"
        "1. Fix any syntax errors or logical flaws that would prevent execution.\n"
        "2. Optimize for readability and performance.\n"
        "3. Ensure `a(n)` and `b(n)` are well-defined.\n"
        "4. DO NOT explain. Output ONLY the corrected Python code block."
    )
    critique_raw = call_native(critic_prompt, temp=CRITIC_TEMP, max_tokens=1024, purpose="critic")
    final_code = clean_code(critique_raw)
    
    # 4. EXECUTION
    console.print("[green]4. Reality Check...[/]")
    try:
        # Inject imports
        if "from decimal" not in final_code:
            final_code = "from decimal import Decimal\n" + final_code
        if "import math" not in final_code:
            final_code = "import math\n" + final_code
        
        final_code = "import sys\nsys.setrecursionlimit(2000)\n" + final_code

        local_scope = {"__builtins__": None, "math": math, "Decimal": Decimal, "sys": sys}
        
        exec(final_code, local_scope)
        
        if 'a' not in local_scope or 'b' not in local_scope:
             console.print("[red]❌ Critic failed to enforce structure.[/]")
             return None

        a, b = local_scope['a'], local_scope['b']
        
        # Eval
        result = Decimal(0)
        depth = 50
        for n in range(depth, 0, -1):
            try:
                # Cast to Decimal to avoid overflow/type errors
                an = Decimal(str(a(n)))
                bn = Decimal(str(b(n)))
            except:
                an, bn = Decimal(1), Decimal(1)

            if result + bn == 0: 
                result = Decimal("NaN")
                break
            result = an / (bn + result)
            
        final_val = Decimal(str(b(0))) + result
        error = abs(final_val - PI_TARGET)
        
        console.print(f"   Result: {final_val}")
        console.print(f"   Error:  {error:.5e}")
        
        return error, final_code

    except Exception as e:
        console.print(f"[red]❌ Execution Failed: {e}[/]")
        return None

async def main():
    best_error = Decimal("Infinity")
    best_code = ""

    for i in range(1, 11): # 10 Loops
        res = await tricameral_cycle(i)
        if res:
            err, code = res
            if err < best_error:
                best_error = err
                best_code = code
                console.print(f"[bold green]🏆 New Best! Error: {best_error}[/]")
    
    console.print(Panel(f"[bold white]Final Best Code (Error {best_error}):[/]\n{best_code}", title="Native Triad Result"))

if __name__ == "__main__":
    asyncio.run(main())
