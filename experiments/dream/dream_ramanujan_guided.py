
import asyncio
import sys
import os
import time
import requests
import json
import argparse
import math
import decimal
from decimal import Decimal, getcontext
from rich.console import Console
from rich.panel import Panel

# Adjust path to match project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.dream_engine import DreamEngine, DreamScenario, DreamMode

console = Console()
BRIDGE_URL = "http://localhost:8766/v1/local/reflex"
EVOLUTION_LOG = os.path.join(os.getcwd(), ".dream_shadow", "ramanujan_evolution.json")

# High precision
getcontext().prec = 100 

CONSTANTS = {
    "pi": Decimal("3.14159265358979323846264338327950288419716939937510"),
    "e": Decimal("2.71828182845904523536028747135266249775724709369995"),
    "phi": (Decimal(1) + Decimal(5).sqrt()) / Decimal(2),
    "sqrt2": Decimal(2).sqrt()
}

def evaluate_continued_fraction(a_func, b_func, depth=100):
    result = Decimal(0)
    try:
        for n in range(depth, 0, -1):
            an = Decimal(a_func(n))
            bn = Decimal(b_func(n))
            if result + bn == 0: return Decimal("NaN")
            result = an / (bn + result)
        b0 = Decimal(b_func(0))
        return b0 + result
    except:
        return Decimal("NaN")

async def bridge_generate_formula_guided(target_name):
    """Asks Gemma for a formula specifically targeting a constant."""
    system_prompt = (
        f"You are the Ramanujan Engine. Derive a Continued Fraction for '{target_name}'.\n"
        "Known pattern hint: For sqrt(2), a(n)=1, b(0)=1, and b(n)=2 for n>0.\n"
        "Output Python lambda functions `a(n)` and `b(n)`.\n"
        "Format:\n"
        "1. Concept: <Description>\n"
        "2. Code: ```python\n"
        "def a(n): return ...\n"
        "def b(n): return ...\n"
        "```"
    )
    
    prompt = f"Generate the formula for {target_name}."
    full_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}\n<start_of_turn>model\n"

    payload = {
        "action": "telepathy",
        "prompt": full_prompt,
        "temperature": 0.7, # Precision mode
        "max_tokens": 512
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=30)
        return response.json().get("response", "") if response.status_code == 200 else ""
    except:
        return ""

async def run_calibration():
    console.print(Panel("[bold cyan]📐 RAMANUJAN CALIBRATION[/]", title="Phase 7"))
    
    targets = ["sqrt2", "phi", "e"]
    
    engine = DreamEngine(os.getcwd())
    engine.set_mode(DreamMode.MANUAL)
    
    for target in targets:
        console.print(f"\n[bold]Calibrating for {target.upper()}...[/]")
        
        captured_raw = ""
        async def wrapped_gen(p):
            nonlocal captured_raw
            res = await bridge_generate_formula_guided(target)
            captured_raw = res
            return res

        results = await engine.dream_manual(wrapped_gen, max_scenarios=1)
        res = results[0]
        
        if not res.phantom_code:
            console.print("[red]❌ No Formula[/]")
            continue
            
        try:
            local_scope = {}
            exec(res.phantom_code, {}, local_scope)
            
            af = local_scope['a']
            bf = local_scope['b']
            
            val = evaluate_continued_fraction(af, bf, depth=100)
            ground_truth = CONSTANTS[target]
            error = abs(val - ground_truth)
            
            console.print(f"   Calculated: {val}")
            console.print(f"   Truth:      {ground_truth}")
            console.print(f"   Error:      {error:.2e}")
            
            if error < 1e-9:
                console.print(f"[bold green]✅ CONVERGED! Model knows {target}.[/]")
            else:
                console.print("[yellow]⚠️  Diverged. Model needs better hints.[/]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")

if __name__ == "__main__":
    asyncio.run(run_calibration())
