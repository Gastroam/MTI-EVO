
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
from rich.table import Table

# Adjust path to match project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.dream_engine import DreamEngine, DreamScenario, DreamMode

console = Console()
BRIDGE_URL = "http://localhost:8766/v1/local/reflex"
EVOLUTION_LOG = os.path.join(os.getcwd(), ".dream_shadow", "ramanujan_evolution.json")

# High internal precision
getcontext().prec = 50

CONSTANTS = {
    "pi": Decimal(math.pi),
    "e": Decimal(math.e),
    "phi": (Decimal(1) + Decimal(5).sqrt()) / Decimal(2),
    "sqrt2": Decimal(2).sqrt(),
    "zeta3": Decimal("1.202056903159594285399738161511449990764986292") # Apéry's constant
}

def log_discovery(formula_id, target, value, error, formula_code):
    entry = {
        "timestamp": time.time(),
        "id": formula_id,
        "target": target,
        "value": str(value),
        "error": str(error),
        "code": formula_code
    }
    with open(EVOLUTION_LOG, 'a') as f:
        f.write(json.dumps(entry) + "\n")

def evaluate_continued_fraction(a_func, b_func, depth=20):
    """
    Evaluates generalized continued fraction:
    b0 + a1 / (b1 + a2 / (b2 + ...))
    Using backward recurrence for stability.
    """
    result = Decimal(0)
    try:
        # Start from the bottom up
        for n in range(depth, 0, -1):
            an = Decimal(a_func(n))
            bn = Decimal(b_func(n))
            if result + bn == 0: return Decimal("NaN")
            result = an / (bn + result)
            
        b0 = Decimal(b_func(0))
        return b0 + result
    except Exception as e:
        return Decimal("NaN")

async def bridge_generate_formula(cycle):
    """Asks Gemma for a continued fraction formula."""
    system_prompt = (
        "You are the Ramanujan Engine. Your goal is to discover Continued Fraction formulas for mathematical constants.\n"
        "Output Python lambda functions `a(n)` and `b(n)`.\n"
        "Format:\n"
        "1. Concept: <Description>\n"
        "2. Code: ```python\n"
        "def a(n): return ...\n"
        "def b(n): return ...\n"
        "```\n"
        "Focus on simple polynomial patterns (e.g., 2*n + 1, n**2)."
    )
    
    prompt = f"Propose Formula Candidate #{cycle}."
    full_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}\n<start_of_turn>model\n"

    payload = {
        "action": "telepathy",
        "prompt": full_prompt,
        "temperature": 1.0,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=30)
        return response.json().get("response", "") if response.status_code == 200 else ""
    except:
        return ""

async def run_ramanujan(cycles=10):
    console.print(Panel("[bold magenta]🧠 THE RAMANUJAN MACHINE[/]", title="Mathematical Discovery"))
    
    engine = DreamEngine(os.getcwd())
    engine.set_mode(DreamMode.MANUAL)
    
    for i in range(1, cycles + 1):
        console.print(f"\n[cyan]Cycle {i}/{cycles}: Dreaming Formula...[/]")
        
        # 1. GENERATE
        captured_raw = ""
        async def wrapped_gen(p):
            nonlocal captured_raw
            res = await bridge_generate_formula(i)
            captured_raw = res
            return res

        results = await engine.dream_manual(wrapped_gen, max_scenarios=1)
        res = results[0]
        
        if not res.phantom_code:
            console.print("[red]❌ No Formula Extracted[/]")
            continue
            
        # 2. COMPILE & EXECUTE
        # We need to execute the string to get a(n) and b(n).
        # We'll use a restricted local scope.
        try:
            local_scope = {}
            exec(res.phantom_code, {}, local_scope)
            
            if 'a' not in local_scope or 'b' not in local_scope:
                console.print("[red]❌ Output missing a(n) or b(n)[/]")
                continue
                
            af = local_scope['a']
            bf = local_scope['b']
            
            # 3. EVALUATE
            val = evaluate_continued_fraction(af, bf, depth=50)
            console.print(f"   Value: [bold white]{val}[/]")
            
            if val.is_nan():
                console.print("   Result: Diverged/Error")
                continue
                
            # 4. MATCH
            best_match = None
            min_error = Decimal("1e-9") # Threshold
            
            for name, const in CONSTANTS.items():
                error = abs(val - const)
                if error < min_error:
                    best_match = (name, error)
                    break
                    
            if best_match:
                name, err = best_match
                console.print(f"   [bold green]🌟 MATCH FOUND![/] Target: {name} (Error: {err:.2e})")
                log_discovery(i, name, val, err, res.phantom_code)
            else:
                # Check for "near integers" or simple rationals
                if abs(val - val.to_integral_value()) < 1e-9:
                     console.print(f"   [yellow]Interesting Integer:[/yellow] {val:.5f}")
                else:
                    console.print(f"   Status: Unknown Constant ({val:.5f}...)")
                    
        except Exception as e:
            console.print(f"[red]❌ Execution Error: {e}[/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_ramanujan(args.cycles))
