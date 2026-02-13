
import asyncio
import sys
import os
import time
import requests
import json
import argparse
import random
import re
from decimal import Decimal, getcontext
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Adjust path to match project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.dream_engine import DreamEngine, DreamScenario, DreamMode
try:
    from mti_evo.mti_broca import MTIBroca
except ImportError:
    from src.mti_evo.mti_broca import MTIBroca

console = Console()
BRIDGE_URL = "http://localhost:8766/v1/local/reflex"
EVO_LOG = os.path.join(os.getcwd(), ".dream_shadow", "ramanujan_evo_stats.json")

getcontext().prec = 50

TARGETS = {
    "e": Decimal("2.71828182845904523536028747135266249775724709369995"),
    "pi": Decimal("3.14159265358979323846264338327950288419716939937510"),
    "phi": (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
}

class Genome:
    def __init__(self, code, fitness=0.0):
        self.code = code
        self.fitness = fitness
        self.error = Decimal("NaN")
        self.value = Decimal("NaN")

    def __repr__(self):
        return f"Genome(fit={self.fitness:.4f}, err={self.error:.2e})"

def evaluate_fraction(code_str, depth=20):
    try:
        local_scope = {}
        # Restricted exec
        exec(code_str, {"__builtins__": None}, local_scope)
        if 'a' not in local_scope or 'b' not in local_scope:
            return Decimal("NaN")
        
        a, b = local_scope['a'], local_scope['b']
        
        # Eval
        result = Decimal(0)
        for n in range(depth, 0, -1):
            an = Decimal(a(n))
            bn = Decimal(b(n))
            if result + bn == 0: return Decimal("NaN")
            result = an / (bn + result)
        b0 = Decimal(b(0))
        return b0 + result
    except:
        return Decimal("NaN")

async def generate_initial_population(target_name, size=5):
    """Asks Gemma for N distinct starting points."""
    pop = []
    
    # Initialize Broca
    try:
        broca = MTIBroca(os.getcwd())
    except:
        broca = None

    engine = DreamEngine(os.getcwd(), broca_store=broca)
    
    # NEW: Dream-Based Seeding
    # Instead of asking for a formula, we drift from the target name and ask for logic.
    
    async def gen_wrapper(p):
        # 1. Perform a Hebbian Drift
        # We shell out to dream_talk logic (simplified here)
        drift_prompt = ""
        try:
            # We assume Broca is available since we are inside the loop, 
            # but Broca is synchronous and we are async. Use simple heuristic or direct access.
            if broca:
                seed = broca.text_to_seed(target_name)
                # Simple random walk
                path = [target_name]
                curr = seed
                for _ in range(5):
                    vec = broca.get_embedding(curr)
                    # Find closest (simplified scan for performance)
                    # In a real run, we'd use the full dream_drift logic.
                    # For now, let's simulated a "Mental Block" that forces the LLM to hallucinate valid math
                    # based on the *concept* of the target.
                    pass
                drift_prompt = f"Concepts: {target_name} -> Geometry -> Ratio -> Infinite"
        except:
             drift_prompt = f"Concepts: {target_name} -> Abstract -> Number"

        # 2. Ask Telepathy to synthesize this drift into code
        prompt = (
            f"You are the Ramanujan Logic Engine.\n"
            f"You have dreamt of this path: {drift_prompt}\n"
            "Task: Crystallize this dream into a Python Continued Fraction.\n"
            "Output ONLY valid Python code with `def a(n)` and `def b(n)`.\n"
            "Explore novel structures (e.g. n^2, 2n+1) inspired by the dream.\n"
        )
        
        payload = {
            "action": "telepathy",
            "prompt": f"<start_of_turn>user\n{prompt}\n<start_of_turn>model\n",
            "temperature": 0.7, # High Temp for Creativity
            "max_tokens": 256
        }
        try:
            r = requests.post(BRIDGE_URL, json=payload, timeout=60)
            if r.status_code == 200:
                raw = r.json().get("response", "")
                console.print(f"[green]Got Dream Seed ({len(raw)} chars).[/]")
                return raw
            else:
                return ""
        except Exception as e:
            console.print(f"[red]Bridge Exception: {e}[/]")
            return ""

    attempts = 0
    while len(pop) < size and attempts < 3:
        console.print(f"   Attempt {attempts+1}: Requesting {size} candidates...")
        results = await engine.dream_manual(gen_wrapper, max_scenarios=size)
        for res in results:
            if res.phantom_code:
                pop.append(Genome(res.phantom_code))
            else:
                console.print(f"[red]   Failed to extract code. Raw Output available in logs.[/]")
        attempts += 1
        
    # FALLBACK / SMART SEEDING
    console.print("[yellow]💉 Injecting Smart Seeds (Known Helpers) to boost diversity...[/]")
    smart_seeds = [
        "def a(n): return 1\ndef b(n): return n",           # Simple Linear
        "def a(n): return n\ndef b(n): return 1",           # Inverted Linear
        "def a(n): return n + 1\ndef b(n): return 2",       # Offset Constant
        "def a(n): return 1\ndef b(n): return 2*n + 1",     # Odd numbers
        "def a(n): return 1\ndef b(n): return n*n",         # Squares
        "def a(n): return 1\ndef b(n): return 2 if n%2==0 else 1" # Alternating
    ]
    
    # Fill remaining slots with smart seeds if needed, or just append them to ensure diversity
    for seed_code in smart_seeds:
        if len(pop) < size * 1.5: # Allow slightly larger initial pop
             pop.append(Genome(seed_code))
        
    return pop

def mutate(code, rate=0.4):
    """Refined mutation: Numbers, Operators, and Complexity Injection."""
    if random.random() > rate: return code
    
    # 1. Number Tweak (Fine tuning)
    code = re.sub(r'\d+', lambda m: str(max(1, int(m.group(0)) + random.choice([-1, 1]))), code)
    
    # 2. Operator Flip
    if random.random() < 0.3:
        code = re.sub(r'\+', lambda m: random.choice(['+', '-', '*']), code)

    # 3. Complexity Injection (Add Term) - The "Growth" Factor
    if random.random() < 0.2:
        lines = code.split('\n')
        target_line_idx = random.randint(0, len(lines)-1)
        line = lines[target_line_idx]
        if "return" in line:
            # Append a term: return x  -> return x + 1  or  return x * n
            term = random.choice([" + 1", " - 1", " * n", " + n", " * 2"])
            lines[target_line_idx] = line + term
            code = "\n".join(lines)
            
    return code

def crossover(parent1, parent2):
    """Splits two codes and swaps halves (naive)."""
    lines1 = parent1.code.split('\n')
    lines2 = parent2.code.split('\n')
    
    if len(lines1) > 2 and len(lines2) > 2:
        split = random.randint(1, min(len(lines1), len(lines2))-1)
        child_code = "\n".join(lines1[:split] + lines2[split:])
        return Genome(child_code)
    return Genome(parent1.code) # Fail safe

async def run_evolution(target_name="e", generations=10, pop_size=10):
    target_val = TARGETS[target_name]
    console.print(Panel(f"[bold green]🧬 EVOLVING FORMULA FOR {target_name.upper()}[/]", title="Genetic Algorithm"))
    
    # 1. Initialize
    console.print("[cyan]🌱 Seeding Population...[/]")
    population = await generate_initial_population(target_name, pop_size)
    
    best_genome = None
    history = []
    
    for gen in range(1, generations + 1):
        # 2. Evaluate
        for genome in population:
            val = evaluate_fraction(genome.code)
            genome.value = val
            if val.is_nan():
                genome.error = Decimal("Infinity")
                genome.fitness = 0.0
            else:
                genome.error = abs(val - target_val)
                # Fitness = 1 / (error + small_epsilon)
                try:
                    genome.fitness = float(1 / (genome.error + Decimal("1e-12")))
                except:
                    genome.fitness = 0.0
                    
        # Sort by fitness desc
        population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = population[0]
        
        if best_genome is None or current_best.fitness > best_genome.fitness:
            best_genome = current_best
            
        console.print(f"Gen {gen}: Best Error = {current_best.error:.2e} (Fit: {current_best.fitness:.2f})")
        
        log_entry = {
            "gen": gen,
            "best_error": str(current_best.error),
            "best_code": current_best.code
        }
        history.append(log_entry)
        
        if current_best.error < 1e-9:
            console.print("[bold green]🏆 CONVERGED![/]")
            break
            
        # 3. Selection (Top 50% survive)
        survivors = population[:pop_size//2]
        
        # 4. Reproduction
        new_pop = []
        while len(new_pop) < pop_size:
            # Elitism: Keep best
            if len(new_pop) == 0:
                new_pop.append(survivors[0])
                continue
                
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            child.code = mutate(child.code)
            new_pop.append(child)
            
        population = new_pop
        
    console.print(Panel(f"[bold white]Final Best Formula:[/]\n{best_genome.code}\nError: {best_genome.error}", title="Evolution Result"))
    
    # Save Log
    with open(EVO_LOG, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="e")
    parser.add_argument("--gen", type=int, default=10)
    args = parser.parse_args()
    
    asyncio.run(run_evolution(args.target, args.gen))
