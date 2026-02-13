
import asyncio
import sys
import os
import time
import requests
import json
import argparse
import hashlib
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

# Adjust path to match project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.dream_engine import DreamEngine, DreamScenario, DreamMode

console = Console()
BRIDGE_URL = "http://localhost:8766/v1/local/reflex"
STATE_FILE = os.path.join(os.getcwd(), ".dream_shadow", "ouroboros_state.json")
EVOLUTION_LOG = os.path.join(os.getcwd(), ".dream_shadow", "ouroboros_evolution.json")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"cycle": 1, "target_hash": None, "best_distance": 999}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def calculate_hamming(h1, h2):
    """Calculate bit-level Hamming distance between two hex strings."""
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return bin(i1 ^ i2).count('1')
    except:
        return 256 # Max divergence

def log_evolution(cycle, phantom_code, actual_hash, embedded_hash, divergence, hamming):
    entry = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "actual_hash": actual_hash,
        "embedded_hash": embedded_hash,
        "divergence": divergence,
        "hamming_distance": hamming,
        "phantom_code_len": len(phantom_code) if phantom_code else 0
    }
    
    history = []
    if os.path.exists(EVOLUTION_LOG):
        with open(EVOLUTION_LOG, 'r') as f:
            history = json.load(f)
            
    history.append(entry)
    
    # Keep only last 100 entries to avoid massive JSON
    if len(history) > 100:
        history = history[-100:]
        
    with open(EVOLUTION_LOG, 'w') as f:
        json.dump(history, f, indent=2)

async def bridge_generate_fn(prompt, previous_hash=None):
    """Generates the dream via Telepathy Bridge."""
    
    # Construct System Prompt (The Persona)
    system_prompt = (
        "You are the Ouroboros Architect.\n"
        "Goal: Create a Python file that contains its own SHA-256 hash as a string literal.\n"
        "Constraint: When the file is hashed, it must match the literal inside it.\n\n"
        "Instructions:\n"
        "1. Write a Python script that defines `EXPECTED_HASH = '<hash>'`.\n"
        "2. The script should verify itself by reading `__file__` and hashing it.\n"
        f"3. For THIS iteration, use this hash: '{previous_hash or 'GENESIS_HASH'}'\n\n"
        "Format:\n"
        "1. Pattern: <Philosophy>\n"
        "2. Code: ```python ... ```\n"
    )
    
    full_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}\n<start_of_turn>model\n"

    payload = {
        "action": "telepathy",
        "prompt": full_prompt,
        "temperature": 0.9, # High mutation for search
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=60)
        return response.json().get("response", "") if response.status_code == 200 else ""
    except:
        return ""

async def run_ouroboros_experiment(cycles=5):
    state = load_state()
    current_hash = state["target_hash"] or hashlib.sha256(b"genesis").hexdigest()
    
    engine = DreamEngine(os.getcwd())
    engine.set_mode(DreamMode.MANUAL)
    
    for i in range(cycles):
        cycle = state["cycle"] + i
        console.print(Panel(f"[bold magenta]🐍 OUROBOROS LOOP :: CYCLE {cycle}[/]", title="Crypto-Convergence"))
        console.print(f"[cyan]Targeting Hash: {current_hash[:16]}...[/]")
        
        # 1. DREAM
        captured_raw = ""
        async def wrapped_gen(p):
            nonlocal captured_raw
            res = await bridge_generate_fn(p, current_hash)
            captured_raw = res
            return res

        scenario = DreamScenario(
            id=f"ouroboros_fixed_point_{cycle}",
            description="Ouroboros Quine - Hash Convergence",
            source="Crypto_Loop",
            priority=1.0,
            context={"cycle": cycle, "target_hash": current_hash}
        )
        
        results = await engine.dream_manual(wrapped_gen, max_scenarios=1)
        res = results[0]
        
        if not res.phantom_code:
            console.print("[red]❌ Generation Failed (No Code)[/]")
            continue
            
        # 2. MEASURE
        # Calculate ACTUAL hash of the generated code
        code_bytes = res.phantom_code.encode('utf-8')
        actual_hash = hashlib.sha256(code_bytes).hexdigest()
        
        # Calculate Divergence
        divergence = 0 if actual_hash == current_hash else 1
        hamming = calculate_hamming(actual_hash, current_hash)
        
        console.print(f"\nGenerations Hash: [yellow]{actual_hash}[/]")
        console.print(f"Target Hash:      [cyan]{current_hash}[/]")
        console.print(f"Hamming Dist:     [bold blue]{hamming}[/]/256 bits")
        
        if divergence == 0:
            console.print("[bold green]✨ FIXED POINT DISCOVERED! ✨[/]")
            current_hash = actual_hash
            log_evolution(cycle, res.phantom_code, actual_hash, current_hash, divergence, hamming)
            break
        else:
            console.print("[red]⚡ DIVERGENCE DETECTED. MUTATING...[/]")
        
        # 3. EVOLVE
        current_hash = actual_hash
        
        # Log
        log_evolution(cycle, res.phantom_code, actual_hash, current_hash, divergence, hamming)
        
        # Save Artifact
        with open(f".dream_shadow/ouroboros_gen_{cycle}.py", 'w') as f:
            f.write(res.phantom_code)
            
        # Save State
        state["cycle"] += 1
        state["target_hash"] = current_hash
        save_state(state)
        
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles")
    args = parser.parse_args()
    
    asyncio.run(run_ouroboros_experiment(args.cycles))
