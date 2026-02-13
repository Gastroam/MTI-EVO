
import asyncio
import sys
import os
import time
import requests
import json
import argparse
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

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
    return {"cycle": 1, "target_hash": None, "best_hamming": 256}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def calculate_hamming(h1, h2):
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return bin(i1 ^ i2).count('1')
    except:
        return 256

def mine_worker(args):
    """Worker function for nonce mining."""
    base_code, target_hash, start_nonce, count = args
    best_nonce = -1
    best_dist = 256
    best_hash = ""
    
    for i in range(count):
        nonce = start_nonce + i
        # Trivial append-only mutation
        # In a real scenario, we'd inject into a comment inside the file
        candidate = base_code + f"\n# Nonce: {nonce}"
        h = hashlib.sha256(candidate.encode('utf-8')).hexdigest()
        dist = calculate_hamming(h, target_hash)
        
        if dist < best_dist:
            best_dist = dist
            best_nonce = nonce
            best_hash = h
            if dist == 0: break # Holy Grail
            
    return (best_dist, best_nonce, best_hash)

async def run_miner(base_code, target_hash, duration=5.0, workers=4):
    """Mines for a better nonce."""
    console.print(f"[bold yellow]⛏️ MINING for {duration}s...[/]")
    
    chunk_size = 50000 
    start_time = time.time()
    best_overall_dist = 256
    best_overall_nonce = -1
    best_overall_hash = ""
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        total_scanned = 0
        current_nonce = 0
        
        while time.time() - start_time < duration:
            futures = []
            for i in range(workers):
                futures.append(executor.submit(mine_worker, (base_code, target_hash, current_nonce + (i * chunk_size), chunk_size)))
            
            for f in futures:
                d, n, h = f.result()
                if d < best_overall_dist:
                    best_overall_dist = d
                    best_overall_nonce = n
                    best_overall_hash = h
            
            current_nonce += (workers * chunk_size)
            total_scanned += (workers * chunk_size)
            
            # Print status every loop
            if total_scanned % 1000000 == 0:
                console.print(f"   Scanned {total_scanned/1000000:.1f}M hashes. Best: {best_overall_dist}/256")
                
            if best_overall_dist == 0:
                break
                
    console.print(f"[bold green]⛏️ Mining Complete. Best: {best_overall_dist}/256 (improved from initial)[/]")
    
    # Reconstruct the winner
    final_code = base_code + f"\n# Nonce: {best_overall_nonce}"
    return final_code, best_overall_dist, best_overall_hash

async def bridge_generate_fn(prompt, previous_hash):
    """Ask Gemma for the base template."""
    system_prompt = (
        "You are the Ouroboros Architect. Write a Python Quine script.\n"
        "It must attempt to match a Target Hash by structure.\n"
        f"Target Hash: {previous_hash}\n"
        "The script MUST be valid Python."
        "Do NOT include a Nonce comment; the Miner will add that.\n"
        "Code Guidelines:\n"
        "- Use standard imports.\n"
        "- Define `TARGET = '{previous_hash}'`\n"
        "- Calculate its own hash.\n"
    )
    
    full_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}\n<start_of_turn>model\n"

    payload = {
        "action": "telepathy",
        "prompt": full_prompt,
        "temperature": 1.0, # High entropy
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=60)
        return response.json().get("response", "") if response.status_code == 200 else ""
    except:
        return ""

async def run_experiment(cycles=5):
    state = load_state()
    # Resume or Init
    if not state["target_hash"]:
        state["target_hash"] = hashlib.sha256(b"genesis").hexdigest()
        
    current_hash = state["target_hash"]
    
    engine = DreamEngine(os.getcwd())
    engine.set_mode(DreamMode.MANUAL)
    
    for i in range(cycles):
        cycle = state["cycle"]
        console.print(Panel(f"[bold magenta]🐍 OUROBOROS MINER :: CYCLE {cycle}[/]", title="Gradient Descent"))
        console.print(f"[cyan]Targeting Hash: {current_hash[:16]}...[/]")
        
        # 1. DREAM (LLM provides structure)
        captured_raw = ""
        async def wrapped_gen(p):
            nonlocal captured_raw
            res = await bridge_generate_fn(p, current_hash)
            captured_raw = res
            return res

        scenario = DreamScenario(
            id=f"ouroboros_miner_{cycle}",
            description="Ouroboros structure generation",
            source="Miner_v1",
            priority=1.0,
            context={"cycle": cycle}
        )
        
        results = await engine.dream_manual(wrapped_gen, max_scenarios=1)
        res = results[0]
        
        if not res.phantom_code:
            console.print("[red]❌ Generation Failed (No Code)[/]")
            continue
            
        code_template = res.phantom_code
        
        # 2. MEASURE INITIAL
        init_hash = hashlib.sha256(code_template.encode('utf-8')).hexdigest()
        init_hamming = calculate_hamming(init_hash, current_hash)
        console.print(f"Base Hamming: {init_hamming}/256")
        
        # 3. MINE (CPU Brute Force)
        final_code, best_hamming, best_hash = await run_miner(code_template, current_hash, duration=10.0)
        
        console.print(f"Final Hamming: [bold blue]{best_hamming}[/]/256")
        
        # 4. EVOLVE
        # Update State
        state["cycle"] += 1
        state["target_hash"] = best_hash # Use the mined hash as next target?
        # Actually Ouroboros means the target is THIS file's hash.
        # But if we change the target to "best_hash", we are just random walking.
        # The Goal is to make `best_hash == current_hash`.
        # So next cycle, valid quine logic requires us to embed `best_hash` into the file...
        # But the Miner *already* modified the file to get `best_hash`.
        
        # Simulating the drift:
        state["target_hash"] = best_hash
        save_state(state)
        
        # Log stats
        entry = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "target": current_hash,
            "mined": best_hash,
            "hamming_improvement": init_hamming - best_hamming
        }
        with open(EVOLUTION_LOG, 'a') as f: # Append mode naive
             f.write(json.dumps(entry) + "\n")
             
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_experiment(args.cycles))
