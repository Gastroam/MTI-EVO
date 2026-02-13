"""
Integration Test: Symbiosis Coherence Gating
============================================
Verifies that MTI_Symbiosis respects the Health Index.
1. Injects a 'Sick' concept (Weight=80, Health=0.1).
2. Queries it.
3. Expects state 'LEARNING' (demoted from FLOW).
"""

import sys
import os
import time

# Ensure path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MTI_Symbiosis import MTISymbiosis
from rich.console import Console

console = Console()

def run_test():
    console.print("[bold magenta]🧬 SYMBIOSIS COHERENCE TEST[/]")
    
    agent = MTISymbiosis()
    
    # 1. Inject Malignant Concept
    seed = agent.instinct.text_to_seed("corruption")
    console.print(f"💉 Injecting 'Corruption' (Seed {seed})")
    
    # Force creation
    agent.instinct.cortex.stimulate([seed], 1.0)
    neuron = agent.instinct.cortex.active_tissue[seed]
    
    # Make it massive (FLOW candidate)
    neuron.weights[:] = 80.0
    neuron.last_accessed = time.time()
    
    # Make it SICK (Health = 0.1)
    # We must hack the engine logic or force the value
    # The engine calculates on the fly.
    # To force Low Health, we can set 'critic_history' to low values.
    neuron.critic_history = [0.1, 0.1, 0.1, 0.1] 
    # This yields mu_critic = 0.1
    # Health = 0.4*Metabolic(1.0) + 0.3*0.1 + 0.3*0.1 = 0.4 + 0.03 + 0.03 = 0.46
    # 0.46 < 0.7 (Threshold) -> Should degrade to LEARNING.
    
    console.print("   Set Weights=80.0, Critic=0.1 -> Expected Health ~0.46")
    
    # 2. Query
    state, res, score, bias, health = agent.get_biological_state("corruption")
    
    console.print(f"\n🧪 DIAGNOSIS:")
    console.print(f"   State:  {state}")
    console.print(f"   Score:  {score:.2f} (Should be > 20)")
    console.print(f"   Health: {health:.2f} (Should be < 0.7)")
    
    # 3. Assert
    if score > 20.0 and health < 0.7 and state == "LEARNING":
        console.print("\n✅ PASS: High-Score Malignant Concept was DEMOTED to LEARNING.")
    elif state == "FLOW":
        console.print("\n❌ FAIL: Malignant Concept reached FLOW state.")
    else:
        console.print(f"\n⚠️ UNEXPECTED: State={state}")

if __name__ == "__main__":
    run_test()
