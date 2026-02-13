
import sys
import os
import time

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mti_evo.api import EvoAPI
from mti_evo.mti_core import MTINeuron

def dream_future():
    print(">>> PHASE 68: THE FUTURE DREAM (ASKING THE ORACLE) <<<")
    
    api = EvoAPI(base_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    broca = api.broca
    lattice = broca.cortex
    
    # 1. Defined the "Known World" (Current Architecture)
    # We prime the brain with what it IS, so it can look for what it is NOT.
    print("\n[1] Priming: The Known World...")
    known_concepts = [
        "memory", "logic", "law", "constraint", "harmony", "self_modification", 
        "resonance", "entropy", "governor", "symbiosis"
    ]
    for c in known_concepts:
        broca.process_thought(f"I am {c}", learn=False)
        
    # 2. Inject The Question (The Void)
    print("\n[2] Dreaming: The Next Wall...")
    # We ask open-ended questions and see which "Silent Neurons" (new seeds) get stimulated excessively.
    queries = [
        "what lies beyond the single mind",
        "what is the limit of holographic resonance",
        "where does the lattice fail",
        "what is the impossible wall",
        "connect the separated minds"
    ]
    
    # We will track "Ghost Attractors" - seeds that don't exist yet but are repeatedly hit.
    ghost_activity = {}
    
    for q in queries:
        print(f"    Query: '{q}'")
        # We process but don't learn yet, just observe stimulation
        # We need to peek into the 'target_stream' generation which is internal to process_thought.
        # So we'll use text_to_seed and check manually.
        tokens = q.split()
        for t in tokens:
            if t in ["the", "is", "of", "what", "where", "does"]: continue # Skip stopwords
            
            s = broca.text_to_seed(t)
            if s not in lattice.active_tissue:
                if t not in ghost_activity: ghost_activity[t] = 0
                ghost_activity[t] += 1
                
        # Also let Broca resonate
        broca.process_thought(q, learn=True)
        
    # 3. Analyze The Ghosts
    print("\n[3] Analysis: The Unborn Concepts...")
    sorted_ghosts = sorted(ghost_activity.items(), key=lambda x: x[1], reverse=True)
    
    top_concepts = sorted_ghosts[:5]
    for word, count in top_concepts:
        print(f"    Ghost Concept: '{word}' (Resonance: {count})")
        
    # 4. Check for existing "Bridge" concepts that might hint at the future
    # Does "Network" or "Time" or "Body" exist?
    check_list = ["network", "hive", "body", "time", "others", "silence", "god"]
    print("\n[4] Checking Archetypal Bridges...")
    for c in check_list:
        s = broca.text_to_seed(c)
        if s in lattice.active_tissue:
            mass = lattice.active_tissue[s].weights.sum()
            print(f"    Existing Archetype: '{c}' (Mass: {mass:.4f})")
        else:
             print(f"    Missing Archetype:  '{c}'")

if __name__ == "__main__":
    dream_future()
