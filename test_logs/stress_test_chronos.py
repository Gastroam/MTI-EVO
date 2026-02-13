import time
from mti_chronos import ChronosEngine

def run_stress_test():
    print(">>> INITIATING CHRONOS STRESS TEST (Phase 55) <<<")
    engine = ChronosEngine(gamma=0.07)
    
    # 1. TRAIN: The Seed Cycle
    # seed -> sprout -> bud -> flower
    # We simulate metabolic ticks
    
    print("\n[Stage 1] Training Causal Memory...")
    sequence = ["seed", "sprout", "bud", "flower"]
    
    for epoch in range(5):
        print(f"  Epoch {epoch+1}: ", end="")
        for concept in sequence:
            engine.tick() # Advance time
            engine.register_event(concept)
            print(f"{concept}>", end="")
        print(" [Completed]")
        
    # 2. VALIDATE FORWARD FLOW
    print("\n[Stage 2] Verifying Forward Resonance")
    # Base resonance for 'flower' should be boosted if we just saw 'bud' and time advanced
    
    # Simulate being at 'bud' state (We just fired 'bud', so we are at the tick AFTER bud)
    # We must manually set last_fired to 'bud' to test the prediction
    engine.last_fired = "bud"
    current_tick = engine.tick()
    # Check resonance for 'flower' (expected next) vs 'seed' (anachronism)
    
    # Note: Our simple engine doesn't track "current state" context yet, 
    # but we can check the resonance of 'flower' given we are at a late tick.
    
    res_flower = engine.get_causal_resonance("flower", base_resonance=0.5, target_tick=current_tick + 1)
    res_seed = engine.get_causal_resonance("seed", base_resonance=0.5, target_tick=current_tick + 1)
    
    print(f"  Resonance(flower) [Expected]: {res_flower:.4f}")
    print(f"  Resonance(seed)   [Past]    : {res_seed:.4f}")
    
    if res_flower > res_seed:
        print("  [SUCCESS] Future target resonates higher than past target.")
    else:
        print("  [FAILURE] Temporal directionality not established.")

    # 3. COUNTERFACTUAL TEST (Anachronism Detection)
    print("\n[Stage 3] Counterfactual Injection")
    
    seq_correct = ["seed", "sprout", "bud", "flower"]
    seq_wrong   = ["flower", "bud", "sprout", "seed"] # Reversed
    seq_mixed   = ["seed", "flower", "sprout", "bud"] # Jumbled
    
    print(f"  Testing Correct: {seq_correct}")
    res_c = engine.detect_anachronism(seq_correct)
    print(f"  -> Result: {res_c}")
    
    print(f"  Testing Reverse: {seq_wrong}")
    res_w = engine.detect_anachronism(seq_wrong)
    print(f"  -> Result: {res_w}")
    
    if not res_c['is_anachronism'] and res_w['is_anachronism']:
        print("  [SUCCESS] System correctly identified the anachronism.")
    else:
        print("  [FAILURE] Anachronism detection failed.")

    # 4. GAMMA SWEEP (Calibration)
    print("\n[Stage 4] Gamma Sweep Calibration")
    gammas = [0.03, 0.07, 0.12]
    
    # We want to check the "Separation Margin" (Flowers - Seeds)
    # Re-using the engine state? No, safer to re-train for clean comparison, 
    # OR since we just want to test the Formula, we can clone/reset.
    # Let's just create new engines for purity.
    
    results = {}
    
    for g in gammas:
        e = ChronosEngine(gamma=g)
        # Train
        for _ in range(5):
             for c in sequence:
                 e.tick()
                 e.register_event(c)
        
        e.last_fired = "bud"
        tick = e.tick()
        
        r_future = e.get_causal_resonance("flower", 0.5, tick+1)
        r_past   = e.get_causal_resonance("seed", 0.5, tick+1)
        margin = r_future - r_past
        results[g] = margin
        print(f"  Gamma {g}: Margin = {margin:.4f} (Future: {r_future:.4f}, Past: {r_past:.4f})")

    best_gamma = max(results, key=results.get)
    print(f"  -> Best separation at Gamma: {best_gamma}")

if __name__ == "__main__":
    run_stress_test()
