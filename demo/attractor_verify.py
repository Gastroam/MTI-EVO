#!/usr/bin/env python3
"""
MTI-EVO Attractor Verification
==============================
Verifies that Configured Anchors (IDRE Seeds) behave as "Strange Attractors".
Attributes verified:
1. Auto-creation: They appear when stimulated.
2. Stability: They initialize with High Weight (80.0).
3. Resilience: They survive Stochastic Pruning better than noise.
"""

import sys
import time
import random
import numpy as np

# Ensure src path
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

from src.mti_idre import IDREStream
from src.mti_config import MTIConfig
from src.mti_logger import get_logger

log = get_logger("Attractor-Test")

def verify_attractors():
    log.info("🧪 INITIALIZING ATTRACTOR TEST...")
    
    # 1. Inspect Config
    config = MTIConfig()
    anchors = config.idre_anchor_seeds
    log.info(f"   [Config] Loaded Anchors: {anchors}")
    
    if not anchors:
        log.error("❌ No anchors defined in Config!")
        return False

    # 2. Initialize IDRE Stream (Broca Wrapper)
    stream = IDREStream()
    broca = stream.broca
    cortex = broca.cortex
    
    # 3. Stimulate Anchors (Lazy Init check)
    log.info("🌊 Stimulating Anchors...")
    for seed in anchors:
        # Before
        # stream.generate_fingerprint_B calls stimulate internally
        _ = stream.generate_fingerprint_B(seed)
        
        # Verify
        if seed not in cortex.active_tissue:
            log.error(f"❌ Anchor {seed} failed to materialize!")
            return False
            
        neuron = cortex.active_tissue[seed]
        weight = float(np.mean(neuron.weights))
        log.info(f"   Anchor {seed} Weight: {weight:.2f}")
        
        if weight < 79.0: # Expecting 80.0
            log.error(f"❌ Anchor {seed} is too weak! (Exp: 80.0, Got: {weight})")
            return False
            
    log.info("✅ Anchors Materialized Correctly.")
    
    # 4. Stress Test (Pruning Resilience)
    log.info("🌪️ RUNNING PRUNING STRESS TEST...")
    
    # Fill cortex with noise
    noise_seeds = random.sample(range(10000, 99999), 200)
    for s in noise_seeds:
        cortex.stimulate([s], input_signal=0.1, learn=True)
        # Manually lower their weight to ensure they are weaker than anchors
        cortex.active_tissue[s].weights *= 0.1 
        
    log.info(f"   Cortex Population: {len(cortex.active_tissue)} neurons")
    
    # Force heavy pruning
    # We will trigger pruning loop manually
    initial_count = len(cortex.active_tissue)
    
    # Simulate heavy load causing over-capacity or just manual pruning calls
    # HolographicLattice usually prunes if capacity exceeded. 
    # We will force call _prune_weakest 50 times.
    
    evicted_anchors = 0
    for _ in range(50):
        cortex._prune_weakest()
        
    final_count = len(cortex.active_tissue)
    log.info(f"   Pruned {initial_count - final_count} neurons.")
    
    # Check Anchors
    failures = 0
    for seed in anchors:
        if seed in cortex.active_tissue:
            w = np.mean(cortex.active_tissue[seed].weights)
            log.info(f"   ⚓ Anchor {seed} SURVIVED (W={w:.1f})")
        else:
            log.error(f"   💀 Anchor {seed} was EVICTED!")
            failures += 1
            
    if failures > 0:
        log.error("❌ Attractor Resilience Failed.")
        return False
        
    log.info("✅ Attractors Survived Stochastic Pruning.")
    return True

if __name__ == "__main__":
    success = verify_attractors()
    sys.exit(0 if success else 1)
