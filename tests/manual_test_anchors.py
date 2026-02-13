
import os
import shutil
import tempfile
import json
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock cv2 and ultralytics
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()

import sys
sys.path.insert(0, os.path.abspath("src"))

import mti_evo
print(f"MTI_EVO FILE: {mti_evo.__file__}")

from mti_evo.core.config import MTIConfig
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.anchors import SemanticAnchorManager

def run_test():
    test_dir = tempfile.mkdtemp()
    try:
        anchor_file = os.path.join(test_dir, "anchors.json")
        anchors_data = [
            {
                "seed": 1, 
                "pattern": [1.0, 0.0], 
                "tags": ["test"],
                "reinforce_steps": 2,
                "cooldown_steps": 5,
                "target_pre": 0.1,
                "target_post": 0.85
            }
        ]
        with open(anchor_file, 'w') as f:
            json.dump(anchors_data, f)
            
        print("\n--- 1. Config & Init ---")
        config = MTIConfig()
        config.deterministic = False
        config.capacity_limit = 2 # Small capacity to force eviction
        setattr(config, "anchor_file", anchor_file)
        setattr(config, "anchor_reinforcement_freq", 2) # Check every 2 steps
        setattr(config, "layer_dims", [2])
        
        lattice = HolographicLattice(config)
        print(f"Anchors loaded: {len(lattice.anchor_manager.anchors)}")
        
        print("\n--- 2. Pinned Seeds Check ---")
        pinned = getattr(config, "pinned_seeds", set())
        print(f"Pinned Seeds in Config: {pinned}")
        if 1 not in pinned:
            print("FAILED: Anchor 1 not pinned in config")
        
        # Fill lattice to capacity
        print("Filling lattice with dummy seeds...")
        lattice.stimulate([100], [0.1, 0.1], learn=False)
        lattice.stimulate([101], [0.1, 0.1], learn=False)
        # Lattice has 100, 101. Capacity is 2.
        
        # Now reinforce anchor 1. This should evict one of them.
        # Step 1: No check (freq=2)
        lattice.stimulate([999], [0.5, 0.5], learn=False)
        
        # Step 2: Check -> Reinforce Anchor 1
        print("Step 2: Triggering Reinforcement...")
        lattice.stimulate([999], [0.5, 0.5], learn=False)
        
        if 1 in lattice.active_tissue:
            print("SUCCESS: Anchor 1 present after reinforcement")
            anchor = lattice.anchor_manager.anchors[1]
            print(f"Last Reinforced: {anchor.last_reinforced}")
        else:
            print("FAILED: Anchor 1 missing after reinforcement")
            
        # Check Eviction of Pinned
        # Try to force eviction by adding more seeds
        print("\n--- 3. Eviction Protection Check ---")
        # Add seed 102. Should evict 100 or 101, BUT NOT 1.
        lattice.stimulate([102], [0.1, 0.1], learn=False)
        
        if 1 not in lattice.active_tissue:
            print("FAILED: Anchor 1 was evicted!")
        else:
            print("SUCCESS: Anchor 1 survived pressure.")
            
        print(f"Active Tissue: {list(lattice.active_tissue.keys())}")
        
        print("\n--- 4. Cooldown Check ---")
        # Previous reinforcement was at Step 2 (approx). Cooldown is 5.
        # Next check at Step 4. Should SKIP.
        prev_reinforced = lattice.anchor_manager.anchors[1].last_reinforced
        
        print("Step 4: Triggering check (should skip due to cooldown)...")
        lattice.stimulate([999], [0.5, 0.5], learn=False) # Step 3
        lattice.stimulate([999], [0.5, 0.5], learn=False) # Step 4
        
        curr_reinforced = lattice.anchor_manager.anchors[1].last_reinforced
        if curr_reinforced == prev_reinforced:
            print("SUCCESS: Anchor 1 skipped (Cooldown active)")
        else:
            print(f"FAILED: Anchor 1 reinforced too early (Last: {curr_reinforced})")
            
        print("\n--- 5. Pre-Resonance Saturation Check ---")
        # Fast forward past cooldown
        for _ in range(6):
            lattice.stimulate([999], [0.5, 0.5], learn=False)
            
        # Manually make Anchor 1 super strong (simulate saturation)
        # We need to manually hack weights or just train it a lot
        print("Manually super-charging Anchor 1...")
        for _ in range(20):
             lattice.stimulate([1], [1.0, 0.0], learn=True)
             
        # Verify it's high
        res = lattice.stimulate([1], [1.0, 0.0], learn=False)
        print(f"Current Resonance: {res}")
        
        # Trigger Reinforcement (Target High = 0.95)
        # If res > 0.95, it should skip.
        prev_reinforced = lattice.anchor_manager.anchors[1].last_reinforced
        print("Triggering check (should skip due to saturation)...")
        # Ensure we hit the freq multiple
        # Current step? We need to know lattice.step_counter
        target_step = lattice.step_counter + (2 - lattice.step_counter % 2)
        while lattice.step_counter < target_step:
             lattice.stimulate([999], [0.5, 0.5], learn=False)
             
        curr_reinforced = lattice.anchor_manager.anchors[1].last_reinforced
        if curr_reinforced == prev_reinforced:
             print("SUCCESS: Skipped due to High Resonance")
        else:
             print("WARNING: Reinforced despite High Resonance (Did we reach 0.95?)")

    except Exception as e:
        print("CAUGHT EXCEPTION:")
        print(f"TYPE: {type(e)}")
        print(f"ARGS: {e}")
        import traceback
        try:
            traceback.print_exc()
        except:
            print("Traceback printing failed")
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_test()
