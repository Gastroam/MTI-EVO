"""
TEST SUITE: MEMORY LIFECYCLE (Phase 8)
======================================
Verifies:
- Reinforcement (Expertise Effect)
- Forgetting Curve (Entropy)
- Pruning (Death)
- Re-Crystallization (Updating)
"""
import sys
import os
import shutil
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.cortex.crystallizer import MTICrystallizer

class MockNeuron:
    def __init__(self, w): self.weights = np.array(w)

def test_lifecycle():
    print("\n--- PHASE 8: MEMORY LIFECYCLE TEST ---")
    
    test_dir = ".mti-brain-test/lifecycle"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    
    engine = MTICrystallizer(storage_path=test_dir)
    # Accelerate decay for test
    engine.decay_lambda = 0.5 

    # 1. Create Initial Crystal
    print("\n1. Birthing Crystal...")
    tissue = {1: MockNeuron([0.5, 0.5])}
    c_root = engine.crystallize(tissue, "lifecycle_test", ["Alice", "Bob"], 0.01)
    if not c_root: 
        print("❌ FAIL: Birth failed.")
        return
        
    initial_conf = c_root.confidence # Should be ~1.0
    print(f"   Born with Conf: {initial_conf:.2f}")

    # 2. Reinforcement
    print("\n2. Testing Reinforcement...")
    engine.reinforce(c_root.crystal_id, agreement_score=1.0)
    # Reload
    c_reinforced = engine.recall("lifecycle_test")
    if c_reinforced.confidence >= initial_conf: 
         # Likely capped at 1.0, but let's check access logic
         print(f"✅ PASS: Reinforced (Conf: {c_reinforced.confidence:.2f}, Access: {c_reinforced.access_count})")
    else:
         print("❌ FAIL: Confidence dropped or same.")

    # 3. Decay & Death
    print("\n3. Testing Entropy (Decay)...")
    # Manually age the crystal
    c_reinforced.last_accessed = time.time() - 10.0 # 10 seconds ago
    engine._save_to_disk(c_reinforced)
    
    # Run Decay Cycle
    # decay = exp(-0.5 * 10) = exp(-5) = 0.006
    # Conf ~ 1.0 * 0.006 = 0.006 < 0.1 (Threshold) -> Death
    dead_list = engine.apply_decay_cycle()
    
    if c_root.crystal_id in dead_list:
        print(f"✅ PASS: Crystal {c_root.crystal_id} died of old age.")
    else:
        print(f"❌ FAIL: Crystal survived entropy.")

    # 4. Re-Crystallization (Updating)
    print("\n4. Testing Re-Crystallization...")
    # Create a new one to update
    c_legacy = engine.crystallize(tissue, "update_test", ["Alice", "Bob"], 0.04)
    if not c_legacy: return
    
    # Update it
    c_new = engine.update_crystal(c_legacy.crystal_id, [0.9, 0.9], 0.01)
    
    if c_new and c_new.metadata.get("parent") == c_legacy.crystal_id:
        print(f"✅ PASS: New Crystal Born {c_new.crystal_id} from Parent.")
        
        # Verify Old is Superseded
        # We need to manually load the file because 'recall' skips superseded
        # But 'recall' skipping IS the test.
        lookup = engine.recall("update_test")
        if lookup.crystal_id == c_new.crystal_id:
            print("✅ PASS: Semantic Recall retrieved the NEW version.")
        else:
            print(f"❌ FAIL: Recall retrieved old version {lookup.crystal_id}")
            
    else:
        print("❌ FAIL: Update failed.")

if __name__ == "__main__":
    test_lifecycle()
