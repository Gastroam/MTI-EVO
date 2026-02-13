"""
TEST SUITE: CRYSTALLIZATION (Layer 5.2)
=======================================
Verifies:
- Invariant 80: No Crystal without Stability.
- Invariant 84: No Crystal without Consensus.
- Engram Formation & Recall.
"""
import sys
import os
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.cortex.crystallizer import MTICrystallizer

# Mock Neuron
class MockNeuron:
    def __init__(self, w):
        self.weights = np.array(w)

def test_crystallization():
    print("\n--- LAYER 5.2: CRYSTALLIZATION TEST ---")
    
    # Setup
    test_dir = ".mti-brain-test/crystals"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    engine = MTICrystallizer(storage_path=test_dir)
    
    # Mock Tissue (Stable Pattern)
    tissue = {
        100: MockNeuron([0.8, 0.2, 0.5]),
        101: MockNeuron([0.8, 0.2, 0.4]), # Close vector
    }
    
    # Scenario 1: High Pressure (Chaos)
    print("\n1. Testing Invariant 80 (Stability)...")
    c1 = engine.crystallize(
        tissue, 
        context_str="chaos_theory", 
        node_ids=["Alice", "Bob"], 
        pressure_reading=0.20 # > 0.05
    )
    if c1 is None:
        print("✅ PASS: Rejected Chaos State.")
    else:
        print("❌ FAIL: Crystallized Chaos!")

    # Scenario 2: Low Consensus (Lonely)
    print("\n2. Testing Invariant 84 (Consensus)...")
    c2 = engine.crystallize(
        tissue,
        context_str="lonely_thought",
        node_ids=["Alice"], # < 2
        pressure_reading=0.01
    )
    if c2 is None:
        print("✅ PASS: Rejected Low Consensus.")
    else:
        print("❌ FAIL: Crystallized without Consensus!")

    # Scenario 3: Success (Stable & Shared)
    print("\n3. Testing Successful Engram...")
    c3 = engine.crystallize(
        tissue,
        context_str="golden_ratio",
        node_ids=["Alice", "Bob", "Charlie"],
        pressure_reading=0.01
    )
    
    if c3 and c3.confidence > 0.9:
        print(f"✅ PASS: Crystal Created {c3.crystal_id}")
    else:
        print("❌ FAIL: Failed to crystallize stable state.")

    # Scenario 4: Recall
    print("\n4. Testing Recall...")
    recalled = engine.recall("golden_ratio")
    if recalled and recalled.crystal_id == c3.crystal_id:
        print("✅ PASS: Instant Knowledge Retrieval.")
        print(f"   Vector Centroid: {recalled.vector_centroid}")
    else:
        print("❌ FAIL: Amnesia.")

if __name__ == "__main__":
    test_crystallization()
