import os
import sys
import time

# Add playground to path
sys.path.append(os.path.dirname(__file__))

from resonance_loader_27b import ResonanceModel27B

# PATH FOR 27B
MODEL_PATH = r"D:\VMTIDE\MTI-EVO\models\gemma-3-27b"

TEST_PROMPTS = [
    ("Mathematical Proof of Stability", "PILLAR"),
    ("Logic requires connection between disparate axioms", "BRIDGE"),
    ("Imagine a color that implies the sound of silence", "GHOST"),
    ("Why does entropy increase in a closed system?", "BRIDGE/PILLAR")
]

def test_resonance_27b():
    print("[DEBUG] Starting Resonance Test (Gemma-27B)...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model path not found: {MODEL_PATH}")
        return

    try:
        model = ResonanceModel27B(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to init model: {e}")
        return
    
    total_savings = 0
    total_layers_full = 62 # 27B has 62 layers
    
    for prompt, expected in TEST_PROMPTS:
        print(f"\n------------------------------------------------")
        print(f"Thinking about: '{prompt}'")
        print(f"Expected Sector: {expected}")
        
        # 1. Predict
        active_indices = model.loader.predict_active_layers(prompt)
        
        count = len(active_indices)
        savings = total_layers_full - count
        percentage = (savings / total_layers_full) * 100
        total_savings += savings
        
        print(f" > Active Layers: {count} / {total_layers_full}")
        print(f" > Skipped Layers: {savings} ({percentage:.1f}%)")
        
        # 2. Simulate Load (Sample first 3 active)
        if active_indices:
            # print(" > Sampling Load (First 3)...")
            start_t = time.time()
            for idx in active_indices[:3]:
                l = model.get_layer(idx)
                l.forward("test")
                l.reset_reality() # Immediate release for test
            duration = time.time() - start_t
            # print(f" > Sample Load Time: {duration:.4f}s")
            
    avg_savings = (total_savings / (len(TEST_PROMPTS) * total_layers_full)) * 100
    print(f"\n[SUCCESS] Resonance Test 27B Complete. Avg Savings: {avg_savings:.1f}%")

if __name__ == "__main__":
    test_resonance_27b()
