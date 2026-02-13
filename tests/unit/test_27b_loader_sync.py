import sys
import os
import time

# Add playground to path
sys.path.append(os.path.dirname(__file__))

from resonance_loader_27b import ResonanceGuidedLoader27B

MODEL_PATH = r"D:\VMTIDE\MTI-EVO\models\gemma-3-27b"

def test_sync_load():
    print(f"Testing Sync Loader for 27B at {MODEL_PATH}")
    start = time.time()
    loader = ResonanceGuidedLoader27B(MODEL_PATH)
    print(f"Loader Initialized in {time.time() - start:.4f}s")
    
    # Predict
    prompt = "What is the nature of the Quantum Lattice?"
    layers = loader.predict_active_layers(prompt)
    print(f"Predicted Layers: {layers}")
    
    # Load first layer
    first_layer = layers[0]
    print(f"Loading Layer {first_layer} (Sync)...")
    s = time.time()
    weights = loader.load_layer(first_layer)
    print(f"Loaded Layer {first_layer} in {time.time() - s:.4f}s")
    
    if weights:
        print("Weights loaded successfully.")
        print(f"Keys: {list(weights.keys())[:5]}")
    else:
        print("Failed to load weights.")

if __name__ == "__main__":
    test_sync_load()
