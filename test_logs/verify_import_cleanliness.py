
import sys
import time
import os

# Ensure we are testing the local src
sys.path.insert(0, os.path.abspath("src"))

print("Testing 'import mti_evo' cleanliness...")
t0 = time.time()
import mti_evo
t1 = time.time()

print(f"Import time: {(t1-t0)*1000:.2f} ms")

forbidden = [
    "torch", 
    "transformers", 
    "llama_cpp", 
    "cv2", 
    "numpy" # Numpy is allowed in core, but checking just to see
]

# Numpy IS allowed in core, so we expect it.
# The others should NOT be in sys.modules

errors = []
for mod in forbidden:
    if mod == "numpy": continue
    if mod in sys.modules:
        errors.append(mod)

if errors:
    print(f"❌ FAILED: Found heavy modules loaded: {errors}")
    sys.exit(1)
else:
    print("✅ SUCCESS: No heavy modules loaded.")
    # Verify core is accessible
    print(f"MTI Version: {mti_evo.__version__}")
    sys.exit(0)
