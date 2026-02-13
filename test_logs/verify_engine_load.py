
import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from mti_evo.engines import NativeEngine

print("Testing NativeEngine lazy load logic...")

config = {
    "model_path": "./dummy_model",
    "check_deps": True # Trigger the check in init if present
}

# 1. Instantiate (Should be safe)
engine = NativeEngine(config)
print("✅ Instantiated NativeEngine.")

# 2. Check sys.modules if lazy imported in init
if "torch" in sys.modules:
    print("⚠️ Warning: torch loaded during init (expected if check_deps=True and import succeeds)")
else:
    print("✅ torch not loaded yet (if check_deps=False or import failed silently)")
    
# 3. Call load_model (Should trigger import and fail gracefully on path)
print("Calling load_model()...")
engine.load_model()

# It should print error about path, but NOT crash on import if torch exists.
# If torch doesn't exist, it prints that error.

print("✅ Called load_model (survived).")

# 4. Check if torch is loaded now
if "torch" in sys.modules:
    print("✅ torch is now in sys.modules (lazy load worked).")
else:
    print("ℹ️ torch not loaded (maybe import failed or skipped).")
