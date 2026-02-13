
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from mti_evo.llm_adapter import LLMAdapter, HAS_TRANSFORMERS

print(f"Transformers Support: {HAS_TRANSFORMERS}")

# Test Native Path (Hybrid Backend)
native_path = r"H:\models\gemma-3-4b-unq"
print(f"Testing Native Path: {native_path}")

try:
    # Initialize with Native Path override
    adapter = LLMAdapter(config={"model_path": native_path})
    print(f"Backend -> {adapter.backend}")
    
    if adapter.backend == "transformers":
        print("SUCCESS: Native Backend Engaged.")
    else:
        print("FAILURE: Backend did not switch to Transformers.")
except Exception as e:
    print(f"Verification Failed: {e}")
