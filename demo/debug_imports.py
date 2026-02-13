import sys
import os
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('.'))

print("--- DEBUG IMPORTS ---")
try:
    print("1. Importing mti_idre...")
    import src.mti_idre
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL: {e}")

try:
    print("2. Importing pynvml...")
    import pynvml
    print("   SUCCESS (pynvml)")
except ImportError:
    print("   FAIL (pynvml) - Expected if not installed")
except Exception as e:
    print(f"   CRASH (pynvml): {e}")

try:
    print("3. Importing llama_cpp...")
    from llama_cpp import Llama
    print("   SUCCESS (llama_cpp)")
except ImportError:
    print("   FAIL (llama_cpp) - Expected if not installed")
except Exception as e:
    print(f"   CRASH (llama_cpp): {e}")
    
try:
    print("4. Importing llm_adapter...")
    try:
        from src.llm_adapter import LLMAdapter
    except ImportError:
        print("   'src.llm_adapter' failed, trying 'llm_adapter'...")
        from llm_adapter import LLMAdapter
        
    print("   SUCCESS (llm_adapter)")
    adapter = LLMAdapter() # This triggers pynvml.nvmlInit()
    print("   Initialized Adapter")
    res = adapter.infer("test")
    print(f"   Inference: {res}")
except Exception as e:
    print(f"   FAIL (llm_adapter): {e}")
    import traceback
    traceback.print_exc()

print("--- END DEBUG ---")
