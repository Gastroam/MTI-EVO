
import sys
import os
sys.path.append(os.getcwd())
try:
    import src.mti_broca
    print("Success")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
