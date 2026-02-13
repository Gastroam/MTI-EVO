import os
import sys
import numpy
import torch

def diagnose():
    print(f"Python: {sys.version}")
    print(f"Numpy version: {numpy.__version__}")
    print(f"Numpy file: {numpy.__file__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch file: {torch.__file__}")
    
    # Check for libiomp5md.dll conflict
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    numpy_lib = os.path.join(os.path.dirname(numpy.__file__), ".libs")
    
    print(f"\nTorch lib path: {torch_lib}")
    if os.path.exists(torch_lib):
        try:
            print("Torch lib contents (iomp):", [f for f in os.listdir(torch_lib) if "iomp" in f.lower()])
        except Exception as e:
            print(f"Could not list torch lib: {e}")
    
    print(f"\nNumpy lib path: {numpy_lib}")
    if os.path.exists(numpy_lib):
        try:
            print("Numpy lib contents:", os.listdir(numpy_lib))
        except Exception as e:
            print(f"Could not list numpy lib: {e}")
        
    # Try loading the offending module
    print("\nAttempting direct import of _multiarray_umath...")
    try:
        from numpy.core import _multiarray_umath
        print("SUCCESS: _multiarray_umath imported.")
    except ImportError as e:
        print(f"FAILURE: {e}")
        # Try to use dependency walker approach or check PATH
        print("\nPATH environment variable (filtered):")
        for p in os.environ.get("PATH", "").split(os.pathsep):
            lower_p = p.lower()
            if "python" in lower_p or "nvidia" in lower_p or "system32" in lower_p:
                print(f"  {p}")

if __name__ == "__main__":
    diagnose()
