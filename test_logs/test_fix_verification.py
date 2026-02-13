import numpy as np
import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    print(f"NumPy Version: {np.__version__}")
    print(f"NumPy Path: {np.__file__}")
except Exception as e:
    print(f"NumPy Import Fail: {e}")

try:
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch Import Fail: {e}")

print("-" * 20)
print("Testing Interop...")

try:
    # Trigger the specific path that might fail
    # Creating a tensor from numpy usually triggers the check
    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)
    print("Success: torch.from_numpy(arr)")
    print(f"Tensor: {t}")
except Exception as e:
    print(f"FAIL: torch.from_numpy(arr)")
    print(f"Error: {e}")

try:
    # Trigger functional tensor checks
    t2 = torch.tensor([1, 2, 3])
    print("Success: torch.tensor([1, 2, 3])")
except Exception as e:
    print(f"FAIL: torch.tensor")
    print(f"Error: {e}")
