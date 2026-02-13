import torch
import numpy as np
import sys
import os

print(f"NumPy Version: {np.__version__}")
print(f"NumPy Path: {np.__file__}")

try:
    print(f"PyTorch Version: {torch.__version__}")
except:
    pass

print("Environment Variables:")
for k, v in os.environ.items():
    if "PATH" in k or "PYTHON" in k:
        print(f"{k}: {v}")
