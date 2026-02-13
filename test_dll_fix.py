import os
# Force allow duplicate OpenMP runtimes (Fix for ctranslate2/torch conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("--- Testing Fix ---")
try:
    import numpy as np
    print(f"NumPy loaded: {np.__version__}")
    # Force load of mkl if possible
    np.array([1])
except ImportError as e:
    print(f"NumPy load failed: {e}")

try:
    import torch
    print(f"PyTorch loaded: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch load failed: {e}")

try:
    t = torch.tensor([1,2,3])
    print(f"Tensor created: {t}")
    n = t.numpy()
    print(f"Conversion back to numpy: {n}")
    print("SUCCESS: Interop working.")
except Exception as e:
    print(f"FAIL: {e}")
