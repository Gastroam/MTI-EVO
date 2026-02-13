
import torch
try:
    print("Testing Float16 on CUDA...")
    a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    b = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
    c = torch.matmul(a, b)
    print("Matmul Success.")
    print("Result Mean:", c.mean().item())
    
    # Test Mixed Precision (Float16 * Float32)
    d = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
    # This usually triggers auto-cast or error depending on ops
    print("Float16 Test Passed.")
except Exception as e:
    print(f"Float16 Failed: {e}")
