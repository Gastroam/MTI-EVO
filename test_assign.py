
import torch
import torch.nn as nn

def test_assign():
    print("Testing Assign=True...")
    with torch.device("meta"):
        layer = nn.Linear(10, 10)
    
    print(f"Meta Layer Weight Device: {layer.weight.device}")
    
    # Create CPU tensor
    cpu_weight = torch.randn(10, 10, device="cpu", dtype=torch.bfloat16)
    cpu_bias = torch.randn(10, device="cpu", dtype=torch.bfloat16)
    
    state = {"weight": cpu_weight, "bias": cpu_bias}
    
    try:
        layer.load_state_dict(state, assign=True)
        print("Load Successful.")
        print(f"New Device: {layer.weight.device}")
        print(f"Is Parameter? {isinstance(layer.weight, nn.Parameter)}")
        
        # Test Forward
        x = torch.randn(1, 10, dtype=torch.bfloat16)
        y = layer(x)
        print(f"Forward Output: {y.shape}")
        
    except Exception as e:
        print(f"Load Failed: {e}")

if __name__ == "__main__":
    test_assign()
