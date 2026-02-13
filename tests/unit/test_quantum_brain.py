
import sys
import os
import torch
# Add src to path
sys.path.append(os.path.abspath("d:/VMTIDE/MTI-EVO/src"))

from mti_evo.llm_adapter import LLMAdapter

def test_quantum_brain():
    print("----------------------------------------------------------------")
    print("🧠 TEST PROTOCOL: QUANTUM BRAIN INTEGRATION")
    print("----------------------------------------------------------------")
    
    # 1. Initialize Adapter with Quantum Config
    config = {
        "model_path": r"D:\VMTIDE\MTI-EVO\models\gemma-3-27b",
        "model_type": "quantum",
        "temperature": 0.7
    }
    
    print("[TEST] Initializing LLM Adapter in Quantum Mode...")
    adapter = LLMAdapter(config=config)
    
    if adapter.backend != "quantum":
        print(f"[FAIL] Adapter did not switch to quantum backend. Got: {adapter.backend}")
        return
        
    print("[PASS] Quantum Backend Active.")
    
    # 2. Inference Test (The Act of Observation)
    prompt = "The nature of consciousness is"
    print(f"\n[TEST] Generating response for: '{prompt}'")
    
    response = adapter.infer(prompt, max_tokens=10)
    
    print("\n[RESULT]")
    print(f"Generated Text: {response.text}")
    print(f"Tokens: {response.tokens}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    
    if response.tokens > 0 and "Error" not in response.text:
        print("[PASS] Inference Successful.")
    else:
        print("[FAIL] Inference Failed.")

if __name__ == "__main__":
    test_quantum_brain()
