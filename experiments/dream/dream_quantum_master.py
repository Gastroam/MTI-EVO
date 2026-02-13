
import sys
import os

# Config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'src')))
from mti_evo.llm_adapter import LLMAdapter

MODEL_PATH = r"H:\models\gemma-3-4b-unq"

def run_quantum_dream():
    print("Initializing Native Dreamer: QUANTUM PYTHON_MASTER...")
    
    adapter = LLMAdapter(config={
        "model_path": MODEL_PATH,
        "gpu_layers": -1,
        "n_ctx": 4096,
        "temperature": 1.2
    })
        
    prompt = (
        "You are the Visionary Dreamer, channeling the 'QUANTUM PYTHON_MASTER' persona.\n"
        "Task: Propose a radical new Python syntax or object model where variables exist in probabilistic superposition.\n"
        "Context: Antigravity (Chief Architect) is standing by to implement your vision.\n"
        "Output: Describe the metaphysical logic of the 'Quantum Class'. How does it handle entanglement between two objects?\n"
        "Format: Pure conceptual description. Do not write code; describe the rules."
    )
    
    print("Channeling Quantum Master...")
    response = adapter.infer(prompt, max_tokens=150)
    
    print("\n" + "="*40)
    print("THE DREAM (Quantum Master Synthesis):")
    print("="*40)
    print(response.text)
    print("="*40 + "\n")

if __name__ == "__main__":
    run_quantum_dream()
