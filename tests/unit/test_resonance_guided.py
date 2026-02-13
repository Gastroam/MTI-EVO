import torch
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mti_evo.resonance_model import ResonanceGemmaForCausalLM

console = Console()
MODEL_PATH = r"H:\models\gemma-3-12B"

def test_resonance():
    console.print(Panel("[bold cyan]🌌 TESTING RESONANCE ARCHITECTURE[/]", subtitle="Sparse Activation + Tiered Memory"))
    
    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]❌ Model path not found: {MODEL_PATH}[/]")
        return

    # 1. Initialize
    console.print("[dim]Initializing Resonance Brain...[/]")
    t0 = time.time()
    try:
        model = ResonanceGemmaForCausalLM(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda")
        console.print(f"[green]✓ Brain Initialized in {time.time()-t0:.2f}s[/]")
    except Exception as e:
        console.print(f"[red]❌ Init Failed: {e}[/]")
        import traceback
        traceback.print_exc()
        return

    # 2. Test Prompts (Cognitive Topology)
    test_prompts = [
        ("Math/Logic (Pillar)", "Calculate the eigenvalue of a matrix."),
        ("Creative (Ghost)", "Write a surreal poem about a ghost in the machine."),
        ("Code (Bridge)", "Write a Python function to fix a memory leak.")
    ]
    
    # Dummy tokens
    dummy_input = torch.tensor([[101, 2045, 2003, 1037, 2038]]).long()
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    for category, prompt in test_prompts:
        console.print(f"\n[bold yellow]>>> Testing Resonance: {category}[/]")
        console.print(f"Prompt: \"{prompt}\"")
        
        # PREDICT LAYERS
        model.prepare_resonance(prompt)
        
        # RUN INFERENCE (Sparse)
        console.print("[dim]Running Sparse Forward Pass...[/]")
        t_start = time.time()
        
        with torch.no_grad():
            logits = model(dummy_input)
            
        t_end = time.time()
        console.print(f"[green]✓ Forward Complete in {t_end - t_start:.2f}s[/]")
        console.print(f"Output Shape: {logits.shape}")
        
    console.print("\n[bold green]✅ RESONANCE TEST COMPLETE[/]")

if __name__ == "__main__":
    test_resonance()
