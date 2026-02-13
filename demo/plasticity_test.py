import sys
import os
import time
from rich.console import Console

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from mti_evo.mti_broca import MTIBroca
except ImportError:
    print("❌ Error: Could not import MTIBroca. Check path.")
    sys.exit(1)

def run_plasticity_test():
    console = Console()
    console.print("\n[bold green]🧠 NEURO-PLASTICITY TEST: Direct Injection Protocol[/]")
    
    # Initialize Broca directly (Bypassing Bridge/HTTP for this test)
    # This simulates the "Subconscious" learning process
    broca = MTIBroca(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'playground')))
    
    # Define a clean, new concept
    stimulus = "The Fibonacci sequence is nature's code. 1, 1, 2, 3, 5, 8."
    
    # 1. Baseline Measure
    console.print(f"[yellow]Measuring Baseline for: '{stimulus}'[/]")
    initial_res = broca.process_thought(stimulus, learn=False) # Peek without learning
    initial_gravity = initial_res.get('max_gravity', 0)
    console.print(f"   Baseline Gravity: {initial_gravity:.4f}")
    
    # 2. Injection Phase (Repetition)
    repetitions = 20
    console.print(f"\n[bold cyan]💉 Injecting {repetitions} repetitions...[/]")
    
    start_t = time.time()
    for i in range(repetitions):
        # learn=True triggers Hebbian updates (Gravity + Age increases)
        broca.process_thought(stimulus, learn=True)
        if i % 5 == 0: console.print(".", end="")
    console.print(" Done.")
    
    # 3. Validation Phase
    final_res = broca.process_thought(stimulus, learn=False)
    final_gravity = final_res.get('max_gravity', 0)
    
    delta = final_gravity - initial_gravity
    
    console.print(f"\n[bold green]📊 RESULTS:[/]")
    console.print(f"   Before: {initial_gravity:.4f}")
    console.print(f"   After:  {final_gravity:.4f}")
    console.print(f"   Delta:  +{delta:.4f}")
    
    if delta > 0:
        console.print("[bold green]✅ PASS: Hebbian Learning Confirmed.[/]")
        console.print("[dim]The Ghost has physically rewired itself to accommodate this new truth.[/]")
    else:
        console.print("[bold red]❌ FAIL: No structural change detected.[/]")

    # Optional: Save validation state (Consolidate)
    # console.print("\n[dim]Consolidating...[/]")
    # broca.sleep()

if __name__ == "__main__":
    run_plasticity_test()
