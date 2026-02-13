
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

# Add playground to path
sys.path.append(os.path.dirname(__file__))

from quantum_engine import QuantumMaster, QuantumLayerRouter, EntanglementManager

console = Console()

def test_schrodingers_cat():
    console.print(Panel("[bold cyan]🧪 Test 1: Schrödinger's Cat (Superposition)[/]"))
    
    class SchrodingerBox(QuantumMaster):
        def __init__(self):
            super().__init__(
                cat_state={"ALIVE": 0.5, "DEAD": 0.5},
                decay_atom={"DECAYED": 0.5, "SAFE": 0.5}
            )

    box = SchrodingerBox()
    
    # 1. Check Superposition (without observing)
    wf_cat = box.get_state('cat_state')
    console.print(f"Initial State (Unobserved): {wf_cat}")
    console.print(f"Entropy: {wf_cat.get_entropy():.4f} bits")
    
    # 2. Observation
    console.print("\n[bold yellow]👀 Opening the box (Observation)...[/]")
    state = box.cat_state
    console.print(f"Observed Reality: [bold magenta]{state}[/]")
    console.print(f"Wavefunction after collapse: {wf_cat}")
    console.print(f"Entropy after collapse: {wf_cat.get_entropy():.4f} bits")
    
    # 3. Persistence
    console.print("\n[dim]Checking persistence...[/]")
    state_2 = box.cat_state
    if state == state_2:
        console.print(f"✅ Reality is stable: {state_2}")
    else:
        console.print(f"❌ Reality drift detected! {state} != {state_2}")

    # 4. Reset
    console.print("\n[bold blue]🔄 Resetting Reality...[/]")
    box.reset_reality()
    console.print(f"State after reset: {wf_cat}")
    
def test_entanglement():
    console.print(Panel("\n[bold cyan]🧪 Test 2: Spooky Action (Entanglement)[/]"))
    
    manager = EntanglementManager()
    
    class ParticlePair(QuantumMaster):
        def __init__(self, mgr):
            super().__init__(
                spin_a={"UP": 0.5, "DOWN": 0.5},
                spin_b={"UP": 0.5, "DOWN": 0.5},
                _q_manager=mgr
            )
            
    pair = ParticlePair(manager)
    
    # Entangle them
    wf_a = pair.get_state('spin_a')
    wf_b = pair.get_state('spin_b')
    manager.entangle(wf_a, wf_b)
    console.print("🔗 Particles Entangled.")
    
    # Observe A
    console.print("\n[bold yellow]👀 Observing Particle A...[/]")
    val_a = pair.spin_a
    console.print(f"Particle A collapsed to: [bold magenta]{val_a}[/]")
    
    # Check B without explicitly observing? No, accessing it observes it, but it should be PRE-DETERMINED now.
    # We check the internal state of B before access to prove it collapsed.
    console.print(f"Particle B internal state: {wf_b}")
    
    # Observe B
    val_b = pair.spin_b
    console.print(f"Observing Particle B: [bold magenta]{val_b}[/]")
    
    if val_a == val_b:
        console.print("✅ Correlation Verified (Simulated Identical Collapse).")
    else:
        console.print("❌ Entanglement Failed (Decorrelated).")

def test_quantum_router():
    console.print(Panel("\n[bold cyan]🧪 Test 3: Quantum Layer Router[/]"))
    
    blocks = ["Linear", "Conv1D", "Attention", "Mamba"]
    router = QuantumLayerRouter(blocks)
    
    console.print(f"Router initialized with {len(blocks)} blocks in superposition.")
    
    # Forward Pass 1
    input_token = "Hello"
    output = router.forward(input_token)
    console.print(f"Forward(Hello) -> [bold green]{output}[/]")
    
    # Forward Pass 2 (Should be same block due to collapse)
    output2 = router.forward("World")
    console.print(f"Forward(World) -> [bold green]{output2}[/]")
    
    block_id_1 = output.split('_')[1]
    block_id_2 = output2.split('_')[1]
    
    if block_id_1 == block_id_2:
        console.print("✅ Deterministic Routing Verified.")
    else:
        console.print("❌ Router drifted!")
        
    # Reset
    console.print("\n[bold blue]🔄 Resetting Router Reality...[/]")
    router.reset_reality()
    output3 = router.forward("NewTimeline")
    console.print(f"Forward(NewTimeline) -> [bold green]{output3}[/]")

if __name__ == "__main__":
    test_schrodingers_cat()
    test_entanglement()
    test_quantum_router()
