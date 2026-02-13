
import asyncio
import sys
import os
import time
import requests
import json
from datetime import datetime

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mti_evo.dream_engine import DreamEngine, DreamScenario, DreamMode
try:
    from mti_evo.mti_broca import MTIBroca
except ImportError:
    from src.mti_evo.mti_broca import MTIBroca
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BRIDGE_URL = "http://localhost:8800/v1/local/reflex"

# --- JOURNALING SYSTEM ---

class DreamJournal:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.stats_file = os.path.join(base_dir, "journal.json")
        self.entries = []
        
        # Ensure dir exists
        os.makedirs(base_dir, exist_ok=True)

    def log_cycle(self, cycle, success, scenario_id, patterns, raw_text):
        """Log a dream cycle."""
        entry = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "scenario": scenario_id,
            "insight": patterns[0] if patterns else None,
            "raw_len": len(raw_text),
            "code_len": len(raw_text.split("```")[1]) if "```" in raw_text else 0
        }
        self.entries.append(entry)
        
        # Save JSON
        with open(self.stats_file, 'w') as f:
            json.dump(self.entries, f, indent=2)
            
        # Save RAW Output
        raw_file = os.path.join(self.base_dir, f"raw_cycle_{cycle:02d}.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(f"--- CYCLE {cycle} ---\n")
            f.write(raw_text)

    def visualize(self):
        """Visualize evolution."""
        console.print("\n[bold magenta]📈 EVOLUTION GRAPH[/]")
        
        successes = sum(1 for e in self.entries if e['success'])
        fails = len(self.entries) - successes
        
        # Simple stats
        console.print(f"Total Cycles: {len(self.entries)}")
        console.print(f"Crystallized: [green]{successes}[/]  Collapsed: [red]{fails}[/]")
        
        # Bar Chart
        max_len = 50
        ratio = successes / len(self.entries) if self.entries else 0
        bar_s = "█" * int(ratio * max_len)
        bar_f = "░" * (max_len - int(ratio * max_len))
        
        console.print(f"\nStability: [{bar_s}[dim]{bar_f}[/]] ({ratio*100:.1f}%)")
        
        # Timeline
        console.print("\n[bold]Timeline:[/]")
        timeline = ""
        for e in self.entries:
            timeline += "✅" if e['success'] else "❌"
        console.print(timeline)


# --- GLOBAL CAPTURE ---
last_raw_capture = ""

async def bridge_generate_fn(prompt):
    """Routes the dream prompt to Gemma via the Telepathy Bridge."""
    global last_raw_capture
    
    # Construct the JSON thought packet
    # Force Python output + NL Pattern for DreamEngine parsing
    system_prompt = (
        "You are the MTI Dream Weaver. Your goal is to solve the user's paradox.\n"
        "Format your response exactly like this:\n"
        "1. A single line starting with 'Pattern:' explaining your approach in plain English.\n"
        "2. A Python code block (```python ... ```) implementing the solution.\n"
        "Do not add other text."
    )
    
    full_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}\n<start_of_turn>model\n"

    payload = {
        "action": "telepathy",
        "prompt": full_prompt,
        "temperature": 0.8, # High creativity
        "max_tokens": 1024  # Ensure enough space for code!
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            raw_text = data.get("response", "")
            
            # CAPTURE
            last_raw_capture = raw_text
            
            return raw_text
        else:
            last_raw_capture = f"[HTTP ERROR {response.status_code}]"
            return f"Error: Bridge returned {response.status_code}"
    except Exception as e:
        last_raw_capture = f"[CONNECTION ERROR {e}]"
        return f"Error: Bridge Connection Failed - {e}"

async def run_simulation():
    console.print(Panel("[bold magenta]🔮 STARTING 10-STEP DREAM SIMULATION[/]", title="Cycle 11: Chrono-Causal Loop"))
    
    # Setup Journal
    journal = DreamJournal(os.path.join(os.getcwd(), ".dream_shadow"))
    
    # Initialize Broca for Memory Storage
    try:
        broca = MTIBroca(os.getcwd())
    except:
        console.print("[yellow]⚠️ Warning: Could not init Broca. Memories will not be persisted.[/]")
        broca = None

    engine = DreamEngine(os.getcwd(), broca_store=broca)
    engine.set_mode(DreamMode.MANUAL)
    
    # Define the "Impossible" Scenario
    scenario = DreamScenario(
        id="chrono_loop_01",
        description="Solve the Chrono-Causal Loop: Create a function that returns a value BEFORE it is calculated.",
        source="Paradox_Injection",
        priority=1.0,
        context={"cycle": 11}
    )
    
    # 10 Iterations
    for i in range(1, 11):
        console.print(f"\n[bold cyan]⚡ Iteration {i}/10[/] - Injecting Paradox...")
        
        # We manually trigger the dream, simulating evolution by updating the prompt context
        scenario.context["surrounding"] = [f"# Previous attempt failed at Cycle {i-1}", f"# Current Cycle: {i}"]
        
        # Reset Capture
        global last_raw_capture
        last_raw_capture = ""
        
        # Run Dream
        results = await engine.dream_manual(bridge_generate_fn, max_scenarios=1)
        
        # Log Results
        for res in results:
            # Determine success based on phantom code presence, not just 'success' flag
            is_crystallized = res.success and res.phantom_code is not None
            
            journal.log_cycle(i, is_crystallized, res.scenario_id, res.patterns_learned, last_raw_capture)
            
            if is_crystallized:
                console.print(f"   ✅ Dream Crystallized: [green]{res.scenario_id}[/]")
                if res.patterns_learned:
                    console.print(f"   🧠 Insight: [italic yellow]{res.patterns_learned[0]}[/]")
                console.print(f"   📜 Phantom Code:\n{res.phantom_code[:100]}...") # Truncate for display
            else:
                 console.print(f"   ❌ Collapse: {res.error or 'Parsing Failed (No Code Block)'}")
        
        # Artificial "REM Sleep" delay
        time.sleep(0.5)

    # Visualize Final Results
    journal.visualize()
    
    if broca:
        console.print("[bold cyan]💾 Consolidating Memories to Disk...[/]")
        broca.sleep()
        console.print("[green]✅ Memories Saved.[/]")

    console.print(Panel("[bold green]✨ SIMULATION COMPLETE.[/]", title="Evolution Status"))

if __name__ == "__main__":
    asyncio.run(run_simulation())
