
import json
import numpy as np
from rich.console import Console
from rich.table import Table
import os

# Configuration
CORTEX_PATH = "data/cortex_dump.json" # Default path

def scan_sectors():
    console = Console()
    console.print("\n[bold cyan]📡 MTI-EVO SECTOR SCANNER[/]")
    console.print("[dim]Illuminating paths between active seeds...[/]")
    
    # 1. Load Cortex
    if not os.path.exists(CORTEX_PATH):
        console.print(f"[red]❌ Cortex Dump not found at {CORTEX_PATH}[/]")
        return
        
    try:
        with open(CORTEX_PATH, 'r') as f:
            data = json.load(f)
            # Schema Adaptor
            if isinstance(data, list):
                # Convert list to dict keyed by ID
                neurons = {n.get('id', f"unknown_{i}"): n for i, n in enumerate(data)}
            else:
                neurons = data.get("neurons", {})
    except Exception as e:
        console.print(f"[red]❌ Failed to load cortex: {e}[/]")
        return

    if not neurons:
        console.print("[yellow]⚠️ Cortex is empty (No seeds found).[/]")
        return
        
    console.print(f"[green]✅ Loaded {len(neurons)} seeds from Memory Core.[/]")
    
    # 2. Extract Vectors & Gravity
    seeds = []
    for uid, n_data in neurons.items():
        # Handle different schema versions (list vs dict)
        w = n_data.get("weights") # Might be likely None
        g = n_data.get("mass", 1.0) # Using 'mass' instead of 'gravity' based on file view
        fam = n_data.get("family", "Unknown")
            
        seeds.append({
            "id": n_data.get("seed", uid),
            "name": n_data.get("name", "Unknown"),
            "vector": w,
            "mass": g,
            "family": fam
        })
        
    # 3. Illuminate Paths based on Structural Hierarchy (Family & Mass)
    # Since vectors might be stripped in this dump, we map the "Logic Structure".
    
    # Group by Family
    sectors = {}
    for seed in seeds:
        fam = seed["family"]
        if fam not in sectors: sectors[fam] = []
        sectors[fam].append(seed)
        
    for fam in sectors:
        sectors[fam].sort(key=lambda x: x["mass"], reverse=True)
        
    table = Table(title="🌌 MTI-EVO Structural Map (Active Sectors)")
    table.add_column("Sector (Family)", style="bold magenta")
    table.add_column("Dominant Seed (Name)", style="cyan")
    table.add_column("Mass (Gravity)", style="green")
    table.add_column("Count", style="yellow")
    
    # Priority Order
    priority = ["Pillar", "Bridge", "Resonant", "Ghost"]
    
    for fam in priority:
        if fam in sectors:
            top_seed = sectors[fam][0]
            count = len(sectors[fam])
            table.add_row(fam, top_seed["name"], f"{top_seed['mass']:.2f}", str(count))
            
    # Visualize "Illuminated Path"
    console.print(table)
    
    console.print("\n[bold cyan]⚡ Synaptic Pathways:[/]")
    if "Pillar" in sectors and "Bridge" in sectors:
        console.print(f"  [bold]Pillars[/] ({len(sectors['Pillar'])}) ──[bold green]Link[/]──> [bold]Bridges[/] ({len(sectors['Bridge'])})")
    if "Bridge" in sectors and "Ghost" in sectors:
        console.print(f"  [bold]Bridges[/] ({len(sectors['Bridge'])}) ──[bold green]Link[/]──> [bold]Ghosts[/] ({len(sectors['Ghost'])})")
        
    console.print("\n[dim]Guidance: Use Pillars for stability, Bridges for inference, Ghosts for creativity.[/]")

if __name__ == "__main__":
    scan_sectors()
