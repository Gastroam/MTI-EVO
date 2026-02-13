
import random
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class ModelSimulator:
    def __init__(self, name="Llama-3-8B", layers=32, dim=4096):
        self.name = name
        self.layers = layers
        self.dim = dim
        self.param_count = 8_000_000_000
        # Rough FLOPs per token per layer ~ 2 * P / L (very rough approx for simple connection)
        # Better: 2 * (6 * dim^2) or similar for Transformer block
        # Let's use a standard heuristic: 8B model ~ 16 GFLOPs per token forward pass (Dense)
        self.flops_per_token_dense = 16.0 # GFLOPs
        self.layer_cost = self.flops_per_token_dense / layers

    def run_simulation(self, num_tokens=1000, sparse_threshold=0.6):
        console.print(Panel(f"[bold cyan]Testing Viability: {self.name}[/]", subtitle=f"Tokens: {num_tokens} | Threshold: {sparse_threshold}"))
        
        # 1. Generate Synthetic Entropy Stream (Zipfian distribution logic simulator)
        # Most tokens are common (low entropy), some are rare/complex (high entropy)
        entropies = np.random.beta(a=2, b=5, size=num_tokens) * 2.0 # Skewed towards low 
        
        total_flops_dense = 0.0
        total_flops_sparse = 0.0
        active_layers_log = []
        
        # Overhead of Entropy Calculation (Very cheap, O(1) relative to Matrix Mult)
        entropy_overhead = 0.0001 # 0.1 MFLOPs
        
        for i, entropy in enumerate(entropies):
            # Dense Cost
            total_flops_dense += self.flops_per_token_dense
            
            # Sparse Cost
            if entropy < sparse_threshold:
                # Fast Path: Skip 75% of layers (Use top 4 + bottom 4 = 8 layers)
                active_layers = 8
            else:
                # Deep Path: Use all 32 layers
                active_layers = 32
            
            active_layers_log.append(active_layers)
            total_flops_sparse += (active_layers * self.layer_cost) + entropy_overhead

        # Results
        savings = (1 - (total_flops_sparse / total_flops_dense)) * 100
        avg_layers = sum(active_layers_log) / len(active_layers_log)
        
        # Table
        table = Table(title="Viability Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Dense (Standard)", style="red")
        table.add_column("Sparse (Chiral)", style="green")
        table.add_column("Delta", style="yellow")
        
        table.add_row("Total FLOPs (Teras)", f"{total_flops_dense/1000:.2f}", f"{total_flops_sparse/1000:.2f}", f"-{savings:.1f}%")
        table.add_row("Avg Active Layers", str(self.layers), f"{avg_layers:.1f}", f"-{self.layers - avg_layers:.1f}")
        table.add_row("Latency (ms/tok)*", "40.0", f"{40.0 * (1-savings/100):.1f}", f"-{savings:.1f}%")
        
        console.print(table)
        
        if savings > 40:
            console.print(f"\n[bold green]✅ VIABLE. {savings:.1f}% Compute Savings detected.[/]")
            console.print("The Entropy Overhead is negligible compared to Layer skipping.")
        else:
            console.print(f"\n[bold red]❌ NOT VIABLE. Only {savings:.1f}% savings.[/]")

if __name__ == "__main__":
    sim = ModelSimulator()
    sim.run_simulation(sparse_threshold=0.8) # If entropy < 0.8 (e.g. "The", "is"), skip layers
