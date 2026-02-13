
import sys
import os
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mti_evo.llm_adapter import LLMAdapter

console = Console()

MODEL_PATH = r"D:\VMTIDE\MTI-EVO\models\meta-llama-3-8b-instruct.Q4_K_M.gguf"

def test_llama():
    console.print(Panel(f"[bold cyan]Testing Model: {os.path.basename(MODEL_PATH)}[/]", subtitle="Llama-3-8B 4-bit"))

    if not os.path.exists(MODEL_PATH):
        console.print(f"[red]Error: Model not found at {MODEL_PATH}[/]")
        return

    try:
        console.print("[yellow]Initializing Adapter (Loading Weights)...[/]")
        # n_gpu_layers=-1 tries to offload all to GPU. 
        # If VRAM is constrained, this might fail or fallback.
        # LLMAdapter expects a config dict, not kwargs
        config = {
            "model_path": MODEL_PATH,
            "gpu_layers": -1,
            "n_ctx": 2048,
            "temperature": 0.7
        }
        adapter = LLMAdapter(config=config)
        
        console.print("[green]Model Loaded Successfully.[/]")
        
        prompt = "User: Hello! Who are you?\nModel:"
        console.print(f"\n[dim]Prompt: {prompt}[/]")
        
        console.print("[yellow]Generating...[/]")
        response = adapter.infer(prompt, max_tokens=100, stop=["User:"])
        
        console.print(Panel(response.text, title="Response", style="green"))
        console.print(f"[dim]Latency: {response.latency_ms:.2f}ms | Tokens: {response.tokens}[/]")

    except Exception as e:
        console.print(f"[red]Inference Failed: {e}[/]")
        console.print("[dim]Tip: Check VRAM usage or if another process is holding the GPU.[/]")

if __name__ == "__main__":
    test_llama()
