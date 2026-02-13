"""
VMT-CLI Intelligence Commands
=============================

Commands to configure intelligence settings:
- Mode selection (local, balanced, cloud)
- Temperature
- Model configuration
- Provider settings
"""
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from pathlib import Path
import json
import os

console = Console()

# Create the intel command group
intel_app = typer.Typer(
    name="intel",
    help="🧠 Intelligence settings (mode, model, temperature)"
)

# Config file for CLI intelligence settings
CONFIG_FILE = Path.home() / ".mti-brain" / "intel_config.json"

# Intelligence modes
MODES = {
    1: {"name": "Local Only", "desc": "100% local inference (Qwen)", "file": "mode_1_local_only.md"},
    2: {"name": "Restricted", "desc": "Local + limited cloud fallback", "file": "mode_2_restricted.md"},
    3: {"name": "Balanced", "desc": "Smart routing local/cloud", "file": "mode_3_balanced.md"},
    4: {"name": "Permissive", "desc": "Cloud preferred, local backup", "file": "mode_4_permissive.md"},
    5: {"name": "Cloud Only", "desc": "100% cloud (Gemini/GPT)", "file": "mode_5_cloud_only.md"},
}


def load_config() -> dict:
    """Load intelligence configuration."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "mode": 3,
        "temperature": 0.7,
        "max_tokens": 4096,
        "local_model": None,
        "cloud_provider": "gemini",
        "gpu_layers": -1,
        "context_window": 8192
    }


def save_config(config: dict) -> None:
    """Save intelligence configuration."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


@intel_app.command("status")
def intel_status():
    """Show current intelligence configuration."""
    config = load_config()
    
    mode_info = MODES.get(config.get("mode", 3), MODES[3])
    
    console.print(Panel.fit("🧠 [bold cyan]Intelligence Configuration[/bold cyan]", border_style="cyan"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")
    
    table.add_row("Mode", f"{config.get('mode', 3)} - {mode_info['name']}", mode_info['desc'])
    table.add_row("Temperature", str(config.get("temperature", 0.7)), "Creativity (0=focused, 1=creative)")
    table.add_row("Max Tokens", str(config.get("max_tokens", 4096)), "Max output length")
    table.add_row("Context Window", str(config.get("context_window", 8192)), "Memory context size")
    table.add_row("GPU Layers", str(config.get("gpu_layers", -1)), "-1 = all layers on GPU")
    table.add_row("Cloud Provider", config.get("cloud_provider", "gemini"), "For cloud inference")
    table.add_row("Local Model", config.get("local_model") or "auto-detect", "GGUF model path")
    
    console.print(table)
    
    console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")


@intel_app.command("mode")
def intel_mode(
    mode: Optional[int] = typer.Argument(None, help="Mode number (1-5)")
):
    """Get or set intelligence mode."""
    config = load_config()
    
    if mode is None:
        # Show modes
        console.print(Panel.fit("🧠 [bold]Intelligence Modes[/bold]", border_style="cyan"))
        
        current = config.get("mode", 3)
        
        for num, info in MODES.items():
            marker = "[green]→[/green]" if num == current else " "
            privacy = "🔒" if num <= 2 else "☁️" if num >= 4 else "⚖️"
            console.print(f"  {marker} [{num}] {privacy} [bold]{info['name']}[/bold]")
            console.print(f"      [dim]{info['desc']}[/dim]")
        
        console.print(f"\n[dim]Set with: vmt intel mode <number>[/dim]")
    else:
        if mode < 1 or mode > 5:
            console.print("[red]Mode must be 1-5[/red]")
            return
        
        config["mode"] = mode
        save_config(config)
        
        info = MODES[mode]
        console.print(f"[green]✓ Mode set to {mode}: {info['name']}[/green]")
        console.print(f"[dim]{info['desc']}[/dim]")


@intel_app.command("temperature")
def intel_temperature(
    value: Optional[float] = typer.Argument(None, help="Temperature (0.0-1.0)")
):
    """Get or set generation temperature."""
    config = load_config()
    
    if value is None:
        console.print(f"[bold]Current temperature:[/bold] {config.get('temperature', 0.7)}")
        console.print("[dim]0.0 = deterministic, 0.7 = balanced, 1.0 = creative[/dim]")
    else:
        if value < 0.0 or value > 2.0:
            console.print("[red]Temperature should be 0.0-2.0[/red]")
            return
        
        config["temperature"] = value
        save_config(config)
        console.print(f"[green]✓ Temperature set to {value}[/green]")


@intel_app.command("model")
def intel_model(
    path: Optional[str] = typer.Argument(None, help="Path to GGUF model")
):
    """Get or set local model path."""
    config = load_config()
    
    if path is None:
        current = config.get("local_model")
        if current:
            console.print(f"[bold]Current model:[/bold] {current}")
        else:
            console.print("[yellow]No local model configured (using auto-detect)[/yellow]")
        
        # List available models
        console.print("\n[bold]Available models:[/bold]")
        model_dirs = [
            Path(__file__).parent.parent.parent / "models",
            Path("D:/VMTIDE/mti-rlm/models"),
        ]
        
        found = []
        for model_dir in model_dirs:
            if model_dir.exists():
                for gguf in model_dir.glob("*.gguf"):
                    size_gb = gguf.stat().st_size / (1024**3)
                    found.append(f"  • {gguf.name} ({size_gb:.1f} GB)")
        
        if found:
            for f in found:
                console.print(f)
        else:
            console.print("[dim]No GGUF models found in default locations[/dim]")
    else:
        # Validate path
        p = Path(path)
        if not p.exists():
            console.print(f"[red]Model not found: {path}[/red]")
            return
        
        config["local_model"] = str(p.absolute())
        save_config(config)
        console.print(f"[green]✓ Local model set to: {p.name}[/green]")


@intel_app.command("provider")
def intel_provider(
    provider: Optional[str] = typer.Argument(None, help="Cloud provider (gemini/openai/deepseek)")
):
    """Get or set cloud provider."""
    config = load_config()
    
    providers = ["gemini", "openai", "deepseek", "anthropic", "typegpt", "oxlo"]
    
    if provider is None:
        console.print(f"[bold]Current provider:[/bold] {config.get('cloud_provider', 'gemini')}")
        console.print(f"[dim]Available: {', '.join(providers)}[/dim]")
    else:
        if provider not in providers:
            console.print(f"[red]Unknown provider. Use: {', '.join(providers)}[/red]")
            return
        
        config["cloud_provider"] = provider
        save_config(config)
        console.print(f"[green]✓ Cloud provider set to: {provider}[/green]")


@intel_app.command("context")
def intel_context(
    size: Optional[int] = typer.Argument(None, help="Context window size")
):
    """Get or set context window size."""
    config = load_config()
    
    if size is None:
        console.print(f"[bold]Context window:[/bold] {config.get('context_window', 8192)} tokens")
        console.print("[dim]Common sizes: 4096, 8192, 16384, 32768[/dim]")
    else:
        if size < 512 or size > 131072:
            console.print("[red]Context size should be 512-131072[/red]")
            return
        
        config["context_window"] = size
        save_config(config)
        console.print(f"[green]✓ Context window set to: {size}[/green]")


@intel_app.command("gpu")
def intel_gpu(
    layers: Optional[int] = typer.Argument(None, help="GPU layers (-1 for all)")
):
    """Get or set GPU layers for local model."""
    config = load_config()
    
    if layers is None:
        current = config.get("gpu_layers", -1)
        console.print(f"[bold]GPU layers:[/bold] {current}")
        console.print("[dim]-1 = all layers, 0 = CPU only, N = specific layers[/dim]")
    else:
        config["gpu_layers"] = layers
        save_config(config)
        
        if layers == -1:
            console.print("[green]✓ All layers on GPU[/green]")
        elif layers == 0:
            console.print("[yellow]✓ CPU only mode[/yellow]")
        else:
            console.print(f"[green]✓ {layers} layers on GPU[/green]")


@intel_app.command("reset")
def intel_reset():
    """Reset to default intelligence settings."""
    from rich.prompt import Confirm
    
    if not Confirm.ask("Reset all intelligence settings to defaults?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    
    default_config = {
        "mode": 3,
        "temperature": 0.7,
        "max_tokens": 4096,
        "local_model": None,
        "cloud_provider": "gemini",
        "gpu_layers": -1,
        "context_window": 8192
    }
    
    save_config(default_config)
    console.print("[green]✓ Reset to default settings[/green]")
    intel_status()
