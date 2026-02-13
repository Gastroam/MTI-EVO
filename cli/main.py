"""
VMT-CLI Main Entry Point
Console-first interface for the 9-lobe RLM architecture.

Usage:
    vmt brain status          # Show all lobe activation
    vmt query "prompt"        # Interactive chat with agent
    vmt search "query"        # Hippocampus vector search
    vmt health [path]         # Homeostasis code analysis
    vmt telemetry             # Parietal hardware stats
    vmt index [path]          # Wernicke codebase indexing
    vmt verify <file>         # Occipital AST verification
"""
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.brain_client import BrainClient
from cli.ui.cognition_stream import CognitionStreamUI

# Initialize
app = typer.Typer(
    name="vmt",
    help="🧠 VMT-CLI: Console interface for the 9-lobe RLM brain",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Sub-apps
brain_app = typer.Typer(help="Brain-level commands")
app.add_typer(brain_app, name="brain")

agent_app = typer.Typer(help="Local agent control (mount/unmount VRAM)")
app.add_typer(agent_app, name="agent")

# Import modular command groups
try:
    from cli.memory import memory_app
    from cli.pain import pain_app
    from cli.dream import dream_app
    from cli.intel import intel_app
    
    app.add_typer(memory_app, name="memory")
    app.add_typer(pain_app, name="pain")
    app.add_typer(dream_app, name="dream")
    app.add_typer(intel_app, name="intel")
except ImportError as e:
    print(f"[Warning] Some CLI modules not available: {e}")


# ============================================================================
# AGENT COMMANDS (Mount/Unmount Local LLM)
# ============================================================================

# Global engine reference for CLI session
_gguf_engine = None

@agent_app.command("status")
def agent_status():
    """Show local agent status (VRAM usage, model loaded)."""
    global _gguf_engine
    
    console.print(Panel.fit("🤖 [bold cyan]Agent Status[/bold cyan]", border_style="cyan"))
    
    # Check VRAM
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = memory_info.used / (1024**3)
        vram_total = memory_info.total / (1024**3)
        pynvml.nvmlShutdown()
        
        vram_pct = (vram_used / vram_total) * 100
        console.print(f"[bold]GPU:[/bold] {gpu_name}")
        console.print(f"[bold]VRAM:[/bold] {vram_used:.1f}/{vram_total:.1f} GB ({vram_pct:.0f}%)")
    except Exception as e:
        console.print(f"[yellow]GPU info unavailable: {e}[/yellow]")
    
    # Check local engine
    if _gguf_engine and _gguf_engine.is_loaded:
        console.print(f"\n[green]✓ Model mounted:[/green] {_gguf_engine.model_name}")
        config = _gguf_engine.get_config()
        console.print(f"[dim]  GPU Layers: {config.get('n_gpu_layers', 'all')}[/dim]")
    else:
        console.print("\n[yellow]⚫ No local model mounted[/yellow]")
        console.print("[dim]Use 'vmt agent mount <model_path>' to load a model[/dim]")


@agent_app.command("mount")
def agent_mount(
    model_path: str = typer.Argument(..., help="Path to .gguf model file"),
    gpu_layers: int = typer.Option(-1, "--gpu-layers", "-g", help="GPU layers (-1 for all)"),
    context: int = typer.Option(8192, "--context", "-c", help="Context window size"),
    flash_attn: bool = typer.Option(False, "--flash", help="Enable flash attention")
):
    """Mount a local GGUF model to VRAM."""
    global _gguf_engine
    
    console.print(Panel.fit(f"🚀 [bold]Mounting:[/bold] {os.path.basename(model_path)}", border_style="green"))
    
    # Unmount existing if any
    if _gguf_engine and _gguf_engine.is_loaded:
        console.print("[yellow]Unmounting previous model...[/yellow]")
        _gguf_engine.unload()
    
    try:
        from llm.gguf_engine import GGUFEngine
        
        _gguf_engine = GGUFEngine(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            context_window=context,
            flash_attn=flash_attn
        )
        
        with console.status("[bold green]Loading model to VRAM..."):
            _gguf_engine.load()
        
        console.print(f"[green]✓ Model mounted successfully![/green]")
        console.print(f"[dim]Model: {_gguf_engine.model_name}[/dim]")
        
        # Show VRAM after loading
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used = memory_info.used / (1024**3)
            vram_total = memory_info.total / (1024**3)
            pynvml.nvmlShutdown()
            console.print(f"[bold]VRAM used:[/bold] {vram_used:.1f}/{vram_total:.1f} GB")
        except:
            pass
            
    except FileNotFoundError as e:
        console.print(f"[red]❌ Model not found: {e}[/red]")
    except ImportError:
        console.print("[red]❌ llama-cpp-python not installed. Run: pip install llama-cpp-python[/red]")
    except Exception as e:
        console.print(f"[red]❌ Failed to mount: {e}[/red]")


@agent_app.command("unmount")
def agent_unmount():
    """Unmount local model from VRAM (free memory)."""
    global _gguf_engine
    
    if not _gguf_engine or not _gguf_engine.is_loaded:
        console.print("[yellow]No model currently mounted.[/yellow]")
        return
    
    model_name = _gguf_engine.model_name
    
    console.print(Panel.fit(f"💤 [bold]Unmounting:[/bold] {model_name}", border_style="yellow"))
    
    _gguf_engine.unload()
    _gguf_engine = None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    console.print("[green]✓ Model unmounted, VRAM freed[/green]")
    
    # Show VRAM after unloading
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = memory_info.used / (1024**3)
        vram_total = memory_info.total / (1024**3)
        pynvml.nvmlShutdown()
        console.print(f"[bold]VRAM now:[/bold] {vram_used:.1f}/{vram_total:.1f} GB")
    except:
        pass


@agent_app.command("generate")
def agent_generate(
    prompt: str = typer.Argument(..., help="Prompt for generation"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Temperature (0-1)")
):
    """Generate text using the mounted local model."""
    global _gguf_engine
    
    if not _gguf_engine or not _gguf_engine.is_loaded:
        console.print("[red]❌ No model mounted. Use 'vmt agent mount <path>' first.[/red]")
        return
    
    console.print(Panel.fit(f"[bold]Prompt:[/bold] {prompt[:50]}...", border_style="blue"))
    
    with console.status("[bold green]Generating..."):
        response = _gguf_engine.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    
    console.print(Panel(response, title="Response", border_style="green"))


@agent_app.command("list")
def agent_list(
    path: str = typer.Argument(None, help="Path to search for models (default: common locations)")
):
    """List available GGUF models."""
    from pathlib import Path
    
    console.print(Panel.fit("📂 [bold]Available Models[/bold]", border_style="yellow"))
    
    # Common model locations
    search_paths = []
    if path:
        search_paths.append(Path(path))
    else:
        # Check common locations - mti-rlm/models is the primary location
        script_dir = Path(__file__).parent.parent.parent  # mti-rlm root
        home = Path.home()
        common = [
            script_dir / "models",           # mti-rlm/models (primary)
            Path("D:/VMTIDE/mti-rlm/models"), # Absolute fallback
            home / "models",
            home / ".cache" / "huggingface",
            Path("D:/models"),
            Path("D:/AI/models"),
            Path(".") / "models",
        ]
        search_paths = [p for p in common if p.exists()]
    
    if not search_paths:
        console.print("[yellow]No model directories found.[/yellow]")
        console.print("[dim]Specify a path: vmt agent list /path/to/models[/dim]")
        return
    
    models_found = []
    for search_path in search_paths:
        console.print(f"[dim]Searching: {search_path}[/dim]")
        for gguf_file in search_path.rglob("*.gguf"):
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            size_gb = size_mb / 1024
            models_found.append({
                "path": str(gguf_file),
                "name": gguf_file.name,
                "size": f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_mb:.0f} MB"
            })
    
    if not models_found:
        console.print("[yellow]No .gguf models found.[/yellow]")
        return
    
    table = Table(title="GGUF Models", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    for i, model in enumerate(models_found, 1):
        table.add_row(str(i), model["name"], model["size"])
    
    console.print(table)
    console.print(f"\n[dim]Mount with: vmt agent mount <full_path>[/dim]")


@agent_app.command("benchmark")
def agent_benchmark(
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of iterations"),
    tokens: int = typer.Option(100, "--tokens", "-t", help="Tokens per iteration")
):
    """Benchmark mounted model speed (tokens/sec)."""
    global _gguf_engine
    import time
    
    if not _gguf_engine or not _gguf_engine.is_loaded:
        console.print("[red]❌ No model mounted. Use 'vmt agent mount <path>' first.[/red]")
        return
    
    console.print(Panel.fit(f"⏱️ [bold]Benchmarking:[/bold] {_gguf_engine.model_name}", border_style="yellow"))
    console.print(f"[dim]Iterations: {iterations} | Tokens/iter: {tokens}[/dim]\n")
    
    prompt = "The quick brown fox jumps over the lazy dog. Continue this story:"
    
    times = []
    total_tokens = 0
    
    for i in range(iterations):
        console.print(f"[dim]Run {i+1}/{iterations}...[/dim]", end=" ")
        
        start = time.perf_counter()
        response = _gguf_engine.generate(prompt, max_new_tokens=tokens, temperature=0.7)
        elapsed = time.perf_counter() - start
        
        # Estimate tokens (rough: chars / 4)
        gen_tokens = len(response) // 4
        tps = gen_tokens / elapsed if elapsed > 0 else 0
        
        times.append(elapsed)
        total_tokens += gen_tokens
        
        console.print(f"[green]{elapsed:.2f}s[/green] ({tps:.1f} tok/s)")
    
    # Summary
    avg_time = sum(times) / len(times)
    avg_tps = total_tokens / sum(times) if sum(times) > 0 else 0
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Average time: {avg_time:.2f}s")
    console.print(f"  Average speed: [bold green]{avg_tps:.1f} tokens/sec[/bold green]")
    console.print(f"  Total tokens: {total_tokens}")


# ============================================================================
# CONFIG COMMAND
# ============================================================================

CONFIG_FILE = os.path.expanduser("~/.vmt_config.json")

@app.command("config")
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current config"),
    model_dir: str = typer.Option(None, "--model-dir", help="Set default model directory"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Set default workspace"),
    port: int = typer.Option(None, "--port", "-p", help="Set default server port")
):
    """Configure VMT-CLI defaults."""
    import json
    
    # Load existing config
    current_config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                current_config = json.load(f)
        except:
            pass
    
    if show or (not model_dir and not workspace and not port):
        console.print(Panel.fit("⚙️ [bold]VMT Configuration[/bold]", border_style="cyan"))
        if current_config:
            for key, value in current_config.items():
                console.print(f"[bold]{key}:[/bold] {value}")
        else:
            console.print("[dim]No configuration set.[/dim]")
        console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")
        return
    
    # Update config
    if model_dir:
        current_config["model_dir"] = os.path.abspath(model_dir)
        console.print(f"[green]✓ model_dir set to: {current_config['model_dir']}[/green]")
    
    if workspace:
        current_config["workspace"] = os.path.abspath(workspace)
        console.print(f"[green]✓ workspace set to: {current_config['workspace']}[/green]")
    
    if port:
        current_config["port"] = port
        console.print(f"[green]✓ port set to: {port}[/green]")
    
    # Save config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(current_config, f, indent=2)
    
    console.print(f"[dim]Saved to {CONFIG_FILE}[/dim]")


# ============================================================================
# BRAIN COMMANDS
# ============================================================================

@brain_app.command("status")
def brain_status():
    """Show activation status of all 9 lobes."""
    console.print(Panel.fit("🧠 [bold cyan]VMT Brain Status[/bold cyan]", border_style="cyan"))
    
    async def fetch_status():
        client = BrainClient()
        try:
            await client.connect()
            status = await client.get_brain_status()
            return status
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    with console.status("[bold green]Connecting to brain..."):
        status = asyncio.run(fetch_status())
    
    if "error" in status:
        console.print(f"[red]❌ Error: {status['error']}[/red]")
        console.print("[dim]Make sure the RLM server is running: python -m src.main[/dim]")
        return
    
    # Create lobe status table
    table = Table(title="Lobe Activation", show_header=True, header_style="bold magenta")
    table.add_column("Lobe", style="cyan", width=15)
    table.add_column("Status", width=10)
    table.add_column("Activation", width=30)
    table.add_column("Info", style="dim")
    
    lobes = status.get("lobes", {})
    for name, data in lobes.items():
        activation = data.get("activation", 0)
        bar_len = int(activation * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        status_icon = "🟢" if data.get("active", False) else "⚫"
        info = data.get("info", "")
        table.add_row(name.title(), status_icon, f"[green]{bar}[/green] {activation:.0%}", info)
    
    console.print(table)


# ============================================================================
# QUERY COMMAND (Main interactive chat)
# ============================================================================

@app.command("query")
def query(
    prompt: str = typer.Argument(..., help="Query to send to the agent"),
    depth: int = typer.Option(1, "--depth", "-d", help="Reasoning depth (1-5)"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response")
):
    """Send a query to the RLM agent and get a response."""
    console.print(Panel.fit(f"[bold]Query:[/bold] {prompt}", border_style="blue"))
    
    async def run_query():
        client = BrainClient()
        ui = CognitionStreamUI(console)
        
        try:
            await client.connect()
            
            if stream:
                # Stream response with live UI
                with Live(ui.get_renderable(), console=console, refresh_per_second=10):
                    async for event in client.query_stream(prompt, depth=depth):
                        ui.update(event)
                        if event.get("type") == "final":
                            break
            else:
                # Single response
                with console.status("[bold green]Thinking..."):
                    response = await client.query(prompt, depth=depth)
                console.print(Markdown(response.get("text", "")))
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
        finally:
            await client.disconnect()
    
    asyncio.run(run_query())


# ============================================================================
# HIPPOCAMPUS SEARCH
# ============================================================================

@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query for vector memory"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results")
):
    """Search the Hippocampus vector memory."""
    console.print(Panel.fit(f"🔍 [bold]Searching:[/bold] {query}", border_style="magenta"))
    
    async def run_search():
        client = BrainClient()
        try:
            await client.connect()
            results = await client.hippocampus_search(query, limit=limit)
            return results
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    with console.status("[bold green]Searching memories..."):
        results = asyncio.run(run_search())
    
    if "error" in results:
        console.print(f"[red]❌ {results['error']}[/red]")
        return
    
    memories = results.get("memories", [])
    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        return
    
    for i, mem in enumerate(memories, 1):
        score = mem.get("score", 0)
        content = mem.get("content", "")[:200]
        source = mem.get("source", "unknown")
        console.print(Panel(
            f"[dim]{source}[/dim]\n{content}...",
            title=f"[cyan]#{i}[/cyan] Score: {score:.3f}",
            border_style="dim"
        ))


# ============================================================================
# HOMEOSTASIS HEALTH
# ============================================================================

@app.command("health")
def health(
    path: str = typer.Argument(".", help="Path to analyze")
):
    """Run Homeostasis code health analysis."""
    console.print(Panel.fit(f"🏥 [bold]Analyzing:[/bold] {path}", border_style="green"))
    
    async def run_health():
        client = BrainClient()
        try:
            await client.connect()
            result = await client.homeostasis_health(path)
            return result
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    with console.status("[bold green]Analyzing code health..."):
        result = asyncio.run(run_health())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    health_pct = result.get("health_percentage", 0)
    color = "green" if health_pct >= 80 else "yellow" if health_pct >= 60 else "red"
    
    console.print(f"\n[bold {color}]Health: {health_pct:.1f}%[/bold {color}]")
    
    # Show issues
    issues = result.get("issues", [])
    if issues:
        console.print(f"\n[yellow]Issues ({len(issues)}):[/yellow]")
        for issue in issues[:10]:
            console.print(f"  • {issue}")


# ============================================================================
# PARIETAL TELEMETRY
# ============================================================================

@app.command("telemetry")
def telemetry(
    watch: bool = typer.Option(False, "--watch", "-w", help="Live monitoring"),
    local: bool = typer.Option(False, "--local", "-l", help="Local mode (no server needed)")
):
    """Show Parietal hardware telemetry (GPU/VRAM/CPU)."""
    
    def get_local_telemetry() -> dict:
        """Get telemetry directly without server."""
        import psutil
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # RAM info
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        
        # GPU info
        gpu_info = {"name": "N/A", "utilization": 0, "vram": {"used_gb": 0, "total_gb": 0}}
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info["name"] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_info["name"], bytes):
                gpu_info["name"] = gpu_info["name"].decode()
            
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info["utilization"] = utilization.gpu
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info["vram"]["used_gb"] = memory_info.used / (1024**3)
            gpu_info["vram"]["total_gb"] = memory_info.total / (1024**3)
            
            pynvml.nvmlShutdown()
        except:
            pass
        
        return {
            "cpu": {"utilization": cpu_percent, "cores": cpu_count},
            "ram": {"used_gb": ram_used_gb, "total_gb": ram_total_gb},
            "gpu": gpu_info
        }
    
    async def get_telemetry():
        client = BrainClient()
        try:
            await client.connect()
            return await client.parietal_telemetry()
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    def display_telemetry(data: dict):
        if "error" in data:
            console.print(f"[red]❌ {data['error']}[/red]")
            return
        
        table = Table(title="🖥️ Hardware Telemetry", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Usage", width=20)
        
        # GPU
        gpu = data.get("gpu", {})
        gpu_util = gpu.get("utilization", 0)
        gpu_bar = "█" * int(gpu_util / 5) + "░" * (20 - int(gpu_util / 5))
        table.add_row("GPU", gpu.get("name", "N/A"), f"[{'red' if gpu_util > 80 else 'green'}]{gpu_bar}[/] {gpu_util}%")
        
        # VRAM
        vram = gpu.get("vram", {})
        vram_used = vram.get("used_gb", 0)
        vram_total = vram.get("total_gb", 1)
        vram_pct = (vram_used / vram_total) * 100 if vram_total > 0 else 0
        vram_bar = "█" * int(vram_pct / 5) + "░" * (20 - int(vram_pct / 5))
        table.add_row("VRAM", f"{vram_used:.1f}/{vram_total:.1f} GB", f"[{'red' if vram_pct > 80 else 'green'}]{vram_bar}[/] {vram_pct:.0f}%")
        
        # CPU
        cpu = data.get("cpu", {})
        cpu_util = cpu.get("utilization", 0)
        cpu_bar = "█" * int(cpu_util / 5) + "░" * (20 - int(cpu_util / 5))
        table.add_row("CPU", f"{cpu.get('cores', 'N/A')} cores", f"[{'red' if cpu_util > 80 else 'green'}]{cpu_bar}[/] {cpu_util}%")
        
        # RAM
        ram = data.get("ram", {})
        ram_used = ram.get("used_gb", 0)
        ram_total = ram.get("total_gb", 1)
        ram_pct = (ram_used / ram_total) * 100 if ram_total > 0 else 0
        ram_bar = "█" * int(ram_pct / 5) + "░" * (20 - int(ram_pct / 5))
        table.add_row("RAM", f"{ram_used:.1f}/{ram_total:.1f} GB", f"[{'red' if ram_pct > 80 else 'green'}]{ram_bar}[/] {ram_pct:.0f}%")
        
        console.print(table)
    
    if watch:
        console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
        try:
            while True:
                console.clear()
                if local:
                    data = get_local_telemetry()
                else:
                    data = asyncio.run(get_telemetry())
                display_telemetry(data)
                import time
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
    else:
        if local:
            data = get_local_telemetry()
        else:
            data = asyncio.run(get_telemetry())
        display_telemetry(data)


# ============================================================================
# WERNICKE INDEX
# ============================================================================

@app.command("index")
def index(
    path: str = typer.Argument(".", help="Path to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reindex")
):
    """Index codebase with Wernicke."""
    console.print(Panel.fit(f"📚 [bold]Indexing:[/bold] {path}", border_style="yellow"))
    
    async def run_index():
        client = BrainClient()
        try:
            await client.connect()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Indexing...", total=None)
                result = await client.wernicke_index(path, force=force)
                progress.update(task, completed=True)
            
            return result
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    result = asyncio.run(run_index())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    files = result.get("files_indexed", 0)
    symbols = result.get("symbols", 0)
    console.print(f"[green]✓ Indexed {files} files, {symbols} symbols[/green]")


# ============================================================================
# OCCIPITAL VERIFY
# ============================================================================

@app.command("verify")
def verify(
    file: str = typer.Argument(..., help="File to verify"),
    local: bool = typer.Option(True, "--local/--server", help="Local mode (no server)")
):
    """Run Occipital AST verification on a file (works offline)."""
    console.print(Panel.fit(f"🔬 [bold]Verifying:[/bold] {file}", border_style="cyan"))
    
    def verify_local(filepath: str) -> dict:
        """Verify AST locally without server."""
        import ast
        
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}
        
        ext = os.path.splitext(filepath)[1]
        
        if ext == ".py":
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                # Count structures
                functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
                
                return {
                    "valid": True,
                    "errors": [],
                    "structure": {"functions": functions, "classes": classes, "imports": imports}
                }
            except SyntaxError as e:
                return {
                    "valid": False,
                    "errors": [{"line": e.lineno, "message": str(e.msg)}],
                    "structure": {}
                }
        else:
            return {"valid": True, "errors": [], "structure": {}, "note": "Non-Python files assumed valid"}
    
    async def run_verify():
        client = BrainClient()
        try:
            await client.connect()
            return await client.occipital_verify(file)
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    if local:
        result = verify_local(file)
    else:
        with console.status("[bold green]Parsing AST..."):
            result = asyncio.run(run_verify())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    valid = result.get("valid", False)
    if valid:
        console.print("[green]✓ AST valid[/green]")
    else:
        console.print("[red]✗ AST errors found:[/red]")
        for err in result.get("errors", []):
            console.print(f"  [red]• Line {err.get('line', '?')}: {err.get('message', '')}[/red]")
    
    # Show structure summary
    structure = result.get("structure", {})
    if structure:
        console.print(f"\n[dim]Functions: {structure.get('functions', 0)} | Classes: {structure.get('classes', 0)} | Imports: {structure.get('imports', 0)}[/dim]")


# ============================================================================
# STATS - Code Statistics (Offline)
# ============================================================================

@app.command("stats")
def stats(
    path: str = typer.Argument(".", help="Path to analyze")
):
    """Show code statistics for a directory (works offline)."""
    from pathlib import Path
    
    console.print(Panel.fit(f"📊 [bold]Code Stats:[/bold] {path}", border_style="yellow"))
    
    target = Path(path).resolve()
    if not target.exists():
        console.print(f"[red]❌ Path not found: {path}[/red]")
        return
    
    # Count files by extension
    ext_counts: dict = {}
    total_lines = 0
    total_files = 0
    
    ignore_dirs = {'node_modules', '.git', '__pycache__', 'venv', 'dist', '.venv', 'target'}
    
    for file_path in target.rglob("*"):
        if file_path.is_file():
            # Skip ignored directories
            if any(ignored in file_path.parts for ignored in ignore_dirs):
                continue
            
            ext = file_path.suffix.lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_files += 1
            
            # Count lines for code files
            if ext in ['.py', '.js', '.ts', '.tsx', '.rs', '.go', '.java', '.c', '.cpp', '.h']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += sum(1 for _ in f)
                except:
                    pass
    
    # Create table
    table = Table(title="Code Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Extension", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:15]:
        table.add_row(ext, str(count))
    
    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {total_files} files, ~{total_lines:,} lines of code")


# ============================================================================
# TREE - File Tree (Offline)
# ============================================================================

@app.command("tree")
def tree_cmd(
    path: str = typer.Argument(".", help="Path to show"),
    depth: int = typer.Option(3, "--depth", "-d", help="Max depth"),
    show_files: bool = typer.Option(True, "--files/--dirs-only", help="Show files")
):
    """Show file tree structure (works offline)."""
    from rich.tree import Tree
    from pathlib import Path
    
    target = Path(path).resolve()
    if not target.exists():
        console.print(f"[red]❌ Path not found: {path}[/red]")
        return
    
    ignore_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv', 'dist', 'target'}
    
    def add_to_tree(tree: Tree, directory: Path, current_depth: int = 0):
        if current_depth >= depth:
            return
        
        try:
            items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for item in items:
            if item.name.startswith('.') and item.name not in ['.env', '.gitignore']:
                continue
            
            if item.is_dir():
                if item.name in ignore_dirs:
                    continue
                branch = tree.add(f"📁 [bold blue]{item.name}[/bold blue]")
                add_to_tree(branch, item, current_depth + 1)
            elif show_files:
                # Icon based on extension
                ext = item.suffix.lower()
                icon = "📄"
                if ext == ".py":
                    icon = "🐍"
                elif ext in [".js", ".ts", ".tsx"]:
                    icon = "📜"
                elif ext == ".rs":
                    icon = "🦀"
                elif ext == ".md":
                    icon = "📝"
                elif ext == ".json":
                    icon = "📋"
                
                tree.add(f"{icon} {item.name}")
    
    root_tree = Tree(f"📂 [bold]{target.name}[/bold]")
    add_to_tree(root_tree, target)
    
    console.print(root_tree)


# ============================================================================
# GHOST MODE
# ============================================================================

@app.command("ghost")
def ghost():
    """Enter Ghost Mode (headless, maximizes VRAM for inference)."""
    console.print(Panel.fit(
        "👻 [bold]Ghost Mode[/bold]\n"
        "[dim]Headless operation - maximum VRAM allocation[/dim]",
        border_style="magenta"
    ))
    
    async def enter_ghost():
        client = BrainClient()
        try:
            await client.connect()
            result = await client.enable_ghost_mode()
            return result
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.disconnect()
    
    result = asyncio.run(enter_ghost())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    vram_freed = result.get("vram_freed_mb", 0)
    console.print(f"[green]✓ Ghost Mode active. Freed {vram_freed}MB VRAM for inference.[/green]")
    console.print("[dim]The brain is now running headless. Use 'vmt query' to interact.[/dim]")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

@app.command("interactive")
def interactive():
    """Enter interactive REPL mode."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    
    console.print(Panel.fit(
        "🧠 [bold cyan]VMT Interactive Mode[/bold cyan]\n"
        "[dim]Type your queries. Use 'exit' or Ctrl+D to quit.[/dim]",
        border_style="cyan"
    ))
    
    history_file = os.path.expanduser("~/.vmt_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory()
    )
    
    async def run_interactive():
        client = BrainClient()
        ui = CognitionStreamUI(console)
        
        try:
            await client.connect()
            console.print("[green]✓ Connected to brain[/green]\n")
            
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: session.prompt("vmt> ")
                    )
                    
                    if user_input.lower() in ("exit", "quit", "q"):
                        break
                    
                    if not user_input.strip():
                        continue
                    
                    # Handle special commands
                    if user_input.startswith("/"):
                        cmd = user_input[1:].split()[0]
                        if cmd == "status":
                            status = await client.get_brain_status()
                            console.print(status)
                        elif cmd == "clear":
                            console.clear()
                        elif cmd == "help":
                            console.print("[dim]/status - Brain status | /clear - Clear screen | /help - This help[/dim]")
                        continue
                    
                    # Regular query
                    with Live(ui.get_renderable(), console=console, refresh_per_second=10):
                        async for event in client.query_stream(user_input):
                            ui.update(event)
                            if event.get("type") == "final":
                                break
                    
                    console.print()
                    
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except Exception as e:
            console.print(f"[red]❌ Connection error: {e}[/red]")
        finally:
            await client.disconnect()
            console.print("\n[dim]Goodbye! 👋[/dim]")
    
    asyncio.run(run_interactive())


# ============================================================================
# SERVE - Start RLM Server
# ============================================================================

@app.command("serve")
def serve(
    port: int = typer.Option(8765, "--port", "-p", help="WebSocket port"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Workspace path"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background")
):
    """Start the RLM brain server."""
    import subprocess
    import sys
    
    workspace_path = os.path.abspath(workspace)
    
    console.print(Panel.fit(
        f"🧠 [bold cyan]Starting RLM Server[/bold cyan]\n"
        f"[dim]Port: {port} | Workspace: {workspace_path}[/dim]",
        border_style="cyan"
    ))
    
    # Set environment variables
    env = os.environ.copy()
    env["MTI_PORT"] = str(port)
    env["MTI_WORKSPACE"] = workspace_path
    
    # Build command - call the server module, not main
    cmd = [sys.executable, "-m", "src.server"]
    
    if background:
        # Run detached
        if sys.platform == "win32":
            subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                cmd,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                start_new_session=True
            )
        console.print("[green]✓ Server started in background[/green]")
        console.print(f"[dim]Connect with: vmt query 'hello'[/dim]")
    else:
        # Run in foreground
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        try:
            subprocess.run(
                cmd,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped.[/yellow]")


# ============================================================================
# PUPPETEER COMMANDS (Inter-LLM Communication)
# ============================================================================

@app.command("discuss")
def discuss(
    content: str = typer.Argument(..., help="Content to discuss with Cloud AI"),
    context: str = typer.Option("Review this implementation plan", "--context", "-c", help="Context for discussion"),
    model: str = typer.Option("qwen3-thinking", "--model", "-m", help="Model to use (qwen3-thinking, deepseek-r1, etc.)")
):
    """Discuss an implementation plan with the Cloud AI (Qwen/TypeGPT)."""
    import websockets
    import json
    
    console.print(Panel.fit(
        f"🧠 [bold cyan]Prefrontal Discussion[/bold cyan]\n"
        f"[dim]Model: {model} | Context: {context[:50]}...[/dim]",
        border_style="cyan"
    ))
    
    async def run_discuss():
        uri = 'ws://localhost:8766/ws?token=whatever'
        try:
            async with websockets.connect(uri) as ws:
                msg = {
                    'type': 'prefrontal:discuss',
                    'payload': {
                        'content': content,
                        'context': context,
                        'model': model
                    }
                }
                await ws.send(json.dumps(msg))
                
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(response)
                    
                    if data.get('type') == 'status':
                        console.print(f"[dim]{data.get('payload')}[/dim]")
                    elif data.get('type') == 'answer':
                        return {"response": data.get('payload'), "model": data.get('model', model)}
                    elif data.get('type') == 'error':
                        return {"error": data.get('payload')}
        except Exception as e:
            return {"error": str(e)}
    
    with console.status("[bold green]Cloud AI thinking..."):
        result = asyncio.run(run_discuss())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    from rich.markdown import Markdown
    console.print(Panel(
        Markdown(result.get("response", "")),
        title=f"[cyan]🧠 {result.get('model', model)}[/cyan]",
        border_style="green"
    ))


@app.command("reflex")
def reflex(
    action: str = typer.Argument(..., help="Action: scan_ghosts, apply_edit, read_file, run_cmd"),
    path: str = typer.Option(None, "--path", "-p", help="File path for scan/read/edit"),
    target_id: str = typer.Option(None, "--target", "-t", help="Ghost Hash ID for apply_edit"),
    new_code: str = typer.Option(None, "--code", "-c", help="New code for apply_edit"),
    command: str = typer.Option(None, "--command", help="Command for run_cmd")
):
    """Execute local reflex actions (The Hands - $0 cost)."""
    import httpx
    
    console.print(Panel.fit(
        f"🛠️ [bold yellow]Local Reflex[/bold yellow]\n"
        f"[dim]Action: {action}[/dim]",
        border_style="yellow"
    ))
    
    async def run_reflex():
        payload = {"action": action}
        
        if action == "scan_ghosts":
            if not path:
                return {"error": "path required for scan_ghosts"}
            payload["path"] = path
        elif action == "apply_edit":
            if not all([path, target_id, new_code]):
                return {"error": "path, target_id (--target), and new_code (--code) required for apply_edit"}
            payload["path"] = path
            payload["target_id"] = target_id
            payload["new_code"] = new_code
        elif action == "read_file":
            if not path:
                return {"error": "path required for read_file"}
            payload["path"] = path
        elif action == "run_cmd":
            if not command:
                return {"error": "command (--command) required for run_cmd"}
            payload["command"] = command
        else:
            return {"error": f"Unknown action: {action}. Valid: scan_ghosts, apply_edit, read_file, run_cmd"}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:8766/v1/local/reflex",
                json={"payload": payload}
            )
            return resp.json()
    
    with console.status(f"[bold green]Executing {action}..."):
        result = asyncio.run(run_reflex())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    if action == "scan_ghosts":
        console.print(Panel(
            result.get("skeleton", ""),
            title="[cyan]👻 Ghost Skeleton[/cyan]",
            border_style="cyan"
        ))
        anchors = result.get("anchors", {})
        if anchors:
            table = Table(title="Anchors", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Line", style="green")
            table.add_column("Name", style="yellow")
            table.add_column("Type", style="dim")
            for hash_id, info in anchors.items():
                table.add_row(f"§{hash_id}", str(info.get('line', '')), info.get('name', ''), info.get('type', ''))
            console.print(table)
        console.print(f"[dim]Tokens saved: ~{result.get('tokens_saved_est', 0)}[/dim]")
    else:
        console.print(result)


@app.command("consult")
def consult(
    query: str = typer.Argument(..., help="Question for the memory system"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max memories to return"),
    summarize: bool = typer.Option(False, "--summarize", "-s", help="Summarize results with Cloud AI")
):
    """Consult the Hippocampus memory (The Advisor)."""
    import httpx
    
    console.print(Panel.fit(
        f"🔮 [bold magenta]Cloud Consult[/bold magenta]\n"
        f"[dim]Query: {query[:50]}...[/dim]",
        border_style="magenta"
    ))
    
    async def run_consult():
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:8766/v1/cloud/consult",
                json={"payload": {"query": query, "limit": limit, "summarize": summarize}}
            )
            return resp.json()
    
    with console.status("[bold green]Consulting memory..."):
        result = asyncio.run(run_consult())
    
    if "error" in result:
        console.print(f"[red]❌ {result['error']}[/red]")
        return
    
    advice = result.get("advice", [])
    if not advice:
        console.print("[yellow]No memories found.[/yellow]")
        return
    
    console.print(f"[green]Found {len(advice)} memories from {result.get('source', 'unknown')}[/green]\n")
    for i, mem in enumerate(advice, 1):
        content = mem.get("content", str(mem))[:200]
        console.print(Panel(content, title=f"[cyan]#{i}[/cyan]", border_style="dim"))


# ============================================================================
# VERSION
# ============================================================================

@app.command("version")
def version():
    """Show VMT-CLI version."""
    console.print("[bold cyan]VMT-CLI[/bold cyan] v1.1.0 [dim](Puppeteer Edition)[/dim]")
    console.print("[dim]Console interface for the 9-lobe RLM architecture[/dim]")
    console.print("[dim]Commands: discuss | reflex | consult[/dim]")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
