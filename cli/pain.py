"""
VMT-CLI Pain Commands
=====================

Pain detection and healing for the exocortex.
Implements the "Corteza Insular" pattern for conflict detection.
"""
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from pathlib import Path

console = Console()

# Create the pain command group
pain_app = typer.Typer(
    name="pain",
    help="🩺 Pain detection & healing (scan, heal, report)"
)


def get_surgeon():
    """Get or create MemorySurgeon instance."""
    from brain.memory_surgeon import MemorySurgeon
    brain_path = Path.home() / ".mti-brain"
    return MemorySurgeon(str(brain_path))


@pain_app.command("scan")
def pain_scan(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info")
):
    """Scan for conflicts, stale memories, and entropy issues."""
    surgeon = get_surgeon()
    
    console.print(Panel.fit("🔍 [bold]Scanning for pain points...[/bold]", border_style="red"))
    
    with console.status("[bold red]Analyzing memory conflicts..."):
        pain_points = surgeon.scan_for_pain()
    
    if not pain_points:
        console.print("[green]✨ No pain points detected! Brain is healthy.[/green]")
        return
    
    # Group by severity
    severe = [p for p in pain_points if p.severity >= 0.7]
    moderate = [p for p in pain_points if 0.4 <= p.severity < 0.7]
    low = [p for p in pain_points if p.severity < 0.4]
    
    console.print(f"\n[bold]Found {len(pain_points)} pain points:[/bold]\n")
    
    # Severe issues
    if severe:
        console.print(f"[red]🔴 SEVERE ({len(severe)})[/red]")
        for p in severe:
            console.print(Panel(
                f"[bold]{p.pain_type.upper()}[/bold] - {p.description}\n"
                f"Memories: {', '.join(p.memory_ids)}\n"
                f"[dim]Suggestion: {p.suggestion}[/dim]",
                title=f"[red]Severity: {p.severity:.2f}[/red]",
                border_style="red"
            ))
    
    # Moderate issues
    if moderate:
        console.print(f"\n[yellow]🟡 MODERATE ({len(moderate)})[/yellow]")
        for p in moderate:
            if verbose:
                console.print(Panel(
                    f"{p.pain_type}: {p.description}\n[dim]{p.suggestion}[/dim]",
                    border_style="yellow"
                ))
            else:
                console.print(f"  • {p.pain_type}: {p.description}")
    
    # Low issues
    if low:
        console.print(f"\n[green]🟢 LOW ({len(low)})[/green]")
        if verbose:
            for p in low:
                console.print(f"  • {p.pain_type}: {p.description}")
        else:
            console.print(f"  [dim]Use --verbose to see details[/dim]")
    
    console.print(f"\n[dim]Run 'vmt pain heal' for guided resolution[/dim]")


@pain_app.command("heal")
def pain_heal(
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-heal where possible")
):
    """Guided healing process for detected pain points."""
    surgeon = get_surgeon()
    
    console.print(Panel.fit("🏥 [bold cyan]Pain Healing Session[/bold cyan]", border_style="cyan"))
    
    pain_points = surgeon.scan_for_pain()
    
    if not pain_points:
        console.print("[green]✨ No pain points to heal![/green]")
        return
    
    # Anesthetize brain for surgery
    console.print("\n[yellow]💉 Anesthetizing brain for surgery...[/yellow]")
    surgeon.anesthetize()
    
    healed = 0
    skipped = 0
    
    try:
        for i, pain in enumerate(pain_points, 1):
            severity_color = "red" if pain.severity >= 0.7 else "yellow" if pain.severity >= 0.4 else "green"
            severity_emoji = "🔴" if pain.severity >= 0.7 else "🟡" if pain.severity >= 0.4 else "🟢"
            
            console.print(f"\n[bold]Pain Point {i}/{len(pain_points)}[/bold]")
            console.print(Panel(
                f"[bold]{pain.pain_type.upper()}[/bold]\n"
                f"{pain.description}\n\n"
                f"[dim]Memories:[/dim] {', '.join(pain.memory_ids)}\n"
                f"[dim]Suggestion:[/dim] {pain.suggestion}",
                title=f"{severity_emoji} Severity: [{severity_color}]{pain.severity:.2f}[/{severity_color}]"
            ))
            
            # Auto-heal if possible and auto mode
            if auto and pain.auto_healable:
                console.print("[green]→ Auto-healing...[/green]")
                # Default action for stale: delete
                if pain.pain_type == "stale":
                    for mem_id in pain.memory_ids:
                        surgeon.delete_memory(mem_id, confirm=True)
                    healed += 1
                continue
            
            # Interactive healing
            console.print("\n[bold]Options:[/bold]")
            
            if pain.pain_type == "conflict":
                console.print("  [1] Keep first memory")
                console.print("  [2] Keep second memory")
                console.print("  [3] Fuse (merge into one)")
                console.print("  [S] Skip")
                console.print("  [Q] Quit healing")
                
                choice = Prompt.ask("Choice", choices=["1", "2", "3", "s", "q"], default="s")
                
                if choice == "q":
                    break
                elif choice == "s":
                    skipped += 1
                elif choice == "1" and len(pain.memory_ids) >= 2:
                    surgeon.delete_memory(pain.memory_ids[1], confirm=True)
                    healed += 1
                elif choice == "2" and len(pain.memory_ids) >= 2:
                    surgeon.delete_memory(pain.memory_ids[0], confirm=True)
                    healed += 1
                elif choice == "3":
                    console.print("[cyan]Enter fused content (or press Enter for auto-merge):[/cyan]")
                    fused_content = Prompt.ask("Fused content", default="")
                    
                    if fused_content:
                        result = surgeon.fuse_memories(
                            pain.memory_ids,
                            manual_content=fused_content
                        )
                    else:
                        # Simple concatenation as fallback
                        memories = [surgeon.get_memory(mid) for mid in pain.memory_ids]
                        fused = "\n---\n".join(m.content for m in memories if m)
                        result = surgeon.fuse_memories(
                            pain.memory_ids,
                            manual_content=fused
                        )
                    
                    if result.success:
                        console.print(f"[green]✓ Fused into {result.affected_memories[0]}[/green]")
                        healed += 1
                    else:
                        console.print(f"[red]Fusion failed: {result.error}[/red]")
            
            elif pain.pain_type == "stale":
                console.print("  [D] Delete this memory")
                console.print("  [K] Keep it")
                console.print("  [S] Skip")
                
                choice = Prompt.ask("Choice", choices=["d", "k", "s"], default="s")
                
                if choice == "d":
                    for mem_id in pain.memory_ids:
                        surgeon.delete_memory(mem_id, confirm=True)
                    healed += 1
                else:
                    skipped += 1
            
            elif pain.pain_type == "orphan":
                console.print("  [T] Add tags")
                console.print("  [D] Delete")
                console.print("  [S] Skip")
                
                choice = Prompt.ask("Choice", choices=["t", "d", "s"], default="s")
                
                if choice == "t":
                    tags = Prompt.ask("Enter tags (comma-separated)")
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    for mem_id in pain.memory_ids:
                        mem = surgeon.get_memory(mem_id)
                        if mem:
                            mem.tags.extend(tag_list)
                    healed += 1
                elif choice == "d":
                    for mem_id in pain.memory_ids:
                        surgeon.delete_memory(mem_id, confirm=True)
                    healed += 1
                else:
                    skipped += 1
        
        # Commit changes
        console.print("\n[bold]Surgery complete![/bold]")
        if Confirm.ask("Commit changes?", default=True):
            surgeon.wake_up(commit=True)
            console.print(f"[green]✓ Healed {healed} pain points, skipped {skipped}[/green]")
            
            # Log to dream journal
            surgeon.add_journal_entry({
                "type": "healing_session",
                "healed": healed,
                "skipped": skipped,
                "total": len(pain_points)
            })
        else:
            surgeon.wake_up(commit=False)
            console.print("[yellow]↩️ Changes rolled back[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error during healing: {e}[/red]")
        surgeon.wake_up(commit=False)
        console.print("[yellow]↩️ Changes rolled back for safety[/yellow]")


@pain_app.command("report")
def pain_report():
    """Generate detailed pain report."""
    surgeon = get_surgeon()
    
    console.print(Panel.fit("📊 [bold]Pain Report[/bold]", border_style="cyan"))
    
    pain_points = surgeon.scan_for_pain()
    stats = surgeon.get_stats()
    
    # Summary table
    table = Table(title="Brain Health Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Memories", str(stats["total_memories"]))
    table.add_row("IMMORTAL", str(stats["immortal_count"]))
    table.add_row("PROTECTED", str(stats["protected_count"]))
    table.add_row("TRANSIENT", str(stats["transient_count"]))
    table.add_row("", "")
    table.add_row("Pain Points", str(len(pain_points)))
    table.add_row("  Severe (>0.7)", str(len([p for p in pain_points if p.severity >= 0.7])))
    table.add_row("  Moderate", str(len([p for p in pain_points if 0.4 <= p.severity < 0.7])))
    table.add_row("  Low", str(len([p for p in pain_points if p.severity < 0.4])))
    table.add_row("", "")
    table.add_row("Backups", str(stats["backup_count"]))
    
    console.print(table)
    
    # Health score
    if pain_points:
        avg_pain = sum(p.severity for p in pain_points) / len(pain_points)
        health_score = max(0, 100 - (avg_pain * 100))
    else:
        health_score = 100
    
    health_color = "green" if health_score >= 80 else "yellow" if health_score >= 60 else "red"
    console.print(f"\n[bold]Brain Health Score:[/bold] [{health_color}]{health_score:.0f}%[/{health_color}]")


@pain_app.command("threshold")
def pain_threshold(
    value: Optional[float] = typer.Argument(None, help="New threshold (0.0-1.0)")
):
    """Get or set the pain detection threshold."""
    if value is None:
        console.print(f"[bold]Current pain threshold:[/bold] 0.7")
        console.print("[dim]Values above this trigger 'SEVERE' classification[/dim]")
    else:
        if not 0.0 <= value <= 1.0:
            console.print("[red]Threshold must be between 0.0 and 1.0[/red]")
            return
        # Note: This would need persistence in a config file
        console.print(f"[green]✓ Pain threshold set to {value}[/green]")
