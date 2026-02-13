"""
VMT-CLI Dream Commands
======================

Dream engine commands for memory consolidation.
Implements the "Semantic Sleep" pattern.
"""
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from pathlib import Path
from datetime import datetime

console = Console()

# Create the dream command group
dream_app = typer.Typer(
    name="dream",
    help="💤 Dream engine (consolidate, prune, journal)"
)


def get_surgeon():
    """Get or create MemorySurgeon instance."""
    from brain.memory_surgeon import MemorySurgeon
    brain_path = Path.home() / ".mti-brain"
    return MemorySurgeon(str(brain_path))


def get_dream_engine(workspace: str = "."):
    """Get DreamEngine instance if available."""
    try:
        from dream_engine import DreamEngine
        return DreamEngine(workspace_path=workspace)
    except ImportError:
        return None


@dream_app.command("status")
def dream_status():
    """Show dream engine status."""
    surgeon = get_surgeon()
    stats = surgeon.get_stats()
    journal = surgeon.get_journal(limit=5)
    
    console.print(Panel.fit("💤 [bold cyan]Dream Engine Status[/bold cyan]", border_style="cyan"))
    
    # Memory stats
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Memories", str(stats["total_memories"]))
    table.add_row("Transient (pruneable)", str(stats["transient_count"]))
    table.add_row("Surgery Mode", "🔴 Active" if stats["is_anesthetized"] else "🟢 Normal")
    
    console.print(table)
    
    # Recent journal entries
    if journal:
        console.print("\n[bold]Recent Dream Journal:[/bold]")
        for entry in reversed(journal):
            date = datetime.fromisoformat(entry.get("date", "")).strftime("%Y-%m-%d %H:%M")
            entry_type = entry.get("type", "unknown")
            console.print(f"  [{date}] {entry_type}")
    
    # Dream engine status
    engine = get_dream_engine()
    if engine:
        status = engine.get_status()
        console.print(f"\n[bold]Engine Mode:[/bold] {status.get('mode', 'unknown')}")
        console.print(f"[bold]Scenarios Queued:[/bold] {status.get('scenarios_queued', 0)}")
        
        if status.get('last_dream_time'):
            last = datetime.fromtimestamp(status['last_dream_time']).strftime("%Y-%m-%d %H:%M")
            console.print(f"[bold]Last Dream:[/bold] {last}")
    else:
        console.print("\n[dim]Dream engine not initialized[/dim]")


@dream_app.command("consolidate")
def dream_consolidate():
    """
    Trigger manual memory consolidation.
    
    This process:
    1. Identifies similar transient memories
    2. Merges duplicates
    3. Promotes important memories to PROTECTED
    4. Logs the session to the dream journal
    """
    surgeon = get_surgeon()
    
    console.print(Panel.fit("🌙 [bold]Memory Consolidation[/bold]", border_style="purple"))
    
    memories = surgeon.list_memories()
    transient = [m for m in memories if m.tier.value == "transient"]
    
    console.print(f"[dim]Found {len(transient)} transient memories to analyze[/dim]")
    
    if not transient:
        console.print("[yellow]No transient memories to consolidate.[/yellow]")
        return
    
    # Simple consolidation logic
    consolidated = 0
    promoted = 0
    
    with console.status("[bold purple]Dreaming..."):
        # Find frequently accessed transient memories -> promote
        for mem in transient:
            if mem.access_count >= 5:
                # Promote to PROTECTED
                from brain.memory_surgeon import MemoryTier
                mem.tier = MemoryTier.PROTECTED
                mem.metadata["promoted_from"] = "transient"
                mem.metadata["promoted_reason"] = f"high_access_{mem.access_count}"
                promoted += 1
        
        # Find very similar memories -> merge
        # (Simplified: just count unique content hashes)
        content_hashes = {}
        for mem in transient:
            h = hash(mem.content[:100])  # Simple hash
            if h in content_hashes:
                # Duplicate found
                content_hashes[h].access_count += mem.access_count
                surgeon.delete_memory(mem.id, confirm=True, force=True)
                consolidated += 1
            else:
                content_hashes[h] = mem
        
        surgeon._save_memories()
    
    # Log to journal
    surgeon.add_journal_entry({
        "type": "consolidation",
        "transient_processed": len(transient),
        "consolidated": consolidated,
        "promoted": promoted
    })
    
    console.print(f"\n[green]✨ Consolidation complete![/green]")
    console.print(f"  Promoted to PROTECTED: {promoted}")
    console.print(f"  Duplicates merged: {consolidated}")
    
    # Show what dream journal would say
    console.print(Panel(
        f"[italic]Last night I processed {len(transient)} recent memories. "
        f"I promoted {promoted} important ones and merged {consolidated} duplicates. "
        f"The brain feels cleaner now.[/italic]",
        title="💭 Dream Journal Entry",
        border_style="purple"
    ))


@dream_app.command("prune")
def dream_prune(
    days: int = typer.Option(2, "--days", "-d", help="Prune memories older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be pruned")
):
    """
    Prune old transient memories.
    
    By default, removes TRANSIENT memories not accessed in 48 hours.
    """
    import time
    surgeon = get_surgeon()
    
    threshold = time.time() - (days * 24 * 60 * 60)
    memories = surgeon.list_memories()
    
    to_prune = [
        m for m in memories 
        if m.tier.value == "transient" and m.last_accessed < threshold
    ]
    
    if not to_prune:
        console.print(f"[green]No stale memories older than {days} days.[/green]")
        return
    
    console.print(Panel.fit(f"🧹 [bold]Pruning {len(to_prune)} stale memories[/bold]", border_style="yellow"))
    
    table = Table(show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Last Access", style="cyan")
    table.add_column("Content Preview")
    
    for mem in to_prune[:10]:
        last = datetime.fromtimestamp(mem.last_accessed).strftime("%Y-%m-%d")
        content = mem.content[:40] + "..." if len(mem.content) > 40 else mem.content
        table.add_row(mem.id, last, content)
    
    if len(to_prune) > 10:
        table.add_row("...", "...", f"and {len(to_prune) - 10} more")
    
    console.print(table)
    
    if dry_run:
        console.print("\n[yellow]Dry run - no memories deleted.[/yellow]")
        return
    
    if not Confirm.ask(f"Delete {len(to_prune)} memories?"):
        console.print("[dim]Cancelled.[/dim]")
        return
    
    for mem in to_prune:
        surgeon.delete_memory(mem.id, confirm=True)
    
    surgeon.add_journal_entry({
        "type": "prune",
        "pruned_count": len(to_prune),
        "threshold_days": days
    })
    
    console.print(f"[green]✓ Pruned {len(to_prune)} stale memories[/green]")


@dream_app.command("journal")
def dream_journal(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of entries")
):
    """View the dream journal (consolidation history)."""
    surgeon = get_surgeon()
    
    journal = surgeon.get_journal(limit=limit)
    
    if not journal:
        console.print("[yellow]Dream journal is empty.[/yellow]")
        console.print("[dim]Run 'vmt dream consolidate' to start dreaming.[/dim]")
        return
    
    console.print(Panel.fit("📔 [bold]Dream Journal[/bold]", border_style="purple"))
    
    for entry in reversed(journal):
        date = entry.get("date", "Unknown")
        if isinstance(date, str) and "T" in date:
            date = datetime.fromisoformat(date).strftime("%Y-%m-%d %H:%M")
        
        entry_type = entry.get("type", "unknown")
        
        if entry_type == "consolidation":
            msg = f"Processed {entry.get('transient_processed', 0)} memories, promoted {entry.get('promoted', 0)}, merged {entry.get('consolidated', 0)}"
        elif entry_type == "prune":
            msg = f"Pruned {entry.get('pruned_count', 0)} memories older than {entry.get('threshold_days', '?')} days"
        elif entry_type == "healing_session":
            msg = f"Healed {entry.get('healed', 0)}/{entry.get('total', 0)} pain points"
        else:
            msg = str(entry)
        
        emoji = "🌙" if entry_type == "consolidation" else "🧹" if entry_type == "prune" else "🏥"
        console.print(f"[{date}] {emoji} {msg}")


@dream_app.command("scenarios")
def dream_scenarios():
    """List pending dream scenarios (TODOs, design notes)."""
    engine = get_dream_engine(".")
    
    if not engine:
        console.print("[yellow]Dream engine not available.[/yellow]")
        return
    
    console.print(Panel.fit("💭 [bold]Dream Scenarios[/bold]", border_style="cyan"))
    
    # Scan for TODOs
    with console.status("[bold cyan]Scanning for scenarios..."):
        engine.scan_todos()
    
    status = engine.get_status()
    scenarios = status.get("scenarios_queued", 0)
    
    if scenarios == 0:
        console.print("[dim]No scenarios queued.[/dim]")
    else:
        console.print(f"[bold]{scenarios} scenarios ready to dream about[/bold]")
        console.print("[dim]Run 'vmt dream consolidate' to process them[/dim]")


@dream_app.command("replay")
def dream_replay(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to replay")
):
    """Replay a past session for learning reinforcement."""
    surgeon = get_surgeon()
    
    console.print(Panel.fit("🔄 [bold]Session Replay[/bold]", border_style="green"))
    
    if session_id:
        console.print(f"[dim]Replaying session: {session_id}[/dim]")
        # Would need session storage to implement fully
        console.print("[yellow]Session replay not yet implemented.[/yellow]")
    else:
        console.print("[dim]Available sessions:[/dim]")
        console.print("[yellow]Session storage not yet implemented.[/yellow]")
