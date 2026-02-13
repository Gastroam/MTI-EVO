"""
VMT-CLI Memory Commands
=======================

Memory management commands for the exocortex.
"""
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt
from pathlib import Path

console = Console()

# Create the memory command group
memory_app = typer.Typer(
    name="memory",
    help="🧠 Memory management (list, search, insert, delete)"
)


def get_surgeon():
    """Get or create MemorySurgeon instance."""
    from brain.memory_surgeon import MemorySurgeon
    brain_path = Path.home() / ".mti-brain"
    return MemorySurgeon(str(brain_path))


@memory_app.command("list")
def memory_list(
    tier: Optional[str] = typer.Option(None, "--tier", "-t", help="Filter by tier (immortal/protected/transient)")
):
    """List all memories by tier."""
    from brain.memory_surgeon import MemoryTier
    
    surgeon = get_surgeon()
    
    # Filter by tier if specified
    filter_tier = None
    if tier:
        try:
            filter_tier = MemoryTier(tier.lower())
        except ValueError:
            console.print(f"[red]Invalid tier: {tier}. Use immortal/protected/transient[/red]")
            return
    
    memories = surgeon.list_memories(tier=filter_tier)
    
    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        return
    
    # Create table by tier
    tier_colors = {
        MemoryTier.IMMORTAL: "red",
        MemoryTier.PROTECTED: "yellow",
        MemoryTier.TRANSIENT: "green"
    }
    
    for tier_type in MemoryTier:
        tier_memories = [m for m in memories if m.tier == tier_type]
        if not tier_memories:
            continue
            
        color = tier_colors[tier_type]
        table = Table(
            title=f"[{color}]{tier_type.value.upper()}[/{color}] ({len(tier_memories)})",
            show_header=True,
            header_style=f"bold {color}"
        )
        table.add_column("ID", style="dim", width=10)
        table.add_column("Content", max_width=50)
        table.add_column("Tags", style="cyan")
        table.add_column("Access", justify="right")
        
        for mem in tier_memories[:10]:  # Show max 10 per tier
            content_preview = mem.content[:47] + "..." if len(mem.content) > 50 else mem.content
            tags = ", ".join(mem.tags[:3]) if mem.tags else "-"
            table.add_row(mem.id, content_preview, tags, str(mem.access_count))
        
        console.print(table)
        console.print()
    
    # Stats
    stats = surgeon.get_stats()
    console.print(f"[dim]Total: {stats['total_memories']} memories | Backups: {stats['backup_count']}[/dim]")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results")
):
    """Semantic search through memories."""
    surgeon = get_surgeon()
    
    console.print(Panel.fit(f"🔍 [bold]Searching:[/bold] {query}", border_style="magenta"))
    
    results = surgeon.search_memories(query, limit=limit)
    
    if not results:
        console.print("[yellow]No matching memories found.[/yellow]")
        return
    
    for mem, score in results:
        if mem:
            console.print(Panel(
                f"[dim]{mem.tier.value}[/dim] | {mem.content[:200]}",
                title=f"[cyan]{mem.id}[/cyan] Score: {score:.3f}",
                border_style="dim"
            ))


@memory_app.command("inspect")
def memory_inspect(
    memory_id: str = typer.Argument(..., help="Memory ID to inspect")
):
    """View full details of a memory."""
    surgeon = get_surgeon()
    
    memory = surgeon.get_memory(memory_id)
    
    if not memory:
        console.print(f"[red]Memory {memory_id} not found.[/red]")
        return
    
    # Build detail panel
    from datetime import datetime
    created = datetime.fromtimestamp(memory.created_at).strftime("%Y-%m-%d %H:%M")
    accessed = datetime.fromtimestamp(memory.last_accessed).strftime("%Y-%m-%d %H:%M")
    
    details = f"""[bold]ID:[/bold] {memory.id}
[bold]Tier:[/bold] {memory.tier.value}
[bold]Source:[/bold] {memory.source}
[bold]Created:[/bold] {created}
[bold]Last Accessed:[/bold] {accessed}
[bold]Access Count:[/bold] {memory.access_count}
[bold]Tags:[/bold] {', '.join(memory.tags) if memory.tags else 'None'}

[bold]Content:[/bold]
{memory.content}
"""
    
    if memory.metadata:
        details += f"\n[bold]Metadata:[/bold] {memory.metadata}"
    
    console.print(Panel(details, title=f"Memory {memory_id}", border_style="cyan"))


@memory_app.command("insert")
def memory_insert(
    content: str = typer.Argument(..., help="Memory content"),
    tier: str = typer.Option("transient", "--tier", "-t", help="Tier: immortal/protected/transient"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    force: bool = typer.Option(False, "--force", help="Force for IMMORTAL tier")
):
    """Insert a new memory."""
    from brain.memory_surgeon import MemoryTier
    
    surgeon = get_surgeon()
    
    try:
        memory_tier = MemoryTier(tier.lower())
    except ValueError:
        console.print(f"[red]Invalid tier: {tier}[/red]")
        return
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    
    # Confirm for IMMORTAL
    if memory_tier == MemoryTier.IMMORTAL and not force:
        console.print("[yellow]⚠️ Creating IMMORTAL memory (cannot be deleted)[/yellow]")
        if not Confirm.ask("Are you sure?"):
            console.print("[dim]Cancelled.[/dim]")
            return
        force = True
    
    result = surgeon.insert_memory(
        content=content,
        tier=memory_tier,
        tags=tag_list,
        force=force
    )
    
    if result.success:
        console.print(f"[green]✓ Memory created: {result.affected_memories[0]}[/green]")
    else:
        console.print(f"[red]❌ {result.error}[/red]")


@memory_app.command("delete")
def memory_delete(
    memory_id: str = typer.Argument(..., help="Memory ID to delete"),
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Confirm deletion"),
    force: bool = typer.Option(False, "--force", help="Force delete IMMORTAL")
):
    """Delete a memory (with safety checks)."""
    surgeon = get_surgeon()
    
    memory = surgeon.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found.[/red]")
        return
    
    # Show what we're deleting
    console.print(Panel(
        f"[bold]Tier:[/bold] {memory.tier.value}\n{memory.content[:200]}",
        title=f"🗑️ Delete {memory_id}?",
        border_style="red"
    ))
    
    # Confirm interactively if not --confirm
    if not confirm:
        confirm = Confirm.ask("Delete this memory?")
    
    if not confirm:
        console.print("[dim]Cancelled.[/dim]")
        return
    
    result = surgeon.delete_memory(memory_id, confirm=True, force=force)
    
    if result.success:
        console.print(f"[green]✓ Memory {memory_id} deleted[/green]")
        if result.backup_path:
            console.print(f"[dim]Backup: {result.backup_path}[/dim]")
    else:
        console.print(f"[red]❌ {result.error}[/red]")


@memory_app.command("demote")
def memory_demote(
    memory_id: str = typer.Argument(..., help="Memory ID to demote"),
    force: bool = typer.Option(False, "--force-demote", help="Force demote IMMORTAL")
):
    """Demote a memory to a lower protection tier."""
    from brain.memory_surgeon import MemoryTier
    
    surgeon = get_surgeon()
    
    memory = surgeon.get_memory(memory_id)
    if not memory:
        console.print(f"[red]Memory {memory_id} not found.[/red]")
        return
    
    # Determine target tier
    if memory.tier == MemoryTier.IMMORTAL:
        target = MemoryTier.PROTECTED
    elif memory.tier == MemoryTier.PROTECTED:
        target = MemoryTier.TRANSIENT
    else:
        console.print("[yellow]Memory is already TRANSIENT (lowest tier).[/yellow]")
        return
    
    console.print(f"[yellow]Demoting {memory.tier.value} → {target.value}[/yellow]")
    
    result = surgeon.demote_memory(memory_id, target, force=force)
    
    if result.success:
        console.print(f"[green]✓ Memory demoted to {target.value}[/green]")
    else:
        console.print(f"[red]❌ {result.error}[/red]")


@memory_app.command("export")
def memory_export(
    output: str = typer.Option("memories_export.json", "--output", "-o", help="Output file")
):
    """Export all memories to JSON."""
    import json
    
    surgeon = get_surgeon()
    memories = surgeon.list_memories()
    
    data = {
        "exported_at": __import__("time").time(),
        "memories": [m.to_dict() for m in memories]
    }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    console.print(f"[green]✓ Exported {len(memories)} memories to {output}[/green]")


@memory_app.command("backup")
def memory_backup():
    """Create a manual backup of all memories."""
    surgeon = get_surgeon()
    
    backup_path = surgeon._create_backup("manual")
    console.print(f"[green]✓ Backup created: {backup_path}[/green]")
    
    stats = surgeon.get_stats()
    console.print(f"[dim]Total backups: {stats['backup_count']}[/dim]")


@memory_app.command("backups")
def memory_backups():
    """List available backups."""
    surgeon = get_surgeon()
    
    backups = surgeon.list_backups()
    
    if not backups:
        console.print("[yellow]No backups found.[/yellow]")
        return
    
    table = Table(title="Available Backups", show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("Label", style="green")
    table.add_column("Memories", justify="right")
    
    from datetime import datetime
    for backup in backups[:10]:
        date = datetime.fromtimestamp(backup["created_at"]).strftime("%Y-%m-%d %H:%M")
        table.add_row(date, backup["label"], str(backup["memory_count"]))
    
    console.print(table)
