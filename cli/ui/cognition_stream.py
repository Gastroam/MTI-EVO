"""
Cognition Stream UI - Rich terminal visualization of agent thinking.
Shows lobe activation, reasoning steps, and streaming response.
"""
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, BarColumn, TextColumn
from typing import Dict, Optional, List


class CognitionStreamUI:
    """Real-time visualization of agent cognition in terminal."""
    
    def __init__(self, console: Console):
        self.console = console
        self.current_text = ""
        self.thinking = True
        self.current_lobe = "prefrontal"
        self.lobe_activations: Dict[str, float] = {
            "prefrontal": 0.0,
            "parietal": 0.0,
            "hippocampus": 0.0,
            "wernicke": 0.0,
            "broca": 0.0,
            "occipital": 0.0,
            "limbic": 0.0,
            "motor": 0.0,
            "homeostasis": 0.0,
        }
        self.steps: List[str] = []
        self.token_count = 0
        self.status_message = "Initializing..."
    
    def update(self, event: Dict):
        """Update UI state from event."""
        event_type = event.get("type", "")
        
        if event_type == "thinking":
            self.thinking = True
            self.status_message = event.get("message", "Thinking...")
            
        elif event_type == "lobe_activation":
            lobe = event.get("lobe", "").lower()
            activation = event.get("activation", 0.0)
            if lobe in self.lobe_activations:
                self.lobe_activations[lobe] = activation
            self.current_lobe = lobe
            
        elif event_type == "step":
            step = event.get("step", "")
            if step:
                self.steps.append(step)
                # Keep last 5 steps
                self.steps = self.steps[-5:]
            
        elif event_type == "token":
            token = event.get("token", "")
            self.current_text += token
            self.token_count += 1
            self.thinking = False
            
        elif event_type == "chunk":
            chunk = event.get("text", "")
            self.current_text += chunk
            self.thinking = False
            
        elif event_type == "final":
            self.thinking = False
            self.current_text = event.get("text", self.current_text)
            self.status_message = "Complete"
            
        elif event_type == "error":
            self.thinking = False
            self.current_text = f"[red]Error: {event.get('error', 'Unknown error')}[/red]"
    
    def _render_lobe_bars(self) -> Table:
        """Render lobe activation bars."""
        table = Table.grid(expand=True)
        table.add_column(width=12)
        table.add_column(width=22)
        
        for lobe, activation in self.lobe_activations.items():
            bar_len = int(activation * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # Color based on whether this is the active lobe
            if lobe == self.current_lobe and activation > 0:
                color = "bold cyan"
            elif activation > 0.5:
                color = "green"
            elif activation > 0:
                color = "yellow"
            else:
                color = "dim"
            
            table.add_row(
                Text(lobe[:10].title(), style=color),
                Text(f"{bar} {activation:.0%}", style=color)
            )
        
        return table
    
    def _render_steps(self) -> Text:
        """Render reasoning steps."""
        text = Text()
        for i, step in enumerate(self.steps):
            if i == len(self.steps) - 1:
                text.append(f"→ {step}\n", style="bold cyan")
            else:
                text.append(f"  {step}\n", style="dim")
        return text
    
    def get_renderable(self) -> RenderableType:
        """Get the complete renderable for Live display."""
        # Header with status
        if self.thinking:
            header = Text()
            header.append("🧠 ", style="bold")
            header.append(self.status_message, style="bold cyan")
            header.append(" ")
            header.append("⣾", style="bold yellow")  # Simple spinner char
        else:
            header = Text("🧠 Response", style="bold green")
        
        # Build content
        content = Text()
        
        # Steps section (if any)
        if self.steps:
            content.append("\n[Reasoning]\n", style="dim")
            content.append_text(self._render_steps())
            content.append("\n")
        
        # Response text
        if self.current_text:
            # Truncate for live display
            display_text = self.current_text[-2000:] if len(self.current_text) > 2000 else self.current_text
            content.append(display_text)
        elif self.thinking:
            content.append("Processing...", style="dim italic")
        
        # Footer with token count
        if self.token_count > 0:
            content.append(f"\n\n[dim]Tokens: {self.token_count}[/dim]")
        
        # Combine into panel
        main_panel = Panel(
            content,
            title=header,
            border_style="cyan" if self.thinking else "green",
            padding=(0, 1)
        )
        
        # Side panel with lobe activations (if any activity)
        if any(v > 0 for v in self.lobe_activations.values()):
            from rich.columns import Columns
            lobe_panel = Panel(
                self._render_lobe_bars(),
                title="[dim]Lobe Activity[/dim]",
                border_style="dim",
                width=40
            )
            return Columns([main_panel, lobe_panel], expand=True)
        
        return main_panel
