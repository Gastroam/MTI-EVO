
from typing import Any, Protocol, Dict, List, Optional

class MTIPlugin(Protocol):
    """
    Protocol for MTI-EVO Plugins.
    Plugins attach to the lattice and subscribe to events.
    """
    def attach(self, lattice: Any) -> None:
        """
        Called when the plugin is attached to the lattice.
        Use this to subscribe to events or initialize plugin state.
        """
        ...
        
    def detach(self) -> None:
        """
        Called when the plugin is removed or the system shuts down.
        """
        ...

class PluginManager:
    """
    Manages plugin lifecycle and event dispatch.
    """
    def __init__(self):
        self.plugins: List[MTIPlugin] = []
        self.subscribers: Dict[str, List[Any]] = {}
        
    def register(self, plugin: MTIPlugin, lattice: Any):
        """Register and attach a plugin."""
        self.plugins.append(plugin)
        plugin.attach(lattice)
        
    def subscribe(self, event_name: str, callback):
        """Subscribe a callback to an event."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)
        
    def trigger(self, event_name: str, payload: Any = None):
        """Trigger an event, notifying all subscribers."""
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                try:
                    callback(payload)
                except Exception as e:
                    # Plugins should not crash the core
                    # In a real system we'd log this, but avoiding circular logging imports for now
                    print(f"Plugin error handling {event_name}: {e}")

class NullPlugin:
    """No-op plugin for default configuration."""
    def attach(self, lattice: Any) -> None:
        pass
        
    def detach(self) -> None:
        pass
