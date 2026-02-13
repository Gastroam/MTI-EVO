"""
MTI-EVO Engine Registry
=======================
Central registry for pluggable inference engines.
"""
from typing import Dict, Any, Type, Optional
from mti_evo.core.logger import get_logger

logger = get_logger("EngineRegistry")

# Protocol definition would ideally be in core.interfaces, but duck typing works for now
# We expect engines to have .load_model(), .infer(), .unload()

class EngineRegistry:
    _engines: Dict[str, Type] = {}
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type, metadata: Dict = None):
        """Register a new engine class."""
        cls._engines[name] = engine_class
        logger.debug(f"Registered engine: {name}")
        
    @classmethod
    def get_engine_class(cls, name: str) -> Optional[Type]:
        return cls._engines.get(name)
        
    @classmethod
    def create(cls, name: str, config: Any) -> Any:
        """Instantiate an engine by name."""
        engine_cls = cls._engines.get(name)
        if not engine_cls:
            raise ValueError(f"Engine '{name}' not found in registry.")
            
        instance = engine_cls(config)
        cls._instances[name] = instance
        return instance

    @classmethod
    def list_engines(cls):
        return list(cls._engines.keys())

# Auto-discovery
def discover_engines():
    """Import known engines to trigger registration."""
    # This is where we import the engines so their decorators (if any) or manual calls run.
    # For now, we manually import the known ones.
    try:
        from mti_evo.engines.gguf_engine import GGUFEngine
        # Register them
        EngineRegistry.register("gguf", GGUFEngine)

        
        # Try loading experimental ones
        try:
             from mti_evo.engines.quantum_engine import QuantumEngine
             EngineRegistry.register("quantum", QuantumEngine)
        except ImportError: pass

        try:
             from mti_evo.engines.api_engine import APIEngine
             EngineRegistry.register("api", APIEngine)
        except ImportError: pass

    except ImportError as e:
        logger.warning(f"Could not discover some core engines: {e}")

    # Discover Plugins
    # We look for engines in mti_evo_plugins.engines namespace
    try:
        import pkgutil
        import importlib
        import mti_evo_plugins.engines as plugin_engines
        
        # Determine path
        if hasattr(plugin_engines, "__path__"):
            for _, name, _ in pkgutil.iter_modules(plugin_engines.__path__):
                try:
                    module = importlib.import_module(f"mti_evo_plugins.engines.{name}")
                    # Convention: Engine class name usually matches file name (CamelCase)
                    # or we scan for subclasses of BaseEngine/EngineProtocol
                    
                    # Heuristic: Scan module for classes ending in 'Engine'
                    for attr_name in dir(module):
                        if attr_name.endswith("Engine") and attr_name != "BaseEngine":
                            cls = getattr(module, attr_name)
                            # Register it using lower case name (e.g. QuantumEngine -> quantum)
                            # Or use specific mapping.
                            key = attr_name.replace("Engine", "").lower()
                            EngineRegistry.register(key, cls)
                            
                except Exception as ex:
                    logger.debug(f"Failed to load plugin engine {name}: {ex}")
                        
    except ImportError:
        # mti_evo_plugins might not exist yet
        pass
    except Exception as e:
        logger.warning(f"Plugin discovery error: {e}")

# Run discovery on import? Or lazy?
# Let's be lazy to keep startup fast.
