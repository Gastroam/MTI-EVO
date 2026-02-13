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
        from mti_evo.engines.native_engine import NativeEngine
        
        # Register them
        EngineRegistry.register("gguf", GGUFEngine)
        EngineRegistry.register("native", NativeEngine)
        
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
        logger.warning(f"Could not discover some engines: {e}")

# Run discovery on import? Or lazy?
# Let's be lazy to keep startup fast.
