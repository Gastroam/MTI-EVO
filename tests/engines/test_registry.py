import pytest
from mti_evo.engines.registry import EngineRegistry, discover_engines
from mti_evo.adapters.llm_adapter import LLMAdapter
from mti_evo.engines.gguf_engine import GGUFEngine

def test_registry_discovery():
    """Verify registry finds core and plugin engines."""
    discover_engines()
    
    engines = EngineRegistry.list_engines()
    print(f"Discovered Engines: {engines}")
    
    assert "gguf" in engines, "Core GGUF engine not found"
    assert "api" in engines, "Core API engine not found (if present)"
    # assert "quantum" in engines, "Plugin Quantum engine not found" 
    # check if plugins are importable in test env

def test_llm_adapter_gguf_creation():
    """Verify LLMAdapter creates GGUF engine via Registry."""
    config = {"model_type": "gguf", "model_path": "mock.gguf", "check_deps": False}
    adapter = LLMAdapter(config, auto_load=False)
    adapter.load_model()
    
    assert isinstance(adapter.engine, GGUFEngine)
    assert adapter.engine.backend_name == "gguf"

def test_llm_adapter_auto_detection():
    """Verify Auto-detection logic still works."""
    config = {"model_type": "auto", "model_path": "mock.gguf", "check_deps": False}
    adapter = LLMAdapter(config, auto_load=False)
    adapter.load_model()
    
    assert isinstance(adapter.engine, GGUFEngine)
