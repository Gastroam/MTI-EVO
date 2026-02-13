import pytest
from mti_evo.engines.protocol import EngineProtocol, EngineResult
from mti_evo.engines.gguf_engine import GGUFEngine
from mti_evo.engines.base import BaseEngine

# Mock config for testing
MOCK_CONFIG = {
    "model_path": "test_model.gguf",
    "n_ctx": 512,
    "check_deps": False
}

def test_gguf_engine_protocol_adherence():
    """Verify GGUFEngine implements EngineProtocol structure."""
    engine = GGUFEngine(MOCK_CONFIG)
    
    # Check inheritance/typing (runtime check)
    assert isinstance(engine, BaseEngine)
    # Protocol compliance
    assert hasattr(engine, "load")
    assert hasattr(engine, "infer")
    assert hasattr(engine, "embed")
    assert hasattr(engine, "unload")
    assert hasattr(engine, "capabilities") # Property
    
    # Check capabilities structure
    caps = engine.capabilities
    assert isinstance(caps, dict)
    assert "device" in caps

def test_gguf_engine_api_signature():
    """Verify method signatures match expectation."""
    engine = GGUFEngine(MOCK_CONFIG)
    
    # Infer signature check
    # We can't easily check runtime typing, but we can call it (it will fail internal logic if no model, but check return type)
    
    # Mock LLM to avoid actual load
    class MockLLM:
        def __call__(self, *args, **kwargs):
            return {
                "choices": [{"text": "Hello World"}],
                "usage": {"completion_tokens": 2}
            }
        def create_embedding(self, text):
            return {"data": [{"embedding": [0.1, 0.2]}]}
            
    engine.llm = MockLLM()
    
    result = engine.infer("Test prompt", max_tokens=10)
    assert isinstance(result, EngineResult)
    assert result.text == "Hello World"
    assert result.tokens == 2
    assert result.metrics is not None
    
    # Check embed
    vec = engine.embed("foo")
    assert vec == [0.1, 0.2]
    
    engine.unload()
    assert engine.llm is None
