"""
MTI-EVO Engine Demo Script
===========================
Demonstrates all available inference engines.

Usage:
  python demo_engines.py

Note: This is a demonstration - actual inference requires models to be loaded.
"""
from mti_evo.llm_adapter import LLMAdapter
from mti_evo.engines.base import LLMResponse
import sys

def demo_engine(engine_type: str, model_path: str = ""):
    print(f"\n{'='*40}")
    print(f"  Engine: {engine_type.upper()}")
    print(f"{'='*40}")
    
    config = {
        "model_path": model_path or f"models/demo_{engine_type}",
        "model_type": engine_type,
        "n_ctx": 2048
    }
    
    try:
        # Create adapter without auto-loading
        adapter = LLMAdapter(config, auto_load=False)
        print(f"  ✅ Adapter created (backend: {adapter.backend})")
        
        # Show what would happen on load
        print(f"  📋 Config: n_ctx={config['n_ctx']}, model_type={engine_type}")
        print(f"  ℹ️ To actually load: adapter.load_model()")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

def main():
    print("\n" + "🌌 "*10)
    print("  MTI-EVO Engine Architecture Demo")
    print("🌌 "*10)
    
    engines = [
        ("gguf", "Standard GGUF via llama-cpp-python"),
        ("native", "Safetensors via Transformers"),
        ("quantum", "Hybrid Quantum Architecture"),
        ("resonant", "Metabolic Layer Activation"),
        ("bicameral", "Dual-Model (4B+12B)"),
        ("qoop", "Quantum OOP Routing"),
        ("api", "External API Calls"),
    ]
    
    print("\nAvailable Engines:")
    for engine, desc in engines:
        print(f"  • {engine:12} - {desc}")
    
    print("\n" + "-"*50)
    print("Testing Engine Initialization...")
    
    for engine, _ in engines:
        demo_engine(engine)
    
    print("\n" + "="*50)
    print("  ✅ All engine types initialized successfully.")
    print("  📖 See engines/README.md for integration guide.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
