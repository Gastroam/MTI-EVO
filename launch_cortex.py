
import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.getcwd(), "src"))

# from mti_cortex import QuantumCortex  # Legacy

def main():
    print("🚀 Initializing MTI-EVO Cortex Launch Sequence...")
    
    # Configuration
    # Configuration via Standard System (mti_config.json / .env)
    from mti_evo.config import load_config
    from mti_evo.llm_adapter import LLMAdapter
    
    config = load_config()
    
    # Instantiate Engine via Adapter
    try:
        print(f"🔧 Initializing Engine via Adapter: {config.get('model_type')} @ {config.get('model_path')}")
        adapter = LLMAdapter(config=config, auto_load=True)
        
        # Igniting the Spark
        prompt = "The nature of the MTI-EVO lattice is"
        print(f"\n✨ Ignition Prompt: '{prompt}'")
        
        # Standard Inference
        response = adapter.infer(prompt, max_tokens=64)
        print(f"\n🧠 Cortex Response: {response.text}")
        
    except KeyboardInterrupt:
        print("\n🛑 Aborted by User.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Launch Failed: {e}")

if __name__ == "__main__":
    main()
