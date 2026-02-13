"""
MTI-EVO Setup Wizard (CLI)
===========================
Interactive configuration wizard for MTI-EVO.

Usage:
  python -m mti_evo.setup

This creates/updates mti_config.json with your settings.
"""
import os
import sys
import json


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_screen()
    print("=" * 60)
    print("    🧠 MTI-EVO Configuration Wizard")
    print("=" * 60)
    print()


def prompt(question: str, default: str = "", secret: bool = False) -> str:
    """Prompt user for input with optional default."""
    if default:
        display = f"{question} [{default}]: "
    else:
        display = f"{question}: "
    
    if secret:
        import getpass
        value = getpass.getpass(display)
    else:
        value = input(display)
    
    return value.strip() if value.strip() else default


def prompt_choice(question: str, options: list, default: int = 0) -> str:
    """Prompt user to choose from options."""
    print(f"\n{question}")
    for i, opt in enumerate(options):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i+1}] {opt}")
    
    while True:
        choice = input(f"Choose [1-{len(options)}] (default: {default+1}): ").strip()
        if not choice:
            return options[default]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice.")


def scan_models(models_dir: str = "models") -> list:
    """Scan for available models."""
    models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            path = os.path.join(models_dir, f)
            if f.endswith(".gguf"):
                models.append(f)
            elif os.path.isdir(path):
                # Check for safetensors
                if any(sf.endswith(".safetensors") for sf in os.listdir(path) if os.path.isfile(os.path.join(path, sf))):
                    models.append(f)
    return models


def main():
    from mti_evo.config import load_config, save_config, DEFAULT_CONFIG
    
    print_header()
    print("This wizard will help you configure MTI-EVO.\n")
    print("Configuration is saved to: mti_config.json")
    print("API keys are stored in environment variables.\n")
    
    input("Press Enter to continue...")
    
    config = load_config()
    
    # === Step 1: Engine Selection ===
    print_header()
    print("STEP 1: Choose Your Inference Engine\n")
    
    engines = [
        "gguf - Local GGUF model (recommended for most users)",
        "native - Local Safetensors (requires torch)",
        "hybrid - Local + Cloud API fusion (best of both)",
        "api - Cloud API only (no local model)",
        "quantum - Advanced: Hybrid Quantum architecture",
        "resonant - Advanced: Metabolic Layer Activation",
        "bicameral - Advanced: Dual-Model (4B+12B)",
        "qoop - Experimental: Quantum OOP routing"
    ]
    
    engine_choice = prompt_choice("Select engine type:", engines, 0)
    engine_type = engine_choice.split(" - ")[0]
    config["model_type"] = engine_type
    
    # === Step 2: Model Path ===
    print_header()
    print("STEP 2: Model Configuration\n")
    
    if engine_type in ["gguf", "native", "hybrid", "bicameral", "resonant", "quantum"]:
        models_dir = "models"
        available = scan_models(models_dir)
        
        if available:
            print(f"Found models in {models_dir}/:")
            for m in available:
                print(f"  • {m}")
            print()
        
        model_path = prompt("Local model path", config.get("model_path", "models/gemma-3-4b-it-q4_0.gguf"))
        config["model_path"] = model_path
    
    # === Step 3: API Configuration (if hybrid or api) ===
    if engine_type in ["hybrid", "api"]:
        print_header()
        print("STEP 3: API Configuration\n")
        
        providers = ["openai", "anthropic", "google", "ollama"]
        provider = prompt_choice("Select API provider:", providers, 0)
        config["api_provider"] = provider
        
        # Model name
        default_models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-1.5-flash",
            "ollama": "llama3.2"
        }
        api_model = prompt(f"API model name", default_models.get(provider, ""))
        config["api_model"] = api_model
        
        # API Key
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": None
        }
        
        env_key = env_keys.get(provider)
        if env_key:
            existing = os.environ.get(env_key, "")
            if existing:
                print(f"\n✅ {env_key} is already set in environment.")
            else:
                print(f"\n⚠️ {env_key} not found in environment.")
                print(f"   Set it with: export {env_key}=your_key")
                print(f"   Or add to .env file.")
                
                set_now = prompt("Set API key now? (y/n)", "n")
                if set_now.lower() == "y":
                    api_key = prompt(f"Enter {provider.title()} API key", secret=True)
                    if api_key:
                        # Write to .env file
                        env_file = ".env"
                        with open(env_file, "a") as f:
                            f.write(f"\n{env_key}={api_key}\n")
                        print(f"   ✅ Saved to {env_file}")
        
        # Hybrid mode
        if engine_type == "hybrid":
            modes = ["local_first", "api_first", "parallel", "escalate"]
            mode = prompt_choice("Hybrid mode:", modes, 0)
            config["mode"] = mode
    
    # === Step 4: Performance Settings ===
    print_header()
    print("STEP 4: Performance Settings\n")
    
    n_ctx = prompt("Context window size", str(config.get("n_ctx", 4096)))
    config["n_ctx"] = int(n_ctx)
    
    gpu_layers = prompt("GPU layers (-1 = all)", str(config.get("gpu_layers", -1)))
    config["gpu_layers"] = int(gpu_layers)
    
    temperature = prompt("Temperature", str(config.get("temperature", 0.7)))
    config["temperature"] = float(temperature)
    
    # === Save Configuration ===
    print_header()
    print("CONFIGURATION SUMMARY\n")
    print("-" * 40)
    
    for key, value in config.items():
        if key != "engine_defaults":
            print(f"  {key}: {value}")
    
    print("-" * 40)
    
    confirm = prompt("\nSave configuration? (y/n)", "y")
    if confirm.lower() == "y":
        save_config(config)
        print("\n✅ Configuration saved to mti_config.json")
        print("\nTo start the server:")
        print("  python -m mti_evo.server --port 8800")
    else:
        print("\n❌ Configuration not saved.")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
