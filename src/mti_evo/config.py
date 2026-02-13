"""
MTI-EVO Configuration System
=============================
Priority: 1) Environment Variables  2) mti_config.json  3) Defaults

Environment Variables:
- MTI_MODEL_PATH, MTI_MODEL_TYPE, MTI_N_CTX, MTI_GPU_LAYERS, MTI_TEMPERATURE
- OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (for hybrid/api engines)
"""
import json
import os
from typing import Any, Dict, Mapping, Union

from mti_evo.core.config import MTIConfig

CONFIG_FILE = "mti_config.json"

# Load .env file if exists
def load_dotenv():
    """Load environment variables from .env file."""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Call on module load
load_dotenv()

DEFAULT_CONFIG = {
    "model_path": "models/gemma-3-4b-it-q4_0.gguf",
    "fast_model_path": "models/gemma-3-4b-unq",
    "model_type": "auto",
    "n_ctx": 8192,
    "temperature": 0.7,
    "max_tokens": 1024,
    "gpu_layers": -1,
    "api_provider": "openai",
    "api_model": "gpt-4o-mini",
    "mode": "local_first",
    "enable_hive": False,
    "enable_ghost": False,
    "engine_defaults": {
        "quantum": {"n_ctx": 8192, "gpu_layers": -1, "temperature": 0.7, "cache_type_k": "f16"},
        "gguf": {"n_ctx": 4096, "gpu_layers": 33, "temperature": 0.8, "cache_type_k": "q8_0"},
        "native": {"n_ctx": 2048, "gpu_layers": 0, "temperature": 0.7, "cache_type_k": "f16"},
        "api": {"n_ctx": 32768, "gpu_layers": 0, "temperature": 1.0, "cache_type_k": "f16"},
        "hybrid": {"n_ctx": 4096, "gpu_layers": -1, "temperature": 0.7, "mode": "local_first"},
        "auto": {"n_ctx": 4096, "gpu_layers": -1, "temperature": 0.7, "cache_type_k": "f16"}
    }
}

# Environment variable overrides
ENV_MAPPING = {
    "MTI_MODEL_PATH": ("model_path", str),
    "MTI_MODEL_TYPE": ("model_type", str),
    "MTI_N_CTX": ("n_ctx", int),
    "MTI_GPU_LAYERS": ("gpu_layers", int),
    "MTI_TEMPERATURE": ("temperature", float),
    "MTI_API_PROVIDER": ("api_provider", str),
    "MTI_API_MODEL": ("api_model", str),
    "MTI_MODE": ("mode", str),
}


def load_config() -> MTIConfig:
    """Load configuration with priority: ENV > file > defaults."""
    # Start with defaults
    config = dict(DEFAULT_CONFIG)
    
    # Override with file config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"[Config] ⚠️ Failed to load {CONFIG_FILE}: {e}")
    
    # Override with environment variables
    for env_key, (config_key, converter) in ENV_MAPPING.items():
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                config[config_key] = converter(env_value)
            except ValueError:
                pass
    
    # Inject API keys from environment
    config["_api_keys"] = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "google": os.environ.get("GOOGLE_API_KEY", ""),
    }
    
    return MTIConfig.from_dict(config)


def _json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(list(value))
    if isinstance(value, tuple):
        return list(value)
    return value


def save_config(new_config: Union[MTIConfig, Mapping[str, Any]]):
    """Save configuration to file (excludes API keys for security)."""
    if isinstance(new_config, MTIConfig):
        config_dict = new_config.to_dict(include_private=True)
    else:
        config_dict = dict(new_config)

    # Remove sensitive data before saving
    safe_config = {k: _json_safe(v) for k, v in config_dict.items() if not k.startswith("_")}
    
    # Merge with existing to prevent data loss
    current = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                current = json.load(f)
        except:
            pass
    
    updated = {**current, **safe_config}
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(updated, f, indent=2)
    
    return MTIConfig.from_dict(updated)


def get_api_key(provider: str) -> str:
    """Get API key for a provider from environment."""
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    return os.environ.get(env_keys.get(provider, ""), "")


def print_config_summary():
    """Print current configuration (for debugging)."""
    config = load_config()
    print("\n[MTI-EVO Configuration]")
    print("-" * 40)
    for key, value in config.to_dict().items():
        if key == "engine_defaults":
            continue
        print(f"  {key}: {value}")
    
    # API key status
    print("\n[API Keys]")
    for provider in ["openai", "anthropic", "google"]:
        key = get_api_key(provider)
        status = "set" if key else "not set"
        print(f"  {provider}: {status}")
    print("-" * 40)

