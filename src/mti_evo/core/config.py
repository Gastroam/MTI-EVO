"""
MTI-EVO Configuration Module
============================
Central definition of system parameters and runtime settings.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Mapping, Optional


def _default_engine_defaults() -> Dict[str, Dict[str, Any]]:
    return {
        "quantum": {"n_ctx": 8192, "gpu_layers": -1, "temperature": 0.7, "cache_type_k": "f16"},
        "gguf": {"n_ctx": 4096, "gpu_layers": 33, "temperature": 0.8, "cache_type_k": "q8_0"},
        "native": {"n_ctx": 2048, "gpu_layers": 0, "temperature": 0.7, "cache_type_k": "f16"},
        "api": {"n_ctx": 32768, "gpu_layers": 0, "temperature": 1.0, "cache_type_k": "f16"},
        "hybrid": {"n_ctx": 4096, "gpu_layers": -1, "temperature": 0.7, "mode": "local_first"},
        "auto": {"n_ctx": 4096, "gpu_layers": -1, "temperature": 0.7, "cache_type_k": "f16"},
    }


@dataclass
class MTIConfig:
    # --- Biological Physics (The DNA) ---
    gravity: float = 20.0
    momentum: float = 0.9
    decay_rate: float = 0.15
    initial_lr: float = 0.5
    weight_cap: float = 80.0
    diminishing_returns: bool = True
    passive_decay_rate: float = 0.00001
    random_seed: int = 1337
    deterministic: bool = True

    # --- Structural Architecture (The Brain) ---
    capacity_limit: int = 5000
    grace_period: int = 100
    embedding_dim: int = 64
    eviction_sample_size: int = 50
    eviction_mode: str = "deterministic_sample"

    # --- System Ops ---
    log_level: str = "INFO"
    telemetry_enabled: bool = True
    stimulate_return_metrics: bool = False

    # --- Security / IDRE ---
    idre_anchor_seeds: tuple = (7245, 8888)
    anchor_file: Optional[str] = None
    anchor_reinforcement_freq: int = 0
    pinned_seeds: set = field(default_factory=set)
    anchor_saturation_threshold: float = 0.95

    # --- Persistence ---
    persistence_dir: str = "cortex_data"
    mmap_capacity: int = 1024 * 1024

    # --- Runtime / Engine ---
    model_path: str = "models/gemma-3-4b-it-q4_0.gguf"
    fast_model_path: str = "models/gemma-3-4b-unq"
    model_type: str = "auto"
    n_ctx: int = 8192
    temperature: float = 0.7
    max_tokens: int = 1024
    gpu_layers: int = -1
    api_provider: str = "openai"
    api_model: str = "gpt-4o-mini"
    mode: str = "local_first"
    engine_defaults: Dict[str, Dict[str, Any]] = field(default_factory=_default_engine_defaults)
    check_deps: bool = False
    n_batch: int = 512
    _api_keys: Dict[str, str] = field(default_factory=dict, repr=False)

    # Plugins
    enable_hive: bool = False
    hive_config: Any = None
    enable_ghost: bool = False

    # Persistence Injection
    persistence_manager: Any = None

    @property
    def seed(self) -> int:
        return self.random_seed

    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        out = asdict(self)
        out["pinned_seeds"] = sorted(list(self.pinned_seeds))
        out["idre_anchor_seeds"] = list(self.idre_anchor_seeds)
        if not include_private:
            out = {k: v for k, v in out.items() if not k.startswith("_")}
        return out

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "MTIConfig":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data

        valid = {f.name for f in fields(cls)}
        payload: Dict[str, Any] = {}
        for key, value in dict(data).items():
            if key in valid:
                payload[key] = value

        if isinstance(payload.get("pinned_seeds"), list):
            payload["pinned_seeds"] = set(payload["pinned_seeds"])
        if isinstance(payload.get("idre_anchor_seeds"), list):
            payload["idre_anchor_seeds"] = tuple(payload["idre_anchor_seeds"])

        return cls(**payload)

    # Compatibility helpers while older call sites migrate away from dict-style access.
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def items(self):
        return self.to_dict(include_private=True).items()

    def update(self, other: Optional[Any] = None, **kwargs: Any) -> None:
        updates: Dict[str, Any] = {}

        if other:
            if isinstance(other, MTIConfig):
                updates.update(other.to_dict(include_private=True))
            elif isinstance(other, Mapping):
                updates.update(dict(other))
            else:
                updates.update(vars(other))

        updates.update(kwargs)

        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
