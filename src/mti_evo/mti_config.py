"""
MTI-EVO Configuration Module
============================
Central Definition of Biological Laws and System Parameters.
"""
from dataclasses import dataclass

@dataclass
class MTIConfig:
    # --- Biological Physics (The DNA) ---
    gravity: float = 20.0        # Pain / Error Magnitude
    momentum: float = 0.9        # Kinetic Memory / Persistence
    decay_rate: float = 0.15     # Simulated Annealing / Cooling
    initial_lr: float = 0.5      # Neuroplasticity Base Rate
    weight_cap: float = 80.0     # Max Absolute Resonance (Stability Ceiling)
    diminishing_returns: bool = True # Logarithmic Dampening Enabled
    passive_decay_rate: float = 0.00001 # Metabolic Burn (per second) (~1 day)
    random_seed: int = 1337      # Seed used for deterministic experiments
    deterministic: bool = True   # If False, RNG uses entropy from OS
    
    # --- Structural Architecture (The Brain) ---
    capacity_limit: int = 5000   # Max Neurons in Holographic Lattice
    grace_period: int = 100      # 'Infant Mortality' Protection (Cycles)
    embedding_dim: int = 64      # Hebbian Vector Dimension (Phase 10)
    eviction_sample_size: int = 50
    eviction_mode: str = "deterministic_sample"  # full_scan | sample | deterministic_sample
    
    # --- System Ops ---
    log_level: str = "INFO"      # DEBUG, INFO, WARNING, ERROR
    telemetry_enabled: bool = True
    stimulate_return_metrics: bool = False
    
    # --- Security / IDRE ---
    idre_anchor_seeds: tuple = (7245, 8888) # Configured Anchors (Masked in Env in Prod)

    @property
    def seed(self) -> int:
        return self.random_seed
    
    @seed.setter
    def seed(self, value: int):
        self.random_seed = value

@dataclass
class HiveConfig:
    """
    MTI-HIVE Operational Parameters (Layer 4/5)
    Tuned for Stability over Chaos.
    """
    # --- Field Dynamics ---
    query_influence_kappa: float = 0.05  # Low: Nudge, don't slam
    field_noise_sigma: float = 0.001     # Low: Minimal background radiation
    mutation_rate_nu: float = 0.01       # Low: Slow drift
    
    # --- Governance ---
    governance_damping_kappa: float = 0.5 # High: Strong parental control
    pressure_high_threshold: float = 0.8  # Trigger DAMP
    pressure_low_threshold: float = 0.1   # Trigger RELAX
    
    # --- Neurogenesis ---
    neurogenesis_gradient_threshold: float = 0.9 # Very High: Hard to create new concepts
    
    # --- Grounding (The Anchors) ---
    grounding_seeds: list = ("STABILITY", "SAFETY", "CLARITY", "HUMILITY")
