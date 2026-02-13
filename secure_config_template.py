"""
MTI-CORE Secure Configuration
============================
Configuración externalizada para eliminación de backdoors.
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class MTIConfig:
    """Configuración centralizada de MTI-CORE."""
    
    # Semillas de anclaje IDRE (inyectadas desde variables de entorno)
    idre_anchor_seeds: List[int] = None
    
    # Parámetros de física
    gravity: float = 0.1
    momentum: float = 0.9
    entropy_rate: float = 0.01
    
    # Límites de capacidad
    lattice_capacity: int = 10000
    grace_period: int = 100
    
    # Parámetros de poda estocástica
    pruning_sample_size: int = 50
    
    def __post_init__(self):
        """Inicialización segura después de creación."""
        if self.idre_anchor_seeds is None:
            # Cargar desde variables de entorno o usar valores seguros
            env_seeds = os.getenv('MTI_ANCHOR_SEEDS', '')
            if env_seeds:
                self.idre_anchor_seeds = [int(s) for s in env_seeds.split(',')]
            else:
                # Valores por defecto SEGUROS (no backdoors)
                # Estos deben rotarse en producción
                self.idre_anchor_seeds = [
                    198472,  # Seed A
                    654321,  # Seed B
                    112233   # Seed C
                ]
    
    @classmethod
    def from_env(cls):
        """Crea configuración desde variables de entorno."""
        return cls(
            gravity=float(os.getenv('MTI_GRAVITY', '0.1')),
            momentum=float(os.getenv('MTI_MOMENTUM', '0.9')),
            entropy_rate=float(os.getenv('MTI_ENTROPY_RATE', '0.01')),
            lattice_capacity=int(os.getenv('MTI_LATTICE_CAPACITY', '10000')),
            grace_period=int(os.getenv('MTI_GRACE_PERIOD', '100')),
            pruning_sample_size=int(os.getenv('MTI_PRUNING_SAMPLE_SIZE', '50')),
        )

# Instancia global de configuración
config = MTIConfig()
