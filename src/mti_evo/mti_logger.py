"""
MTI-EVO Logger
==============
Structured, beautiful logging using Rich.
"""
import logging
from rich.logging import RichHandler
from mti_evo.mti_config import MTIConfig

def get_logger(name="MTI-EVO", config: MTIConfig = None):
    # Set level based on config
    level_str = config.log_level if config else "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True
    )
    
    return logging.getLogger(name)
