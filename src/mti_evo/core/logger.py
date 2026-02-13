"""
MTI-EVO Logger
==============
Structured, beautiful logging using Rich.
"""
import logging
from rich.logging import RichHandler
from mti_evo.core.config import MTIConfig

def get_logger(name="MTI-EVO", config: MTIConfig = None):
    # Set level based on config
    level_str = config.log_level if config else "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # We check if root logger already configured to avoid duplicate handlers?
    # Python logging basicConfig does this check if force=False.
    # We use force=True to override.
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True
    )
    
    return logging.getLogger(name)
