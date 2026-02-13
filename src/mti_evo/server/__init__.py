"""
MTI-EVO Server Module
=====================
Exports server runners.
"""
from .dev_server import run_dev_server
from .substrate import run_substrate_server

# Official runtime entrypoint (mainline)
run_server = run_substrate_server

# Legacy alias for backward compatibility
run_unified_server = run_dev_server

__all__ = [
    "run_server",
    "run_substrate_server",
    "run_dev_server",
    "run_unified_server",
]
