#!/usr/bin/env python
"""
MTI-EVO Launcher
================
Single entry point to start the MTI-EVO Control Plane.

Usage:
    python run_evo.py           # Start on default port 8800
    python run_evo.py --port 9000  # Custom port
"""
import sys
import os

# Ensure the src package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mti_evo.server import run_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MTI-EVO: The Bicameral Mind Control Plane",
        epilog="Visit /status for health check, /control/dream for dreaming."
    )
    parser.add_argument("--port", type=int, default=8800, help="Server port (default: 8800)")
    args = parser.parse_args()
    
    print("""
    ███╗   ███╗████████╗██╗      ███████╗██╗   ██╗ ██████╗ 
    ████╗ ████║╚══██╔══╝██║      ██╔════╝██║   ██║██╔═══██╗
    ██╔████╔██║   ██║   ██║█████╗█████╗  ██║   ██║██║   ██║
    ██║╚██╔╝██║   ██║   ██║╚════╝██╔══╝  ╚██╗ ██╔╝██║   ██║
    ██║ ╚═╝ ██║   ██║   ██║      ███████╗ ╚████╔╝ ╚██████╔╝
    ╚═╝     ╚═╝   ╚═╝   ╚═╝      ╚══════╝  ╚═══╝   ╚═════╝ 
                    The Bicameral Mind v2.0
    """)
    
    run_server(args.port)
