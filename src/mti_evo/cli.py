"""
MTI-EVO CLI (Command Line Interface)
=====================================
Main entry point for the mti-evo command.

Usage:
  mti-evo server       Start the HTTP server
  mti-evo setup        Run interactive setup wizard
  mti-evo config       Show current configuration
  mti-evo engines      List available engines
  mti-evo help         Show this help message
"""
import sys
import argparse


def cmd_server(args):
    """Start the MTI-EVO server."""
    from mti_evo.server import run_server
    port = args.port if hasattr(args, 'port') else 8800
    run_server(port=port)


def cmd_setup(args):
    """Run interactive setup wizard."""
    from mti_evo.setup import main as setup_main
    setup_main()


def cmd_config(args):
    """Show current configuration."""
    from mti_evo.config import print_config_summary
    print_config_summary()


def cmd_engines(args):
    """List available engines."""
    engines = [
        ("gguf", "GGUF models via llama-cpp-python (default)"),
        ("native", "Safetensors via Transformers"),
        ("quantum", "Hybrid Quantum architecture"),
        ("resonant", "Metabolic Layer Activation (sparse loading)"),
        ("bicameral", "Dual-Model (4B+12B parallel streams)"),
        ("qoop", "Quantum OOP (probabilistic routing)"),
        ("hybrid", "Local + API Fusion"),
        ("api", "External API calls"),
    ]
    
    print("\n🌌 MTI-EVO Available Engines\n")
    print("Set via: model_type in mti_config.json\n")
    for engine, desc in engines:
        print(f"  {engine:12} - {desc}")
    print()


def cmd_help(args=None):
    """Show help message."""
    print(__doc__)


def main():
    parser = argparse.ArgumentParser(
        prog="mti-evo",
        description="MTI-EVO: Holographic Lattice Cognitive Mesh"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the HTTP server")
    server_parser.add_argument("-p", "--port", type=int, default=8800, help="Port (default: 8800)")
    server_parser.set_defaults(func=cmd_server)
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_parser.set_defaults(func=cmd_setup)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    config_parser.set_defaults(func=cmd_config)
    
    # Engines command
    engines_parser = subparsers.add_parser("engines", help="List available engines")
    engines_parser.set_defaults(func=cmd_engines)
    
    # Help command (additional to --help)
    help_parser = subparsers.add_parser("help", help="Show help")
    help_parser.set_defaults(func=cmd_help)
    
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        # Default: show help
        parser.print_help()
        print("\nQuick Start:")
        print("  mti-evo setup     # Run setup wizard")
        print("  mti-evo server    # Start server")


if __name__ == "__main__":
    main()
