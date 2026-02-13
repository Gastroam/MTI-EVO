"""
TEST: Infrastructure Hardening
==============================
Verifies:
1. MTIConfig loading and defaults.
2. MTILogger output and levels.
3. Telemetry metrics collection.
4. Integration with HolographicLattice.
"""
import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mti_evo.mti_config import MTIConfig
from mti_evo.mti_logger import get_logger
from mti_evo.mti_core import HolographicLattice

def test_config():
    print("\n[TEST] Configuration Management...")
    # Default
    default_cfg = MTIConfig()
    assert default_cfg.gravity == 20.0, "Default gravity mismatch"
    print("   -> Default Config Loaded OK.")
    
    # Custom
    custom_cfg = MTIConfig(gravity=50.0, capacity_limit=100)
    assert custom_cfg.gravity == 50.0
    assert custom_cfg.capacity_limit == 100
    print("   -> Custom Config Loaded OK.")

def test_logger():
    print("\n[TEST] Logging System...")
    cfg = MTIConfig(log_level="DEBUG")
    logger = get_logger("TestLogger", cfg)
    
    assert logger.getEffectiveLevel() == logging.DEBUG
    print("   -> Logger Level set to DEBUG OK.")
    # We can't easily assert stdout here without detailed capturing, 
    # but successful instantiation is key.

def test_integration_and_telemetry():
    print("\n[TEST] Core Integration & Telemetry...")
    cfg = MTIConfig(
        capacity_limit=5, 
        grace_period=2, 
        telemetry_enabled=True,
        log_level="ERROR" # Quiet logs for this test
    )
    
    lattice = HolographicLattice(config=cfg)
    
    # Verify Injection
    assert lattice.capacity_limit == 5
    assert lattice.telemetry is not None
    print("   -> Config Injected OK.")
    
    # Generate Activity
    print("   -> Generating Neural Activity...")
    lattice.stimulate([1, 2, 3], 0.5)
    
    # Check Telemetry
    snapshot = lattice.telemetry.metrics.snapshot()
    print(f"   -> Telemetry Snapshot: {snapshot}")
    
    assert snapshot['neuron_count'] == 3
    assert snapshot['neurogenesis'] == 3
    assert snapshot['pulses_processed'] == 1
    assert snapshot['latency_ms'] > 0
    print("✅ Telemetry recording accurately.")

if __name__ == "__main__":
    try:
        test_config()
        test_logger()
        test_integration_and_telemetry()
        print("\n[SUCCESS] Infrastructure Verified.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FAILURE] {repr(e)}")
        exit(1)
