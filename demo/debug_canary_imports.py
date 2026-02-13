import sys
import os
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('.'))

print("--- DEBUG CANARY IMPORTS ---")
try:
    print("1. Importing TelemetrySink...")
    from src.telemetry_sink import TelemetrySink
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL TelemetrySink: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Importing HiveManager...")
    from src.hive_manager import HiveManager
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL HiveManager: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Importing LLMAdapter...")
    from src.llm_adapter import LLMAdapter
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL LLMAdapter: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Importing CognitiveRouter...")
    from src.cognitive_router import CognitiveRouter
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL CognitiveRouter: {e}")
    import traceback
    traceback.print_exc()

try:
    print("5. Importing HiveNode...")
    from src.mti_hive_node import HiveNode
    print("   SUCCESS")
except Exception as e:
    print(f"   FAIL HiveNode: {e}")
    # import traceback
    # traceback.print_exc()

print("--- END DEBUG ---")
