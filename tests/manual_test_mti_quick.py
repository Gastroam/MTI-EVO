"""
MTI-EVO Quick Test Script
==========================
Run this to verify your MTI-EVO installation is working.

Usage:
  python test_mti_quick.py

Expected: All tests pass with green checkmarks.
"""
import requests
import time
import sys

SERVER_URL = "http://localhost:8800"

def test(name, condition, details=""):
    if condition:
        print(f"  ✅ {name}")
        return True
    else:
        print(f"  ❌ {name}" + (f" - {details}" if details else ""))
        return False

def main():
    print("\n" + "="*50)
    print("  MTI-EVO Quick Test Suite")
    print("="*50 + "\n")
    
    passed = 0
    failed = 0
    
    # 1. Server Reachability
    print("🔹 Server Connection")
    try:
        r = requests.get(f"{SERVER_URL}/help", timeout=5)
        if test("Server reachable", r.status_code == 200):
            passed += 1
        else:
            failed += 1
    except requests.exceptions.ConnectionError:
        test("Server reachable", False, "Is the server running? Start with: python -m mti_evo.server")
        failed += 1
        print("\n⚠️ Server not running. Start it first.")
        sys.exit(1)
    
    # 2. Help Endpoint
    print("\n🔹 Help API")
    data = r.json()
    if test("Version present", "version" in data):
        passed += 1
    else:
        failed += 1
    if test("Engines listed", "engines" in data and len(data["engines"]) >= 5):
        passed += 1
    else:
        failed += 1
    if test("API tiers defined", "tiers" in data):
        passed += 1
    else:
        failed += 1
    
    # 3. Status Endpoint
    print("\n🔹 Brain Status")
    r = requests.get(f"{SERVER_URL}/status")
    data = r.json()
    if test("Status online", data.get("status") == "online"):
        passed += 1
    else:
        failed += 1
    if test("Neurons loaded", data.get("neurons", 0) > 0):
        passed += 1
    else:
        failed += 1
    
    # 4. Graph Topology
    print("\n🔹 Graph Topology")
    r = requests.get(f"{SERVER_URL}/api/graph")
    data = r.json()
    if test("Nodes present", "nodes" in data):
        passed += 1
    else:
        failed += 1
    if test("Edges present", "edges" in data):
        passed += 1
    else:
        failed += 1
    
    # 5. Metrics
    print("\n🔹 Telemetry")
    r = requests.get(f"{SERVER_URL}/api/metrics")
    data = r.json()
    if test("History available", "history" in data):
        passed += 1
    else:
        failed += 1
    
    # Summary
    print("\n" + "="*50)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed")
    if failed == 0:
        print("  🎉 All tests passed! MTI-EVO is ready.")
    else:
        print(f"  ⚠️ {failed} test(s) failed. Check server logs.")
    print("="*50 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
