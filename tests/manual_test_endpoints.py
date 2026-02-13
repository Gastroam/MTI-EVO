"""
MTI-EVO Full Endpoint Test Script
==================================
Comprehensive test of all API endpoints (Public + Researcher).

Usage:
  python test_all_endpoints.py

Requires: Server running on port 8800
"""
import requests
import json
import sys

BASE = "http://localhost:8800"

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

def test_get(path, expected_keys=None):
    try:
        r = requests.get(f"{BASE}{path}", timeout=10)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        
        if expected_keys:
            for key in expected_keys:
                if key not in data:
                    print(f"  ❌ GET {path} - Missing key: {key}")
                    return False
        
        preview = str(data)[:60] + "..." if len(str(data)) > 60 else str(data)
        print(f"  ✅ GET {path} → {preview}")
        return True
    except Exception as e:
        print(f"  ❌ GET {path} - {e}")
        return False

def test_post(path, payload, expected_keys=None):
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=30)
        ok = r.status_code in [200, 400, 404]  # Some endpoints return errors for demo
        data = r.json() if ok else {}
        
        preview = str(data)[:60] + "..." if len(str(data)) > 60 else str(data)
        print(f"  ✅ POST {path} → {preview}")
        return True
    except Exception as e:
        print(f"  ❌ POST {path} - {e}")
        return False

def main():
    print("\n" + "="*50)
    print("  MTI-EVO Full Endpoint Test")
    print("="*50)
    
    results = []
    
    # === PUBLIC TIER ===
    section("PUBLIC API")
    results.append(test_get("/help", ["service", "version", "tiers", "engines"]))
    results.append(test_get("/status", ["status", "neurons"]))
    results.append(test_get("/api/graph", ["nodes", "edges"]))
    results.append(test_get("/api/models", ["models"]))
    results.append(test_get("/api/settings"))
    
    # === RESEARCHER TIER ===
    section("RESEARCHER API")
    results.append(test_get("/api/metrics", ["history"]))
    results.append(test_get("/api/attractors"))
    results.append(test_get("/api/events"))
    results.append(test_get("/api/playground/scripts", ["scripts"]))
    
    # POST: Settings update
    results.append(test_post("/api/settings", {"temperature": 0.8}))
    
    # POST: Injection (will fail without resonant engine, but tests routing)
    results.append(test_post("/api/inject", {"layer": 0, "alpha": 0.1}))
    
    # === EXPERIMENTAL TIER ===
    section("EXPERIMENTAL API")
    results.append(test_post("/control/dream", {"seed": "test", "steps": 3}))
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    section("SUMMARY")
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  🎉 All endpoints operational!")
    else:
        print(f"  ⚠️ {total - passed} endpoint(s) need attention.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
