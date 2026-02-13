
import requests
import time
import sys

BASE_URL = "http://localhost:8800"

def test_api():
    print("Testing MTI-EVO Production Refactor API...")
    
    # 1. Check Status (Should be fast)
    try:
        res = requests.get(f"{BASE_URL}/status")
        if res.status_code == 200:
            print("✅ GET /status: OK")
            print(f"   Response: {res.json()}")
        else:
            print(f"❌ GET /status Error: {res.status_code}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. Check Models List
    try:
        res = requests.get(f"{BASE_URL}/api/models")
        if res.status_code == 200:
            models = res.json().get("models", [])
            print("✅ GET /api/models: OK")
            print(f"   Found {len(models)} models: {models[:3]}...")
        else:
            print(f"❌ GET /api/models Error: {res.status_code}")
    except Exception as e:
        print(f"❌ Models Fetch Failed: {e}")

    # 3. Check Settings (Verify Lazy Mode)
    try:
        res = requests.get(f"{BASE_URL}/api/settings")
        if res.status_code == 200:
            config = res.json()
            print("✅ GET /api/settings: OK")
            print(f"   Current Model Path: {config.get('model_path')}")
        else:
            print(f"❌ GET /api/settings Error: {res.status_code}")
    except Exception as e:
        print(f"❌ Settings Fetch Failed: {e}")

    # 4. Test Model Load with Custom Params (K-Cache)
    print("\n4. Testing Model Load with Params (inc. K-Cache)...")
    try:
        payload = {
            "path": "gemma-3-4b-it-q4_0.gguf", # Assuming this exists or mocking behavior
            "n_ctx": 4096,
            "gpu_layers": 33,
            "temperature": 0.8,
            "cache_type_k": "q4_0"
        }
        res = requests.post(f"{BASE_URL}/api/model/load", json=payload)
        if res.status_code == 200:
            data = res.json()
            print("✅ POST /api/model/load: OK")
            returned_config = data.get("config", {})
            print(f"   Backend: {data.get('backend')}")
            print(f"   Config Verification:")
            print(f"     - n_ctx:     {returned_config.get('n_ctx')} (Expected 4096)")
            print(f"     - gpu:       {returned_config.get('gpu_layers')} (Expected 33)")
            print(f"     - k-cache:   {returned_config.get('cache_type_k')} (Expected q4_0)")
        else:
            print(f"❌ POST /api/model/load Error: {res.status_code}")
            print(f"   Response: {res.text}")
    except Exception as e:
        print(f"❌ Load Test Failed: {e}")

if __name__ == "__main__":
    test_api()
