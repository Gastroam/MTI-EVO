import requests
import time
import sys

BASE_URL = "http://localhost:8800"

def test_playground():
    print("1. Listing Scripts...")
    try:
        res = requests.get(f"{BASE_URL}/api/playground/scripts")
        if res.status_code == 200:
            scripts = res.json().get("scripts", [])
            print(f"SUCCESS: Found {len(scripts)} scripts.")
            if "debug_seed.py" in scripts:
                print("Confirmed 'debug_seed.py' is available.")
        else:
            print(f"FAIL: Status {res.status_code}")
            return
    except Exception as e:
        print(f"FAIL: Connection refused {e}")
        return

    print("\n2. Running Script (debug_seed.py)...")
    try:
        res = requests.post(f"{BASE_URL}/api/playground/run", json={"script": "debug_seed.py"})
        if res.status_code == 200:
            data = res.json()
            pid = data.get("pid")
            print(f"SUCCESS: Started PID {pid}, Log: {data.get('log_file')}")
            
            # Wait a sec for logs
            time.sleep(1)
            
            print("\n3. Checking Logs...")
            res_log = requests.get(f"{BASE_URL}/api/playground/logs?pid={pid}")
            if res_log.status_code == 200:
                log_data = res_log.json()
                lines = log_data.get("lines", [])
                print(f"SUCCESS: Retrieved {len(lines)} log lines.")
                print("Tail:", "".join(lines[-3:]))
            else:
                print(f"FAIL Log: {res_log.status_code}")
        else:
            print(f"FAIL Run: {res.status_code} {res.text}")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_playground()
