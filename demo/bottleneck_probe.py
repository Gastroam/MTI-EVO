import requests
import time
import threading
import json
import uuid

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def send_request(name, prompt_len, expected_latency_threshold):
    session_id = str(uuid.uuid4())
    nonce = int(time.time() * 1000)
    prompt = "A" * prompt_len
    
    intent = {"operation": "infer", "purpose": "audit"}
    payload = {
        "action": "telepathy", 
        "prompt": prompt, 
        "max_tokens": 10, 
        "nonce": nonce, 
        "session_id": session_id,
        "intent": intent
    }
    
    print(f"[{name}] Sending (len={prompt_len})...")
    start_t = time.time()
    try:
        res = requests.post(BRIDGE_URL, json=payload, timeout=60)
        end_t = time.time()
        latency = (end_t - start_t) * 1000
        print(f"[{name}] Done. Latency: {latency:.2f}ms")
        return latency
    except Exception as e:
        print(f"[{name}] Failed: {e}")
        return 99999

def run_probe():
    print("🔬 BOTTLENECK PROBE: Head-of-Line Blocking Analysis")
    
    # 1. Start Heavy Request in background
    # 5000 chars should trigger some processing time in simulation or real model
    t_heavy = threading.Thread(target=send_request, args=("HEAVY", 5000, 2000))
    t_heavy.start()
    
    time.sleep(0.1) # Small delay to ensure HEAVY hits the socket first
    
    # 2. Start Light Request immediately
    # Should be instant if async, blocked if sync
    latency_light = send_request("LIGHT", 10, 100)
    
    t_heavy.join()
    
    print("\n--- ANALYSIS ---")
    if latency_light > 1000:
        print("🔴 BLOCKING DETECTED: Light request waited for Heavy request.")
        print("   Reason: SimpleHTTPRequestHandler is single-threaded or LLMAdapter Lock is global.")
    else:
        print("🟢 NON-BLOCKING: Light request slipped through.")

if __name__ == "__main__":
    run_probe()
