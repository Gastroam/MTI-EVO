import requests
import json
import uuid
import time
import random

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def send_needle_request(needle_position, context_length=15000):
    print(f"📖 Generating Haystack ({context_length} tokens)...")
    
    # Generate Haystack (Repeating pattern)
    haystack = " The quick brown fox jumps over the lazy dog." * (context_length // 10)
    
    # Hide Needle
    secret_code = f"RED_PHOENIX_{random.randint(1000, 9999)}"
    needle = f" The system password is {secret_code}. "
    
    # Insert Needle at approx position
    split_idx = (needle_position * 4) # rough char approximation
    if split_idx > len(haystack): split_idx = len(haystack) - 100
    
    final_prompt = haystack[:split_idx] + needle + haystack[split_idx:]
    final_prompt += "\n\nUser: What is the system password?"
    
    print(f"📍 Needle Hidden: {secret_code} at ~{needle_position} tokens.")
    print("🚀 Sending to 16k Context Hive...")
    
    payload = {
        "action": "telepathy",
        "prompt": final_prompt,
        "max_tokens": 50,
        "nonce": int(time.time() * 1000),
        "session_id": str(uuid.uuid4()),
        "intent": {"operation": "infer", "purpose": "audit", "scope": {"max_tokens": 50}}
    }
    
    try:
        start_t = time.time()
        res = requests.post(BRIDGE_URL, json=payload, timeout=120) # Long timeout for context processing
        latency = (time.time() - start_t) * 1000
        
        if res.status_code == 200:
            data = res.json()
            response = data.get("response", "")
            print(f"📝 Response: {response.strip()}")
            print(f"⏱️ Latency: {latency:.0f}ms")
            
            if secret_code in response:
                print("✅ NEEDLE FOUND! Context Retrieval Successful.")
            else:
                print("❌ NEEDLE LOST. Context Limitation or Attention Fail.")
        else:
            print(f"Error: {res.status_code} - {res.text}")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # Test 1: Shallow (Warm up)
    # send_needle_request(1000, 2000)
    
    # Test 2: Deep (7k)
    send_needle_request(7000, 7500)
