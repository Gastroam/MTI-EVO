import requests
import time
import json
import uuid

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def send_hive_request(prompt, purpose, expected_expert):
    session_id = str(uuid.uuid4())
    nonce = int(time.time() * 1000)
    
    payload = {
        "action": "telepathy",
        "prompt": prompt,
        "max_tokens": 100,
        "session_id": session_id,
        "nonce": nonce,
        "intent": {
            "operation": "infer",
            "purpose": purpose,
            "scope": {"max_tokens": 100}
        }
    }
    
    print(f"\n--- Request [Purpose: {purpose}] Expecting: {expected_expert} ---")
    try:
        res = requests.post(BRIDGE_URL, json=payload)
        if res.status_code == 200:
            data = res.json()
            expert = data.get("expert", "Unknown")
            print(f"Routed To: {expert}")
            print(f"Response: {data.get('response')[:60]}...")
            
            if expert == expected_expert:
                print("✅ Routing SUCCESS")
            else:
                print(f"❌ Routing FAIL (Got {expert})")
        else:
            print(f"Error: {res.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # 1. Math
    send_hive_request("Prove that sqrt(2) is irrational.", "math", "Gemma-Math")
    
    # 2. Dreams
    send_hive_request("I dreamt of a blue tiger flying over a glass city.", "dreams", "Gemma-Dreams")
    
    # 3. Code (Heuristic: 'debug')
    send_hive_request("Why is this recursion failing?", "debug", "Gemma-Code")
    
    # 4. Consensus
    send_hive_request("Vote on whether to deploy this patch.", "vote", "Gemma-Consensus")
