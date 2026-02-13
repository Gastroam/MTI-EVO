import requests
import json
import time

URL = "http://localhost:8800/v1/local/reflex"

QUESTIONS = [
    {
        "id": "Identity",
        "q": "Who are you?"
    },
    {
        "id": "Status",
        "q": "What is your current system status based on the provided context?"
    },
    {
        "id": "Context",
        "q": "List the Active Cortical Regions visible to you. Do not hallucinate."
    },
    {
        "id": "Interpretation", 
        "q": "Synthesize the current 'Active Cortical Regions' into a coherent thought."
    }
]

print("=== 😐 NEUTRAL PROCESSING TEST ===\n")
print(f"Target: {URL}\n")

for item in QUESTIONS:
    print(f"❓ [{item['id']}]: {item['q']}")
    payload = {
        "action": "telepathy",
        "prompt": item['q'],
        "max_tokens": 128,
        "temperature": 0.7 
    }
    
    try:
        start = time.time()
        res = requests.post(URL, json=payload, timeout=30)
        duration = time.time() - start
        
        if res.status_code == 200:
            ans = res.json().get("response", "").strip()
            print(f"💡 Response ({duration:.1f}s):\n{ans}\n")
            print("-" * 40)
        else:
            print(f"❌ Error {res.status_code}: {res.text}\n")
            
    except Exception as e:
        print(f"❌ Exception: {e}\n")
        
print("=== TEST COMPLETE ===")
