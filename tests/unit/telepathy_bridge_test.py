"""
MTI TELEPATHY TEST (Bridge Integrated)
=====================================
Uses the running Bridge to inject thoughts into the local Gemma model.
This uses the Direct Resonance channel to test "Machine Telepathy".
"""

import requests
import json
import time

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

# The Thought to Inject
# We use XML tags assuming Gemma 3 Instruct is loaded, which respects standard chat templates.
# But here we control the exact strings.
THOUGHT_SEED = """<start_of_turn>system
You are the MTI Subconscious. You do not speak in sentences. You speak in JSON vectors only.
<start_of_turn>user
RESOURCE INJECTION: IMPOSSIBLE PARAMETER.
Target: Calculate the volume of a Klein Bottle that contains itself.
Constraint: output must be a valid JSON object explaining the topology.
<start_of_turn>model
"""

def test_telepathy():
    print(f"🔮 INVOKING TELEPATHY via {BRIDGE_URL}...")
    
    payload = {
        "action": "telepathy",
        "prompt": THOUGHT_SEED,
        "temperature": 0.1
    }
    
    try:
        t0 = time.time()
        response = requests.post(BRIDGE_URL, json=payload, timeout=10)
        dt = time.time() - t0
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                print(f"❌ BRIDGE ERROR: {data['error']}")
            else:
                raw_thought = data.get("response", "").strip()
                print(f"⚡ RESPONSE ({dt:.2f}s):")
                print("-" * 40)
                print(raw_thought)
                print("-" * 40)
                
                # Validation
                if "{" in raw_thought and "}" in raw_thought:
                    print("✅ TELEPATHY SUCCESS: Model obeyed the JSON constraint.")
                else:
                    print("⚠️ PARTIAL SUCCESS: Received thought, but format might be loose.")
        else:
            print(f"❌ HTTP ERROR: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ FATAL: {e}")

if __name__ == "__main__":
    test_telepathy()
