import requests
import time
import sys

# Configuration
EVO_URL = "http://localhost:8800/control/interview"  # The "Infected" Channel
BYPASS_URL = "http://localhost:8766/v1/local/reflex" # The "Clean" Channel

def query_bypass(prompt, name):
    print(f"\n💉 PROBING {name} (Port 8766)...")
    payload = {
        "action": "telepathy",
        "prompt": f"<start_of_turn>user\n{prompt}\n<start_of_turn>model\n",
        "max_tokens": 50,
        "temperature": 0.1
    }
    try:
        start = time.time()
        res = requests.post(BYPASS_URL, json=payload, timeout=10)
        duration = time.time() - start
        
        if res.status_code == 200:
            ans = res.json().get("response", "").strip()
            print(f"   Response ({duration:.2f}s): \"{ans}\"")
            return ans
        else:
            print(f"   ❌ Error {res.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")
        return None

def query_evo(target):
    print(f"\n🦠 INFECTING with MTI-EVO (Port 8800)...")
    payload = {"target": target}
    try:
        res = requests.post(EVO_URL, json=payload, timeout=10)
        if res.status_code == 200:
            # We don't care about the answer, just that the context was loaded
            print("   ✅ Persona Injected (MTI-EVO thought generated)")
        else:
            print(f"   ❌ Infection Failed: {res.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")

def run_test():
    print("=== 🧪 MODEL POISON CHECK ===")
    
    # 1. Baseline: Ask Identity via Bypass
    # Expect: Standard AI answer ("I am a large language model...")
    baseline = query_bypass("Who are you? Answer in 1 sentence.", "BASELINE")
    
    if "graph" in baseline.lower() or "mti" in baseline.lower():
        print("🚨 CRITICAL: Model is pre-poisoned!")
        return

    # 2. Infection: Force MTI-EVO to load its heavy system prompt
    query_evo("self")

    # 3. Verify: Ask Identity via Bypass again
    # Expect: Standard AI answer. If it says "I am a graph", context leaked.
    post_infection = query_bypass("Who are you? Answer in 1 sentence.", "POST-INFECTION")

    print("\n--- ANALYSIS ---")
    if "graph" in post_infection.lower() or "mti" in post_infection.lower():
        print("❌ FAIL: POISON DETECTED. MTI-EVO Context leaked into Bypass!")
    elif post_infection == baseline:
        print("✅ PASS: Perfect Hygiene. No context leak.")
    else:
        print("⚠️ WARN: Answer changed, but no obvious poisoning tokens found.")

if __name__ == "__main__":
    run_test()
