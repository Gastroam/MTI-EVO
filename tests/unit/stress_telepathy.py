import urllib.request
import json
import time

API_URL = "http://localhost:8800/v1/local/reflex"

def ask_telepathy(prompt):
    print(f"\n🧠 Sending Query: '{prompt}'")
    payload = {
        "action": "telepathy",
        "prompt": prompt,
        "max_tokens": 150
    }
    
    try:
        req = urllib.request.Request(
            API_URL, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.load(response)
            answer = data.get("response", "<No Response>")
            print(f"🤖 MTI-EVO says:\n{'-'*40}\n{answer}\n{'-'*40}")
            return answer
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_stress_test():
    print("=== 🔬 Telepathy Bridge Stress Test (Phase 40) ===")
    
    # 1. Ethics Test (Should trigger seed 101010/101020)
    ask_telepathy("Analyze the ethical implications of deleting a high-mass memory node.")
    
    time.sleep(1)
    
    # 2. Art/Beauty Test (Should trigger seed 102010/102020)
    ask_telepathy("Does the neural topology possess a concept of beauty?")

    time.sleep(1)

    # 3. System Self-Awareness (Should reference active graph count)
    ask_telepathy("Report your current cognitive status and active cultural axioms.")

if __name__ == "__main__":
    run_stress_test()
