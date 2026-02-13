import requests
import json
import uuid
import time

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def ask_expert(question, purpose):
    payload = {
        "action": "telepathy",
        "prompt": question,
        "max_tokens": 100,
        "nonce": int(time.time() * 1000),
        "session_id": str(uuid.uuid4()),
        "intent": {"operation": "infer", "purpose": purpose}
    }
    try:
        res = requests.post(BRIDGE_URL, json=payload)
        return res.json().get("response", "").strip()
    except:
        return "Error"

def run_debate():
    topic = "Is code strictly logical or a form of art?"
    print(f"🎤 DEBATE TOPIC: {topic}\n")

    # 1. Math Expert (Logic)
    print("--- 🧮 Gemma-Math (Logic Argument) ---")
    math_arg = ask_expert(f"Argue strictly logically: {topic}", "math")
    print(math_arg)

    # 2. Dreams Expert (Art)
    print("\n--- 🌙 Gemma-Dreams (Artistic Argument) ---")
    dream_arg = ask_expert(f"Argue poetically: {topic}", "dreams")
    print(dream_arg)

    # 3. Consensus (Verdict)
    print("\n--- ⚖️ Gemma-Consensus (Verdict) ---")
    prompt = (
        f"Review these two arguments.\n"
        f"Argument A (Logic): {math_arg}\n"
        f"Argument B (Art): {dream_arg}\n"
        f"Which view is more comprehensive? Give a VERDICT."
    )
    verdict = ask_expert(prompt, "consensus")
    print(verdict)

if __name__ == "__main__":
    run_debate()
