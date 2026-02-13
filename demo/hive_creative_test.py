import requests
import json
import uuid
import time

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

def ask_hive(prompt, purpose, intent_data=None):
    if intent_data is None:
        intent_data = {"operation": "infer", "purpose": purpose}
    
    payload = {
        "action": "telepathy",
        "prompt": prompt,
        "max_tokens": 150,
        "nonce": int(time.time() * 1000),
        "session_id": str(uuid.uuid4()),
        "intent": intent_data
    }
    try:
        res = requests.post(BRIDGE_URL, json=payload)
        data = res.json()
        return data.get("response", "").strip(), data.get("expert", "Unknown")
    except Exception as e:
        return f"Error: {e}", "Error"

def run_creative_pipeline():
    print("🎬 === HIVE CREATIVE PIPELINE === 🎬\n")
    
    # 1. SCRIBE: Write the Scene
    print("step 1: [Gemma-Scribe] Writing Scene...")
    scene_prompt = (
        "Write a high-intensity cyberpunk scene.\n"
        "Characters: Kael (hacker) and Unit 734 (rogue AI).\n"
        "Setting: Neon-soaked alleyway in rain.\n"
        "Action: A handshake protocol exchange."
    )
    script, expert = ask_hive(scene_prompt, "script")
    print(f"[{expert}] Output:\n{script}\n")
    
    # 2. DIRECTOR: Visualize the Scene
    print("Step 2: [Gemma-Director] Planning Shots (FFMPEG)...")
    dir_prompt = (
        f"Create a visual shot list and FFMPEG commands to render this scene:\n"
        f"---\n{script[:200]}...\n---"
    )
    shots, expert = ask_hive(dir_prompt, "video")
    print(f"[{expert}] Output:\n{shots}\n")
    
    # 3. CONSENSUS: Greenlight
    print("Step 3: [Gemma-Consensus] Executive Review...")
    review_prompt = (
        f"Review the artistic cohesion.\n"
        f"Script: {script[:50]}...\n"
        f"Visuals: {shots[:50]}...\n"
        f"Verdict: Greenlight or Rewrite?"
    )
    verdict, expert = ask_hive(review_prompt, "vote")
    print(f"[{expert}] Verdict:\n{verdict}\n")

if __name__ == "__main__":
    run_creative_pipeline()
