import requests
import json
import uuid
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor

BRIDGE_URL = "http://localhost:8766/v1/local/reflex"

# Mix of intents to hit different experts
INTENTS = [
    {"purpose": "math", "prompt": "Calculate 123 * 456"},
    {"purpose": "dreams", "prompt": "I saw a red door in the sky."},
    {"purpose": "code", "prompt": "Check this loop for errors."},
    {"purpose": "script", "prompt": "Kael hacks the terminal."},
    {"purpose": "video", "prompt": "Cut to black."}
]

RESULTS = {
    "total": 0,
    "success": 0,
    "errors": 0,
    "latencies": []
}

def worker(worker_id):
    scenario = random.choice(INTENTS)
    payload = {
        "action": "telepathy",
        "prompt": scenario["prompt"],
        "max_tokens": 50, # Keep it light for high RPS simulation
        "nonce": int(time.time() * 1000),
        "session_id": str(uuid.uuid4()),
        "intent": {"operation": "infer", "purpose": scenario["purpose"]}
    }
    
    start_t = time.time()
    try:
        res = requests.post(BRIDGE_URL, json=payload, timeout=30)
        latency = (time.time() - start_t) * 1000
        
        if res.status_code == 200:
            RESULTS["success"] += 1
            RESULTS["latencies"].append(latency)
            # print(f"[Worker {worker_id}] ✅ {scenario['purpose']} ({latency:.0f}ms)")
        else:
            RESULTS["errors"] += 1
            print(f"[Worker {worker_id}] ❌ {res.status_code}")
    except Exception as e:
        RESULTS["errors"] += 1
        print(f"[Worker {worker_id}] 💥 {e}")
    finally:
        RESULTS["total"] += 1

def run_stress_test(concurrent_users=50, total_requests=100):
    print(f"🚀 Starting HIVE STRESS TEST")
    print(f"   Users: {concurrent_users}")
    print(f"   Requests: {total_requests}")
    print("------------------------------------------------")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(worker, i) for i in range(total_requests)]
        
        # Wait for all
        for f in futures:
            f.result()
            
    total_time = time.time() - start_time
    avg_latency = sum(RESULTS["latencies"]) / len(RESULTS["latencies"]) if RESULTS["latencies"] else 0
    throughput = RESULTS["success"] / total_time
    
    print("------------------------------------------------")
    print(f"🏁 TEST COMPLETE in {total_time:.2f}s")
    print(f"   Total Requests: {RESULTS['total']}")
    print(f"   Successful:     {RESULTS['success']}")
    print(f"   Errors:         {RESULTS['errors']}")
    print(f"   Avg Latency:    {avg_latency:.2f}ms")
    print(f"   Throughput:     {throughput:.2f} req/s")
    print("------------------------------------------------")

if __name__ == "__main__":
    # Ensure bridge is up manually before running
    run_stress_test(concurrent_users=50, total_requests=100)
